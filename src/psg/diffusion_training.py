"""Training loop for the masked-diffusion denoiser (NELBO of psg.models.masked_diffusion).

AdamW with linear warm-up, gradient clipping, and an exponential moving average (EMA) of the weights; the EMA
weights are the ones evaluated and kept. Validation: NELBO per site on the validation set with FIXED noise
(same times and masks at every evaluation), so successive epochs are compared on identical draws.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .models.masked_diffusion import nelbo_per_sequence, sample_times, save_diffusion


@dataclass
class DiffusionTrainConfig:
    epochs: int = 100
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    val_repeats: int = 4          # noise draws per validation sequence
    patience: int = 10
    max_steps: int | None = None
    amp: bool = False             # mixed precision (fp16 autocast + gradient scaling), CUDA only
    save_every: int = 10          # snapshot of the EMA model every N epochs (checkpoints/epoch_XXX.pt); 0 = never
    seed: int = 0


class EMA:
    """shadow = decay * shadow + (1 - decay) * weights, after every optimiser step."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)


@torch.no_grad()
def evaluate_nelbo(model, S: torch.Tensor, repeats: int = 4, seed: int = 12345, batch_size: int = 2048) -> float:
    """Mean NELBO per site (nats) with fixed noise (seeded generator on S.device)."""
    model.eval()
    g_cpu = torch.Generator().manual_seed(seed)
    g_dev = torch.Generator(device=S.device).manual_seed(seed)
    tot, n = 0.0, 0
    for _ in range(repeats):
        for k in range(0, len(S), batch_size):
            x0 = S[k:k + batch_size]
            t = sample_times(len(x0), generator=g_cpu, device=S.device)
            tot += nelbo_per_sequence(model, x0, t, model.mask_id, g_dev).sum().item()
            n += x0.numel()
    return tot / n


def train_diffusion(model, S_train: np.ndarray, S_val: np.ndarray, cfg: DiffusionTrainConfig,
                    device: torch.device | None = None, verbose: bool = True,
                    checkpoint_dir=None, resume: bool = False) -> tuple[dict, torch.nn.Module]:
    """Returns (history, best EMA model).

    Checkpoints (if checkpoint_dir is given):
        model.pt                  best EMA model so far, rewritten whenever the validation NELBO improves
        last.pt                   full training state after every epoch (weights, EMA, optimiser, scaler, schedule,
                                  RNG states, history): resume=True continues from it
        checkpoints/epoch_XXX.pt  EMA model every cfg.save_every epochs (snapshots of the training trajectory)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    g_cpu = torch.Generator().manual_seed(cfg.seed)
    g_dev = torch.Generator(device=device).manual_seed(cfg.seed)
    model.to(device).train()
    ema = EMA(model, cfg.ema_decay)
    Str = torch.as_tensor(S_train, dtype=torch.long, device=device)
    Sva = torch.as_tensor(S_val, dtype=torch.long, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup_steps)))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist = {"config": asdict(cfg), "epoch": [], "train_nelbo_site": [], "val_nelbo_site": [], "seconds": []}
    best, best_state, bad, step, t0, start_epoch = np.inf, None, 0, 0, time.time(), 0
    L = Str.shape[1]
    ckdir = None if checkpoint_dir is None else __import__("pathlib").Path(checkpoint_dir)
    if ckdir is not None:
        (ckdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if resume and ckdir is not None and (ckdir / "last.pt").exists():
        st = torch.load(ckdir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        ema.shadow.load_state_dict(st["ema"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        g_cpu.set_state(st["g_cpu"].cpu())          # RNG states are CPU byte tensors, even for a CUDA generator
        g_dev.set_state(st["g_dev"].cpu())
        hist, best, best_state, bad, step = st["hist"], st["best"], st["best_state"], st["bad"], st["step"]
        start_epoch = st["epoch"] + 1
        t0 = time.time() - (hist["seconds"][-1] if hist["seconds"] else 0.0)
        if verbose:
            print(f"resumed from epoch {st['epoch']} (best val {best:.4f})", flush=True)
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        perm = torch.randperm(len(Str), generator=g_cpu).to(device)
        tot, nb = 0.0, 0
        for k in range(0, len(Str), cfg.batch_size):
            x0 = Str[perm[k:k + cfg.batch_size]]
            t = sample_times(len(x0), generator=g_cpu, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = nelbo_per_sequence(model, x0, t, model.mask_id, g_dev).mean()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            tot, nb, step = tot + loss.item(), nb + 1, step + 1
            if cfg.max_steps is not None and step >= cfg.max_steps:
                break
        val = evaluate_nelbo(ema.shadow, Sva, cfg.val_repeats)
        hist["epoch"].append(epoch)
        hist["train_nelbo_site"].append(tot / nb / L)
        hist["val_nelbo_site"].append(val)
        hist["seconds"].append(round(time.time() - t0, 1))
        if verbose:
            print(f"epoch {epoch:3d}  train NELBO/site {tot / nb / L:.4f}  val NELBO/site (EMA) {val:.4f}  "
                  f"({hist['seconds'][-1]}s)", flush=True)
        improved = val < best - 1e-4
        if improved:
            best, best_state, bad = val, copy.deepcopy(ema.shadow.state_dict()), 0
        else:
            bad += 1
        if ckdir is not None:
            if improved:
                save_diffusion(ema.shadow, ckdir / "model.pt", epoch=epoch, val_nelbo_site=val)
            if cfg.save_every and (epoch + 1) % cfg.save_every == 0:
                save_diffusion(ema.shadow, ckdir / "checkpoints" / f"epoch_{epoch:03d}.pt", epoch=epoch,
                               val_nelbo_site=val)
            torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "g_cpu": g_cpu.get_state(),
                        "g_dev": g_dev.get_state(), "hist": hist, "best": best, "best_state": best_state,
                        "bad": bad, "step": step, "epoch": epoch}, ckdir / "last.pt")
        if bad >= cfg.patience:
            break
        if cfg.max_steps is not None and step >= cfg.max_steps:
            break
    ema.shadow.load_state_dict(best_state)
    hist["best_val_nelbo_site"] = best
    hist["best_epoch"] = int(hist["epoch"][int(np.argmin(hist["val_nelbo_site"]))])
    return hist, ema.shadow
