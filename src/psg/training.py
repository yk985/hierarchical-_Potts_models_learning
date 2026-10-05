"""Training loop for the factored-attention Potts models (pseudo-likelihood + L2).

Adam on minibatches; after every epoch the validation pseudo-NLL per site is computed (true tags, no dropout);
the best epoch's weights are kept (early stopping with `patience`).
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .models.factored_attention import ConditionalFactoredAttentionPotts, drop_tags, pseudo_nll


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 512
    lr: float = 5e-3
    lam_h: float = 0.01
    lam_J: float = 0.01
    lam_label: float = 0.01                      # conditional model only
    tag_dropout: tuple = (0.1, 0.1, 0.15)        # P(drop family), P(drop subfamily), P(drop clade)
    patience: int = 8
    max_steps: int | None = None                 # stop after this many optimiser steps (smoke tests)
    seed: int = 0


@torch.no_grad()
def evaluate_pnll(model, S: torch.Tensor, tags: torch.Tensor | None = None, batch_size: int = 2048) -> float:
    """Mean pseudo-NLL per site (no regularisation)."""
    model.eval()
    tot = 0.0
    for k in range(0, len(S), batch_size):
        s = S[k:k + batch_size]
        logits = model.logits(s, tags[k:k + batch_size]) if tags is not None else model.logits(s)
        tot += pseudo_nll(logits, s).sum().item()
    model.train()
    return tot / (len(S) * S.shape[1])


def train(model, S_train: np.ndarray, S_val: np.ndarray, cfg: TrainConfig,
          tags_train: np.ndarray | None = None, tags_val: np.ndarray | None = None,
          device: torch.device | None = None, verbose: bool = True) -> dict:
    conditional = isinstance(model, ConditionalFactoredAttentionPotts)
    if conditional and (tags_train is None or tags_val is None):
        raise ValueError("the conditional model needs tags")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model.to(device).train()
    Str = torch.as_tensor(S_train, dtype=torch.long, device=device)
    Sva = torch.as_tensor(S_val, dtype=torch.long, device=device)
    Ttr = torch.as_tensor(tags_train, dtype=torch.long, device=device) if conditional else None
    Tva = torch.as_tensor(tags_val, dtype=torch.long, device=device) if conditional else None
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    hist = {"config": asdict(cfg), "epoch": [], "train_loss": [], "train_pnll_site": [], "val_pnll_site": [],
            "seconds": []}
    best, best_state, bad, step, t0 = np.inf, None, 0, 0, time.time()
    L = Str.shape[1]
    for epoch in range(cfg.epochs):
        perm = torch.randperm(len(Str), generator=gen).to(device)
        tl, tp, nb = 0.0, 0.0, 0
        for k in range(0, len(Str), cfg.batch_size):
            idx = perm[k:k + cfg.batch_size]
            if conditional:
                tags = drop_tags(Ttr[idx], *cfg.tag_dropout, generator=gen)
                loss, pnll = model.loss(Str[idx], tags, cfg.lam_h, cfg.lam_J, cfg.lam_label)
            else:
                loss, pnll = model.loss(Str[idx], cfg.lam_h, cfg.lam_J)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl, tp, nb, step = tl + loss.item(), tp + pnll.item(), nb + 1, step + 1
            if cfg.max_steps is not None and step >= cfg.max_steps:
                break
        val = evaluate_pnll(model, Sva, Tva)
        hist["epoch"].append(epoch)
        hist["train_loss"].append(tl / nb)
        hist["train_pnll_site"].append(tp / nb / L)
        hist["val_pnll_site"].append(val)
        hist["seconds"].append(round(time.time() - t0, 1))
        if verbose:
            print(f"epoch {epoch:3d}  loss {tl / nb:.4f}  train pNLL/site {tp / nb / L:.4f}  "
                  f"val pNLL/site {val:.4f}  ({hist['seconds'][-1]}s)", flush=True)
        if val < best - 1e-5:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
        if cfg.max_steps is not None and step >= cfg.max_steps:
            break
    model.load_state_dict(best_state)
    hist["best_val_pnll_site"] = best
    hist["best_epoch"] = int(hist["epoch"][int(np.argmin(hist["val_pnll_site"]))])
    return hist
