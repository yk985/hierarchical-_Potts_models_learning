"""Step I - train the masked (absorbing-state) diffusion model on data/splits/train.npz (all 18 clades, no labels),
early stopping on the validation NELBO. The test split is not touched.

Outputs (under --root):
    models/<run_name>/model.pt                  best EMA model (rewritten during training whenever validation improves)
    models/<run_name>/last.pt                   full training state after every epoch (for --resume)
    models/<run_name>/checkpoints/epoch_XXX.pt  EMA model every `save_every` epochs
    models/<run_name>/config.json, history.json, summary.txt
    figures/models/<run_name>/training_curve.png

    python scripts/s09_train_diffusion.py --config configs/md_unconditional.json
    python scripts/s09_train_diffusion.py --config configs/md_unconditional.json --resume     # continue from last.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psg import PROJECT_ROOT  # noqa: E402
from psg.data import LabelledSequences  # noqa: E402
from psg.diffusion_training import DiffusionTrainConfig, train_diffusion  # noqa: E402
from psg.models.masked_diffusion import MaskedDiffusionTransformer, save_diffusion  # noqa: E402
from psg.plots import SERIES, plt, save  # noqa: E402


def main(run_cfg: dict, data_cfg: dict, root: Path = PROJECT_ROOT, max_steps: int | None = None,
         epochs: int | None = None, verbose: bool = True, resume: bool = False) -> dict:
    hier = data_cfg["hierarchy"]
    tr = LabelledSequences.load(root / "data" / "splits" / "train.npz")
    va = LabelledSequences.load(root / "data" / "splits" / "val.npz")
    kw = {k: v for k, v in run_cfg["model"].items() if k != "class"}
    model = MaskedDiffusionTransformer(hier["L"], hier["q"], **kw)
    summary = model.describe()
    if verbose:
        print(summary, flush=True)
    tcfg = dict(run_cfg["train"])
    if max_steps is not None:
        tcfg["max_steps"] = max_steps
    if epochs is not None:
        tcfg["epochs"] = epochs
    out = root / "models" / run_cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(run_cfg, indent=2))
    hist, best = train_diffusion(model, tr.S, va.S, DiffusionTrainConfig(**tcfg), verbose=verbose,
                                 checkpoint_dir=out, resume=resume)

    save_diffusion(best, out / "model.pt", run_config=run_cfg, best_epoch=hist["best_epoch"])
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    (out / "summary.txt").write_text(summary + f"\n\ntrain sequences: {len(tr)}, val sequences: {len(va)}\n")

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(hist["epoch"], hist["train_nelbo_site"], color=SERIES[1], label="train (one noise draw per step)")
    ax.plot(hist["epoch"], hist["val_nelbo_site"], color=SERIES[0], label="validation (EMA weights, fixed noise)")
    ax.axvline(hist["best_epoch"], color="#52514e", lw=1, ls="--", label=f"best epoch ({hist['best_epoch']})")
    ax.set(xlabel="epoch", ylabel="NELBO per site (nats)", title=f"{run_cfg['run_name']}: training curve")
    ax.legend(fontsize=7)
    save(fig, root / "figures" / "models" / run_cfg["run_name"] / "training_curve.png")
    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-config", default=str(PROJECT_ROOT / "configs" / "ground_truth.json"))
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="continue from models/<run>/last.pt")
    a = ap.parse_args()
    main(json.loads(Path(a.config).read_text()), json.loads(Path(a.data_config).read_text()), Path(a.root),
         a.max_steps, a.epochs, resume=a.resume)
