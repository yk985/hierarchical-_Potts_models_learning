"""Step E - train a factored-attention Potts model (unconditioned or conditioned on hierarchical tags)
on data/splits/train.npz, with early stopping on data/splits/val.npz. The test split is not touched.

Outputs (under --root):
    models/<run_name>/model.pt, config.json, history.json, summary.txt
    figures/models/<run_name>/training_curve.png

    python scripts/s05_train_factored_attention.py --config configs/fa_unconditional.json
    python scripts/s05_train_factored_attention.py --config configs/fa_conditional.json
    python scripts/s05_train_factored_attention.py --config configs/fa_conditional.json --max-steps 20   # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psg import PROJECT_ROOT  # noqa: E402
from psg.data import LabelledSequences, global_tags  # noqa: E402
from psg.models.factored_attention import (  # noqa: E402
    ConditionalFactoredAttentionPotts,
    FactoredAttentionPotts,
    save_model,
)
from psg.plots import SERIES, plt, save  # noqa: E402
from psg.training import TrainConfig, train  # noqa: E402


def build_model(model_cfg: dict, hier_cfg: dict):
    kw = {k: v for k, v in model_cfg.items() if k != "class"}
    L, q = hier_cfg["L"], hier_cfg["q"]
    if model_cfg["class"] == "FactoredAttentionPotts":
        return FactoredAttentionPotts(L, q, **kw)
    nF, nS, nC = hier_cfg["n_families"], hier_cfg["n_subfamilies"], hier_cfg["n_clades"]
    return ConditionalFactoredAttentionPotts(L, q, n_families=nF, n_subfamilies=nF * nS, n_clades=nF * nS * nC, **kw)


def main(run_cfg: dict, data_cfg: dict, root: Path = PROJECT_ROOT, max_steps: int | None = None,
         epochs: int | None = None, verbose: bool = True) -> dict:
    hier = data_cfg["hierarchy"]
    tr = LabelledSequences.load(root / "data" / "splits" / "train.npz")
    va = LabelledSequences.load(root / "data" / "splits" / "val.npz")
    model = build_model(run_cfg["model"], hier)
    conditional = isinstance(model, ConditionalFactoredAttentionPotts)
    summary = model.describe()
    if verbose:
        print(summary, flush=True)

    tcfg = dict(run_cfg["train"])
    if "tag_dropout" in tcfg:
        tcfg["tag_dropout"] = tuple(tcfg["tag_dropout"])
    if max_steps is not None:
        tcfg["max_steps"] = max_steps
    if epochs is not None:
        tcfg["epochs"] = epochs
    cfg = TrainConfig(**tcfg)
    tags = (global_tags(tr, hier["n_subfamilies"]), global_tags(va, hier["n_subfamilies"])) if conditional else (None, None)
    hist = train(model, tr.S, va.S, cfg, tags[0], tags[1], verbose=verbose)

    out = root / "models" / run_cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    save_model(model, out / "model.pt", run_config=run_cfg)
    (out / "config.json").write_text(json.dumps(run_cfg, indent=2))
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    (out / "summary.txt").write_text(summary + f"\n\ntrain sequences: {len(tr)}, val sequences: {len(va)}\n")

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(hist["epoch"], hist["train_pnll_site"], color=SERIES[1], label="train (with tag dropout)" if conditional else "train")
    ax.plot(hist["epoch"], hist["val_pnll_site"], color=SERIES[0], label="validation")
    ax.axvline(hist["best_epoch"], color="#52514e", lw=1, ls="--", label=f"best epoch ({hist['best_epoch']})")
    ax.set(xlabel="epoch", ylabel="pseudo-NLL per site", title=f"{run_cfg['run_name']}: training curve")
    ax.legend()
    save(fig, root / "figures" / "models" / run_cfg["run_name"] / "training_curve.png")
    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-config", default=str(PROJECT_ROOT / "configs" / "ground_truth.json"))
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N optimiser steps (smoke test)")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    main(json.loads(Path(a.config).read_text()), json.loads(Path(a.data_config).read_text()), Path(a.root),
         a.max_steps, a.epochs)
