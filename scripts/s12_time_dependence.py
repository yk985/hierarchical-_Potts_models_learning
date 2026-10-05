"""Step L (diagnostic) - does the time-conditioned denoiser use t?

Theory (docs, eq. md_opt): the optimal denoiser P(x_0^i | visible sites) does not depend on t. We feed the SAME
partially masked test sequences with different values of t and measure how much the output distribution of the
masked sites changes (mean total variation), for several masking fractions.

Output: evaluations/<run>/time_dependence.json

    python scripts/s12_time_dependence.py --config configs/md_unconditional.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psg import PROJECT_ROOT  # noqa: E402
from psg.data import LabelledSequences  # noqa: E402
from psg.models.masked_diffusion import load_diffusion, mask_tokens  # noqa: E402


@torch.no_grad()
def main(run_cfg: dict, root: Path = PROJECT_ROOT, n: int = 2000, seed: int = 0) -> dict:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_diffusion(root / "models" / run_cfg["run_name"] / "model.pt", dev).eval()
    test = LabelledSequences.load(root / "data" / "splits" / "test.npz")
    x0 = torch.as_tensor(test.S[:n], dtype=torch.long, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    ts = [0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    out = {"t_values": ts, "by_mask_fraction": {}}
    for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
        xt, m = mask_tokens(x0, torch.full((n,), frac, device=dev), model.mask_id, g)
        probs = {t: torch.softmax(model(xt, torch.full((n,), t, device=dev)).float(), -1) for t in ts}
        ref = probs[frac] if frac in probs else probs[min(ts, key=lambda u: abs(u - frac))]
        tv = {str(t): float((0.5 * (probs[t] - ref).abs().sum(-1))[m].mean()) for t in ts}
        # scale of a meaningful change: TV between the predictions for two different sequences at the same site
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).to(dev)
        baseline = float((0.5 * (ref - ref[perm]).abs().sum(-1))[m & m[perm]].mean())
        out["by_mask_fraction"][str(frac)] = {"tv_vs_t_matched": tv, "tv_between_sequences": baseline}
        print(frac, {k: round(v, 4) for k, v in tv.items()}, "between-sequence scale", round(baseline, 3))
    ev = root / "evaluations" / run_cfg["run_name"]
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "time_dependence.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    a = ap.parse_args()
    main(json.loads(Path(a.config).read_text()), Path(a.root))
