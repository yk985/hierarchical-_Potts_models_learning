"""Step H (diagnostic) - are wrong family proportions in generated sequences due to the MODEL or to the SAMPLER?

Gibbs chains of the learned model are started (i) from random sequences and (ii) from the test sequences of each
family, and the family proportions of the chains (nearest family centroid on the reference PCA) are tracked over
sweeps. If all starts converge to the same proportions, the sampler mixes between families and the proportions
belong to the learned distribution itself. If chains stay where they started, the sampler is stuck.

Outputs: evaluations/<run_name>/mode_mixing.json, figures/models/<run_name>/mode_mixing.png

    python scripts/s08_mode_mixing.py --config configs/fa_unconditional.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from psg import PROJECT_ROOT  # noqa: E402
from psg.data import LabelledSequences  # noqa: E402
from psg.evaluation import family_assignment  # noqa: E402
from psg.models.factored_attention import load_model  # noqa: E402
from psg.plots import INK_2, SERIES, plt, save  # noqa: E402
from psg.sampling import gibbs_sample  # noqa: E402
from psg.stats import mean_pairwise_identity, site_frequencies  # noqa: E402
from s06_evaluate_model import load_reference_pca  # noqa: E402


def main(run_cfg: dict, data_cfg: dict, root: Path = PROJECT_ROOT, n_chains: int = 600, n_checkpoints: int = 5,
         every: int = 100, seed: int = 0) -> dict:
    run = run_cfg["run_name"]
    q = data_cfg["hierarchy"]["q"]
    P = load_model(root / "models" / run / "model.pt").to_potts()
    test = LabelledSequences.load(root / "data" / "splits" / "test.npz")
    pca = load_reference_pca(root, q)
    Zt = pca.transform(test.S)
    rng = np.random.default_rng(seed)
    starts = {"random": None} | {f"F{f}": test.S[test.family == f][:n_chains] for f in np.unique(test.family)}
    out = {"sweeps": [every * (k + 1) for k in range(n_checkpoints)], "test": family_assignment(Zt, Zt, test.family)}
    for name, init in starts.items():
        r = gibbs_sample(P, n_chains, 0, n_samples_per_chain=n_checkpoints, thin=every, rng=rng, init=init)
        out[f"start_{name}"] = [family_assignment(pca.transform(r.samples[k * n_chains:(k + 1) * n_chains]), Zt, test.family)
                                for k in range(n_checkpoints)]

    # ------------------------------------------------ energy vs entropy: why these proportions?
    # Under the learned model P(F) = sum_{s in F} exp(-E(s)) / Z = exp(-(<E>_F - S_F)) / Z, where <E>_F and S_F are
    # the mean energy and the entropy of the model restricted to family region F. Hence for two families
    #     ln P(F)/P(F') = -(<E>_F - <E>_F')  +  (S_F - S_F').
    # <E>_F is measured on the model's own samples assigned to F; the entropy difference is what is left.
    final = gibbs_sample(P, 4 * n_chains, 200, rng=rng).samples
    Zf = pca.transform(final)
    fam_ids = np.unique(test.family)
    cent = np.stack([Zt[test.family == f].mean(0) for f in fam_ids])
    near = np.linalg.norm(Zf[:, None] - cent[None], axis=2).argmin(1)
    E = P.energy(final)
    ref = int(np.argmax(np.bincount(near, minlength=len(fam_ids))))       # most populated family as reference
    dec = {}
    for k, f in enumerate(fam_ids):
        share, share_ref = np.mean(near == k), np.mean(near == ref)
        dE = float(E[near == k].mean() - E[near == ref].mean())
        log_ratio = float(np.log(share / share_ref))
        fr = site_frequencies(test.S[test.family == f], q)
        dec[f"F{int(f)}"] = {
            "share_generated": float(share),
            "mean_E_learned_generated": float(E[near == k].mean()),
            "mean_E_learned_test": float(P.energy(test.S[test.family == f]).mean()),
            f"ln_share_ratio_vs_F{int(fam_ids[ref])}": log_ratio,
            "energy_term": -dE,
            "entropy_term_implied": log_ratio + dE,
            "site_entropy_test": float(-(fr * np.log(fr + 1e-12)).sum(1).mean()),
            "identity_test": mean_pairwise_identity(test.S[test.family == f], max_pairs=20000),
        }
    out["energy_entropy_decomposition"] = dec

    fams = [k for k in out["test"] if k != "between"]
    fig, axes = plt.subplots(1, len(fams), figsize=(4 * len(fams), 3.3), sharey=True)
    for ax, fam, color in zip(axes, fams, SERIES):
        for s, ls in zip(starts, ["-", "--", ":", "-."]):
            ax.plot([0] + out["sweeps"], [float(s == fam) if s != "random" else np.nan] + [c[fam] for c in out[f"start_{s}"]],
                    ls=ls, marker="o", ms=4, color=color if s == fam else INK_2, label=f"chains started: {s}")
        ax.axhline(out["test"][fam], color=color, lw=1, alpha=0.5)
        ax.text(out["sweeps"][-1], out["test"][fam], "true share", fontsize=7, va="bottom", ha="right", color=INK_2)
        ax.set(xlabel="sweeps", title=f"share of chains in {fam}", ylim=(-0.02, 1.02))
    axes[0].set_ylabel("fraction of chains")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, root / "figures" / "models" / run / "mode_mixing.png")
    ev = root / "evaluations" / run
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "mode_mixing.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-config", default=str(PROJECT_ROOT / "configs" / "ground_truth.json"))
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    a = ap.parse_args()
    print(json.dumps(main(json.loads(Path(a.config).read_text()), json.loads(Path(a.data_config).read_text()),
                          Path(a.root)), indent=2))
