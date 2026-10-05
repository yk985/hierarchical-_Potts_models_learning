"""Step J - evaluate the masked-diffusion model against the HELD-OUT test sequences.

1. Likelihood: NELBO per site on the test set (an upper bound on -log p_theta(x) / L).
2. Direct probabilities: P_theta(x_i | x_-i) (one site masked) on test sequences
   - pseudo-NLL per site, comparable to the Potts models and to the true clade models (0.80 on validation);
   - agreement with the exact conditional of the TRUE clade model the sequence was drawn from.
3. Samples: the exact continuous-time sampler (T -> infinity, one site revealed at a time) is the main one; the
   finite-T ancestral sampler is run for several T as an ablation (the number of steps matters: see test_13).
   For the main sampler, the full pooled evaluation of s06 (statistics, family weights, distances, figures).

Outputs: sampling/<run>/all_T<T>.fasta, evaluations/<run>/metrics.json, figures/models/<run>/*.png

    python scripts/s10_evaluate_diffusion.py --config configs/md_unconditional.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from psg import PROJECT_ROOT  # noqa: E402
from psg.alphabet import write_fasta  # noqa: E402
from psg.data import LabelledSequences  # noqa: E402
from psg.diffusion_training import evaluate_nelbo  # noqa: E402
from psg.distances import nearest_neighbor_distance  # noqa: E402
from psg.evaluation import compare_samples, family_assignment  # noqa: E402
from psg.hierarchy import leaves, load_hierarchy  # noqa: E402
from psg.models.masked_diffusion import (  # noqa: E402
    ancestral_sample,
    continuous_time_sample,
    load_diffusion,
    site_conditionals,
)
from psg.plots import INK_2, SERIES, plt, save  # noqa: E402
from s06_evaluate_model import evaluate_pooled, load_reference_pca, oracle_samples  # noqa: E402


def true_clade_conditionals(S: np.ndarray, leaf: np.ndarray, lv) -> np.ndarray:
    """Exact P(x_i | x_-i) under the TRUE model of the clade each sequence was drawn from. (M, L, q)"""
    M, L = S.shape
    out = np.empty((M, L, lv[0].model.q))
    for k, node in enumerate(lv):
        sel = np.flatnonzero(leaf == k)
        for i in range(L):
            out[sel, i] = node.model.conditional(S[sel], i)
    return out


def main(run_cfg: dict, data_cfg: dict, root: Path = PROJECT_ROOT, verbose: bool = True) -> dict:
    run, sc = run_cfg["run_name"], run_cfg["sampling"]
    q, L = data_cfg["hierarchy"]["q"], data_cfg["hierarchy"]["L"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_diffusion(root / "models" / run / "model.pt", dev).eval()
    nodes, contacts, _ = load_hierarchy(root / "data" / "ground_truth_potts")
    lv = leaves(nodes)
    test = LabelledSequences.load(root / "data" / "splits" / "test.npz")
    train = LabelledSequences.load(root / "data" / "splits" / "train.npz")
    pca = load_reference_pca(root, q)
    Zt = pca.transform(test.S)
    fig_dir, samp_dir = root / "figures" / "models" / run, root / "sampling" / run
    metrics: dict = {"run": run, "n_test": int(len(test))}
    St = torch.as_tensor(test.S, dtype=torch.long, device=dev)
    amp = bool(sc.get("amp", False)) and dev.type == "cuda"

    def autocast():
        return torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp)

    # ------------------------------------------------ 1. likelihood bound
    metrics["test_nelbo_site"] = evaluate_nelbo(model, St, repeats=8)

    # ------------------------------------------------ 2. direct probabilities P(x_i | x_-i)
    with autocast():
        Pm = site_conditionals(model, St, q).numpy()                              # (M, L, q)
    Pt = true_clade_conditionals(test.S, test.leaf, lv)
    idx = (np.arange(len(test))[:, None], np.arange(L)[None, :], test.S)
    tv = 0.5 * np.abs(Pm - Pt).sum(-1)                                            # (M, L)
    metrics["site_conditionals"] = {
        "pseudo_nll_site_model": float(-np.log(Pm[idx] + 1e-12).mean()),
        "pseudo_nll_site_true_clade_model": float(-np.log(Pt[idx]).mean()),
        "tv_to_true_conditional_mean": float(tv.mean()),
        "tv_to_true_conditional_median": float(np.median(tv)),
        "kl_true_to_model_mean": float((Pt * (np.log(Pt + 1e-12) - np.log(Pm + 1e-12))).sum(-1).mean()),
    }
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    sub = np.random.default_rng(0).choice(Pt.size, min(60000, Pt.size), replace=False)
    ax[0].plot([0, 1], [0, 1], color=INK_2, lw=1, ls="--")
    ax[0].scatter(Pt.ravel()[sub], Pm.ravel()[sub], s=2, alpha=0.2, color=SERIES[0], lw=0, rasterized=True)
    ax[0].set(xlabel="true clade model P(x_i = a | x_-i)", ylabel="diffusion model P(x_i = a | x_-i)",
              title="Direct probabilities on test sequences")
    ax[1].hist(tv.ravel(), bins=60, color=SERIES[0])
    ax[1].set(xlabel="total variation between the two conditionals", ylabel="count (sites x sequences)",
              title=f"mean TV = {tv.mean():.3f}")
    fig.tight_layout()
    save(fig, fig_dir / "site_conditionals_vs_truth.png")

    # ------------------------------------------------ 3. samples, as a function of the number of reverse steps
    oracle = oracle_samples(lv, test, sc["n_burnin_oracle"], sc["seed"])
    by_T, main_gen = {}, None
    runs = [(str(T), T) for T in sc["n_steps"]] + ([("continuous", None)] if sc.get("continuous_time", True) else [])
    for key, T in runs:
        with autocast():
            if T is None:
                gen = continuous_time_sample(model, len(test), L, q, seed=sc["seed"], device=dev).numpy()
            else:
                gen = ancestral_sample(model, len(test), L, q, T, seed=sc["seed"] + T, device=dev).numpy()
        write_fasta(samp_dir / f"all_{'T' + key if T else key}.fasta", gen,
                    [f"{run}|{key}|gen{m:05d}" for m in range(len(gen))])
        cs = compare_samples(gen, test.S, q)
        by_T[key] = {k: cs[k] for k in ["pearson_f1", "pearson_C", "slope_C", "identity_gen_gen"]} | {
            "family_proportions": family_assignment(pca.transform(gen), Zt, test.family),
            "nn_to_train_mean": float(nearest_neighbor_distance(gen, train.S, q).mean())}
        if verbose:
            print(f"sampler {key}: r_f {cs['pearson_f1']:.3f} r_C {cs['pearson_C']:.3f} slope {cs['slope_C']:.3f} "
                  f"families {by_T[key]['family_proportions']}", flush=True)
        main_gen = gen                        # the last one: continuous time if enabled, else the largest T
    metrics["by_n_steps"] = by_T

    Ts = [int(t) for t in by_T if t != "continuous"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for key, lab, c in [("pearson_f1", "r_f", SERIES[0]), ("pearson_C", "r_C", SERIES[1]), ("slope_C", "slope C", SERIES[2])]:
        ax[0].plot(Ts, [by_T[str(t)][key] for t in Ts], marker="o", color=c, label=f"{lab}, T steps")
        if "continuous" in by_T:
            ax[0].axhline(by_T["continuous"][key], color=c, ls=":", lw=1.5, label=f"{lab}, continuous time")
    ax[0].set_xscale("log")
    ax[0].set(xlabel="reverse steps T", title="Pooled statistics vs number of steps")
    ax[0].legend(fontsize=7)
    fam_keys = [k for k in by_T[str(Ts[0])]["family_proportions"] if k != "between"]
    for f, c in zip(fam_keys, SERIES):
        ax[1].plot(Ts, [by_T[str(t)]["family_proportions"][f] for t in Ts], marker="o", color=c, label=f"{f}, T steps")
        if "continuous" in by_T:
            ax[1].axhline(by_T["continuous"]["family_proportions"][f], color=c, ls=":", lw=1.5)
    ax[1].axhline(1 / len(fam_keys), color=INK_2, ls="--", lw=1, label="true share")
    ax[1].set_xscale("log")
    ax[1].set(xlabel="reverse steps T", ylabel="share of generated sequences", title="Family proportions vs T")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, fig_dir / "quality_vs_steps.png")

    metrics |= evaluate_pooled(main_gen, oracle, test, train, nearest_neighbor_distance(test.S, train.S, q), pca,
                               contacts, q, L, fig_dir, sc["seed"],
                               title="Generated (masked diffusion, continuous-time sampler)"
                               if sc.get("continuous_time", True) else f"Generated (masked diffusion, T={runs[-1][1]})")
    ev = root / "evaluations" / run
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-config", default=str(PROJECT_ROOT / "configs" / "ground_truth.json"))
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    a = ap.parse_args()
    main(json.loads(Path(a.config).read_text()), json.loads(Path(a.data_config).read_text()), Path(a.root))
