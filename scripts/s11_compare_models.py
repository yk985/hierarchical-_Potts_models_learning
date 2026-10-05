"""Step K - compare all generative models on the same footing.

For each set of generated sequences (unconditioned Potts H128/H256, conditioned Potts H128 with its 18 clade tags,
masked diffusion) and for fresh true samples (oracle):
  * clade assignment: every sequence is given the clade whose TRUE model gives it the highest pseudo-log-likelihood
    (psg.evaluation.classify_by_pseudo_likelihood; its accuracy on the labelled test set is the reference);
  * clade and family proportions (the truth is 1/18 and 1/3), and their total-variation distance to uniform;
  * per clade: generated sequences assigned to clade c vs the test sequences of clade c (r_C and Frechet distance).
For each model that has explicit conditionals: P(x_i | x_-i) on the test sequences vs the true clade conditionals
(pseudo-NLL per site and mean total variation).

Outputs: evaluations/comparison/summary.json, figures/comparison/*.png

    python scripts/s11_compare_models.py
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
from psg.alphabet import read_fasta  # noqa: E402
from psg.data import LabelledSequences  # noqa: E402
from psg.distances import frechet_distance  # noqa: E402
from psg.evaluation import classify_by_pseudo_likelihood  # noqa: E402
from psg.hierarchy import leaves, load_hierarchy  # noqa: E402
from psg.models.factored_attention import load_model  # noqa: E402
from psg.plots import INK_2, SERIES, plt, save  # noqa: E402
from psg.stats import compare_statistics  # noqa: E402
from s06_evaluate_model import load_reference_pca, oracle_samples  # noqa: E402
from s10_evaluate_diffusion import true_clade_conditionals  # noqa: E402

SETS = [  # (label, run, fasta pattern)
    ("unconditioned Potts H128", "fa_unconditional_H128", "all.fasta"),
    ("unconditioned Potts H256", "fa_unconditional_H256", "all.fasta"),
    ("conditioned Potts H128 (clade tags)", "fa_conditional_H128", "clades"),
    ("masked diffusion", "md_unconditional", "all_continuous.fasta"),
]


def load_generated(root: Path, run: str, pattern: str, lv) -> np.ndarray:
    d = root / "sampling" / run
    if pattern == "clades":
        return np.concatenate([read_fasta(d / f"{n.name}.fasta")[1] for n in lv])
    return read_fasta(d / pattern)[1]


def per_clade_report(S: np.ndarray, assigned: np.ndarray, test: LabelledSequences, pca, q: int, n_clades: int) -> dict:
    props = np.bincount(assigned, minlength=n_clades) / len(S)
    rC, fd = [], []
    for c in range(n_clades):
        sel = S[assigned == c]
        ref = test.S[test.leaf == c]
        if len(sel) >= 50:
            rC.append(compare_statistics(ref, sel, q)["pearson_C"])
            fd.append(frechet_distance(pca.transform(sel), pca.transform(ref)))
        else:
            rC.append(float("nan"))
            fd.append(float("nan"))
    return {"clade_proportions": props.tolist(), "tv_clade_proportions_to_uniform": float(0.5 * np.abs(props - 1 / n_clades).sum()),
            "per_clade_rC": rC, "per_clade_frechet": fd,
            "per_clade_rC_mean": float(np.nanmean(rC)), "per_clade_frechet_mean": float(np.nanmean(fd))}


def main(data_cfg: dict, root: Path = PROJECT_ROOT) -> dict:
    hier = data_cfg["hierarchy"]
    q, L, nS = hier["q"], hier["L"], hier["n_subfamilies"]
    nodes, _, _ = load_hierarchy(root / "data" / "ground_truth_potts")
    lv = leaves(nodes)
    models = [n.model for n in lv]
    n_cl = len(lv)
    fam_of = np.array([n.path[0] for n in lv])
    test = LabelledSequences.load(root / "data" / "splits" / "test.npz")
    pca = load_reference_pca(root, q)
    out: dict = {}

    # the classifier on labelled sequences: its accuracy is the reference
    pred = classify_by_pseudo_likelihood(test.S, models)
    out["classifier_accuracy_on_test"] = {"clade": float((pred == test.leaf).mean()),
                                          "family": float((fam_of[pred] == test.family).mean())}
    sets = {"oracle (true samples)": oracle_samples(lv, test, 200, 7)}
    for lab, run, pat in SETS:
        if (root / "sampling" / run).exists():
            sets[lab] = load_generated(root, run, pat, lv)
    for lab, S in sets.items():
        assigned = classify_by_pseudo_likelihood(S, models)
        rep = per_clade_report(S, assigned, test, pca, q, n_cl)
        rep["family_proportions"] = (np.bincount(fam_of[assigned], minlength=3) / len(S)).tolist()
        if lab.startswith("conditioned"):
            tags = np.repeat(np.arange(n_cl), len(S) // n_cl)
            rep["tag_matches_assigned_clade"] = float((assigned == tags).mean())
        out[lab] = rep
        print(f"{lab}: clade TV to uniform {rep['tv_clade_proportions_to_uniform']:.3f}, "
              f"per-clade r_C {rep['per_clade_rC_mean']:.3f}, FD {rep['per_clade_frechet_mean']:.3f}", flush=True)

    # conditionals of the Potts models on the test sequences, vs the true clade conditionals
    Pt = true_clade_conditionals(test.S, test.leaf, lv)
    idx = (np.arange(len(test))[:, None], np.arange(L)[None, :], test.S)
    cond: dict = {"true clade models": {"pseudo_nll_site": float(-np.log(Pt[idx]).mean()), "tv_mean": 0.0}}
    for lab, run in [("unconditioned Potts H128", "fa_unconditional_H128"), ("unconditioned Potts H256", "fa_unconditional_H256"),
                     ("conditioned Potts H128 (true tags)", "fa_conditional_H128"),
                     ("conditioned Potts H256 (true tags)", "fa_conditional_H256")]:
        p = root / "models" / run / "model.pt"
        if not p.exists():
            continue
        mdl = load_model(p)
        Pm = np.empty_like(Pt)
        if "unconditioned" in lab:
            P = mdl.to_potts()
            for i in range(L):
                Pm[:, i] = P.conditional(test.S, i)
        else:
            for k, n in enumerate(lv):
                P = mdl.to_potts((n.path[0], n.path[0] * nS + n.path[1], k))
                sel = np.flatnonzero(test.leaf == k)
                for i in range(L):
                    Pm[sel, i] = P.conditional(test.S[sel], i)
        cond[lab] = {"pseudo_nll_site": float(-np.log(Pm[idx]).mean()),
                     "tv_mean": float((0.5 * np.abs(Pm - Pt).sum(-1)).mean())}
    md = root / "evaluations" / "md_unconditional" / "metrics.json"
    if md.exists():
        sc = json.loads(md.read_text())["site_conditionals"]
        cond["masked diffusion"] = {"pseudo_nll_site": sc["pseudo_nll_site_model"], "tv_mean": sc["tv_to_true_conditional_mean"]}
    out["site_conditionals_on_test"] = cond

    # ------------------------------------------------ figures
    fig_dir = root / "figures" / "comparison"
    show = [k for k in sets if k != "unconditioned Potts H256"][:4]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    w = 0.8 / len(show)
    for k, lab in enumerate(show):
        ax.bar(np.arange(n_cl) + (k - (len(show) - 1) / 2) * w, out[lab]["clade_proportions"], w, color=SERIES[k], label=lab)
    ax.axhline(1 / n_cl, color=INK_2, ls="--", lw=1, label="true share 1/18")
    ax.set_xticks(range(n_cl), [n.name for n in lv], rotation=90, fontsize=7)
    ax.set(ylabel="share of sequences", title="Clade proportions (clade = best pseudo-likelihood under the true models)")
    ax.legend(fontsize=7, ncol=3)
    save(fig, fig_dir / "clade_proportions.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, lab in enumerate(show):
        ax[0].plot(range(n_cl), out[lab]["per_clade_rC"], marker="o", ms=4, lw=1, color=SERIES[k], label=lab)
        ax[1].plot(range(n_cl), out[lab]["per_clade_frechet"], marker="o", ms=4, lw=1, color=SERIES[k], label=lab)
    for a, t, yl in [(ax[0], "r_C per clade (generated assigned to c vs test c)", "Pearson r_C"),
                     (ax[1], "Frechet distance per clade (PCA)", "Frechet distance")]:
        a.set_xticks(range(n_cl), [n.name for n in lv], rotation=90, fontsize=6)
        a.set(title=t, ylabel=yl)
    ax[1].set_yscale("log")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, fig_dir / "per_clade_quality.png")

    ev = root / "evaluations" / "comparison"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "summary.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default=str(PROJECT_ROOT / "configs" / "ground_truth.json"))
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    a = ap.parse_args()
    main(json.loads(Path(a.data_config).read_text()), Path(a.root))
