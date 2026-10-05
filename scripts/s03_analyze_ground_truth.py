"""Step C - analyse the ground-truth sequences: PCA, site frequencies, correlations, metric baselines.

Outputs (under --root):
    evaluations/ground_truth/pca_reference.npz      PCA fitted on all ground-truth sequences (reuse for AI samples)
    evaluations/ground_truth/site_frequencies.npz   f_i(a) per clade
    evaluations/ground_truth/sequence_statistics.json
    figures/ground_truth/03_analysis/*.png

    python scripts/s03_analyze_ground_truth.py --config configs/ground_truth.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psg import PROJECT_ROOT  # noqa: E402
from psg.alphabet import AMINO_ACIDS  # noqa: E402
from psg.hierarchy import RELATION_NAMES, common_ancestor_level  # noqa: E402
from psg.plots import (  # noqa: E402
    plot_correlation_strength,
    plot_distance_by_relation,
    plot_distance_matrix,
    plot_pca,
    plot_site_frequencies,
    plot_stat_comparison,
    plt,
    save,
)
from psg.stats import (  # noqa: E402
    PCA,
    connected_correlations,
    mean_pairwise_identity,
    pearson,
    site_frequencies,
    upper_pairs,
)


def main(config: dict, root: Path = PROJECT_ROOT) -> dict:
    q = config["hierarchy"]["q"]
    d = np.load(root / "data" / "sequences" / "all_sequences.npz")
    S, fam, sub, cla, leaf = d["S"], d["family"], d["subfamily"], d["clade"], d["leaf"]
    names = [str(x) for x in d["leaf_names"]]
    paths = [tuple(int(t[1:]) for t in n.split(".")) for n in names]
    contacts = np.load(root / "data" / "ground_truth_potts" / "contact_map.npy")
    fig_dir = root / "figures" / "ground_truth" / "03_analysis"
    ev = root / "evaluations" / "ground_truth"
    ev.mkdir(parents=True, exist_ok=True)
    report: dict = {"n_sequences": int(len(S)), "n_clades": len(names)}

    # ---------------------------------------------------------------- PCA on all sequences
    pca = PCA(q, n_components=config["analysis"]["n_components"]).fit(S)
    Z = pca.transform(S)
    np.savez_compressed(ev / "pca_reference.npz", mean=pca.mean_, components=pca.components_,
                        explained_variance_ratio=pca.explained_variance_ratio_)
    report["pca_explained_variance_ratio"] = [round(float(x), 4) for x in pca.explained_variance_ratio_]
    save(plot_pca(Z, fam, sub, pca.explained_variance_ratio_, "PCA of all sequences (fit on all clades)"),
         fig_dir / "pca_all_PC1_PC2.png")
    save(plot_pca(Z, fam, sub, pca.explained_variance_ratio_, "PCA of all sequences", pcs=(2, 3)),
         fig_dir / "pca_all_PC3_PC4.png")

    # within each family: PCA fitted on that family only, colour = subfamily, marker = clade
    fams = np.unique(fam)
    fig, axes = plt.subplots(1, len(fams), figsize=(5 * len(fams), 4.6), squeeze=False)
    for ax, f in zip(axes[0], fams):
        k = fam == f
        p = PCA(q, n_components=2).fit(S[k])
        plot_pca(p.transform(S[k]), sub[k], cla[k], p.explained_variance_ratio_, f"Family F{f} only", ax=ax,
                 color_name="subfamily S", marker_name="clade C")
    fig.tight_layout()
    save(fig, fig_dir / "pca_within_family.png")

    # ---------------------------------------------------------------- frequencies and correlations per clade
    n = len(names)
    f1 = np.stack([site_frequencies(S[leaf == k], q) for k in range(n)])
    np.savez_compressed(ev / "site_frequencies.npz", f1=f1, leaf_names=np.array(names))
    Cu = [upper_pairs(connected_correlations(S[leaf == k], q)).astype(np.float32) for k in range(n)]

    C0 = connected_correlations(S[leaf == 0], q)
    save(plot_site_frequencies(f1[0], AMINO_ACIDS, f"Site frequencies f_i(a), clade {names[0]}"),
         fig_dir / f"site_frequencies_{names[0]}.png")
    save(plot_correlation_strength(C0, contacts, f"||C_ij||_F, clade {names[0]}"),
         fig_dir / f"correlation_strength_{names[0]}.png")
    Nn = np.sqrt((C0 ** 2).sum(axis=(2, 3)))
    iu = np.triu_indices(Nn.shape[0], 1)
    report["corr_strength_contacts_vs_noncontacts"] = [float(Nn[iu][contacts[iu]].mean()),
                                                       float(Nn[iu][~contacts[iu]].mean())]

    # ---------------------------------------------------------------- metric matrix between clades
    # noise floor: two disjoint halves of the SAME clade
    rng = np.random.default_rng(0)
    floor_f, floor_C = [], []
    for k in range(n):
        Sk = S[leaf == k][rng.permutation((leaf == k).sum())]
        a, b = Sk[: len(Sk) // 2], Sk[len(Sk) // 2:]
        floor_f.append(pearson(site_frequencies(a, q).ravel(), site_frequencies(b, q).ravel()))
        floor_C.append(pearson(upper_pairs(connected_correlations(a, q)), upper_pairs(connected_correlations(b, q))))
    R_f, R_C, ID = np.eye(n), np.eye(n), np.zeros((n, n))
    for i in range(n):
        R_f[i, i], R_C[i, i] = floor_f[i], floor_C[i]
        ID[i, i] = mean_pairwise_identity(S[leaf == i], max_pairs=50_000)
        for j in range(i + 1, n):
            R_f[i, j] = R_f[j, i] = pearson(f1[i].ravel(), f1[j].ravel())
            R_C[i, j] = R_C[j, i] = pearson(Cu[i], Cu[j])
            ID[i, j] = ID[j, i] = mean_pairwise_identity(S[leaf == i], S[leaf == j], max_pairs=50_000)

    def by_rel(M):
        g: dict = {}
        for i in range(n):
            for j in range(i + 1, n):
                g.setdefault(common_ancestor_level(paths[i], paths[j]), []).append(M[i, j])
        g[3] = list(np.diag(M))
        return {k: np.array(v) for k, v in sorted(g.items())}

    rel_names = {**RELATION_NAMES, 3: "same clade\n(two halves)"}
    for key, M, label in [("pearson_f1", R_f, "Pearson r of f_i(a)"), ("pearson_C", R_C, "Pearson r of C_ij(a,b)"),
                          ("identity", ID, "mean sequence identity")]:
        g = by_rel(M)
        report[key] = {rel_names[k].replace("\n", " "): {"mean": float(v.mean()), "min": float(v.min()),
                                                         "max": float(v.max())} for k, v in g.items()}
        save(plot_distance_matrix(M, names, f"{label} between clades", label), fig_dir / f"{key}_matrix.png")
        save(plot_distance_by_relation(g, rel_names, f"{label} by relation", label), fig_dir / f"{key}_by_relation.png")

    # three example comparisons, from easiest to hardest to tell apart
    k0 = 0
    sib = next(j for j in range(n) if common_ancestor_level(paths[0], paths[j]) == 2)
    far = next(j for j in range(n) if common_ancestor_level(paths[0], paths[j]) == 0)
    S0 = S[leaf == k0]
    half = len(S0) // 2
    for tag, A, B, bname in [("same_clade_halves", S0[:half], S0[half:], f"{names[0]} (2nd half)"),
                             ("sibling_clade", S0, S[leaf == sib], names[sib]),
                             ("other_family", S0, S[leaf == far], names[far])]:
        fa, fb = site_frequencies(A, q), site_frequencies(B, q)
        ca, cb = upper_pairs(connected_correlations(A, q)), upper_pairs(connected_correlations(B, q))
        m = {"pearson_f1": pearson(fa.ravel(), fb.ravel()), "pearson_C": pearson(ca, cb)}
        save(plot_stat_comparison(fa.ravel(), fb.ravel(), ca, cb, m, names[0], bname),
             fig_dir / f"compare_{tag}.png")

    (ev / "sequence_statistics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "ground_truth.json"))
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    a = ap.parse_args()
    print(json.dumps(main(json.loads(Path(a.config).read_text()), Path(a.root)), indent=2))
