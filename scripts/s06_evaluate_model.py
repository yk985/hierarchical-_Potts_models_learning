"""Step F - evaluate a trained factored-attention model against the HELD-OUT test sequences and the true Potts models.

For every label (the whole data set for the unconditioned model; every clade and every family for the conditional
one) we export (h, J), Gibbs-sample as many sequences as the test set holds for that label, and compare:
    * statistics  : r_f, r_C, identity (psg.evaluation.compare_samples)  vs the test sequences
    * oracle      : the same metric for FRESH ground-truth samples of the same size (the best achievable score)
    * parameters  : learned J vs true J (zero-sum gauge), contact precision
    * PCA         : generated sequences projected on the reference PCA axes of Part I

Outputs (under --root):
    sampling/<run_name>/<label>.fasta
    evaluations/<run_name>/metrics.json
    figures/models/<run_name>/*.png

    python scripts/s06_evaluate_model.py --config configs/fa_conditional.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psg import PROJECT_ROOT  # noqa: E402
from psg.alphabet import write_fasta  # noqa: E402
from psg.data import LabelledSequences  # noqa: E402
from psg.alphabet import AMINO_ACIDS  # noqa: E402
from psg.evaluation import (  # noqa: E402
    compare_samples,
    contact_precision,
    contact_scores,
    coupling_recovery,
    family_assignment,
)
from psg.distances import distance_report, frechet_distance, nearest_neighbor_distance, pairwise_distances  # noqa: E402
from psg.hierarchy import common_ancestor_level, leaves, load_hierarchy  # noqa: E402
from psg.models.factored_attention import ConditionalFactoredAttentionPotts, load_model  # noqa: E402
from psg.plots import (  # noqa: E402
    INK_2,
    SERIES,
    plot_correlation_triangles,
    plot_distance_by_relation,
    plot_distance_distributions,
    plot_distance_matrix,
    plot_frequency_maps,
    plot_pca,
    plot_stat_comparison,
    plt,
    save,
)
from psg.potts import Potts  # noqa: E402
from psg.stats import PCA, connected_correlations, pearson, site_frequencies, upper_pairs  # noqa: E402
from psg.torch_sampling import gibbs_sample_torch  # noqa: E402


def sample(model: Potts, n: int, n_burnin: int, seed: int) -> np.ndarray:
    return gibbs_sample_torch(model, n, n_burnin, seed=seed, energy_every=0).samples


def load_reference_pca(root: Path, q: int) -> PCA:
    d = np.load(root / "evaluations" / "ground_truth" / "pca_reference.npz")
    p = PCA(q, n_components=len(d["explained_variance_ratio"]))
    p.mean_, p.components_, p.explained_variance_ratio_ = d["mean"], d["components"], d["explained_variance_ratio"]
    return p


def pca_figure(pca, test: LabelledSequences, gen: np.ndarray, gen_family, gen_sub, title_gen):
    Zt, Zg = pca.transform(test.S), pca.transform(gen)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    plot_pca(Zt, test.family, test.subfamily, pca.explained_variance_ratio_, "Held-out test sequences", ax=ax[0])
    if gen_family is None:
        ax[1].scatter(Zg[:, 0], Zg[:, 1], s=5, color=INK_2, alpha=0.4, lw=0, rasterized=True)
        ax[1].set(title=title_gen, xlabel=ax[0].get_xlabel())
    else:
        plot_pca(Zg, gen_family, gen_sub, pca.explained_variance_ratio_, title_gen, ax=ax[1])
    fig.tight_layout()
    return fig


def oracle_samples(lv, test: LabelledSequences, n_burnin: int, seed: int) -> np.ndarray:
    """Fresh ground-truth samples, as many per clade as the test set holds (the best achievable generator)."""
    return np.concatenate([sample(n.model, int((test.leaf == k).sum()), n_burnin, seed + 1000 + k)
                           for k, n in enumerate(lv)])


def evaluate_pooled(gen: np.ndarray, oracle: np.ndarray, test: LabelledSequences, train: LabelledSequences,
                    nn_test: np.ndarray, pca, contacts: np.ndarray, q: int, L: int, fig_dir: Path, seed: int,
                    title: str = "Generated (unconditioned model)") -> dict:
    """Everything that can be said about a POOLED set of generated sequences (no labels): statistics vs the test
    set, family proportions, per-family statistics, rebalanced statistics, distances, and the figures.
    Shared by the unconditioned Potts model (s06) and the diffusion model (s10)."""
    metrics: dict = {}
    Zt = pca.transform(test.S)
    metrics["all_clades"] = {"generated_vs_test": compare_samples(gen, test.S, q),
                             "oracle_vs_test": compare_samples(oracle, test.S, q)}
    save(pca_figure(pca, test, gen, None, None, title), fig_dir / "pca_generated_vs_test.png")

    # which family does each generated sequence fall in (nearest family centroid on the reference PCA)?
    Zg = pca.transform(gen)
    metrics["family_proportions"] = {
        "test": family_assignment(Zt, Zt, test.family),
        "oracle": family_assignment(pca.transform(oracle), Zt, test.family),
        "generated": family_assignment(Zg, Zt, test.family),
    }
    # the shape of each mode: generated sequences assigned to family f vs the test sequences of family f
    fams = np.unique(test.family)
    cent = np.stack([Zt[test.family == f].mean(0) for f in fams])
    nearest = np.linalg.norm(Zg[:, None] - cent[None], axis=2).argmin(1)
    metrics["per_family_after_assignment"] = {
        f"F{int(f)}": compare_samples(gen[nearest == k], test.S[test.family == f], q) | {"n_generated": int((nearest == k).sum())}
        for k, f in enumerate(fams) if (nearest == k).sum() > 10}
    # rebalanced: equal numbers of generated sequences per assigned family (the true proportions), to separate
    # "wrong family weights" from "wrong statistics inside each family"
    n_min = min(int((nearest == k).sum()) for k in range(len(fams)))
    rs = np.random.default_rng(seed)
    bal = np.concatenate([gen[rs.choice(np.flatnonzero(nearest == k), n_min, replace=False)] for k in range(len(fams))])
    metrics["all_clades"]["generated_rebalanced_vs_test"] = compare_samples(bal, test.S, q) | {"n": int(len(bal))}
    sub = np.concatenate([test.S[rs.choice(np.flatnonzero(test.family == f), n_min, replace=False)] for f in fams])
    metrics["all_clades"]["test_subsample_same_size_vs_test"] = compare_samples(sub, test.S, q) | {"n": int(len(sub))}

    # site frequencies and correlations, pooled over all test sequences
    f_t, f_g, f_o = (site_frequencies(x, q) for x in (test.S, gen, oracle))
    C_t, C_g, C_o = (connected_correlations(x, q) for x in (test.S, gen, oracle))
    for tag, fx, Cx, m in [("generated", f_g, C_g, metrics["all_clades"]["generated_vs_test"]),
                           ("oracle", f_o, C_o, metrics["all_clades"]["oracle_vs_test"])]:
        save(plot_stat_comparison(f_t.ravel(), fx.ravel(), upper_pairs(C_t), upper_pairs(Cx), m, "test", tag),
             fig_dir / f"freq_corr_{tag}_vs_test.png")
    save(plot_frequency_maps(f_t, f_g, AMINO_ACIDS), fig_dir / "site_frequencies_test_vs_generated.png")
    save(plot_correlation_triangles(C_t, C_g, contacts), fig_dir / "correlation_strength_test_vs_generated.png")

    # distances (psg.distances): novelty w.r.t. the training set, pairwise Hamming, Frechet distance in PCA space
    metrics["distances"] = {
        "generated_vs_test": distance_report(gen, test.S, train.S, q, pca, nn_ref=nn_test),
        "oracle_vs_test": distance_report(oracle, test.S, train.S, q, pca, nn_ref=nn_test),
    }
    rs = np.random.default_rng(1)
    save(plot_distance_distributions({
        "generated -> train": nearest_neighbor_distance(gen, train.S, q),
        "test -> train": nn_test,
        "oracle -> train": nearest_neighbor_distance(oracle, train.S, q)},
        "Distance to the nearest training sequence", "d_NN (normalised Hamming)", step=1 / L), fig_dir / "dist_nn_to_train.png")
    save(plot_distance_distributions({
        "within generated": pairwise_distances(gen, rng=rs), "within test": pairwise_distances(test.S, rng=rs),
        "generated - test": pairwise_distances(gen, test.S, rng=rs)},
        "Pairwise Hamming distances", step=1 / L), fig_dir / "dist_pairwise.png")
    return metrics


def main(run_cfg: dict, data_cfg: dict, root: Path = PROJECT_ROOT, verbose: bool = True) -> dict:
    run = run_cfg["run_name"]
    sc = run_cfg["sampling"]
    hier = data_cfg["hierarchy"]
    q, nS = hier["q"], hier["n_subfamilies"]
    model = load_model(root / "models" / run / "model.pt")
    nodes, contacts, _ = load_hierarchy(root / "data" / "ground_truth_potts")
    lv = leaves(nodes)
    test = LabelledSequences.load(root / "data" / "splits" / "test.npz")
    pca = load_reference_pca(root, q)
    fig_dir, samp_dir = root / "figures" / "models" / run, root / "sampling" / run
    metrics: dict = {"run": run, "n_test": int(len(test))}
    Zt = pca.transform(test.S)
    train = LabelledSequences.load(root / "data" / "splits" / "train.npz")
    nn_test = nearest_neighbor_distance(test.S, train.S, q)          # held-out real sequences -> training set

    if not isinstance(model, ConditionalFactoredAttentionPotts):
        # ---------------------------------------------------------------- unconditioned: one model for everything
        P = model.to_potts()
        gen = sample(P, len(test), sc["n_burnin"], sc["seed"])
        write_fasta(samp_dir / "all.fasta", gen, [f"{run}|gen{m:05d}" for m in range(len(gen))])
        oracle = oracle_samples(lv, test, sc["n_burnin"], sc["seed"])
        metrics |= evaluate_pooled(gen, oracle, test, train, nn_test, pca, contacts, q, hier["L"], fig_dir, sc["seed"])
        mean_clade = Potts(np.mean([n.model.h for n in lv], 0), np.mean([n.model.J for n in lv], 0))
        metrics["parameters"] = {
            "vs_root_model": coupling_recovery(P, nodes["root"].model, contacts),
            "vs_mean_of_clade_models": coupling_recovery(P, mean_clade, contacts),
            "contact_precision_long_range": contact_precision(contact_scores(P), contacts, min_sep=4),
        }
    else:
        # ---------------------------------------------------------------- conditional: every clade, every family
        per_clade, gens, oracles, Cg, Ct = {}, [], [], [], []
        for k, node in enumerate(lv):
            f, s, c = node.path
            tag = (f, f * nS + s, k)
            P = model.to_potts(tag)
            ref = test.S[test.leaf == k]
            gen = sample(P, len(ref), sc["n_burnin"], sc["seed"] + k)
            oracle = sample(node.model, len(ref), sc["n_burnin"], sc["seed"] + 1000 + k)
            write_fasta(samp_dir / f"{node.name}.fasta", gen, [f"{node.name}|gen{m:05d}" for m in range(len(gen))])
            per_clade[node.name] = {
                "tag": list(tag),
                "generated_vs_test": compare_samples(gen, ref, q),
                "oracle_vs_test": compare_samples(oracle, ref, q),
                "parameters": coupling_recovery(P, node.model, contacts) | {
                    "contact_precision_long_range": contact_precision(contact_scores(P), contacts, min_sep=4)},
            }
            gens.append(gen)
            oracles.append(oracle)
            Cg.append(upper_pairs(connected_correlations(gen, q)))
            Ct.append(upper_pairs(connected_correlations(ref, q)))
            if verbose:
                g, o = per_clade[node.name]["generated_vs_test"], per_clade[node.name]["oracle_vs_test"]
                print(f"{node.name}: r_C gen {g['pearson_C']:.3f} (oracle {o['pearson_C']:.3f}), "
                      f"r_J {per_clade[node.name]['parameters']['pearson_J_contacts']:.3f}", flush=True)
        metrics["per_clade"] = per_clade

        # which test clade does each generated clade resemble most?  (r_C between all pairs)
        n = len(lv)
        R = np.array([[pearson(Cg[a], Ct[b]) for b in range(n)] for a in range(n)])
        metrics["clade_assignment_accuracy"] = float(np.mean(R.argmax(axis=1) == np.arange(n)))
        names = [x.name for x in lv]
        save(plot_distance_matrix(R, names, "r_C(generated clade [row], test clade [col])", "Pearson r_C"),
             fig_dir / "cross_clade_rC.png")

        per_family = {}
        for f in range(hier["n_families"]):
            ref = test.S[test.family == f]
            gen = sample(model.to_potts((f, -1, -1)), len(ref), sc["n_burnin"], sc["seed"] + 500 + f)
            write_fasta(samp_dir / f"family_F{f}.fasta", gen, [f"F{f}|gen{m:05d}" for m in range(len(gen))])
            per_family[f"F{f}"] = {"generated_vs_test": compare_samples(gen, ref, q),
                                   "assigned_family_proportions": family_assignment(pca.transform(gen), Zt, test.family)}
        metrics["per_family_tag_only"] = per_family

        # no tag at all: the conditional model's own "unconditioned" mode (learned through tag dropout)
        gen0 = sample(model.to_potts((-1, -1, -1)), len(test), sc["n_burnin"], sc["seed"] + 900)
        write_fasta(samp_dir / "no_tag.fasta", gen0, [f"notag|gen{m:05d}" for m in range(len(gen0))])
        metrics["no_tag"] = {"generated_vs_test": compare_samples(gen0, test.S, q),
                             "family_proportions": family_assignment(pca.transform(gen0), Zt, test.family)}
        # all 18 clade-tag samples pooled: equal numbers per clade by construction
        metrics["all_clade_tags_pooled"] = {"generated_vs_test": compare_samples(np.concatenate(gens), test.S, q),
                                            "family_proportions": family_assignment(pca.transform(np.concatenate(gens)), Zt, test.family)}

        # ---------------------------------------------------------------- distances (psg.distances)
        paths = [x.path for x in lv]
        test_by = [test.S[test.leaf == k] for k in range(n)]
        nn_by = [nn_test[test.leaf == k] for k in range(n)]
        dist = {}
        for k, x in enumerate(lv):
            dist[x.name] = {"generated_vs_test": distance_report(gens[k], test_by[k], train.S, q, pca, nn_ref=nn_by[k], seed=k),
                            "oracle_vs_test": distance_report(oracles[k], test_by[k], train.S, q, pca, nn_ref=nn_by[k], seed=k)}
        # Frechet distance in PCA space: generated clade a vs test clade b, and the ground-truth ladder
        Zg = [pca.transform(g) for g in gens]
        Ztk = [pca.transform(t) for t in test_by]
        FD = np.array([[frechet_distance(Zg[a], Ztk[b]) for b in range(n)] for a in range(n)])
        ladder: dict = {3: [frechet_distance(pca.transform(oracles[k]), Ztk[k]) for k in range(n)]}
        W1_ladder: dict = {3: [dist[x.name]["oracle_vs_test"]["W1_within_generated_vs_within_reference"] for x in lv]}
        for a in range(n):
            for b in range(a + 1, n):
                lvl = common_ancestor_level(paths[a], paths[b])
                ladder.setdefault(lvl, []).append(frechet_distance(Ztk[a], Ztk[b]))
        rel = {0: "different\nfamily", 1: "same family,\ndiff. subfamily", 2: "sibling\nclade", 3: "same clade\n(oracle)"}
        metrics["distances"] = {
            "per_clade": dist,
            "frechet_assignment_accuracy": float(np.mean(FD.argmin(axis=1) == np.arange(n))),
            "frechet_generated_own_clade_mean": float(np.mean(np.diag(FD))),
            "frechet_ladder_test_vs_test": {rel[k].replace("\n", " "): {"mean": float(np.mean(v)), "min": float(np.min(v)),
                                                                        "max": float(np.max(v))} for k, v in sorted(ladder.items())},
        }
        save(plot_distance_matrix(FD, names, "Frechet distance in PCA space: generated clade [row] vs test clade [col]",
                                  "Frechet distance"), fig_dir / "cross_clade_frechet.png")
        g = {k: np.array(v) for k, v in sorted(ladder.items())}
        g[4] = np.diag(FD)
        rel[4] = "generated\n(own clade tag)"
        save(plot_distance_by_relation(g, rel, "Frechet distance to a clade's test sequences", "Frechet distance (PCA)"),
             fig_dir / "frechet_ladder.png")
        rs = np.random.default_rng(1)
        allgen, allora = np.concatenate(gens), np.concatenate(oracles)
        save(plot_distance_distributions({
            "generated -> train": nearest_neighbor_distance(allgen, train.S, q),
            "test -> train": nn_test,
            "oracle -> train": nearest_neighbor_distance(allora, train.S, q)},
            "Distance to the nearest training sequence (all clade tags)", "d_NN (normalised Hamming)", step=1 / hier["L"]),
            fig_dir / "dist_nn_to_train.png")
        k0 = 0
        save(plot_distance_distributions({
            "within generated": pairwise_distances(gens[k0], rng=rs), "within test": pairwise_distances(test_by[k0], rng=rs),
            "generated - test": pairwise_distances(gens[k0], test_by[k0], rng=rs),
            "within oracle": pairwise_distances(oracles[k0], rng=rs)},
            f"Pairwise Hamming distances, clade {names[k0]}", step=1 / hier["L"]), fig_dir / f"dist_pairwise_{names[k0]}.png")

        # site frequencies and correlations for one example clade
        k0 = 0
        ref0 = test.S[test.leaf == k0]
        f_t, f_g = site_frequencies(ref0, q), site_frequencies(gens[k0], q)
        C_t, C_g = connected_correlations(ref0, q), connected_correlations(gens[k0], q)
        save(plot_stat_comparison(f_t.ravel(), f_g.ravel(), upper_pairs(C_t), upper_pairs(C_g),
                                  per_clade[lv[k0].name]["generated_vs_test"], f"test {lv[k0].name}",
                                  f"generated, tag {lv[k0].name}"), fig_dir / f"freq_corr_{lv[k0].name}.png")
        save(plot_frequency_maps(f_t, f_g, AMINO_ACIDS, f"test {lv[k0].name}", f"generated {lv[k0].name}"),
             fig_dir / f"site_frequencies_{lv[k0].name}.png")
        save(plot_correlation_triangles(C_t, C_g, contacts, "test", "generated"),
             fig_dir / f"correlation_strength_{lv[k0].name}.png")

        allgen = np.concatenate(gens)
        lab = np.array([x.path for x in lv])[np.repeat(np.arange(n), [len(g) for g in gens])]
        save(pca_figure(pca, test, allgen, lab[:, 0], lab[:, 1], "Generated, conditioned on each clade tag"),
             fig_dir / "pca_generated_vs_test.png")

        # r_C per clade: generated vs oracle
        g = [per_clade[x]["generated_vs_test"]["pearson_C"] for x in names]
        o = [per_clade[x]["oracle_vs_test"]["pearson_C"] for x in names]
        fig, ax = plt.subplots(figsize=(9, 3.4))
        x = np.arange(n)
        ax.bar(x - 0.2, o, 0.4, color=SERIES[0], label="oracle (fresh true samples)")
        ax.bar(x + 0.2, g, 0.4, color=SERIES[1], label="generated by the model")
        ax.set_xticks(x, names, rotation=90, fontsize=7)
        ax.set(ylabel="Pearson r_C vs test", title="Connected correlations reproduced, per clade")
        ax.legend(loc="lower right")
        save(fig, fig_dir / "rC_per_clade.png")

    out = root / "evaluations" / run
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-config", default=str(PROJECT_ROOT / "configs" / "ground_truth.json"))
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    a = ap.parse_args()
    main(json.loads(Path(a.config).read_text()), json.loads(Path(a.data_config).read_text()), Path(a.root))
