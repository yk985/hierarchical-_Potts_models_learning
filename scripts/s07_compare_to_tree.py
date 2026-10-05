"""Step G - compare the learned (h, J) of a trained model with EVERY Potts model of the ground-truth tree:
the root (the ancestor all others were generated from), the 3 families, 9 subfamilies and 18 clades,
plus the average of the 18 clade models. Everything in the zero-sum gauge.

For the conditional model, each label (every clade tag, every family tag, and no tag) is compared.

Outputs (under --root):
    evaluations/<run_name>/tree_comparison.json
    figures/models/<run_name>/tree_*.png

    python scripts/s07_compare_to_tree.py --config configs/fa_unconditional.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psg import PROJECT_ROOT  # noqa: E402
from psg.evaluation import compare_to_nodes, coupling_recovery  # noqa: E402
from psg.hierarchy import coupling_distance, field_distance, leaves, load_hierarchy  # noqa: E402
from psg.models.factored_attention import ConditionalFactoredAttentionPotts, load_model  # noqa: E402
from psg.plots import INK, INK_2, SERIES, plot_distance_matrix, plt, save  # noqa: E402
from psg.potts import Potts  # noqa: E402

LEVEL_NAMES = ["root", "family", "subfamily", "clade"]


def strip_by_level(res: dict, key: str, ylabel: str, title: str, ref: float | None = None, ref_label: str = ""):
    """One column per tree level; one dot per node, coloured by its family (root in ink)."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    rng = np.random.default_rng(0)
    for name, r in res.items():
        x = r["level"] + rng.uniform(-0.12, 0.12) * (r["level"] > 0)
        color = INK if r["family"] < 0 else SERIES[r["family"]]
        ax.scatter(x, r[key], s=40, color=color, edgecolor="white", lw=1, zorder=3)
    best = min(res, key=lambda n: res[n][key]) if key.startswith("d_") else max(res, key=lambda n: res[n][key])
    ax.annotate(best, (res[best]["level"], res[best][key]), textcoords="offset points", xytext=(8, 0),
                fontsize=8, color=INK, va="center")
    if ref is not None:
        ax.axhline(ref, color=INK_2, lw=1, ls="--", label=ref_label)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", color=INK, label="root")]
    h += [Line2D([], [], ls="", marker="o", color=SERIES[f], label=f"family F{f} branch")
          for f in sorted({r["family"] for r in res.values() if r["family"] >= 0})]
    if ref is not None:
        h.append(Line2D([], [], color=INK_2, ls="--", label=ref_label))
    ax.legend(handles=h, fontsize=7, loc="best")
    ax.set_xticks(range(4), LEVEL_NAMES)
    ax.grid(axis="x", visible=False)
    ax.set(ylabel=ylabel, title=title)
    return fig


def scatter_J(learned: Potts, targets: list[tuple[str, Potts]], contacts: np.ndarray):
    """Learned vs true couplings on the contact entries (zero-sum gauge), one panel per target model;
    last panel: the fields."""
    a = learned.zero_sum_gauge()
    iu = np.triu_indices(a.L, 1)
    c = contacts[iu]
    fig, axes = plt.subplots(1, len(targets) + 1, figsize=(4 * (len(targets) + 1), 3.9))
    for ax, (name, t) in zip(axes, targets):
        b = t.zero_sum_gauge()
        x, y = b.J[iu][c].ravel(), a.J[iu][c].ravel()
        r = coupling_recovery(learned, t, contacts)
        lim = np.abs(x).max()
        ax.plot([-lim, lim], [-lim, lim], color=INK_2, lw=1, ls="--", label="y = x")
        ax.scatter(x, y, s=2, alpha=0.25, color=SERIES[0], lw=0, rasterized=True)
        ax.set(xlabel=f"true J ({name})", ylabel="learned J",
               title=f"{name}: r = {r['pearson_J_contacts']:.2f}, slope = {r['slope_J_contacts']:.2f}")
        ax.legend(loc="upper left")
    ax = axes[-1]
    for k, (name, t) in enumerate(targets[:3]):
        b = t.zero_sum_gauge()
        ax.scatter(b.h.ravel(), a.h.ravel(), s=4, alpha=0.4, color=SERIES[k], lw=0,
                   label=f"{name}: r = {coupling_recovery(learned, t)['pearson_h']:.2f}")
    lim = np.abs(a.h).max()
    ax.plot([-lim, lim], [-lim, lim], color=INK_2, lw=1, ls="--")
    ax.set(xlabel="true h", ylabel="learned h", title="fields h_i(a)")
    ax.legend(fontsize=7, markerscale=3)
    fig.tight_layout()
    return fig


def tag_of(node, nS: int) -> tuple:
    """The conditional-model tag that corresponds to a tree node (root = no tag)."""
    p = node.path
    if node.level == 0:
        return (-1, -1, -1)
    if node.level == 1:
        return (p[0], -1, -1)
    if node.level == 2:
        return (p[0], p[0] * nS + p[1], -1)
    return (p[0], p[0] * nS + p[1], None)       # clade index filled by the caller (global leaf index)


def delta_recovery(learned_child: Potts, learned_parent: Potts, true_child: Potts, true_parent: Potts,
                   contacts: np.ndarray) -> float:
    """Pearson r between the learned change of J from parent tag to child tag and the TRUE change from the
    parent node to the child node (zero-sum gauge, contact entries): did the model learn what is specific
    to the child, not only what it shares with its parent?"""
    iu = np.triu_indices(true_child.L, 1)
    c = contacts[iu]
    dl = (learned_child.zero_sum_gauge().J - learned_parent.zero_sum_gauge().J)[iu][c].ravel()
    dt = (true_child.zero_sum_gauge().J - true_parent.zero_sum_gauge().J)[iu][c].ravel()
    return float(np.corrcoef(dl, dt)[0, 1])


def delta_h_recovery(learned_child: Potts, learned_parent: Potts, true_child: Potts, true_parent: Potts) -> float:
    """Same as delta_recovery for the fields h (zero-sum gauge)."""
    dl = (learned_child.zero_sum_gauge().h - learned_parent.zero_sum_gauge().h).ravel()
    dt = (true_child.zero_sum_gauge().h - true_parent.zero_sum_gauge().h).ravel()
    return float(np.corrcoef(dl, dt)[0, 1])


def compare_conditional(model, nodes: dict, contacts: np.ndarray, nS: int, fig_dir: Path) -> dict:
    """Every tree node gets the tag that should reproduce it (root <- no tag, F1 <- family tag, F1.S2 <- family +
    subfamily tag, F1.S2.C0 <- full tag). The learned J(tag) is compared with ALL nodes; the matched node should
    be the closest one (the diagonal of the label x node matrix)."""
    names = list(nodes)
    leaf_index = {n.name: k for k, n in enumerate(leaves(nodes))}
    tags = {}
    for name, n in nodes.items():
        t = tag_of(n, nS)
        tags[name] = t if t[2] is not None else (t[0], t[1], leaf_index[name])
    learned = {name: model.to_potts(t) for name, t in tags.items()}

    R = np.zeros((len(names), len(names)))
    Dm = np.zeros_like(R)
    per_label = {}
    for a, lab in enumerate(names):
        res = compare_to_nodes(learned[lab], nodes, contacts)
        R[a] = [res[n]["pearson_J_contacts"] for n in names]
        Dm[a] = [res[n]["d_J"] for n in names]
        order = sorted(names, key=lambda n: res[n]["d_J"])
        m = res[lab]
        per_label[lab] = {"tag": list(tags[lab]), "matched": {k: m[k] for k in
                          ["pearson_J_contacts", "slope_J_contacts", "pearson_J_all_pairs", "rel_error_J", "d_J",
                           "pearson_h", "d_h", "mean_abs_J_non_contacts"]},
                          "rank_of_matched_by_d_J": order.index(lab) + 1, "closest_node": order[0],
                          "best_other_r_J": max(res[n]["pearson_J_contacts"] for n in names if n != lab)}
        if nodes[lab].parent is not None:
            par = nodes[lab].parent
            per_label[lab]["delta_vs_parent_r"] = delta_recovery(learned[lab], learned[par], nodes[lab].model,
                                                                 nodes[par].model, contacts)
            per_label[lab]["delta_h_vs_parent_r"] = delta_h_recovery(learned[lab], learned[par], nodes[lab].model,
                                                                     nodes[par].model)

    levels = [nodes[n].level for n in names]
    summary = {}
    for lvl, lname in enumerate(LEVEL_NAMES):
        labs = [n for n in names if nodes[n].level == lvl]
        summary[lname] = {
            "n": len(labs),
            "matched_r_J_mean": float(np.mean([per_label[x]["matched"]["pearson_J_contacts"] for x in labs])),
            "matched_slope_mean": float(np.mean([per_label[x]["matched"]["slope_J_contacts"] for x in labs])),
            "matched_d_J_mean": float(np.mean([per_label[x]["matched"]["d_J"] for x in labs])),
            "matched_r_h_mean": float(np.mean([per_label[x]["matched"]["pearson_h"] for x in labs])),
            "fraction_matched_is_closest": float(np.mean([per_label[x]["rank_of_matched_by_d_J"] == 1 for x in labs])),
            "delta_vs_parent_r_mean": (float(np.mean([per_label[x]["delta_vs_parent_r"] for x in labs])) if lvl else None),
            "delta_h_vs_parent_r_mean": (float(np.mean([per_label[x]["delta_h_vs_parent_r"] for x in labs])) if lvl else None),
        }
    save(plot_distance_matrix(R, names, "r_J(learned J(tag) [row], true J(node) [col])", "Pearson r (J, contacts)"),
         fig_dir / "tree_matrix_rJ.png")
    save(plot_distance_matrix(Dm, names, "d_J(learned J(tag) [row], true J(node) [col])", "||dJ||_F"),
         fig_dir / "tree_matrix_dJ.png")

    # matched r_J by level, next to the best NON-matched node for the same tag
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    rng = np.random.default_rng(0)
    for lab in names:
        lvl, x0 = nodes[lab].level, nodes[lab].level + rng.uniform(-0.12, 0.12) * (nodes[lab].level > 0)
        color = INK if nodes[lab].level == 0 else SERIES[nodes[lab].path[0]]
        ax.scatter(x0 - 0.18, per_label[lab]["matched"]["pearson_J_contacts"], s=40, color=color, edgecolor="white", zorder=3)
        ax.scatter(x0 + 0.18, per_label[lab]["best_other_r_J"], s=40, facecolor="none", edgecolor=color, zorder=3)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color=INK_2, label="matched node (right tensor)"),
                       Line2D([], [], ls="", marker="o", mfc="none", color=INK_2, label="best other node")],
              fontsize=7, loc="lower left")
    ax.set_xticks(range(4), [f"{n}\n(tag: {t})" for n, t in zip(LEVEL_NAMES, ["none", "family", "+subfamily", "+clade"])])
    ax.grid(axis="x", visible=False)
    ax.set(ylabel="Pearson r (J, contacts)", title="J(tag) vs the tensor it should reproduce")
    save(fig, fig_dir / "tree_matched_rJ.png")

    # scatter: matched comparisons for one clade, its family, and the root
    ex = leaves(nodes)[0].name
    fam = ex.split(".")[0]
    for name in [ex, fam, "root"]:
        t = [(f"true {name}", nodes[name].model)]
        save(scatter_J(learned[name], t, contacts), fig_dir / f"tree_scatter_matched_{name}.png")
    return {"per_label": per_label, "summary_by_level": summary, "node_order": names, "levels": levels}


def main(run_cfg: dict, data_cfg: dict, root: Path = PROJECT_ROOT) -> dict:
    run = run_cfg["run_name"]
    nS = data_cfg["hierarchy"]["n_subfamilies"]
    model = load_model(root / "models" / run / "model.pt")
    nodes, contacts, _ = load_hierarchy(root / "data" / "ground_truth_potts")
    lv = leaves(nodes)
    mean_clade = Potts(np.mean([n.model.h for n in lv], 0), np.mean([n.model.J for n in lv], 0))

    fig_dir = root / "figures" / "models" / run
    if isinstance(model, ConditionalFactoredAttentionPotts):
        out = compare_conditional(model, nodes, contacts, nS, fig_dir)
        learned = {"learned": model.to_potts((-1, -1, -1))}      # the no-tag model, same figures as unconditioned
    else:
        out, learned = {}, {"learned": model.to_potts()}

    for lab, P in learned.items():
        res = compare_to_nodes(P, nodes, contacts)
        ref = coupling_recovery(P, mean_clade, contacts) | {"d_J": coupling_distance(P, mean_clade),
                                                             "d_h": field_distance(P, mean_clade)}
        out[lab] = {"nodes": res, "mean_of_clades": ref,
                    "closest_node_by_d_J": min(res, key=lambda n: res[n]["d_J"]),
                    "best_node_by_r_J": max(res, key=lambda n: res[n]["pearson_J_contacts"])}
        tag = "" if lab == "learned" else "_" + lab.replace(" ", "_")
        save(strip_by_level(res, "pearson_J_contacts", "Pearson r (J, contacts)",
                            f"Learned J vs every tree model{tag}", ref["pearson_J_contacts"], "mean of 18 clades"),
             fig_dir / f"tree_rJ{tag}.png")
        save(strip_by_level(res, "d_J", "||J_learned - J_node||_F", f"Distance to every tree model{tag}",
                            ref["d_J"], "mean of 18 clades"), fig_dir / f"tree_dJ{tag}.png")
        save(strip_by_level(res, "pearson_h", "Pearson r (h)", f"Learned h vs every tree model{tag}",
                            ref["pearson_h"], "mean of 18 clades"), fig_dir / f"tree_rh{tag}.png")
        best = out[lab]["best_node_by_r_J"]
        targets = [("root", nodes["root"].model), ("mean of clades", mean_clade)]
        if best not in ("root",):
            targets.append((best, nodes[best].model))
        save(scatter_J(P, targets, contacts), fig_dir / f"tree_scatter_J{tag}.png")

    # reference: how far apart the TRUE models are (to read the distances above)
    out["reference_true_distances"] = {
        "root_to_family_mean": float(np.mean([coupling_distance(nodes["root"].model, n.model)
                                              for n in nodes.values() if n.level == 1])),
        "root_to_clade_mean": float(np.mean([coupling_distance(nodes["root"].model, n.model) for n in lv])),
        "mean_of_clades_to_root": coupling_distance(mean_clade, nodes["root"].model),
        "norm_J_root": coupling_distance(nodes["root"].model, Potts(np.zeros_like(mean_clade.h), np.zeros_like(mean_clade.J))),
    }
    ev = root / "evaluations" / run
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "tree_comparison.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-config", default=str(PROJECT_ROOT / "configs" / "ground_truth.json"))
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    a = ap.parse_args()
    main(json.loads(Path(a.config).read_text()), json.loads(Path(a.data_config).read_text()), Path(a.root))
