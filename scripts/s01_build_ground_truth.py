"""Step A - build the hierarchical tree of ground-truth Potts models and check the hierarchy on J.

Outputs (under --root, default = project root):
    data/ground_truth_potts/<node>.npz, contact_map.npy, tree.json
    evaluations/ground_truth/hierarchy_distances.json
    figures/ground_truth/01_hierarchy/*.png

    python scripts/s01_build_ground_truth.py --config configs/ground_truth.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psg import PROJECT_ROOT  # noqa: E402
from psg.hierarchy import (  # noqa: E402
    RELATION_NAMES,
    HierarchyConfig,
    build_hierarchy,
    distance_matrix,
    distances_by_relation,
    expected_distances,
    leaves,
    save_hierarchy,
)
from psg.plots import plot_contact_map, plot_distance_by_relation, plot_distance_matrix, save  # noqa: E402


def main(config: dict, root: Path = PROJECT_ROOT) -> dict:
    cfg = HierarchyConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in config["hierarchy"].items()})
    nodes, contacts = build_hierarchy(cfg)
    save_hierarchy(nodes, contacts, cfg, root / "data" / "ground_truth_potts")

    lv = leaves(nodes)
    names = [n.name for n in lv]
    models = [n.model for n in lv]
    fig_dir = root / "figures" / "ground_truth" / "01_hierarchy"
    report = {"n_nodes": len(nodes), "n_leaves": len(lv), "n_contacts": int(contacts.sum() // 2)}

    theory = expected_distances(cfg, contacts)
    for kind, label in [("J", "||J^a - J^b||_F"), ("h", "||h^a - h^b||_F")]:
        D = distance_matrix(models, kind)
        g = distances_by_relation(lv, D)
        strict = all(g[k + 1].max() < g[k].min() for k in range(len(g) - 1))
        report[kind] = {
            "by_relation": {RELATION_NAMES[k].replace("\n", " "): {"mean": float(v.mean()), "min": float(v.min()),
                                                                    "max": float(v.max()), "n": int(len(v)),
                                                                    "rms": float(np.sqrt((v ** 2).mean())),
                                                                    "theory_rms": theory[kind][k]}
                            for k, v in g.items()},
            "strictly_hierarchical": bool(strict),
        }
        save(plot_distance_matrix(D, names, f"{label} between clades"), fig_dir / f"{kind}_distance_matrix.png")
        save(plot_distance_by_relation(g, RELATION_NAMES, f"{label} by relation of the two clades", label,
                                        theory=theory[kind]),
             fig_dir / f"{kind}_distance_by_relation.png")
    save(plot_contact_map(contacts), fig_dir / "contact_map.png")

    out = root / "evaluations" / "ground_truth" / "hierarchy_distances.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "ground_truth.json"))
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    a = ap.parse_args()
    rep = main(json.loads(Path(a.config).read_text()), Path(a.root))
    print(json.dumps(rep, indent=2))
