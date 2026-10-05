"""Step D - training dataset: top up every clade to n_per_clade sequences (GPU Gibbs), then split each clade
train / val / test. The test split is never used for training or model selection.

Outputs (under --root):
    data/splits/{train,val,test}.npz        LabelledSequences (S + family / subfamily / clade / leaf)
    data/splits/test_fasta/<clade>.fasta    held-out sequences, one file per clade
    evaluations/ground_truth/dataset.json

    python scripts/s04_make_dataset.py --config configs/ground_truth.json
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
from psg.data import LabelledSequences, split_per_clade, top_up  # noqa: E402
from psg.hierarchy import leaves, load_hierarchy  # noqa: E402


def main(config: dict, root: Path = PROJECT_ROOT, verbose: bool = True) -> dict:
    dc = config["dataset"]
    nodes, _, _ = load_hierarchy(root / "data" / "ground_truth_potts")
    lv = leaves(nodes)
    base = LabelledSequences.load(root / "data" / "sequences" / "all_sequences.npz")
    full = top_up(base, lv, dc["n_per_clade"], dc["n_burnin"], dc["seed"], verbose=verbose)
    sp = split_per_clade(full.leaf, tuple(dc["fractions"]), seed=dc["seed"])
    out = root / "data" / "splits"
    names = np.array([n.name for n in lv])
    for k, idx in sp.items():
        full.subset(idx).save(out / f"{k}.npz", leaf_names=names)
    test = full.subset(sp["test"])
    for k, n in enumerate(lv):
        S = test.S[test.leaf == k]
        write_fasta(out / "test_fasta" / f"{n.name}.fasta", S, [f"{n.name}|test{m:05d}" for m in range(len(S))])
    rep = {"n_per_clade": dc["n_per_clade"], "sizes": {k: int(len(v)) for k, v in sp.items()},
           "per_clade": {k: int(len(v) // len(lv)) for k, v in sp.items()}}
    (root / "evaluations" / "ground_truth").mkdir(parents=True, exist_ok=True)
    (root / "evaluations" / "ground_truth" / "dataset.json").write_text(json.dumps(rep, indent=2))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "ground_truth.json"))
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    a = ap.parse_args()
    print(json.dumps(main(json.loads(Path(a.config).read_text()), Path(a.root)), indent=2))
