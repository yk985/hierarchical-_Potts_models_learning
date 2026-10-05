"""Step 5 - end-to-end pipeline (build tree -> sample -> analyse) on a tiny tree.
All outputs are written inside this test's outcome folder (figures/tests/test_05_pipeline/...)."""
import importlib.util
import json

import numpy as np

from psg import PROJECT_ROOT
from psg.alphabet import read_fasta

TINY = {
    "hierarchy": {"L": 15, "q": 5, "n_families": 2, "n_subfamilies": 2, "n_clades": 2,
                  "sigma_h": [1.5, 0.9, 0.5, 0.3], "sigma_J": [0.8, 0.45, 0.27, 0.15],
                  "n_long_contacts": 15, "min_separation": 4, "seed": 0},
    "sampling": {"n_per_clade": 300, "n_burnin": 60, "beta": 1.0, "seed": 1},
    "analysis": {"n_components": 4},
}


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pipeline_tiny(outdir):
    rep1 = load_script("s01_build_ground_truth").main(TINY, outdir)
    assert rep1["n_leaves"] == 8 and rep1["J"]["strictly_hierarchical"]
    assert (outdir / "data" / "ground_truth_potts" / "F1.S1.C1.npz").exists()

    load_script("s02_sample_ground_truth").main(TINY, outdir, verbose=False)
    headers, S = read_fasta(outdir / "data" / "sequences" / "F0.S1.C0.fasta")
    assert S.shape == (300, 15) and headers[0].startswith("F0.S1.C0|")
    d = np.load(outdir / "data" / "sequences" / "all_sequences.npz")
    assert d["S"].shape == (8 * 300, 15)
    assert (np.bincount(d["family"]) == 4 * 300).all()

    rep3 = load_script("s03_analyze_ground_truth").main(TINY, outdir)
    # the sequence statistics must also be hierarchical: same clade > sibling clade > same family > other family
    means = [v["mean"] for v in rep3["pearson_C"].values()]   # keys ordered: diff fam, same fam, same sub, same clade
    assert means == sorted(means), rep3["pearson_C"]
    ident = [v["mean"] for v in rep3["identity"].values()]
    assert ident == sorted(ident)
    assert (outdir / "figures" / "ground_truth" / "03_analysis" / "pca_all_PC1_PC2.png").exists()
    json.dumps(rep3)
