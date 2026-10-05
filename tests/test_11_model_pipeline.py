"""Step 11 - end-to-end: ground truth -> dataset -> train (both models, a few epochs) -> sample -> evaluate,
on a tiny tree. Everything is written inside this test's outcome folder (figures/tests/test_11_model_pipeline/...)."""
import copy
import importlib.util
import json
import math

import numpy as np

from psg import PROJECT_ROOT
from psg.data import LabelledSequences

TINY = {
    "hierarchy": {"L": 15, "q": 5, "n_families": 2, "n_subfamilies": 2, "n_clades": 2,
                  "sigma_h": [1.5, 0.9, 0.5, 0.3], "sigma_J": [0.8, 0.45, 0.27, 0.15],
                  "n_long_contacts": 15, "min_separation": 4, "seed": 0},
    "sampling": {"n_per_clade": 300, "n_burnin": 60, "beta": 1.0, "seed": 1},
    "analysis": {"n_components": 4},
    "dataset": {"n_per_clade": 600, "n_burnin": 60, "fractions": [0.8, 0.1, 0.1], "seed": 100},
}
RUN = {
    "model": {"n_heads": 16, "d_k": 4, "init_v": 0.01, "seed": 0},
    "train": {"epochs": 4, "batch_size": 256, "lr": 0.02, "lam_h": 1e-4, "lam_J": 1e-4, "patience": 8, "seed": 0},
    "sampling": {"n_burnin": 60, "seed": 7},
}


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_cfg(conditional):
    r = copy.deepcopy(RUN)
    r["run_name"] = "tiny_conditional" if conditional else "tiny_unconditional"
    r["model"]["class"] = "ConditionalFactoredAttentionPotts" if conditional else "FactoredAttentionPotts"
    if conditional:
        r["train"].update(lam_label=1e-3, tag_dropout=[0.1, 0.1, 0.15])
    return r


def test_model_pipeline_tiny(outdir):
    load_script("s01_build_ground_truth").main(TINY, outdir)
    load_script("s02_sample_ground_truth").main(TINY, outdir, verbose=False)
    load_script("s03_analyze_ground_truth").main(TINY, outdir)
    rep = load_script("s04_make_dataset").main(TINY, outdir, verbose=False)
    assert rep["sizes"] == {"train": 8 * 480, "val": 8 * 60, "test": 8 * 60}
    test = LabelledSequences.load(outdir / "data" / "splits" / "test.npz")
    train_ = LabelledSequences.load(outdir / "data" / "splits" / "train.npz")
    assert not ({s.tobytes() for s in train_.S} >= {s.tobytes() for s in test.S})   # test is not a subset of train

    s05, s06 = load_script("s05_train_factored_attention"), load_script("s06_evaluate_model")
    s07, s08 = load_script("s07_compare_to_tree"), load_script("s08_mode_mixing")
    for conditional in [False, True]:
        cfg = run_cfg(conditional)
        hist = s05.main(cfg, TINY, outdir, verbose=False)
        assert hist["val_pnll_site"][-1] < hist["val_pnll_site"][0]              # it learns something
        assert (outdir / "models" / cfg["run_name"] / "model.pt").exists()
        m = s06.main(cfg, TINY, outdir, verbose=False)
        json.dumps(m)
        if conditional:
            assert len(m["per_clade"]) == 8 and len(m["per_family_tag_only"]) == 2
            d = m["distances"]["per_clade"]["F0.S0.C0"]["generated_vs_test"]
            assert 0 <= d["nn_to_train_generated_mean"] <= 1 and d["frechet_pca"] >= 0
            assert set(m["distances"]["frechet_ladder_test_vs_test"]) >= {"different family", "same clade (oracle)"}
            assert all(math.isfinite(v["generated_vs_test"]["pearson_C"]) for v in m["per_clade"].values())
            assert (outdir / "sampling" / cfg["run_name"] / "F1.S1.C1.fasta").exists()
        else:
            assert math.isfinite(m["all_clades"]["generated_vs_test"]["pearson_f1"])
            assert 0.0 <= m["parameters"]["contact_precision_long_range"] <= 1.0
            assert 0 <= m["distances"]["generated_vs_test"]["fraction_copies_of_train"] <= 1
        assert (outdir / "figures" / "models" / cfg["run_name"] / "pca_generated_vs_test.png").exists()
        tree = s07.main(cfg, TINY, outdir)
        assert len(tree["learned"]["nodes"]) == 1 + 2 + 4 + 8                   # compared with every tree node
        if conditional:                                                          # one tag per node, matched
            assert len(tree["per_label"]) == 15 and tree["per_label"]["F1.S1.C1"]["tag"] == [1, 3, 7]
            assert tree["per_label"]["F1"]["tag"] == [1, -1, -1] and "delta_vs_parent_r" in tree["per_label"]["F1.S0"]
        mix = s08.main(cfg, TINY, outdir, n_chains=40, n_checkpoints=2, every=10)
        assert len(mix["start_random"]) == 2 and "start_F1" in mix
        assert set(mix["energy_entropy_decomposition"]) == {"F0", "F1"}
        if not conditional:
            assert abs(sum(m["family_proportions"]["generated"].values()) - 1) < 1e-9
