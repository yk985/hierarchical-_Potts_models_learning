"""Step 15 - end-to-end diffusion pipeline on a tiny tree: dataset -> s09 train -> s10 evaluate."""
import importlib.util
import json
import math

from psg import PROJECT_ROOT

TINY = {
    "hierarchy": {"L": 15, "q": 5, "n_families": 2, "n_subfamilies": 2, "n_clades": 2,
                  "sigma_h": [1.5, 0.9, 0.5, 0.3], "sigma_J": [0.8, 0.45, 0.27, 0.15],
                  "n_long_contacts": 15, "min_separation": 4, "seed": 0},
    "sampling": {"n_per_clade": 300, "n_burnin": 60, "beta": 1.0, "seed": 1},
    "analysis": {"n_components": 4},
    "dataset": {"n_per_clade": 600, "n_burnin": 60, "fractions": [0.8, 0.1, 0.1], "seed": 100},
}
RUN = {
    "run_name": "tiny_md",
    "model": {"class": "MaskedDiffusionTransformer", "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64,
              "dropout": 0.0, "seed": 0},
    "train": {"epochs": 4, "batch_size": 128, "lr": 0.001, "weight_decay": 0.0, "warmup_steps": 20,
              "grad_clip": 1.0, "ema_decay": 0.9, "val_repeats": 2, "patience": 10, "seed": 0, "amp": True},
    "sampling": {"n_steps": [15, 45], "continuous_time": True, "seed": 7, "n_burnin_oracle": 60, "amp": True},
}


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_diffusion_pipeline_tiny(outdir):
    for s in ["s01_build_ground_truth", "s02_sample_ground_truth", "s03_analyze_ground_truth", "s04_make_dataset"]:
        mod = load_script(s)
        mod.main(TINY, outdir, verbose=False) if s in ("s02_sample_ground_truth", "s04_make_dataset") else mod.main(TINY, outdir)
    hist = load_script("s09_train_diffusion").main(RUN, TINY, outdir, verbose=False)
    assert hist["val_nelbo_site"][-1] < hist["val_nelbo_site"][0]                  # it learns
    assert (outdir / "models" / "tiny_md" / "model.pt").exists()
    m = load_script("s10_evaluate_diffusion").main(RUN, TINY, outdir, verbose=False)
    json.dumps(m)
    assert math.isfinite(m["test_nelbo_site"]) and m["test_nelbo_site"] > 0
    sc = m["site_conditionals"]
    assert 0 <= sc["tv_to_true_conditional_mean"] <= 1 and sc["pseudo_nll_site_model"] > 0
    assert set(m["by_n_steps"]) == {"15", "45", "continuous"}
    assert abs(sum(m["family_proportions"]["generated"].values()) - 1) < 1e-9
    assert (outdir / "sampling" / "tiny_md" / "all_T45.fasta").exists()
    assert (outdir / "sampling" / "tiny_md" / "all_continuous.fasta").exists()
    assert (outdir / "figures" / "models" / "tiny_md" / "quality_vs_steps.png").exists()
