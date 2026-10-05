"""Step N - refined speciation analysis (companion note docs/notes/speciation_refined.pdf).

1. Exact class posteriors. log Z_c of the 18 true clade models by thermodynamic integration; then
   P(c | x) = exp(-E_c(x) - log Z_c) / sum_c' (...)   for any complete sequence (equal priors).
   Checked on the labelled test set: accuracy, and calibration E[sum_c P^2] = E[P(true c)].
2. Soft-label cloning. As in s13, but every clone is labelled by its exact posterior instead of the pseudo-likelihood
   classifier's argmax. Clone agreement then estimates E[sum_c P(c | x_U)^2] for the LATENT class, with no classifier
   error. Its value at k = L is the exact ceiling E[sum_c P(c | x)^2] < 1 (classes overlap).
   Under-confidence test (no reference needed): soft clone-clone vs soft clone-original agreement.
3. Correlation-weighted profile theory. Every revealed site's log-likelihood is multiplied by a weight that discounts
   evidence shared with correlated revealed sites (modes 'geo', 'prod', 'eff'; psg.regimes.correlation_weights),
   rho = Linfoot's informational correlation of the within-class mutual information. Each variant is judged by its
   calibration gap E[sum Q^2] - E[Q(true)] (0 for a correct posterior).

Outputs: evaluations/ground_truth/log_partition.json, evaluations/<run>/regimes_refined.json,
         figures/models/<run>/regimes/refined_*.png

    python scripts/s14_regimes_refined.py --config configs/md_unconditional.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from psg import PROJECT_ROOT  # noqa: E402
from psg.data import LabelledSequences  # noqa: E402
from psg.evaluation import pseudo_loglik_scores_torch  # noqa: E402
from psg.hierarchy import leaves, load_hierarchy  # noqa: E402
from psg.models.masked_diffusion import continuous_time_sample, load_diffusion  # noqa: E402
from psg.plots import INK_2, SERIES, plt, save  # noqa: E402
from psg.regimes import (  # noqa: E402
    complete,
    correlation_weights,
    crossing,
    exact_log_likelihoods,
    information_correlation,
    pair_agreement,
    random_orders,
    reveal_first_k,
    soft_pair_agreement,
    thermodynamic_log_z,
    weighted_profile_log_posterior,
    within_class_mutual_information,
)
from psg.stats import site_frequencies  # noqa: E402
from s13_dynamical_regimes import LEVELS, level_maps  # noqa: E402

MODES = ["none", "geo", "prod", "eff"]
MODE_NAMES = {"none": "profile (independent sites)", "geo": "weighted: geometric mean of (1-rho)",
              "prod": "weighted: product of (1-rho)", "eff": "weighted: 1/(1+sum rho)"}


def normalise(phi, phi0, phi_end):
    return [(p - phi0) / (phi_end - phi0) for p in phi]


def main(run_cfg: dict, data_cfg: dict, root: Path = PROJECT_ROOT, n_start: int = 1000, n_clones: int = 8,
         n_theory: int = 5000, seed: int = 0, ti: dict | None = None, verbose: bool = True) -> dict:
    t0 = time.time()
    ti = ti or {"n_lambda": 21, "n_chains": 1000, "n_equil": 20, "n_measure": 10, "n_burnin": 200}
    run = run_cfg["run_name"]
    hier = data_cfg["hierarchy"]
    q, L, n_sub = hier["q"], hier["L"], hier["n_subfamilies"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(run_cfg["sampling"].get("amp", False)) and dev.type == "cuda"
    model = load_diffusion(root / "models" / run / "model.pt", dev).eval()

    def den(x, t):
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            return model(x, t)

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    nodes, _, _ = load_hierarchy(root / "data" / "ground_truth_potts")
    lv = leaves(nodes)
    models = [n.model for n in lv]
    maps = level_maps(lv, n_sub)
    K = {lvl: int(maps[lvl].max() + 1) for lvl in LEVELS}
    onehot_level = {lvl: np.eye(K[lvl])[maps[lvl]] for lvl in LEVELS}           # (18, K): leaf -> level
    train = LabelledSequences.load(root / "data" / "splits" / "train.npz")
    test = LabelledSequences.load(root / "data" / "splits" / "test.npz")
    true_level = {"family": test.family, "subfamily": test.family * n_sub + test.subfamily, "clade": test.leaf}
    train_level = {"family": train.family, "subfamily": train.family * n_sub + train.subfamily, "clade": train.leaf}
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    out: dict = {"run": run}

    # ------------------------------------------------ 1. partition functions and exact posteriors
    zfile = root / "evaluations" / "ground_truth" / "log_partition.json"
    if zfile.exists() and json.loads(zfile.read_text()).get("settings") == ti:
        zres = json.loads(zfile.read_text())
    else:
        per = [thermodynamic_log_z(m, seed=seed + c, device=dev, **ti) for c, m in enumerate(models)]
        zres = {"settings": ti, "names": [n.name for n in lv], "log_z": [r["log_z"] for r in per],
                "error": [r["error"] for r in per], "detail": per}
        zfile.parent.mkdir(parents=True, exist_ok=True)
        zfile.write_text(json.dumps(zres, indent=2))
    log_z = np.array(zres["log_z"])
    log(f"log Z: {np.round(log_z, 2).tolist()}, max hysteresis error {max(zres['error']):.3f}")

    def posterior(S: np.ndarray) -> np.ndarray:                               # (M, 18) exact P(clade | x)
        ll = exact_log_likelihoods(S, models, log_z, dev)
        return np.exp(ll - np.logaddexp.reduce(ll, axis=1, keepdims=True))

    P_test = posterior(test.S)
    pll = pseudo_loglik_scores_torch(test.S, models, dev).argmax(1)
    val = {}
    for lvl in LEVELS:
        Pl = P_test @ onehot_level[lvl]
        val[lvl] = {"accuracy_exact_posterior": float((Pl.argmax(1) == true_level[lvl]).mean()),
                    "accuracy_pll_classifier": float((maps[lvl][pll] == true_level[lvl]).mean()),
                    "E_sum_P2": float((Pl ** 2).sum(1).mean()),
                    "E_P_true": float(Pl[np.arange(len(Pl)), true_level[lvl]].mean())}
    out["exact_posterior_on_test"] = val
    log("exact posterior on test: " + ", ".join(f"{l}: acc {v['accuracy_exact_posterior']:.3f} (PLL "
                                                f"{v['accuracy_pll_classifier']:.3f}), E sumP2 {v['E_sum_P2']:.3f} "
                                                f"vs E P(true) {v['E_P_true']:.3f}" for l, v in val.items()))

    # ------------------------------------------------ 2. soft-label cloning
    k_grid = sorted(set(list(range(0, min(L, 20) + 1)) + [k for k in (24, 28, 34, 40, 50) if k < L] + [L]))
    idx_A = rng.choice(len(test), size=min(n_start, len(test)), replace=False)
    ens = {"A_test": torch.as_tensor(test.S[idx_A], dtype=torch.long, device=dev),
           "B_network": continuous_time_sample(den, n_start, L, q, seed=seed + 1, device=dev).to(dev)}
    clon: dict = {"k_grid": k_grid}
    for name, x0 in ens.items():
        rank = random_orders(len(x0), L, g, dev)
        P0 = posterior(x0.cpu().numpy())                                      # posterior of the originals
        leaf0 = pseudo_loglik_scores_torch(x0.cpu().numpy(), models, dev).argmax(1)
        res = {lvl: {k2: [] for k2 in ["hard_clone_clone", "hard_clone_original", "soft_clone_clone",
                                        "soft_clone_original"]} for lvl in LEVELS}
        for k in k_grid:
            if k == L:
                for lvl in LEVELS:
                    Pl = P0 @ onehot_level[lvl]
                    ceiling = float((Pl ** 2).sum(1).mean())
                    res[lvl]["hard_clone_clone"].append(1.0)
                    res[lvl]["hard_clone_original"].append(1.0)
                    res[lvl]["soft_clone_clone"].append(ceiling)
                    res[lvl]["soft_clone_original"].append(ceiling)
                continue
            C = complete(den, reveal_first_k(x0, rank, k, q), n_clones, q, g).reshape(-1, L).cpu().numpy()
            Pc = posterior(C).reshape(len(x0), n_clones, -1)
            leaf = pseudo_loglik_scores_torch(C, models, dev).argmax(1).reshape(len(x0), n_clones)
            for lvl in LEVELS:
                R = Pc @ onehot_level[lvl]                                       # (B, C, K)
                R0 = P0 @ onehot_level[lvl]                                      # (B, K)
                lab = maps[lvl][leaf]
                res[lvl]["hard_clone_clone"].append(float(pair_agreement(lab).mean()))
                res[lvl]["hard_clone_original"].append(float((lab == maps[lvl][leaf0][:, None]).mean()))
                res[lvl]["soft_clone_clone"].append(float(soft_pair_agreement(R).mean()))
                res[lvl]["soft_clone_original"].append(float((R * R0[:, None, :]).sum(-1).mean()))
        for lvl in LEVELS:
            r = res[lvl]
            r["ceiling"] = r["soft_clone_clone"][-1]
            r["psi_soft"] = normalise(r["soft_clone_clone"], r["soft_clone_clone"][0], r["ceiling"])
            r["k_S_soft"] = crossing(k_grid, r["psi_soft"], 0.5)
            r["psi_hard"] = normalise(r["hard_clone_clone"], r["hard_clone_clone"][0], 1.0)
            r["k_S_hard"] = crossing(k_grid, r["psi_hard"], 0.5)
        clon[name] = res
        log(f"cloning {name}: k_S soft " + ", ".join(f"{l} {res[l]['k_S_soft']:.2f}" for l in LEVELS)
            + " | hard " + ", ".join(f"{l} {res[l]['k_S_hard']:.2f}" for l in LEVELS))
    out["cloning"] = clon

    # ------------------------------------------------ 3. correlation-weighted profile theory
    idx_th = rng.choice(len(test), size=min(n_theory, len(test)), replace=False)
    S_th = test.S[idx_th]
    rank_th = np.argsort(rng.random((len(S_th), L)), 1).argsort(1)
    exact_ceiling = {lvl: float(((P_test[idx_th] @ onehot_level[lvl]) ** 2).sum(1).mean()) for lvl in LEVELS}
    theo: dict = {"exact_ceiling": exact_ceiling}
    for lvl in LEVELS:
        lab_tr = train_level[lvl]
        F = []
        for c in range(K[lvl]):
            f = site_frequencies(train.S[lab_tr == c], q) + 1e-3
            F.append(f / f.sum(1, keepdims=True))
        log_f = np.log(np.stack(F))
        mi, null1 = within_class_mutual_information(train.S, lab_tr, q, seed=seed)
        _, null2 = within_class_mutual_information(train.S, lab_tr, q, seed=seed + 1)
        noise = float(np.std((null1 - null2)[np.triu_indices(L, 1)]) / math.sqrt(2))
        partners = mi > 5 * noise
        rho = information_correlation(mi)
        prior = np.full(K[lvl], -math.log(K[lvl]))
        ytrue = true_level[lvl][idx_th]
        lv_out = {"noise_mi": noise, "n_partner_pairs": int(np.triu(partners, 1).sum()),
                  "rho_partners_mean": float(rho[np.triu(partners, 1)].mean()) if partners.any() else 0.0}
        for mode in MODES:
            phi, ptrue, nll, acc = [], [], [], []
            for k in range(L + 1):
                rev = rank_th < k
                w = correlation_weights(rev, rho, partners, mode)
                Q = np.exp(weighted_profile_log_posterior(S_th, rev, log_f, prior, w))
                phi.append(float((Q ** 2).sum(1).mean()))
                pt = Q[np.arange(len(Q)), ytrue]
                ptrue.append(float(pt.mean()))
                nll.append(float(-np.log(np.maximum(pt, 1e-300)).mean()))
                acc.append(float((Q.argmax(1) == ytrue).mean()))
            gap = [a - b for a, b in zip(phi, ptrue)]
            psi = normalise(phi, phi[0], exact_ceiling[lvl])
            psi_cal = normalise(ptrue, ptrue[0], exact_ceiling[lvl])
            lv_out[mode] = {"phi": phi, "E_Q_true": ptrue, "calibration_gap": gap, "log_loss": nll, "accuracy": acc,
                            "k_S": crossing(range(L + 1), psi, 0.5), "k_S_from_E_Q_true": crossing(range(L + 1), psi_cal, 0.5),
                            "mean_abs_gap": float(np.mean(np.abs(gap))), "mean_log_loss": float(np.mean(nll))}
        theo[lvl] = lv_out
        log(f"theory {lvl}: {lv_out['n_partner_pairs']} correlated pairs; " + "; ".join(
            f"{m} k_S {lv_out[m]['k_S']:.2f} |gap| {lv_out[m]['mean_abs_gap']:.4f} logloss {lv_out[m]['mean_log_loss']:.3f}"
            for m in MODES))
    out["theory"] = theo

    ev = root / "evaluations" / run
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "regimes_refined.json").write_text(json.dumps(out, indent=2))
    make_figures(out, root / "figures" / "models" / run / "regimes", L)
    log("done")
    return out


def make_figures(out: dict, fig_dir: Path, L: int) -> None:
    kg = np.array(out["cloning"]["k_grid"])
    kk = np.arange(L + 1)
    kmax = 30
    th, cl = out["theory"], out["cloning"]

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for ax, lvl in zip(axs, LEVELS):
        a = cl["A_test"][lvl]
        s = kg <= kmax
        ax.plot(kg[s], np.array(a["soft_clone_clone"])[s], "o-", ms=3, color=SERIES[0], lw=1.8,
                label="network, exact soft labels")
        ax.plot(kg[s], np.array(a["hard_clone_clone"])[s], "s--", ms=2.5, color=SERIES[0], lw=1, alpha=0.5,
                label="network, classifier labels")
        for j, mode in enumerate(["none", "geo", "eff"]):
            ax.plot(kk[kk <= kmax], np.array(th[lvl][mode]["phi"])[kk <= kmax], color=SERIES[1 + j], lw=1.5,
                    ls=[":", "-.", "-"][j], label="theory: " + MODE_NAMES[mode])
        ax.axhline(th["exact_ceiling"][lvl], color=INK_2, lw=0.8, ls=":")
        ax.text(kmax * 0.98, th["exact_ceiling"][lvl] - 0.02, "exact ceiling (k = L)", fontsize=7, color=INK_2,
                ha="right", va="top")
        ax.set(xlabel="revealed sites k", title=f"{lvl}", xlim=(0, kmax))
    axs[0].set_ylabel("commitment phi = E sum_c P(c | x_U)^2")
    axs[0].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save(fig, fig_dir / "refined_speciation.png")

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for ax, lvl in zip(axs, LEVELS):
        for j, mode in enumerate(MODES):
            ax.plot(kk, th[lvl][mode]["calibration_gap"], color=SERIES[j], lw=1.5, label=MODE_NAMES[mode])
        a = cl["A_test"][lvl]
        ax.plot(kg, np.array(a["soft_clone_clone"]) - np.array(a["soft_clone_original"]), "o", ms=3, color=INK_2,
                label="network: clone-clone minus clone-original")
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set(xlabel="revealed sites k", title=f"{lvl}: calibration gap")
    axs[0].set_ylabel("E sum Q^2 - E Q(true)   (> 0: over-confident)")
    axs[0].legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, fig_dir / "refined_calibration.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-config", default=str(PROJECT_ROOT / "configs" / "ground_truth.json"))
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    a = ap.parse_args()
    main(json.loads(Path(a.config).read_text()), json.loads(Path(a.data_config).read_text()), Path(a.root))
