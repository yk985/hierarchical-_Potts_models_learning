"""Step M - dynamical regimes of the masked diffusion model: speciation and collapse (theory: docs, Part IV).

k = number of revealed sites (k = L(1 - t) on average). All curves are computed at fixed k and converted to the
masking time t with a binomial average (psg.regimes.to_time).

Speciation (does the trajectory already know its family / subfamily / clade?)
  * cloning: start from x_t, complete it C times with the network, classify every completion (pseudo-likelihood of
    the 18 true clade models), measure the probability that two clones share the class. Two starting ensembles:
    (A) held-out TRUE sequences, masked;  (B) the network's OWN trajectories (its samples, masked: identical in law).
  * theory 1 (profile Bayes): class posterior when each class is replaced by its independent-site profile;
  * theory 2 (Gaussian log-odds): psi = E tanh^2(Lambda/2), Lambda ~ N(mu, 2 mu), mu = k D / L, D = sibling KL;
  * data (empirical Bayes): class shares among the training sequences that match x_t exactly (valid while many match).
Collapse (does the trajectory already know WHICH training sequence it will become? - empirical denoiser)
  * E[1/N], N = number of training sequences consistent with x_t, for trajectories planted on training points;
  * quenched theory: N - 1 ~ Poisson((n-1) P(x_U)), P(x_U) from the network's chain rule;  annealed theory with
    the collision probability of the data (Hamming distances);  entropy criterion H(k) = log n;
  * the same for smaller training sets (n = 100 ... 144 000) and, by theory, for any n: the phase diagram;
  * memorisation check of the trained network: training matches along its own trajectories vs held-out data.

Outputs: evaluations/<run>/regimes.json, figures/models/<run>/regimes/*.png

    python scripts/s13_dynamical_regimes.py --config configs/md_unconditional.json
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

from psg import PROJECT_ROOT  # noqa: E402
from psg.data import LabelledSequences  # noqa: E402
from psg.diffusion_training import evaluate_nelbo  # noqa: E402
from psg.evaluation import pseudo_loglik_scores_torch  # noqa: E402
from psg.hierarchy import leaves, load_hierarchy  # noqa: E402
from psg.models.masked_diffusion import continuous_time_sample, load_diffusion  # noqa: E402
from psg.plots import INK_2, SERIES, plt, save  # noqa: E402
from psg.regimes import (  # noqa: E402
    complete,
    consistent_counts,
    crossing,
    gaussian_llr_commitment,
    pair_agreement,
    poisson_commitment,
    profile_log_posterior,
    random_orders,
    reveal_first_k,
    revealed_log_prob,
    sibling_kl,
    solve_mu_star,
    subset_agreement,
    to_time,
)
from psg.stats import site_frequencies  # noqa: E402

LEVELS = ["family", "subfamily", "clade"]


def level_maps(lv, n_sub: int) -> dict[str, np.ndarray]:
    """leaf index -> class index at each level (global subfamily index)."""
    return {"family": np.array([n.path[0] for n in lv]),
            "subfamily": np.array([n.path[0] * n_sub + n.path[1] for n in lv]),
            "clade": np.arange(len(lv))}


def main(run_cfg: dict, data_cfg: dict, root: Path = PROJECT_ROOT, n_start: int = 1000, n_clones: int = 8,
         n_theory: int = 5000, n_traj: int = 6, n_traj_clones: int = 64, seed: int = 0, verbose: bool = True) -> dict:
    t_start = time.time()
    run = run_cfg["run_name"]
    hier = data_cfg["hierarchy"]
    q, L, n_sub = hier["q"], hier["L"], hier["n_subfamilies"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(run_cfg["sampling"].get("amp", False)) and dev.type == "cuda"
    model = load_diffusion(root / "models" / run / "model.pt", dev).eval()

    def den(x, t):
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            return model(x, t)

    nodes, _, _ = load_hierarchy(root / "data" / "ground_truth_potts")
    lv = leaves(nodes)
    models = [n.model for n in lv]
    maps = level_maps(lv, n_sub)
    K = {lvl: int(maps[lvl].max() + 1) for lvl in LEVELS}
    parent = {"family": lambda c: 0, "subfamily": lambda c: c // n_sub, "clade": lambda c: c // hier["n_clades"]}
    train = LabelledSequences.load(root / "data" / "splits" / "train.npz")
    test = LabelledSequences.load(root / "data" / "splits" / "test.npz")
    n = len(train)
    T = torch.as_tensor(train.S, dtype=torch.long, device=dev)
    T_leaf = torch.as_tensor(train.leaf, dtype=torch.long, device=dev)
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    k_grid = sorted(set(list(range(0, min(L, 20) + 1)) + [k for k in (24, 28, 34, 40, 50) if k < L] + [L]))
    t_grid = np.linspace(0.0, 1.0, 201)
    out: dict = {"run": run, "L": L, "n_train": n, "k_grid": k_grid, "n_start": n_start, "n_clones": n_clones}

    def log(msg):
        if verbose:
            print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)

    def classify(S: np.ndarray) -> np.ndarray:
        return pseudo_loglik_scores_torch(S, models, dev).argmax(1)

    acc_leaf = classify(test.S)
    out["classifier_accuracy_test"] = {lvl: float((maps[lvl][acc_leaf] == maps[lvl][test.leaf]).mean()) for lvl in LEVELS}
    log(f"classifier accuracy on test: {out['classifier_accuracy_test']}")

    # ------------------------------------------------ starting ensembles
    idx_A = rng.choice(len(test), size=min(n_start, len(test)), replace=False)
    x_A = torch.as_tensor(test.S[idx_A], dtype=torch.long, device=dev)
    x_B = continuous_time_sample(den, n_start, L, q, seed=seed + 1, device=dev).to(dev)
    ens = {"A_test": x_A, "B_network": x_B}
    rank = {name: random_orders(len(x), L, g, dev) for name, x in ens.items()}

    # ------------------------------------------------ 1. speciation by cloning
    spec: dict = {}
    for name, x0 in ens.items():
        leaf0 = classify(x0.cpu().numpy())
        res = {lvl: {"phi": [], "phi_se": [], "retention": []} for lvl in LEVELS}
        if name == "A_test":
            for lvl in LEVELS:
                res[lvl]["retention_true_label"] = []
        for k in k_grid:
            if k == L:
                for lvl in LEVELS:
                    res[lvl]["phi"].append(1.0)
                    res[lvl]["phi_se"].append(0.0)
                    res[lvl]["retention"].append(1.0)
                    if name == "A_test":
                        res[lvl]["retention_true_label"].append(
                            float((maps[lvl][leaf0] == maps[lvl][test.leaf[idx_A]]).mean()))
                continue
            C = complete(den, reveal_first_k(x0, rank[name], k, q), n_clones, q, g).reshape(-1, L).cpu().numpy()
            leaf = classify(C).reshape(len(x0), n_clones)
            for lvl in LEVELS:
                lab = maps[lvl][leaf]
                pa = pair_agreement(lab)
                res[lvl]["phi"].append(float(pa.mean()))
                res[lvl]["phi_se"].append(float(pa.std() / math.sqrt(len(pa))))
                res[lvl]["retention"].append(float((lab == maps[lvl][leaf0][:, None]).mean()))
                if name == "A_test":
                    res[lvl]["retention_true_label"].append(
                        float((lab == maps[lvl][test.leaf[idx_A]][:, None]).mean()))
        for lvl in LEVELS:
            r = res[lvl]
            phi0 = r["phi"][0]
            r["psi"] = [(p - phi0) / (1 - phi0) for p in r["phi"]]
            r["k_S"] = crossing(k_grid, r["psi"], 0.5)
            psi_t = (to_time(k_grid, r["phi"], t_grid, L) - phi0) / (1 - phi0)
            r["t_S"] = crossing(t_grid[::-1], psi_t[::-1], 0.5)
        spec[name] = res
        log(f"cloning {name}: k_S = " + ", ".join(f"{lvl} {res[lvl]['k_S']:.2f}" for lvl in LEVELS))
    out["speciation_cloning"] = spec

    # ------------------------------------------------ 2. theory: profile Bayes and Gaussian log-odds
    mu_star = solve_mu_star()
    idx_th = rng.choice(len(test), size=min(n_theory, len(test)), replace=False)
    S_th = test.S[idx_th]
    rank_th = np.argsort(rng.random((len(S_th), L)), 1).argsort(1)
    theory: dict = {"mu_star": mu_star}
    for lvl in LEVELS:
        lab_tr = maps[lvl][train.leaf]
        freqs = {}
        for c in range(K[lvl]):
            f = site_frequencies(train.S[lab_tr == c], q) + 1e-3
            freqs[c] = f / f.sum(1, keepdims=True)
        log_f = np.log(np.stack([freqs[c] for c in range(K[lvl])]))
        D = sibling_kl(freqs, {c: parent[lvl](c) for c in range(K[lvl])})
        phi = [float((np.exp(profile_log_posterior(S_th, rank_th < k, log_f, np.full(K[lvl], -math.log(K[lvl]))))
                      ** 2).sum(1).mean()) for k in range(L + 1)]
        psi = [(p - 1 / K[lvl]) / (1 - 1 / K[lvl]) for p in phi]
        psi_t = (to_time(range(L + 1), phi, t_grid, L) - 1 / K[lvl]) / (1 - 1 / K[lvl])
        theory[lvl] = {"sibling_KL_nats": D, "profile_phi": phi, "profile_psi": psi,
                       "profile_k_S": crossing(range(L + 1), psi, 0.5),
                       "profile_t_S": crossing(t_grid[::-1], psi_t[::-1], 0.5),
                       "gaussian_k_S": L * mu_star / D, "gaussian_t_S": 1 - mu_star / D}
        log(f"theory {lvl}: sibling KL {D:.1f} nats, profile k_S {theory[lvl]['profile_k_S']:.2f}, "
            f"Gaussian k_S {theory[lvl]['gaussian_k_S']:.2f}")
    out["speciation_theory"] = theory

    # ------------------------------------------------ 3. training-set matches: empirical Bayes and collapse counts
    to_level = {lvl: torch.nn.functional.one_hot(torch.as_tensor(maps[lvl]), K[lvl]).float().to(dev) for lvl in LEVELS}
    idx_C = rng.choice(n, size=n_start, replace=False)
    x_C = T[idx_C]
    rank_C = random_orders(n_start, L, g, dev)
    counts: dict = {"A_test": [], "B_network": [], "C_train_planted": []}
    emp_bayes = {lvl: {"phi": [], "fraction_valid": []} for lvl in LEVELS}
    n_min = 20
    for k in k_grid:
        cA = consistent_counts(reveal_first_k(x_A, rank["A_test"], k, q), T, q, T_leaf, len(lv))    # (B, 18)
        NA = cA.sum(1)
        counts["A_test"].append(NA.cpu().numpy())
        counts["B_network"].append(consistent_counts(reveal_first_k(x_B, rank["B_network"], k, q), T, q).cpu().numpy())
        counts["C_train_planted"].append(consistent_counts(reveal_first_k(x_C, rank_C, k, q), T, q).cpu().numpy())
        ok = NA >= n_min
        for lvl in LEVELS:
            P = (cA @ to_level[lvl])[ok] / NA[ok, None]
            emp_bayes[lvl]["phi"].append(float((P ** 2).sum(1).mean()) if ok.any() else float("nan"))
            emp_bayes[lvl]["fraction_valid"].append(float(ok.float().mean()))
    for lvl in LEVELS:
        emp_bayes[lvl]["psi"] = [(p - 1 / K[lvl]) / (1 - 1 / K[lvl]) for p in emp_bayes[lvl]["phi"]]
        emp_bayes[lvl]["n_min"] = n_min
    out["speciation_empirical_bayes"] = emp_bayes
    log("training-set matches counted")

    # ------------------------------------------------ 4. collapse: measured and theory
    lp_A = revealed_log_prob(den, x_A, rank["A_test"], q).cpu().numpy()          # (B, L+1)
    lp_C = revealed_log_prob(den, x_C, rank_C, q).cpu().numpy()
    H = -lp_A.mean(0)                                                             # entropy of revealed residues
    ii = rng.integers(0, n, 400_000)
    jj = rng.integers(0, n, 400_000)
    keep = ii != jj
    Dpairs = (train.S[ii[keep]] != train.S[jj[keep]]).sum(1)
    agree = np.array([subset_agreement(Dpairs, k, L) for k in range(L + 1)])
    col = {
        "measured_planted_E_inv_N": [float((1.0 / c).mean()) for c in counts["C_train_planted"]],
        "fresh_E_inv_1_plus_N": [float((1.0 / (1 + c)).mean()) for c in counts["A_test"]],
        "quenched_theory_test": [float(poisson_commitment((n - 1) * np.exp(lp_A[:, k])).mean()) for k in k_grid],
        "quenched_theory_train": [float(poisson_commitment((n - 1) * np.exp(lp_C[:, k])).mean()) for k in k_grid],
        "annealed_theory": [float(poisson_commitment((n - 1) * agree[k])) for k in k_grid],
        "entropy_H_k": H.tolist(),
        "entropy_H_k_train": (-lp_C.mean(0)).tolist(),
        "renyi2_H2_k": (-np.log(agree)).tolist(),
        "log_n": math.log(n),
        "match_prob": {name: [float((c >= 1).mean()) for c in counts[name]] for name in counts},
        "mean_matches": {name: [float(c.mean()) for c in counts[name]] for name in counts},
    }
    for key in ["measured_planted_E_inv_N", "quenched_theory_test", "annealed_theory"]:
        col[key + "_k_C"] = crossing(k_grid, col[key], 0.5)
        col[key + "_t_C"] = crossing(t_grid[::-1], to_time(k_grid, col[key], t_grid, L)[::-1], 0.5)
    col["entropy_k_C"] = crossing(range(L + 1), H, math.log(n))
    col["renyi2_k_C"] = crossing(range(L + 1), -np.log(agree), math.log(n))
    log(f"collapse: measured k_C {col['measured_planted_E_inv_N_k_C']:.2f}, quenched {col['quenched_theory_test_k_C']:.2f}, "
        f"annealed {col['annealed_theory_k_C']:.2f}, entropy criterion {col['entropy_k_C']:.2f}")

    # n-dependence: measured on training subsets, theory for any n
    by_n = {}
    for n_sub_set in sorted({min(m, n) for m in [100, 1000, 10_000, n]}):
        sub = torch.as_tensor(rng.choice(n, size=n_sub_set, replace=False), device=dev)
        Tn = T[sub]
        pl = Tn[torch.as_tensor(rng.integers(0, n_sub_set, n_start), device=dev)]
        rk = random_orders(n_start, L, g, dev)
        e = [float((1.0 / consistent_counts(reveal_first_k(pl, rk, k, q), Tn, q).double()).mean().item()) for k in k_grid]
        th = [float(poisson_commitment((n_sub_set - 1) * np.exp(lp_A[:, k])).mean()) for k in k_grid]
        by_n[str(n_sub_set)] = {"measured": e, "theory": th, "k_C_measured": crossing(k_grid, e, 0.5),
                                "k_C_theory": crossing(k_grid, th, 0.5)}
    col["by_n"] = by_n
    log_n_grid = np.linspace(math.log(10), 60 * math.log(10), 120)
    kfine = np.arange(L + 1)
    col["phase_diagram"] = {"log10_n": (log_n_grid / math.log(10)).tolist(),
                            "k_C_theory": [crossing(kfine, poisson_commitment(np.exp(ln + lp_A)).mean(0), 0.5)
                                           for ln in log_n_grid],
                            "k_C_entropy": [crossing(kfine, H, ln) for ln in log_n_grid]}
    out["collapse"] = col

    # memorisation of the trained network: likelihood gap train vs test
    S_tr = torch.as_tensor(train.S[rng.choice(n, min(n, len(test)), replace=False)], dtype=torch.long, device=dev)
    out["nelbo_site"] = {"train": evaluate_nelbo(model, S_tr, repeats=4),
                         "test": evaluate_nelbo(model, torch.as_tensor(test.S, dtype=torch.long, device=dev), repeats=4),
                         "empirical_denoiser_on_train": math.log(n) / L}
    log(f"NELBO/site train {out['nelbo_site']['train']:.4f}, test {out['nelbo_site']['test']:.4f}")

    # ------------------------------------------------ 5. single trajectories
    x_T = x_B[:n_traj]
    rank_T = rank["B_network"][:n_traj]
    leaf_T = classify(x_T.cpu().numpy())
    k_traj = list(range(0, min(L, 20) + 1))
    traj = {lvl: np.zeros((n_traj, len(k_traj))) for lvl in ["family", "clade"]}
    for j, k in enumerate(k_traj):
        C = complete(den, reveal_first_k(x_T, rank_T, k, q), n_traj_clones, q, g).reshape(-1, L).cpu().numpy()
        leaf = classify(C).reshape(n_traj, n_traj_clones)
        for lvl in traj:
            traj[lvl][:, j] = (maps[lvl][leaf] == maps[lvl][leaf_T][:, None]).mean(1)
    out["trajectories"] = {"k": k_traj, **{lvl: v.tolist() for lvl, v in traj.items()},
                           "final_leaf": leaf_T.tolist()}
    log("single trajectories done")

    ev = root / "evaluations" / run
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "regimes.json").write_text(json.dumps(out, indent=2))
    make_figures(out, root / "figures" / "models" / run / "regimes", L)
    log("figures written")
    return out


# ============================================================================ figures
def _top_t_axis(ax, L):
    sec = ax.secondary_xaxis("top", functions=(lambda k: 1 - np.asarray(k) / L, lambda t: L * (1 - np.asarray(t))))
    sec.set_xlabel("masking time t = 1 - k/L", fontsize=8)


def make_figures(out: dict, fig_dir: Path, L: int) -> None:
    kg = np.array(out["k_grid"])
    spec, th, eb, col = out["speciation_cloning"], out["speciation_theory"], out["speciation_empirical_bayes"], out["collapse"]
    kmax = 24
    sel = kg <= kmax

    # 1. speciation curves
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, lvl in zip(axs, LEVELS):
        a, b = spec["A_test"][lvl], spec["B_network"][lvl]
        ax.errorbar(kg[sel], np.array(a["psi"])[sel], yerr=2 * np.array(a["phi_se"])[sel] / (1 - a["phi"][0]),
                    color=SERIES[0], marker="o", ms=3, lw=1.5, label="clones, from held-out true x_t (A)")
        ax.plot(kg[sel], np.array(b["psi"])[sel], color=SERIES[2], marker="s", ms=3, lw=1, ls="--",
                label="clones, network's own trajectories (B)")
        kk = np.arange(L + 1)
        ax.plot(kk[kk <= kmax], np.array(th[lvl]["profile_psi"])[kk <= kmax], color=SERIES[1], lw=1.5, ls=":",
                label="theory: profile Bayes")
        v = np.array(eb[lvl]["fraction_valid"]) > 0.9
        ax.plot(kg[v & sel], np.array(eb[lvl]["psi"])[v & sel], "D", color=SERIES[3], ms=5, mfc="none",
                label="data: exact matches in training set")
        ax.axvline(th[lvl]["gaussian_k_S"], color=SERIES[1], lw=1, alpha=0.6)
        ax.axvline(a["k_S"], color=SERIES[0], lw=1, alpha=0.6)
        ax.axhline(0.5, color=INK_2, lw=0.8, ls=":")
        ax.set(xlabel="revealed sites k", title=f"{lvl}: k_S = {a['k_S']:.1f} (t_S = {a['t_S']:.3f})", xlim=(0, kmax))
        _top_t_axis(ax, L)
    axs[0].set_ylabel("normalised commitment psi")
    axs[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, fig_dir / "speciation.png")

    # 2. universal curve
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    mus = np.linspace(0, 12, 200)
    ax.plot(mus, gaussian_llr_commitment(mus), color=INK_2, lw=2, label="psi(mu), Gaussian log-odds")
    for j, lvl in enumerate(LEVELS):
        D = th[lvl]["sibling_KL_nats"]
        kk = np.arange(L + 1)
        ax.plot(kk * D / L, th[lvl]["profile_psi"], "o", ms=3, color=SERIES[j], label=f"profile Bayes, {lvl}")
        ax.plot(np.array(kg) * D / L, spec["A_test"][lvl]["psi"], "-", lw=1, color=SERIES[j], alpha=0.6)
    ax.axvline(th["mu_star"], color=INK_2, ls=":", lw=1)
    ax.set(xlabel="class signal-to-noise mu = k D / L", ylabel="normalised commitment", xlim=(0, 12),
           title="Speciation in units of class signal")
    ax.legend(fontsize=7)
    save(fig, fig_dir / "speciation_universal.png")

    # 3. single trajectories
    tr = out["trajectories"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, lvl in zip(axs, ["family", "clade"]):
        for j, row in enumerate(tr[lvl]):
            ax.plot(tr["k"], row, marker="o", ms=2.5, lw=1.2, color=SERIES[j % len(SERIES)])
        ax.axvline(spec["B_network"][lvl]["k_S"], color=INK_2, ls=":", lw=1)
        ax.set(xlabel="revealed sites k", title=f"P(final {lvl} | x_t) along single trajectories")
        _top_t_axis(ax, L)
    axs[0].set_ylabel("fraction of 64 clones in the final class")
    fig.tight_layout()
    save(fig, fig_dir / "trajectories.png")

    # 4. collapse
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axs[0]
    ax.plot(kg[sel], np.array(col["measured_planted_E_inv_N"])[sel], "o-", color=SERIES[0], ms=3,
            label="measured: empirical denoiser (planted on train)")
    ax.plot(kg[sel], np.array(col["quenched_theory_test"])[sel], color=SERIES[1], lw=1.5,
            label="quenched theory: Poisson, P(x_U) from network")
    ax.plot(kg[sel], np.array(col["annealed_theory"])[sel], color=SERIES[3], lw=1.5, ls="--",
            label="annealed theory: collision probability")
    for j, lvl in enumerate(LEVELS):
        ax.axvline(spec["A_test"][lvl]["k_S"], color=INK_2, lw=0.8, ls=":")
        ax.text(spec["A_test"][lvl]["k_S"], 0.97, f" k_S {lvl}", fontsize=7, color=INK_2, rotation=90, va="top")
    ax.axhline(0.5, color=INK_2, lw=0.8, ls=":")
    ax.set(xlabel="revealed sites k", ylabel="P(two clones = same training seq.)", xlim=(0, kmax),
           title=f"Collapse of the empirical denoiser, n={out['n_train']:,}")
    _top_t_axis(ax, L)
    ax.legend(fontsize=7, loc="center right")
    ax = axs[1]
    names = {"C_train_planted": "training seqs (empirical denoiser paths)", "A_test": "held-out true seqs",
             "B_network": "trained network's own paths"}
    for j, (key, lab) in enumerate(names.items()):
        ax.plot(kg[sel], np.array(col["match_prob"][key])[sel], "o-", ms=3, color=SERIES[j], label=lab)
    ax.set(xlabel="revealed sites k", ylabel="P(x_t matches >= 1 training seq.)", xlim=(0, kmax),
           title="Memorisation check")
    _top_t_axis(ax, L)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, fig_dir / "collapse.png")

    # 5. entropy of the revealed residues
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    kk = np.arange(L + 1)
    ax.plot(kk, col["entropy_H_k"], color=SERIES[0], lw=2, label="H(k): Shannon (network chain rule)")
    ax.plot(kk, col["renyi2_H2_k"], color=SERIES[3], lw=1.5, ls="--", label="H2(k): collision (data)")
    for nn, ls in [(1e3, ":"), (out["n_train"], "-"), (1e8, "-.")]:
        ax.axhline(math.log(nn), color=INK_2, lw=0.8, ls=ls)
        ax.text(L * 0.62, math.log(nn) + 1, f"log n, n = {nn:.0e}", fontsize=7, color=INK_2)
    ax.set(xlabel="revealed sites k", ylabel="nats", title="Entropy of the revealed residues", xlim=(0, L))
    ax.legend(fontsize=7, loc="upper left")
    save(fig, fig_dir / "entropy.png")

    # 6. phase diagram
    pdg = col["phase_diagram"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pdg["log10_n"], pdg["k_C_theory"], color=SERIES[1], lw=2, label="collapse k_C(n), quenched theory")
    ax.plot(pdg["log10_n"], pdg["k_C_entropy"], color=SERIES[1], lw=1, ls="--", label="entropy criterion H(k) = log n")
    for key, v in col["by_n"].items():
        ax.plot(math.log10(int(key)), v["k_C_measured"], "o", color=SERIES[0], ms=6)
    ax.plot([], [], "o", color=SERIES[0], label="collapse measured (training subsets)")
    for j, lvl in enumerate(LEVELS):
        ax.axhline(spec["A_test"][lvl]["k_S"], color=SERIES[2 + j], lw=1.5, ls=":", label=f"speciation k_S, {lvl}")
    ax.axvline(math.log10(out["n_train"]), color=INK_2, lw=0.8)
    ax.set(xlabel="log10 (training-set size n)", ylabel="revealed sites k", ylim=(0, L), xlim=(1, 60),
           title="When does the empirical denoiser collapse?")
    sec = ax.secondary_yaxis("right", functions=(lambda k: 1 - np.asarray(k) / L, lambda t: L * (1 - np.asarray(t))))
    sec.set_ylabel("t", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    save(fig, fig_dir / "phase_diagram.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-config", default=str(PROJECT_ROOT / "configs" / "ground_truth.json"))
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--n-start", type=int, default=1000)
    ap.add_argument("--n-clones", type=int, default=8)
    a = ap.parse_args()
    main(json.loads(Path(a.config).read_text()), json.loads(Path(a.data_config).read_text()), Path(a.root),
         n_start=a.n_start, n_clones=a.n_clones)
