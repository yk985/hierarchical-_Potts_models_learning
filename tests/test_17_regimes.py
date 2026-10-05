"""Step 17 - dynamical regimes (speciation and collapse) of masked diffusion: the measurement tools, checked on
systems small enough to enumerate, with the exact denoiser of test_13."""
import itertools
import math

import numpy as np
import torch

from psg.evaluation import pseudo_loglik_scores, pseudo_loglik_scores_torch
from psg.plots import INK_2, SERIES, plt, save
from psg.potts import Potts, enumerate_states, symmetrize
from psg.regimes import (
    complete,
    consistent_counts,
    crossing,
    gaussian_llr_commitment,
    normalised_commitment,
    pair_agreement,
    poisson_commitment,
    profile_log_posterior,
    random_orders,
    reveal_first_k,
    revealed_log_prob,
    solve_mu_star,
    subset_agreement,
    to_time,
)
from test_13_masked_diffusion import exact_denoiser


# ---------------------------------------------------------------------------- small helpers, by hand
def test_pair_agreement_and_normalisation():
    # row 1: all 4 clones agree -> 1; row 2: labels 0,1,0,1 -> 2 equal pairs out of 6
    phi = pair_agreement(np.array([[0, 0, 0, 0], [0, 1, 0, 1]]))
    np.testing.assert_allclose(phi, [1.0, 2 / 6])
    np.testing.assert_allclose(normalised_commitment([0.5, 1.0, 0.75], 0.5), [0.0, 1.0, 0.5])


def test_to_time_and_crossing():
    L = 10
    ks = np.arange(L + 1)
    # f(k) = k: the average number of revealed sites at time t is L (1 - t)
    np.testing.assert_allclose(to_time(ks, ks, [0.0, 0.3, 1.0], L), [10.0, 7.0, 0.0], atol=1e-9)
    assert crossing([0, 1, 2, 3], [0.0, 0.2, 0.6, 1.0], 0.5) == 1.75
    assert math.isnan(crossing([0, 1], [0.0, 0.2], 0.5))


def test_random_orders_reveal_uniform_subsets():
    g = torch.Generator().manual_seed(0)
    rank = random_orders(20000, 5, g)
    assert (rank.sort(1).values == torch.arange(5)).all()              # each row is a permutation
    x0 = torch.arange(5).repeat(20000, 1)
    xt = reveal_first_k(x0, rank, 2, mask_id=9)
    assert ((xt != 9).sum(1) == 2).all()
    # every site is revealed with probability k / L = 0.4
    np.testing.assert_allclose((xt != 9).float().mean(0).numpy(), 0.4, atol=0.015)


def test_profile_posterior_by_hand():
    f = np.array([[[0.9, 0.1]], [[0.1, 0.9]]])                          # 2 classes, 1 site, q = 2
    lp = profile_log_posterior(np.array([[0], [1], [0]]), np.array([[True], [True], [False]]),
                               np.log(f), np.log([0.5, 0.5]))
    np.testing.assert_allclose(np.exp(lp), [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])


def test_consistent_counts_by_hand():
    M = 9
    train = torch.tensor([[0, 0, 0], [0, 1, 1], [1, 1, 1]])
    xt = torch.tensor([[0, M, M], [M, 1, M], [M, M, M], [1, 0, M]])
    assert consistent_counts(xt, train, M).tolist() == [2, 2, 3, 0]
    per_class = consistent_counts(xt, train, M, labels=torch.tensor([0, 0, 1]), n_classes=2)
    assert per_class.tolist() == [[2, 0], [1, 1], [2, 1], [0, 0]]


def test_subset_agreement_and_poisson(rng):
    # L = 4, k = 2: a pair with D mismatches agrees on a random 2-subset with prob C(4-D, 2) / C(4, 2)
    assert abs(subset_agreement(np.array([0, 1, 2]), 2, 4) - (1 + 3 / 6 + 1 / 6) / 3) < 1e-12
    D = rng.integers(0, 8, 400)
    # mismatches on sites 0..D-1; reveal 3 random sites out of 10
    mc = np.mean([[not (set(rng.permutation(10)[:3]) & set(range(d))) for d in D] for _ in range(40)])
    assert abs(mc - subset_agreement(D, 3, 10)) < 0.02
    np.testing.assert_allclose(poisson_commitment([0.0, 1.0]), [1.0, 1 - math.exp(-1)])
    N = rng.poisson(2.5, 200000)
    assert abs(np.mean(1 / (1 + N)) - poisson_commitment(2.5)) < 3e-3


def test_gpu_pseudo_likelihood_matches_numpy(rng):
    models = [Potts(rng.normal(0, 1, (6, 4)), symmetrize(rng.normal(0, 0.5, (6, 6, 4, 4)))) for _ in range(3)]
    S = rng.integers(0, 4, (50, 6))
    np.testing.assert_allclose(pseudo_loglik_scores_torch(S, models, batch_size=16), pseudo_loglik_scores(S, models),
                               atol=1e-4)


# ---------------------------------------------------------------------------- speciation theory
def test_gaussian_mixture_speciation(outdir):
    """Two Gaussian classes +-m in d dims, Ornstein-Uhlenbeck noise x_t = e^-t x_0 + sqrt(1 - e^-2t) xi. The exact
    class log-odds is Gaussian with mean mu(t) = 2|m|^2 e^-2t / Gamma_t and variance 2 mu(t), so the measured
    commitment E[tanh^2(Lambda/2)] must equal psi(mu(t)); psi = 1/2 at mu*; t_S ~ (1/2) log(2 |m|^2 / mu*)."""
    g = np.random.default_rng(0)
    d, sigma = 50, 1.0
    m = g.normal(0, 1, d)
    m *= np.sqrt(400.0) / np.linalg.norm(m)                          # |m|^2 = 400 ~ top eigenvalue Lambda
    ts = np.linspace(0.5, 5.0, 19)
    meas, pred = [], []
    for t in ts:
        Gam = sigma ** 2 * math.exp(-2 * t) + 1 - math.exp(-2 * t)
        sgn = g.choice([-1, 1], 20000)
        x0 = sgn[:, None] * m + sigma * g.normal(size=(20000, d))
        xt = math.exp(-t) * x0 + math.sqrt(1 - math.exp(-2 * t)) * g.normal(size=(20000, d))
        lam = sgn * 2 * math.exp(-t) * (xt @ m) / Gam                  # log-odds of the TRUE class
        meas.append(np.mean(np.tanh(lam / 2) ** 2))
        pred.append(gaussian_llr_commitment(2 * 400 * math.exp(-2 * t) / Gam)[0])
    np.testing.assert_allclose(meas, pred, atol=0.01)
    mu_star = solve_mu_star()
    assert abs(gaussian_llr_commitment(mu_star)[0] - 0.5) < 1e-6 and 0.5 < mu_star < 5
    assert gaussian_llr_commitment(0.0)[0] < 1e-9 and gaussian_llr_commitment(200.0)[0] > 0.99
    t_s = crossing(ts, pred, 0.5)
    assert abs(t_s - 0.5 * math.log(2 * 400 / mu_star)) < 0.05       # Gamma_t ~ 1 at t_S

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(ts, meas, "o", color=SERIES[0], ms=4, label="simulation, exact posterior")
    ax.plot(ts, pred, color=SERIES[1], label=r"$\psi(\mu(t))$, Gaussian log-odds")
    ax.axvline(0.5 * math.log(400), color=INK_2, ls=":", lw=1, label=r"$\frac{1}{2}\log\Lambda$")
    ax.set(xlabel="diffusion time t (OU)", ylabel="normalised commitment",
           title=r"Two Gaussians, $|m|^2=400$, d=50")
    ax.legend(fontsize=7)
    save(fig, outdir / "gaussian_speciation.png")


# ---------------------------------------------------------------------------- the cloning machinery, exact
def two_profile_mixture(rng, L=4, q=3):
    f = rng.dirichlet(np.full(q, 0.7), size=(2, L))                  # (2, L, q)
    states = enumerate_states(L, q)
    pc = np.stack([np.prod(f[c, np.arange(L), states], axis=1) for c in range(2)])   # (2, n_states)
    p = 0.5 * pc.sum(0)
    return states, p, pc


def exact_commitment(states, p, label, k):
    """E_{x ~ p, U random k-subset} sum_y P(label = y | x_U)^2, by enumeration of all states and subsets."""
    L = states.shape[1]
    tot = 0.0
    subsets = list(itertools.combinations(range(L), k))
    for U in subsets:
        key = states[:, list(U)] @ (10 ** np.arange(len(U))) if U else np.zeros(len(states), int)
        for v in np.unique(key):
            sel = key == v
            pU = p[sel].sum()
            py = np.array([p[sel & (label == y)].sum() for y in np.unique(label)]) / pU
            tot += pU * (py ** 2).sum()
    return tot / len(subsets)


def test_cloning_with_exact_denoiser_gives_exact_commitment(rng, outdir):
    """Clone agreement (and agreement with the original's label) = E[sum_y P(y | x_U)^2], for every k."""
    L, q = 4, 3
    states, p, pc = two_profile_mixture(rng, L, q)
    label = (pc[1] > pc[0]).astype(int)                              # a deterministic classifier of full sequences
    den = exact_denoiser(states, p, q)
    g = torch.Generator().manual_seed(0)
    idx = rng.choice(len(states), size=3000, p=p)
    x0 = torch.as_tensor(states[idx])
    rank = random_orders(len(x0), L, g)
    exact, clones, retention = [], [], []
    for k in range(L + 1):
        exact.append(exact_commitment(states, p, label, k))
        C = complete(den, reveal_first_k(x0, rank, k, q), 6, q, g)       # (B, 6, L)
        lab = label[C.reshape(-1, L).numpy() @ (q ** np.arange(L - 1, -1, -1))].reshape(len(x0), 6)
        clones.append(pair_agreement(lab).mean())
        retention.append((lab == label[idx][:, None]).mean())
        seen = rank < k
        for j in range(6):
            assert (C[:, j][seen] == x0[seen]).all()                      # revealed sites are never changed
    np.testing.assert_allclose(clones, exact, atol=0.02)
    np.testing.assert_allclose(retention, exact, atol=0.02)
    assert clones[-1] == 1.0 and abs(exact[0] - (p[label == 0].sum() ** 2 + p[label == 1].sum() ** 2)) < 1e-12

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(range(L + 1), exact, color=SERIES[1], label="exact (enumeration)")
    ax.plot(range(L + 1), clones, "o", color=SERIES[0], ms=5, label="clone pairs, exact denoiser")
    ax.plot(range(L + 1), retention, "s", color=SERIES[2], ms=4, mfc="none", label="clone vs original")
    ax.set(xlabel="revealed sites k", ylabel="commitment", title="Cloning experiment, 2 classes, L=4")
    ax.legend(fontsize=7)
    save(fig, outdir / "cloning_exact.png")


def test_chain_rule_gives_marginal_of_revealed_sites(rng):
    L, q = 4, 3
    states, p, _ = two_profile_mixture(rng, L, q)
    den = exact_denoiser(states, p, q)
    x0 = torch.as_tensor(states[rng.choice(len(states), size=20, p=p)])
    rank = random_orders(20, L, torch.Generator().manual_seed(1))
    lp = revealed_log_prob(den, x0, rank, q)
    for b in range(20):
        for k in range(L + 1):
            U = np.flatnonzero(rank[b].numpy() < k)
            exact = p[(states[:, U] == x0[b, U].numpy()).all(1)].sum()
            assert abs(lp[b, k].item() - math.log(exact)) < 1e-4


def test_empirical_denoiser_collapse_is_one_over_n(rng, outdir):
    """With the empirical denoiser (uniform over n training sequences), two clones end at the same training
    sequence with probability E[1/N], N = number of training sequences consistent with x_t; the planted count is
    1 + Binomial(n-1, P(a_U)) and E[1/N] follows the Poisson formula when the data are fresh."""
    L, q, n = 6, 2, 12
    train = np.unique(rng.integers(0, q, (n, L)), axis=0)
    n = len(train)
    den = exact_denoiser(train, np.full(n, 1 / n), q)
    g = torch.Generator().manual_seed(2)
    planted = torch.as_tensor(train[rng.integers(0, n, 2000)])
    rank = random_orders(len(planted), L, g)
    T = torch.as_tensor(train)
    code = lambda S: S @ (q ** np.arange(L - 1, -1, -1))                  # noqa: E731
    lookup = {c: i for i, c in enumerate(code(train))}
    clones, inv_n = [], []
    for k in range(L + 1):
        xt = reveal_first_k(planted, rank, k, q)
        C = complete(den, xt, 4, q, g).reshape(-1, L).numpy()
        lab = np.array([lookup[c] for c in code(C)]).reshape(-1, 4)       # clones always land on training points
        clones.append(pair_agreement(lab).mean())
        inv_n.append((1.0 / consistent_counts(xt, T, q).double()).mean().item())
    np.testing.assert_allclose(clones, inv_n, atol=0.03)
    assert abs(inv_n[0] - 1 / n) < 1e-12 and inv_n[-1] == 1.0

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(range(L + 1), inv_n, color=SERIES[1], label="E[1/N] (counting)")
    ax.plot(range(L + 1), clones, "o", color=SERIES[0], ms=5, label="clones land on the same training point")
    ax.axhline(1 / n, color=INK_2, ls=":", lw=1, label="1/n")
    ax.set(xlabel="revealed sites k", ylabel="collapse commitment", title=f"Empirical denoiser, n={n}, L={L}")
    ax.legend(fontsize=7)
    save(fig, outdir / "empirical_collapse.png")


# ---------------------------------------------------------------------------- the whole analysis on a tiny tree
def test_regimes_pipeline_tiny(outdir):
    import json

    from test_15_diffusion_pipeline import RUN, TINY, load_script

    for s in ["s01_build_ground_truth", "s02_sample_ground_truth", "s03_analyze_ground_truth", "s04_make_dataset"]:
        mod = load_script(s)
        mod.main(TINY, outdir, verbose=False) if s in ("s02_sample_ground_truth", "s04_make_dataset") else mod.main(TINY, outdir)
    load_script("s09_train_diffusion").main(RUN, TINY, outdir, verbose=False)
    out = load_script("s13_dynamical_regimes").main(RUN, TINY, outdir, n_start=100, n_clones=4, n_theory=300,
                                                     n_traj=2, n_traj_clones=8, verbose=False)
    json.dumps(out)
    L = TINY["hierarchy"]["L"]
    for lvl in ["family", "subfamily", "clade"]:
        psi = out["speciation_cloning"]["A_test"][lvl]["psi"]
        assert psi[0] == 0.0 and psi[-1] == 1.0                      # nothing decided at k=0, everything at k=L
        assert out["speciation_theory"][lvl]["profile_psi"][-1] > out["speciation_theory"][lvl]["profile_psi"][0]
    col = out["collapse"]
    assert col["measured_planted_E_inv_N"][-1] >= col["measured_planted_E_inv_N"][0]
    assert col["match_prob"]["C_train_planted"] == [1.0] * len(out["k_grid"])   # planted paths always match
    assert abs(col["entropy_H_k"][0]) < 1e-9 and col["entropy_H_k"][L] > col["entropy_H_k"][1]
    assert (outdir / "evaluations" / "tiny_md" / "regimes.json").exists()
    for f in ["speciation", "speciation_universal", "trajectories", "collapse", "entropy", "phase_diagram"]:
        assert (outdir / "figures" / "models" / "tiny_md" / "regimes" / f"{f}.png").exists()
