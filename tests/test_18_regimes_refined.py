"""Step 18 - refinements of the speciation analysis: exact class posteriors (partition functions by thermodynamic
integration), soft-label cloning (no classifier error), correlation-weighted profile theory."""
import math

import numpy as np
import torch

from psg.potts import Potts, enumerate_states, exact_distribution, symmetrize
from psg.regimes import (
    complete,
    correlation_weights,
    exact_log_likelihoods,
    information_correlation,
    pair_agreement,
    profile_log_posterior,
    random_orders,
    reveal_first_k,
    soft_pair_agreement,
    thermodynamic_log_z,
    weighted_profile_log_posterior,
    within_class_mutual_information,
)
from test_13_masked_diffusion import exact_denoiser
from test_17_regimes import exact_commitment


def tiny_potts(rng, L=4, q=3, s=1.0):
    return Potts(rng.normal(0, 1.0, (L, q)), symmetrize(rng.normal(0, s, (L, L, q, q))))


def exact_log_z(m):
    states = enumerate_states(m.L, m.q)
    e = -m.energy(states)
    return float(np.log(np.exp(e - e.max()).sum()) + e.max())


def test_thermodynamic_integration_matches_enumeration(rng):
    for s in [0.5, 1.0]:
        m = tiny_potts(rng, 5, 3, s)
        r = thermodynamic_log_z(m, n_lambda=21, n_chains=4000, n_equil=20, n_measure=20, n_burnin=100)
        assert abs(r["log_z"] - exact_log_z(m)) < 0.02, (r["log_z"], exact_log_z(m))
        assert r["error"] < 0.05
        assert abs(r["log_z0"] - np.log(np.exp(m.h).sum(1)).sum()) < 1e-9


def test_exact_log_likelihoods_match_enumeration(rng):
    models = [tiny_potts(rng) for _ in range(2)]
    zs = [exact_log_z(m) for m in models]
    states, p = exact_distribution(models[1])
    ll = exact_log_likelihoods(states, models, zs)
    np.testing.assert_allclose(np.exp(ll[:, 1]), p, atol=1e-5)
    np.testing.assert_allclose(np.exp(ll).sum(0), [1.0, 1.0], atol=1e-4)


def test_soft_pair_agreement_by_hand():
    # one-hot probabilities reproduce hard pair agreement
    lab = np.array([[0, 0, 1, 1]])
    R = np.eye(2)[lab]
    np.testing.assert_allclose(soft_pair_agreement(R), pair_agreement(lab))
    # two clones with probabilities (0.5, 0.5): agreement 0.5 ; (1,0) and (0.5,0.5): 0.5
    np.testing.assert_allclose(soft_pair_agreement(np.array([[[0.5, 0.5], [0.5, 0.5]], [[1, 0], [0.5, 0.5]]])), [0.5, 0.5])


def test_soft_cloning_is_exact_for_latent_classes(rng):
    """Mixture of two Potts models: labelling each clone by its exact posterior P(c | clone) gives clone agreement
    = E[sum_c P(c | x_U)^2] for the LATENT class, including at k = L where it stays below 1 (classes overlap)."""
    L, q = 4, 3
    models = [tiny_potts(rng, L, q, 0.7) for _ in range(2)]
    states = enumerate_states(L, q)
    pc = np.stack([exact_distribution(m)[1] for m in models])            # (2, n_states), each normalised
    p = 0.5 * pc.sum(0)
    post = (0.5 * pc / p).T                                               # P(c | x) for every state
    den = exact_denoiser(states, p, q)
    g = torch.Generator().manual_seed(0)
    idx = rng.choice(len(states), size=3000, p=p)
    x0 = torch.as_tensor(states[idx])
    rank = random_orders(len(x0), L, g)
    code = q ** np.arange(L - 1, -1, -1)
    for k in range(L + 1):
        # exact E sum_c P(c|x_U)^2 by enumeration: P(c|x_U) = sum over consistent states of p(x) P(c|x) / P(x_U)
        tot = 0.0
        import itertools
        for U in itertools.combinations(range(L), k):
            key = states[:, list(U)] @ (10 ** np.arange(k)) if U else np.zeros(len(states), int)
            for v in np.unique(key):
                sel = key == v
                pu = p[sel].sum()
                pcu = (p[sel, None] * post[sel]).sum(0) / pu
                tot += pu * (pcu ** 2).sum()
        exact = tot / math.comb(L, k)
        C = complete(den, reveal_first_k(x0, rank, k, q), 6, q, g).reshape(-1, L).numpy()
        est = soft_pair_agreement(post[C @ code].reshape(len(x0), 6, 2)).mean()
        assert abs(est - exact) < 0.02, (k, est, exact)
    assert exact < 0.999                                                  # full sequences do not fix the class


def test_mutual_information_and_correlation(rng):
    N = 20000
    a = rng.integers(0, 4, N)
    S = np.stack([a, a, rng.integers(0, 4, N), (a + (rng.random(N) < 0.3)) % 4], 1)   # copy, independent, noisy copy
    mi, null = within_class_mutual_information(S, np.zeros(N, int), 4)
    H = math.log(4)
    assert abs(mi[0, 1] - H) < 0.01                                       # copy: MI = entropy
    assert mi[0, 2] < 0.002 and null[0, 2] > 0                            # independent: bias removed
    rho = information_correlation(mi)
    assert abs(rho[0, 1] - math.sqrt(1 - 1 / 16)) < 0.005                 # copy of a 4-state site: sqrt(1 - e^{-2 log 4})
    assert rho[0, 2] < 0.07 and 0.5 < rho[0, 3] < rho[0, 1]
    np.testing.assert_allclose(information_correlation(np.array([0.0])), [0.0])


def test_correlation_weights_by_hand():
    rho = np.array([[0, 0.5, 0.5, 0.9], [0.5, 0, 0, 0], [0.5, 0, 0, 0], [0.9, 0, 0, 0]])
    partners = rho > 0
    revealed = np.array([[True, True, True, False]])                     # site 3 (rho 0.9) hidden: ignored
    w = {m: correlation_weights(revealed, rho, partners, m)[0] for m in ["geo", "prod", "eff", "none"]}
    np.testing.assert_allclose(w["geo"][0], 0.5)                          # sqrt(0.5 * 0.5)
    np.testing.assert_allclose(w["prod"][0], 0.25)
    np.testing.assert_allclose(w["eff"][0], 0.5)                          # 1 / (1 + 0.5 + 0.5)
    np.testing.assert_allclose(w["geo"][1], 0.5)                          # one partner: 1 - 0.5
    np.testing.assert_allclose(w["none"], 1.0)
    # a site with no revealed partner keeps weight 1
    np.testing.assert_allclose(correlation_weights(np.array([[True, False, False, False]]), rho, partners, "geo")[0, 0], 1.0)


def test_weighted_posterior_reduces_to_profile(rng):
    log_f = np.log(rng.dirichlet(np.ones(3), size=(2, 5)))
    S = rng.integers(0, 3, (10, 5))
    rev = rng.random((10, 5)) < 0.6
    prior = np.log([0.5, 0.5])
    np.testing.assert_allclose(weighted_profile_log_posterior(S, rev, log_f, prior, np.ones((10, 5))),
                               profile_log_posterior(S, rev, log_f, prior))
    # weight 0 on every site: back to the prior
    np.testing.assert_allclose(np.exp(weighted_profile_log_posterior(S, rev, log_f, prior, np.zeros((10, 5)))), 0.5)


def test_refined_pipeline_tiny(outdir):
    from test_15_diffusion_pipeline import RUN, TINY, load_script

    for s in ["s01_build_ground_truth", "s02_sample_ground_truth", "s03_analyze_ground_truth", "s04_make_dataset"]:
        mod = load_script(s)
        mod.main(TINY, outdir, verbose=False) if s in ("s02_sample_ground_truth", "s04_make_dataset") else mod.main(TINY, outdir)
    load_script("s09_train_diffusion").main(RUN, TINY, outdir, verbose=False)
    ti = {"n_lambda": 5, "n_chains": 200, "n_equil": 5, "n_measure": 3, "n_burnin": 30}
    out = load_script("s14_regimes_refined").main(RUN, TINY, outdir, n_start=60, n_clones=4, n_theory=200, ti=ti,
                                                   verbose=False)
    L = TINY["hierarchy"]["L"]
    for lvl in ["family", "subfamily", "clade"]:
        v = out["exact_posterior_on_test"][lvl]
        assert v["accuracy_exact_posterior"] > 1 / 4
        r = out["cloning"]["A_test"][lvl]
        assert r["psi_soft"][0] == 0.0 and abs(r["psi_soft"][-1] - 1.0) < 1e-12
        assert 0 < r["ceiling"] <= 1
        for mode in ["none", "geo", "prod", "eff"]:
            assert len(out["theory"][lvl][mode]["phi"]) == L + 1
    assert (outdir / "figures" / "models" / "tiny_md" / "regimes" / "refined_speciation.png").exists()
    assert (outdir / "evaluations" / "ground_truth" / "log_partition.json").exists()
