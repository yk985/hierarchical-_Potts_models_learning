"""Step 10 - evaluation functions: J recovery, contact precision, sample comparison."""
import numpy as np

from psg.evaluation import (
    compare_samples,
    compare_to_nodes,
    contact_precision,
    contact_scores,
    coupling_recovery,
    family_assignment,
)
from psg.hierarchy import HierarchyConfig, build_hierarchy
from psg.potts import Potts


def test_contact_precision_by_hand():
    C = np.zeros((6, 6), bool)
    C[0, 4] = C[4, 0] = C[1, 5] = C[5, 1] = True
    scores = np.zeros((6, 6))
    scores[0, 4] = scores[4, 0] = 2.0
    scores[0, 5] = scores[5, 0] = 1.0          # a wrong pair ranked 2nd
    scores[1, 5] = scores[5, 1] = 0.5
    # pairs with |i-j| >= 4: (0,4), (0,5), (1,5); 2 true contacts -> top-2 = (0,4), (0,5) -> 1/2
    assert contact_precision(scores, C, min_sep=4) == 0.5
    assert contact_precision(scores, C, min_sep=4, k=3) == 2 / 3


def test_true_model_finds_its_contacts():
    """Scores computed from the TRUE couplings recover the contact map perfectly."""
    cfg = HierarchyConfig(L=30, q=5, n_families=1, n_subfamilies=1, n_clades=1, n_long_contacts=20, seed=0)
    nodes, C = build_hierarchy(cfg)
    m = nodes["F0.S0.C0"].model
    assert contact_precision(contact_scores(m), C, min_sep=4) == 1.0


def test_coupling_recovery_is_gauge_invariant(rng):
    cfg = HierarchyConfig(L=12, q=4, n_families=1, n_subfamilies=1, n_clades=1, n_long_contacts=8, seed=1)
    nodes, C = build_hierarchy(cfg)
    m = nodes["F0.S0.C0"].model
    # a gauge transform (eq. 4 of the PDF): J_ij(a,b) += u_ij(a) + u_ji(b), h_i(a) -= sum_j u_ij(a)
    u = rng.normal(0, 1, (12, 12, 4))
    u[np.arange(12), np.arange(12)] = 0
    J = m.J + u[:, :, :, None] + u.transpose(1, 0, 2)[:, :, None, :]
    J[np.arange(12), np.arange(12)] = 0
    h = m.h - u.sum(axis=1)
    moved = Potts(h, J)
    r = coupling_recovery(moved, m, C)
    assert r["pearson_J_all_pairs"] > 0.9999 and r["pearson_h"] > 0.9999 and r["rel_error_J"] < 1e-8


def test_compare_samples(rng):
    ref = rng.integers(0, 4, (500, 10))
    out = compare_samples(ref[:200], ref, 4)
    assert out["fraction_copies_of_reference"] == 1.0
    out2 = compare_samples(rng.integers(0, 4, (200, 10)), ref, 4)
    assert out2["fraction_copies_of_reference"] < 0.01
    assert abs(out2["identity_gen_ref"] - 0.25) < 0.02


def test_compare_to_nodes_finds_itself():
    cfg = HierarchyConfig(L=12, q=4, n_families=2, n_subfamilies=2, n_clades=2, n_long_contacts=8, seed=2)
    nodes, C = build_hierarchy(cfg)
    out = compare_to_nodes(nodes["F1.S0.C1"].model, nodes, C)
    assert len(out) == 1 + 2 + 4 + 8
    assert out["F1.S0.C1"]["d_J"] < 1e-9 and abs(out["F1.S0.C1"]["pearson_J_contacts"] - 1) < 1e-9
    assert abs(out["F1.S0.C1"]["slope_J_contacts"] - 1) < 1e-9
    # closest after itself: its sibling clade, then its parent subfamily beats the other family
    assert out["F1.S0"]["d_J"] < out["F0"]["d_J"]
    assert out["F1.S0.C1"]["level"] == 3 and out["root"]["family"] == -1


def test_family_assignment_toy(rng):
    Zref = np.concatenate([rng.normal(0, 1, (500, 2)), rng.normal(10, 1, (500, 2))])
    fam = np.repeat([0, 1], 500)
    Z = np.concatenate([rng.normal(0, 1, (300, 2)), rng.normal(10, 1, (100, 2)), np.full((100, 2), 5.0)])
    out = family_assignment(Z, Zref, fam)
    assert abs(out["F0"] - 0.6) < 0.02 and abs(out["F1"] - 0.2) < 0.02 and abs(out["between"] - 0.2) < 0.02


def test_pseudo_likelihood_classifier(rng):
    """Sequences sampled from two different Potts models are assigned to the right one; the score equals the
    sum of log conditionals computed with Potts.conditional."""
    from psg.evaluation import classify_by_pseudo_likelihood, pseudo_loglik_scores
    from psg.potts import symmetrize
    from psg.sampling import gibbs_sample

    L, q = 10, 4
    models = [Potts(rng.normal(0, 1.5, (L, q)), symmetrize(rng.normal(0, 0.5, (L, L, q, q)))) for _ in range(2)]
    S = np.concatenate([gibbs_sample(m, 300, 50, rng=rng).samples for m in models])
    lab = np.repeat([0, 1], 300)
    assert (classify_by_pseudo_likelihood(S, models) == lab).mean() > 0.97
    sc = pseudo_loglik_scores(S[:3], models)
    ref = sum(np.log(models[1].conditional(S[:3], i)[np.arange(3), S[:3, i]]) for i in range(L))
    np.testing.assert_allclose(sc[:, 1], ref)
