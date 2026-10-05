"""Step 4 - frequencies, correlations, comparison metric, PCA. Toy alignments with known answers."""
import numpy as np

from psg.plots import plot_pca, plot_stat_comparison, save
from psg.stats import (
    PCA,
    compare_statistics,
    connected_correlations,
    mean_pairwise_identity,
    pair_frequencies,
    site_frequencies,
    upper_pairs,
)


def test_frequencies_by_hand():
    # 4 sequences, L=2, q=2
    S = np.array([[0, 0], [0, 1], [1, 1], [1, 1]])
    f1 = site_frequencies(S, 2)
    np.testing.assert_allclose(f1, [[0.5, 0.5], [0.25, 0.75]])
    f2 = pair_frequencies(S, 2)
    np.testing.assert_allclose(f2[0, 1], [[0.25, 0.25], [0.0, 0.5]])
    np.testing.assert_allclose(f2[1, 0], f2[0, 1].T)
    np.testing.assert_allclose(f2[0, 0], np.diag(f1[0]))          # f_ii(a,b) = delta_ab f_i(a)
    C = connected_correlations(S, 2)
    # C_12(1,1) = 0.5 - 0.5 * 0.75
    assert np.isclose(C[0, 1, 1, 1], 0.125)
    np.testing.assert_allclose(C[0, 1].sum(), 0, atol=1e-12)        # rows/cols of C sum to 0
    np.testing.assert_allclose(C[0, 0], 0)


def test_copied_column_correlation(rng):
    """s_2 = s_1 exactly -> C_12(a,a) = f(a)(1 - f(a)), C_12(a,b) = -f(a) f(b)."""
    x = rng.integers(0, 3, 5000)
    S = np.stack([x, x], axis=1)
    f = site_frequencies(S, 3)[0]
    C = connected_correlations(S, 3)[0, 1]
    np.testing.assert_allclose(C, np.diag(f) - np.outer(f, f), atol=1e-12)


def test_independent_columns_are_uncorrelated(rng):
    M = 20000
    S = rng.integers(0, 4, (M, 6))
    C = upper_pairs(connected_correlations(S, 4))
    assert np.abs(C).max() < 5 * 0.25 / np.sqrt(M) * 2      # ~ several standard errors of f_ij


def test_compare_statistics(rng, outdir):
    """Identical sets -> r = 1, rmse = 0.  Two samples of the same distribution -> r close to 1.
    A distribution with no correlations -> r_C near 0."""
    M, L = 4000, 8
    base = rng.integers(0, 3, (M, L))
    base[:, 1] = base[:, 0]                     # a strong correlation 0-1
    base[:, 5] = (base[:, 4] + 1) % 3           # and 4-5
    m = compare_statistics(base, base, 3)
    assert np.isclose(m["pearson_f1"], 1) and np.isclose(m["pearson_C"], 1) and m["rmse_C"] == 0

    other = rng.integers(0, 3, (M, L))
    other[:, 1] = other[:, 0]
    other[:, 5] = (other[:, 4] + 1) % 3
    m_same = compare_statistics(base, other, 3)
    assert m_same["pearson_C"] > 0.95

    shuffled = np.stack([rng.permutation(base[:, i]) for i in range(L)], axis=1)   # keeps f_i, kills C_ij
    m_shuf = compare_statistics(base, shuffled, 3)
    assert m_shuf["pearson_f1"] > 0.999 and abs(m_shuf["pearson_C"]) < 0.2

    fig = plot_stat_comparison(site_frequencies(base, 3).ravel(), site_frequencies(shuffled, 3).ravel(),
                               upper_pairs(connected_correlations(base, 3)),
                               upper_pairs(connected_correlations(shuffled, 3)), m_shuf,
                               "original", "column-shuffled")
    save(fig, outdir / "original_vs_shuffled.png")


def test_identity():
    A = np.array([[0, 0, 0, 0], [0, 0, 1, 1]])
    assert np.isclose(mean_pairwise_identity(A), 0.5)
    assert np.isclose(mean_pairwise_identity(A[:1], A[:1]), 1.0)


def test_pca_separates_two_clusters(rng, outdir):
    """Two consensus sequences + 15% random mutations: PC1 splits them perfectly."""
    L, q, n = 30, 20, 300
    cons = rng.integers(0, q, (2, L))
    S, lab = [], []
    for k in range(2):
        X = np.repeat(cons[k][None], n, axis=0)
        mut = rng.random(X.shape) < 0.15
        X[mut] = rng.integers(0, q, mut.sum())
        S.append(X)
        lab += [k] * n
    S, lab = np.concatenate(S), np.array(lab)
    pca = PCA(q, n_components=5)
    Z = pca.fit_transform(S)
    assert np.all(np.diff(pca.explained_variance_ratio_) <= 1e-12)
    assert pca.explained_variance_ratio_.sum() <= 1 + 1e-9
    side = Z[:, 0] > 0
    acc = max((side == (lab == 0)).mean(), (side == (lab == 1)).mean())
    assert acc == 1.0
    # transform of new data from the same clusters lands on the same side
    np.testing.assert_allclose(pca.transform(S[:5]), Z[:5])

    save(plot_pca(Z, lab, np.zeros_like(lab), pca.explained_variance_ratio_, "Toy: two clusters", s=14),
         outdir / "pca_two_clusters.png")
