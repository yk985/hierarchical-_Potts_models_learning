"""Step 12 - distances between sequences and between sets of sequences."""
import numpy as np

from psg.distances import (
    distance_report,
    frechet_distance,
    hamming_matrix,
    nearest_neighbor_distance,
    pairwise_distances,
)
from psg.plots import SERIES, plt, save


def test_hamming_by_hand():
    A = np.array([[0, 0, 0, 0]])
    B = np.array([[0, 0, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])
    np.testing.assert_allclose(hamming_matrix(A, B, q=2), [[0.5, 1.0, 0.0]])


def test_nearest_neighbor(rng):
    T = rng.integers(0, 4, (300, 20))
    A = np.concatenate([T[:5], T[5:10].copy()])
    A[5:, :3] = (A[5:, :3] + 1) % 4               # 3 of 20 sites mutated -> distance <= 0.15 to its source
    d = nearest_neighbor_distance(A, T, q=4, device="cpu", chunk=4)
    np.testing.assert_allclose(d[:5], 0)
    assert (d[5:] <= 0.15 + 1e-12).all() and (d[5:] > 0).all()
    # same set with the trivial self-pair excluded: equals the brute-force minimum over the others
    self_d = nearest_neighbor_distance(T[:50], T[:50], q=4, exclude_identical_index=True, device="cpu", chunk=7)
    M = hamming_matrix(T[:50], T[:50], 4)
    np.fill_diagonal(M, np.inf)
    np.testing.assert_allclose(self_d, M.min(1), atol=1e-6)


def test_pairwise_random_sequences(rng):
    """Uniform random sequences over q letters differ at a fraction 1 - 1/q of the sites on average."""
    S = rng.integers(0, 5, (2000, 40))
    assert abs(pairwise_distances(S).mean() - 0.8) < 0.01
    assert pairwise_distances(np.zeros((10, 5), int)).max() == 0


def test_frechet_known_values(rng, outdir):
    d, n = 4, 20000
    X = rng.normal(0, 1, (n, d))
    assert frechet_distance(X, X) < 1e-8
    shift = np.array([1.0, 2.0, 0, 0])
    fd_shift = frechet_distance(X, rng.normal(0, 1, (n, d)) + shift)        # expected ||shift||^2 = 5
    assert abs(fd_shift - 5.0) < 0.15
    fd_scale = frechet_distance(X, 2 * rng.normal(0, 1, (n, d)))          # N(0,I) vs N(0,4I): Tr(I+4I-2*2I) = d
    assert abs(fd_scale - d) < 0.15
    fd_same = frechet_distance(X, rng.normal(0, 1, (n, d)))               # same distribution: only noise
    assert fd_same < 0.01

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["same law", "mean shifted\n(expected 5)", "std x2\n(expected 4)"], [fd_same, fd_shift, fd_scale],
           color=SERIES[0], width=0.5)
    ax.set(ylabel="Frechet distance", title="Frechet distance on known Gaussians")
    save(fig, outdir / "frechet_known_values.png")


def test_distance_report_detects_copies(rng):
    train = rng.integers(0, 4, (500, 15))
    ref = rng.integers(0, 4, (200, 15))
    copies = distance_report(train[:100], ref, train, q=4, n_pairs=5000)
    assert copies["fraction_copies_of_train"] == 1.0 and copies["nn_to_train_generated_mean"] == 0.0
    fresh = distance_report(rng.integers(0, 4, (200, 15)), ref, train, q=4, n_pairs=5000)
    assert fresh["fraction_copies_of_train"] < 0.01
    # fresh sequences from the same law: same NN distance to train and same pairwise distances as the reference
    assert abs(fresh["nn_to_train_generated_mean"] - fresh["nn_to_train_reference_mean"]) < 0.02
    assert fresh["W1_within_generated_vs_within_reference"] < 0.01
