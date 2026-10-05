"""Distances between sequences and between SETS of sequences (definitions in docs/modelization.pdf).

Sequence level
    Hamming distance        d_H(s, s') = (1/L) sum_i 1[s_i != s'_i]        in [0, 1]   (= 1 - identity)
    nearest-neighbour dist. d_NN(s, T) = min_{t in T} d_H(s, t)

Set level (generated set G vs reference set R)
    novelty / memorisation  distribution of d_NN(g, Train) for g in G, compared with d_NN(r, Train) for held-out
                            real sequences r; fraction of g with d_NN = 0 (exact copies of a training sequence)
    pairwise distributions  D_GG, D_RR, D_GR: d_H over random pairs within G, within R, across;
                            W1(D_GG, D_RR) and W1(D_GR, D_RR) (1-D Wasserstein distance = area between CDFs)
    PCA space               Frechet distance between Gaussian fits of the PCA coordinates:
                            FD = ||mu_G - mu_R||^2 + Tr(S_G + S_R - 2 (S_G^1/2 S_R S_G^1/2)^1/2)
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance


def _onehot(S: np.ndarray, q: int, device) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.as_tensor(S, dtype=torch.long, device=device), q).reshape(len(S), -1).float()


def hamming_matrix(A: np.ndarray, B: np.ndarray, q: int, device: str | torch.device = "cpu") -> np.ndarray:
    """Normalised Hamming distances between all rows of A (n, L) and B (m, L). Returns (n, m). Small sets only."""
    L = A.shape[1]
    same = _onehot(A, q, device) @ _onehot(B, q, device).T          # number of identical positions
    return (1.0 - same / L).cpu().numpy()


def nearest_neighbor_distance(A: np.ndarray, B: np.ndarray, q: int, exclude_identical_index: bool = False,
                              chunk: int = 1024, device: str | torch.device | None = None) -> np.ndarray:
    """d_NN(a, B) = min_b d_H(a, b) for every row a of A. Chunked matrix products (GPU if available).
    exclude_identical_index: A and B are the same set; ignore the pair (a, a)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    L = A.shape[1]
    XB = _onehot(B, q, device)
    out = np.empty(len(A))
    for k in range(0, len(A), chunk):
        same = _onehot(A[k:k + chunk], q, device) @ XB.T
        if exclude_identical_index:
            idx = torch.arange(k, min(k + chunk, len(A)), device=device)
            same[torch.arange(len(idx), device=device), idx] = -1.0
        out[k:k + chunk] = (1.0 - same.max(dim=1).values / L).cpu().numpy()
    return out


def pairwise_distances(A: np.ndarray, B: np.ndarray | None = None, n_pairs: int = 50_000,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Normalised Hamming distances of n_pairs random pairs (within A if B is None, else across A x B)."""
    rng = np.random.default_rng(0) if rng is None else rng
    B_ = A if B is None else B
    i = rng.integers(0, len(A), n_pairs)
    j = rng.integers(0, len(B_), n_pairs)
    if B is None:
        keep = i != j
        i, j = i[keep], j[keep]
    return (A[i] != B_[j]).mean(axis=1)


def frechet_distance(Z1: np.ndarray, Z2: np.ndarray) -> float:
    """Frechet distance between Gaussian fits of two point clouds (rows = points). Zero iff same mean and
    covariance. It is the squared 2-Wasserstein distance between the two Gaussians (same idea as the FID)."""
    m1, m2 = Z1.mean(0), Z2.mean(0)
    S1, S2 = np.cov(Z1, rowvar=False), np.cov(Z2, rowvar=False)
    r1 = sqrtm(S1)
    cross = sqrtm(r1 @ S2 @ r1)
    fd = ((m1 - m2) ** 2).sum() + np.trace(S1 + S2 - 2 * np.real(cross))
    return float(max(fd, 0.0))


def distance_report(gen: np.ndarray, ref: np.ndarray, train: np.ndarray, q: int, pca=None,
                    n_pairs: int = 50_000, seed: int = 0, nn_ref: np.ndarray | None = None) -> dict:
    """All set-level distances of a generated set vs a reference (test) set. nn_ref (d_NN of the reference
    sequences to the training set) can be passed to avoid recomputing it."""
    rng = np.random.default_rng(seed)
    nn_gen = nearest_neighbor_distance(gen, train, q)
    nn_ref = nearest_neighbor_distance(ref, train, q) if nn_ref is None else nn_ref
    D_gg, D_rr, D_gr = (pairwise_distances(gen, None, n_pairs, rng), pairwise_distances(ref, None, n_pairs, rng),
                        pairwise_distances(gen, ref, n_pairs, rng))
    out = {
        "nn_to_train_generated_mean": float(nn_gen.mean()),
        "nn_to_train_reference_mean": float(nn_ref.mean()),
        "nn_to_train_W1": float(wasserstein_distance(nn_gen, nn_ref)),
        "fraction_copies_of_train": float((nn_gen == 0).mean()),
        "pairwise_within_generated_mean": float(D_gg.mean()),
        "pairwise_within_reference_mean": float(D_rr.mean()),
        "pairwise_generated_reference_mean": float(D_gr.mean()),
        "W1_within_generated_vs_within_reference": float(wasserstein_distance(D_gg, D_rr)),
        "W1_cross_vs_within_reference": float(wasserstein_distance(D_gr, D_rr)),
    }
    if pca is not None:
        out["frechet_pca"] = frechet_distance(pca.transform(gen), pca.transform(ref))
    return out
