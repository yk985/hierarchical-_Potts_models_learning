"""Alignment statistics: frequencies, correlations, comparison metrics, PCA.

    f_i(a)       = (1/M) sum_m  1[s_i^m = a]
    f_ij(a, b)   = (1/M) sum_m  1[s_i^m = a] 1[s_j^m = b]
    C_ij(a, b)   = f_ij(a, b) - f_i(a) f_j(b)          (connected correlation, i != j)

These are the statistics a Potts model is fitted to reproduce, so they are also the first test a
generative model must pass: its samples should reproduce f_i and C_ij of the reference sequences.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .alphabet import one_hot


def site_frequencies(S: np.ndarray, q: int) -> np.ndarray:
    """(M, L) -> (L, q)."""
    return one_hot(S, q).mean(axis=0)


def pair_frequencies(S: np.ndarray, q: int) -> np.ndarray:
    """(M, L) -> (L, L, q, q), f[i, j, a, b]."""
    M, L = S.shape
    X = one_hot(S, q).reshape(M, L * q)
    return (X.T @ X / M).reshape(L, q, L, q).transpose(0, 2, 1, 3)


def connected_correlations(S: np.ndarray, q: int) -> np.ndarray:
    """(M, L) -> (L, L, q, q). Diagonal blocks i == j are set to 0 (they are not correlations between sites)."""
    f1 = site_frequencies(S, q)
    C = pair_frequencies(S, q) - np.einsum("ia,jb->ijab", f1, f1)
    L = S.shape[1]
    C[np.arange(L), np.arange(L)] = 0.0
    return C


def upper_pairs(X: np.ndarray) -> np.ndarray:
    """Flatten the i<j blocks of an (L, L, q, q) tensor -> (n_pairs * q * q,)."""
    iu = np.triu_indices(X.shape[0], k=1)
    return X[iu].ravel()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x - x.mean(), y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def compare_statistics(S_ref: np.ndarray, S_gen: np.ndarray, q: int) -> dict:
    """Metric between two sets of sequences: how well S_gen reproduces the statistics of S_ref."""
    f_ref, f_gen = site_frequencies(S_ref, q), site_frequencies(S_gen, q)
    c_ref = upper_pairs(connected_correlations(S_ref, q))
    c_gen = upper_pairs(connected_correlations(S_gen, q))
    return {
        "pearson_f1": pearson(f_ref.ravel(), f_gen.ravel()),
        "rmse_f1": float(np.sqrt(((f_ref - f_gen) ** 2).mean())),
        "pearson_C": pearson(c_ref, c_gen),
        "rmse_C": float(np.sqrt(((c_ref - c_gen) ** 2).mean())),
        "slope_C": float((c_ref * c_gen).sum() / (c_ref * c_ref).sum()),   # least squares through origin
    }


def mean_pairwise_identity(A: np.ndarray, B: np.ndarray | None = None, max_pairs: int = 200_000,
                           rng: np.random.Generator | None = None) -> float:
    """Average fraction of identical positions between random pairs (from A x A, or A x B)."""
    rng = np.random.default_rng(0) if rng is None else rng
    B_ = A if B is None else B
    n = min(max_pairs, len(A) * len(B_))
    i = rng.integers(0, len(A), n)
    j = rng.integers(0, len(B_), n)
    if B is None:
        keep = i != j
        i, j = i[keep], j[keep]
    return float((A[i] == B_[j]).mean())


# ---------------------------------------------------------------------- PCA
@dataclass
class PCA:
    """PCA on flattened one-hot sequences (M, L*q). Fit once on the reference data, then reuse
    `transform` to project any other set (e.g. generated sequences) on the same axes."""
    q: int
    n_components: int = 10
    mean_: np.ndarray | None = None
    components_: np.ndarray | None = None          # (n_components, L*q)
    explained_variance_ratio_: np.ndarray | None = None

    def _flat(self, S):
        return one_hot(S, self.q).reshape(len(S), -1)

    def fit(self, S: np.ndarray) -> "PCA":
        X = self._flat(S)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = Xc.T @ Xc / (len(X) - 1)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        self.components_ = evecs[:, : self.n_components].T
        self.explained_variance_ratio_ = evals[: self.n_components] / evals.sum()
        return self

    def transform(self, S: np.ndarray) -> np.ndarray:
        return (self._flat(S) - self.mean_) @ self.components_.T

    def fit_transform(self, S: np.ndarray) -> np.ndarray:
        return self.fit(S).transform(S)
