"""Potts model of a protein sequence.

Convention (same as the PDF, docs/modelization.pdf):

    E(s) = - sum_i h_i(s_i) - sum_{i<j} J_ij(s_i, s_j)
    P(s) = exp(-beta * E(s)) / Z

Storage:
    h : (L, q)          fields
    J : (L, L, q, q)    couplings, with J[i, j, a, b] == J[j, i, b, a] and J[i, i] == 0
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Potts:
    h: np.ndarray
    J: np.ndarray

    def __post_init__(self):
        self.h = np.asarray(self.h, dtype=np.float64)
        self.J = np.asarray(self.J, dtype=np.float64)
        L, q = self.h.shape
        if self.J.shape != (L, L, q, q):
            raise ValueError(f"J has shape {self.J.shape}, expected {(L, L, q, q)}")

    @property
    def L(self) -> int:
        return self.h.shape[0]

    @property
    def q(self) -> int:
        return self.h.shape[1]

    # ------------------------------------------------------------------ checks
    def check_symmetry(self, atol: float = 1e-10) -> None:
        if not np.allclose(self.J, self.J.transpose(1, 0, 3, 2), atol=atol):
            raise ValueError("J must satisfy J[i,j,a,b] == J[j,i,b,a]")
        if np.abs(self.J[np.arange(self.L), np.arange(self.L)]).max() > atol:
            raise ValueError("J[i,i] must be zero")

    def neighbors(self, atol: float = 0.0) -> list[np.ndarray]:
        """For every site i, the sites j whose coupling block J_ij is not identically zero."""
        norms = np.abs(self.J).max(axis=(2, 3))
        return [np.flatnonzero(norms[i] > atol) for i in range(self.L)]

    # ------------------------------------------------------------------ energy
    def energy(self, S: np.ndarray) -> np.ndarray:
        """E(s) for every row of S (M, L). Returns (M,)."""
        S = np.atleast_2d(S)
        M, L = S.shape
        E = -self.h[np.arange(L)[None, :], S].sum(axis=1)
        for i in range(L - 1):
            j = np.arange(i + 1, L)
            # J[i, j, S[:, i], S[:, j]] for every chain and every j > i
            E -= self.J[i, j[None, :], S[:, i][:, None], S[:, j]].sum(axis=1)
        return E

    def local_field(self, S: np.ndarray, i: int, nbrs: np.ndarray | None = None) -> np.ndarray:
        """H_i(a | s_rest) = h_i(a) + sum_{j != i} J_ij(a, s_j).  Returns (M, q).

        The conditional of site i given all the others is softmax(beta * H_i).
        """
        S = np.atleast_2d(S)
        if nbrs is None:
            nbrs = np.delete(np.arange(self.L), i)
        if len(nbrs) == 0:
            return np.broadcast_to(self.h[i], (S.shape[0], self.q)).copy()
        Jn = self.J[i, nbrs]  # (k, q, q)
        # Jn[k, :, S[m, nbrs[k]]] -> (M, k, q)
        contrib = Jn[np.arange(len(nbrs))[None, :], :, S[:, nbrs]]
        return self.h[i][None, :] + contrib.sum(axis=1)

    def conditional(self, S: np.ndarray, i: int, beta: float = 1.0) -> np.ndarray:
        """P(s_i = a | s_{-i}) for every row of S. Returns (M, q), rows sum to 1."""
        return softmax(beta * self.local_field(S, i), axis=1)

    # ------------------------------------------------------------------ gauge
    def zero_sum_gauge(self) -> "Potts":
        """Equivalent model (same P(s)) with sum_a J_ij(a,b) = sum_b J_ij(a,b) = 0 and sum_a h_i(a) = 0.

        This is the gauge in which ||J||_F is minimal, which makes distances between models meaningful.
        """
        J = self.J
        Ja = J.mean(axis=2, keepdims=True)          # J_ij(., b)
        Jb = J.mean(axis=3, keepdims=True)          # J_ij(a, .)
        Jab = J.mean(axis=(2, 3), keepdims=True)    # J_ij(., .)
        J0 = J - Ja - Jb + Jab
        h = self.h + Jb[..., 0].sum(axis=1)         # h_i(a) + sum_j J_ij(a, .)
        h0 = h - h.mean(axis=1, keepdims=True)
        return Potts(h0, J0)

    # ------------------------------------------------------------------ I/O
    def save(self, path: str | Path, **meta) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, h=self.h, J=self.J, **meta)

    @classmethod
    def load(cls, path: str | Path) -> "Potts":
        d = np.load(path)
        return cls(d["h"], d["J"])


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def symmetrize(J: np.ndarray) -> np.ndarray:
    """Force J[i,j,a,b] == J[j,i,b,a] and zero diagonal blocks, keeping the upper triangle i<j."""
    L = J.shape[0]
    iu = np.triu_indices(L, k=1)
    out = np.zeros_like(J)
    out[iu] = J[iu]
    out[iu[1], iu[0]] = J[iu].transpose(0, 2, 1)
    return out


def enumerate_states(L: int, q: int) -> np.ndarray:
    """All q**L sequences, (q**L, L). Only for tiny test systems."""
    return np.array(list(itertools.product(range(q), repeat=L)), dtype=np.int64)


def exact_distribution(model: Potts, beta: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Exact Boltzmann distribution by enumeration. Returns (states, probs)."""
    states = enumerate_states(model.L, model.q)
    logw = -beta * model.energy(states)
    p = np.exp(logw - logw.max())
    return states, p / p.sum()
