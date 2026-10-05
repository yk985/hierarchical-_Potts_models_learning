"""Gibbs sampling from a Potts model.

One sweep = visit every site i = 0..L-1 once and redraw s_i from its exact conditional

    P(s_i = a | s_{-i}) = softmax_a( beta * [h_i(a) + sum_j J_ij(a, s_j)] ),

i.e. the same site conditionals that the pseudo-likelihood (PLM) is built from.
Many independent chains are run in parallel (vectorised over chains).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .potts import Potts, softmax


@dataclass
class GibbsResult:
    samples: np.ndarray        # (n_chains * n_samples_per_chain, L) int
    energy_trace: np.ndarray   # (n_sweeps_total, n_chains) energy of each chain after each sweep
    chain_id: np.ndarray       # (n_samples,) which chain each sample came from


def gibbs_sample(
    model: Potts,
    n_chains: int,
    n_burnin: int,
    n_samples_per_chain: int = 1,
    thin: int = 1,
    beta: float = 1.0,
    rng: np.random.Generator | None = None,
    init: np.ndarray | None = None,
) -> GibbsResult:
    """Run n_chains parallel Gibbs chains.

    Each chain: n_burnin sweeps (discarded), then n_samples_per_chain samples taken every `thin` sweeps.
    Returns samples ordered chain-major within each sampling step.
    """
    rng = np.random.default_rng() if rng is None else rng
    L, q = model.L, model.q
    S = rng.integers(0, q, (n_chains, L)) if init is None else np.array(init, dtype=np.int64)
    nbrs = model.neighbors()
    n_sweeps = n_burnin + n_samples_per_chain * thin
    energies = np.empty((n_sweeps, n_chains))
    out, ids = [], []
    for t in range(n_sweeps):
        for i in range(L):
            p = softmax(beta * model.local_field(S, i, nbrs[i]), axis=1)
            S[:, i] = _draw(p, rng)
        energies[t] = model.energy(S)
        if t >= n_burnin and (t - n_burnin + 1) % thin == 0:
            out.append(S.copy())
            ids.append(np.arange(n_chains))
    return GibbsResult(np.concatenate(out), energies, np.concatenate(ids))


def _draw(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One categorical draw per row of p (M, q) by inverse CDF."""
    u = rng.random((p.shape[0], 1))
    idx = (np.cumsum(p, axis=1) < u).sum(axis=1)
    return np.minimum(idx, p.shape[1] - 1)
