"""Gibbs sampling on GPU (torch) for dense Potts models, e.g. the (h, J) extracted from a trained network.

Same algorithm and same conventions as psg.sampling.gibbs_sample:
one sweep redraws s_1, ..., s_L in order from P(s_i = a | s_-i) = softmax_a(beta * [h_i(a) + sum_j J_ij(a, s_j)]).
The local field of site i is computed as one matrix product  X (M, L*q) @ J_i (L*q, q),
where X is the one-hot encoding of the current sequences.
"""
from __future__ import annotations

import numpy as np
import torch

from .potts import Potts
from .sampling import GibbsResult


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def gibbs_sample_torch(
    model: Potts,
    n_chains: int,
    n_burnin: int,
    n_samples_per_chain: int = 1,
    thin: int = 1,
    beta: float = 1.0,
    seed: int = 0,
    device: torch.device | None = None,
    energy_every: int = 1,
) -> GibbsResult:
    device = default_device() if device is None else device
    gen = torch.Generator(device=device).manual_seed(seed)
    L, q = model.L, model.q
    h = torch.as_tensor(model.h, dtype=torch.float32, device=device)                     # (L, q)
    # Jcol[i] : (L*q, q) with Jcol[i][(j, b), a] = J_ij(a, b)
    Jcol = torch.as_tensor(model.J, dtype=torch.float32, device=device).permute(0, 1, 3, 2).reshape(L, L * q, q)

    S = torch.randint(0, q, (n_chains, L), generator=gen, device=device)
    X = torch.nn.functional.one_hot(S, q).float()                                       # (M, L, q)
    n_sweeps = n_burnin + n_samples_per_chain * thin
    energies, out = [], []
    for t in range(n_sweeps):
        for i in range(L):
            field = h[i] + X.reshape(n_chains, L * q) @ Jcol[i]                         # (M, q); J_ii = 0
            new = torch.multinomial(torch.softmax(beta * field, dim=1), 1, generator=gen).squeeze(1)
            S[:, i] = new
            X[:, i] = torch.nn.functional.one_hot(new, q).float()
        if energy_every and (t % energy_every == 0 or t == n_sweeps - 1):
            energies.append(_energy(X, h, Jcol, L, q).cpu().numpy())
        if t >= n_burnin and (t - n_burnin + 1) % thin == 0:
            out.append(S.cpu().numpy().copy())
    samples = np.concatenate(out)
    chain_id = np.tile(np.arange(n_chains), len(out))
    return GibbsResult(samples, np.stack(energies) if energies else np.empty((0, n_chains)), chain_id)


def _energy(X, h, Jcol, L, q):
    """E = -sum_i h_i(s_i) - 1/2 sum_{i != j} J_ij(s_i, s_j)."""
    M = X.shape[0]
    Xf = X.reshape(M, L * q)
    field_h = (X * h).sum(dim=(1, 2))
    # full field from couplings for every site: (M, L, q)
    H = torch.einsum("mk,ika->mia", Xf, Jcol)
    pair = (H * X).sum(dim=(1, 2))
    return -(field_h + 0.5 * pair)
