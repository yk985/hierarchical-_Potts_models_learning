"""Step 6 - GPU (torch) Gibbs sampler: same answers as exact enumeration and as the numpy sampler."""
import numpy as np
import torch

from psg.plots import SERIES, INK_2, plt, save
from psg.potts import Potts, exact_distribution, symmetrize
from psg.sampling import gibbs_sample
from psg.stats import site_frequencies
from psg.torch_sampling import _energy, gibbs_sample_torch


def state_index(S, q):
    return S @ (q ** np.arange(S.shape[1] - 1, -1, -1))


def test_torch_energy_matches_numpy(rng):
    m = Potts(rng.normal(0, 1, (5, 4)), symmetrize(rng.normal(0, 1, (5, 5, 4, 4))))
    S = rng.integers(0, 4, (50, 5))
    X = torch.nn.functional.one_hot(torch.as_tensor(S), 4).float()
    Jcol = torch.as_tensor(m.J, dtype=torch.float32).permute(0, 1, 3, 2).reshape(5, 20, 4)
    E = _energy(X, torch.as_tensor(m.h, dtype=torch.float32), Jcol, 5, 4).numpy()
    np.testing.assert_allclose(E, m.energy(S), rtol=1e-4, atol=1e-4)


def test_torch_gibbs_matches_exact(rng, outdir):
    L, q = 3, 3
    m = Potts(rng.normal(0, 0.8, (L, q)), symmetrize(rng.normal(0, 0.8, (L, L, q, q))))
    states, p = exact_distribution(m)
    res = gibbs_sample_torch(m, n_chains=5000, n_burnin=20, n_samples_per_chain=10, thin=3, seed=0)
    emp = np.bincount(state_index(res.samples, q), minlength=q**L) / len(res.samples)
    tv = 0.5 * np.abs(emp - p).sum()
    assert tv < 0.02, tv

    order = np.argsort(-p)
    x = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(x - 0.2, p[order], 0.4, color=SERIES[0], label="exact")
    ax.bar(x + 0.2, emp[order], 0.4, color=SERIES[1], label="torch Gibbs")
    ax.set(xlabel="state (sorted)", ylabel="P(s)", title=f"torch Gibbs vs exact, TV = {tv:.4f}")
    ax.legend()
    save(fig, outdir / "state_distribution.png")


def test_torch_matches_numpy_sampler(rng, outdir):
    """Larger sparse model (L=15, q=6): the two samplers give the same site frequencies."""
    L, q = 15, 6
    C = np.zeros((L, L), bool)
    C[np.arange(L - 1), np.arange(1, L)] = True
    C = C | C.T
    m = Potts(rng.normal(0, 1, (L, q)), symmetrize(rng.normal(0, 0.8, (L, L, q, q)) * C[:, :, None, None]))
    a = gibbs_sample(m, 4000, 100, rng=rng).samples
    b = gibbs_sample_torch(m, 4000, 100, seed=1).samples
    fa, fb = site_frequencies(a, q), site_frequencies(b, q)
    assert np.abs(fa - fb).max() < 0.04

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls="--")
    ax.scatter(fa.ravel(), fb.ravel(), s=12, color=SERIES[0])
    ax.set(xlabel="f_i(a), numpy Gibbs", ylabel="f_i(a), torch Gibbs", title="numpy vs torch sampler")
    save(fig, outdir / "freq_numpy_vs_torch.png")
