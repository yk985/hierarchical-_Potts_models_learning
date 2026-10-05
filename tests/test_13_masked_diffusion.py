"""Step 13 - masked (absorbing-state) diffusion: forward process, ELBO, ancestral sampler, denoiser network.
The sampler and the loss are checked with an EXACT denoiser built by enumerating a tiny Potts model."""
import numpy as np
import torch

from psg.models.masked_diffusion import (
    MaskedDiffusionTransformer,
    ancestral_sample,
    load_diffusion,
    mask_tokens,
    nelbo_per_sequence,
    sample_times,
    save_diffusion,
)
from psg.plots import INK_2, SERIES, plt, save
from psg.potts import Potts, exact_distribution, symmetrize


def tiny_potts(rng, L=4, q=3):
    return Potts(rng.normal(0, 1.0, (L, q)), symmetrize(rng.normal(0, 1.0, (L, L, q, q))))


def exact_denoiser(states: np.ndarray, p: np.ndarray, q: int):
    """The optimal denoiser: log P(x_0^i = a | the unmasked sites of x_t), by enumeration of all states."""
    S = torch.as_tensor(states)
    P = torch.as_tensor(p, dtype=torch.float64)
    oh = torch.nn.functional.one_hot(S, q).double()                     # (n_states, L, q)

    def f(xt, t):
        obs = xt != q
        match = ((S[None] == xt[:, None]) | ~obs[:, None]).all(-1)      # (B, n_states): consistent with x_t
        w = P[None] * match
        marg = torch.einsum("bs,slq->blq", w, oh)
        marg = marg / marg.sum(-1, keepdim=True)
        return torch.log(marg + 1e-300).float()
    return f


def state_index(S, q):
    return S @ (q ** np.arange(S.shape[1] - 1, -1, -1))


def test_mask_tokens_and_times():
    g = torch.Generator().manual_seed(0)
    x0 = torch.randint(0, 5, (4000, 30), generator=g)
    for tv in [0.0, 0.3, 0.8, 1.0]:
        xt, m = mask_tokens(x0, torch.full((4000,), tv), mask_id=5, generator=g)
        assert abs(m.float().mean().item() - tv) < 0.01               # fraction masked = t
        assert (xt[~m] == x0[~m]).all() and (xt[m] == 5).all()        # unmasked sites untouched
    t = sample_times(1000, eps=1e-3, generator=g)
    assert t.min() >= 1e-3 and t.max() <= 1.0
    gaps = np.diff(np.sort(t.numpy()))
    assert gaps.max() < 2.0 / 1000                                    # stratified: evenly spread


def test_network_shapes_and_time_conditioning():
    m = MaskedDiffusionTransformer(L=7, q=4, d_model=32, n_layers=2, n_heads=4, d_ff=64)
    x = torch.randint(0, 5, (3, 7))
    out = m(x, torch.tensor([0.1, 0.5, 0.9]))
    assert out.shape == (3, 7, 4)                                     # q classes: no probability on [MASK]
    same_x = x[:1].repeat(2, 1)
    o2 = m(same_x, torch.tensor([0.1, 0.9]))
    assert not torch.allclose(o2[0], o2[1])                           # the output depends on t
    assert "MaskedDiffusionTransformer" in m.describe()


def test_save_load(tmp_path):
    m = MaskedDiffusionTransformer(L=5, q=3, d_model=16, n_layers=1, n_heads=2, d_ff=32)
    save_diffusion(m, tmp_path / "d.pt")
    m2 = load_diffusion(tmp_path / "d.pt")
    x, t = torch.randint(0, 4, (2, 5)), torch.tensor([0.3, 0.7])
    torch.testing.assert_close(m(x, t), m2(x, t))


def test_exact_denoiser_gives_exact_nelbo(rng):
    """With the exact denoiser, E[NELBO(x)] = -log p(x): averaged over data it equals the entropy H(p).
    A denoiser that ignores everything (uniform) gives L log q instead."""
    L, q = 4, 3
    states, p = exact_distribution(tiny_potts(rng, L, q))
    H = float(-(p * np.log(p)).sum())
    den = exact_denoiser(states, p, q)
    idx = rng.choice(len(states), size=4000, p=p)
    x0 = torch.as_tensor(states[idx])
    g = torch.Generator().manual_seed(1)
    est = np.mean([nelbo_per_sequence(den, x0, sample_times(len(x0), generator=g), q, g).mean().item()
                   for _ in range(40)])
    assert abs(est - H) < 0.03 * H, (est, H)
    uniform = lambda xt, t: torch.zeros(xt.shape[0], L, q)             # noqa: E731
    est_u = nelbo_per_sequence(uniform, x0, sample_times(len(x0), generator=g), q, g).mean().item()
    assert abs(est_u - L * np.log(q)) < 0.1 and est_u > est


def test_ancestral_sampler_with_exact_denoiser(rng, outdir):
    """Many steps -> sites revealed one at a time -> exact samples. One step -> all sites revealed together from
    their marginals -> the product of marginals (independent sites)."""
    L, q = 4, 3
    states, p = exact_distribution(tiny_potts(rng, L, q))
    den = exact_denoiser(states, p, q)
    marg = np.stack([[p[states[:, i] == a].sum() for a in range(q)] for i in range(L)])
    p_indep = np.prod([marg[i, states[:, i]] for i in range(L)], axis=0)
    tv_indep = 0.5 * np.abs(p_indep - p).sum()

    steps, tvs = [1, 2, 4, 8, 32, 128], []
    for T in steps:
        S = ancestral_sample(den, 20000, L, q, T, seed=T).numpy()
        emp = np.bincount(state_index(S, q), minlength=q ** L) / len(S)
        tvs.append(0.5 * np.abs(emp - p).sum())
    assert abs(tvs[0] - tv_indep) < 0.03          # T = 1: independent sites
    assert tvs[-1] < 0.03                          # T = 128: exact up to sampling noise
    assert tvs[-1] < tvs[0] / 3

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(steps, tvs, marker="o", color=SERIES[0], label="ancestral sampler, exact denoiser")
    ax.axhline(tv_indep, color=SERIES[1], ls="--", lw=1.5, label="product of marginals")
    ax.axhline(0.5 * np.sqrt(len(p) / 20000), color=INK_2, ls=":", lw=1, label="sampling-noise scale")
    ax.set_xscale("log")
    ax.set(xlabel="number of reverse steps T", ylabel="total variation to exact P",
           title=f"L={L}, q={q}: sampler error vs steps")
    ax.legend(fontsize=7)
    save(fig, outdir / "tv_vs_steps.png")


def test_site_conditionals_with_exact_denoiser(rng):
    """Masking one site and reading the exact denoiser gives exactly the Potts conditional P(s_i | s_-i)."""
    from psg.models.masked_diffusion import site_conditionals

    L, q = 4, 3
    m = tiny_potts(rng, L, q)
    states, p = exact_distribution(m)
    den = exact_denoiser(states, p, q)
    S = states[::5]
    P = site_conditionals(den, torch.as_tensor(S), q, batch_size=8).numpy()
    for i in range(L):
        np.testing.assert_allclose(P[:, i], m.conditional(S, i), atol=1e-6)


def test_continuous_time_sampler_is_exact(rng):
    """The one-site-at-a-time sampler (T -> infinity limit) with the exact denoiser reproduces P exactly."""
    from psg.models.masked_diffusion import continuous_time_sample

    L, q = 4, 3
    states, p = exact_distribution(tiny_potts(rng, L, q))
    den = exact_denoiser(states, p, q)
    S = continuous_time_sample(den, 20000, L, q, seed=3).numpy()
    emp = np.bincount(state_index(S, q), minlength=q ** L) / len(S)
    assert 0.5 * np.abs(emp - p).sum() < 0.03
