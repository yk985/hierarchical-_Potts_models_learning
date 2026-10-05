"""Step 2 - Gibbs sampler. Checked against exact enumeration on tiny systems."""
import numpy as np

from psg.plots import INK_2, SERIES, plt, save
from psg.potts import Potts, exact_distribution, softmax, symmetrize
from psg.sampling import gibbs_sample


def state_index(S, q):
    """Map each row (s_1..s_L) to its index in enumerate_states order (base-q number)."""
    L = S.shape[1]
    return S @ (q ** np.arange(L - 1, -1, -1))


def test_independent_sites(rng, outdir):
    """J = 0  ->  sites independent, P(s_i = a) = softmax(h_i)(a) exactly."""
    L, q = 4, 5
    h = rng.normal(0, 1.0, (L, q))
    m = Potts(h, np.zeros((L, L, q, q)))
    res = gibbs_sample(m, n_chains=20000, n_burnin=2, rng=rng)
    f = np.stack([np.bincount(res.samples[:, i], minlength=q) for i in range(L)]) / len(res.samples)
    p = softmax(h, axis=1)
    # binomial standard error at n=20000 is < 0.0036
    assert np.abs(f - p).max() < 0.015

    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, p.max()], [0, p.max()], color=INK_2, lw=1, ls="--", label="y = x")
    ax.scatter(p.ravel(), f.ravel(), s=36, color=SERIES[0], edgecolor="white", lw=1, label="site i, residue a")
    ax.set(xlabel="exact softmax(h_i)(a)", ylabel="Gibbs frequency", title="J = 0: Gibbs vs exact marginals")
    ax.legend()
    save(fig, outdir / "marginals_vs_exact.png")


def test_gibbs_matches_exact_distribution(rng, outdir):
    """Coupled L=3, q=3 model (27 states): empirical state histogram -> exact Boltzmann distribution."""
    L, q = 3, 3
    m = Potts(rng.normal(0, 0.8, (L, q)), symmetrize(rng.normal(0, 0.8, (L, L, q, q))))
    states, p_exact = exact_distribution(m)
    res = gibbs_sample(m, n_chains=5000, n_burnin=20, n_samples_per_chain=10, thin=3, rng=rng)
    counts = np.bincount(state_index(res.samples, q), minlength=q**L)
    p_emp = counts / counts.sum()
    tv = 0.5 * np.abs(p_emp - p_exact).sum()
    assert tv < 0.02, f"total variation {tv:.4f}"

    order = np.argsort(-p_exact)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(x - 0.2, p_exact[order], width=0.4, color=SERIES[0], label="exact (enumeration)")
    ax.bar(x + 0.2, p_emp[order], width=0.4, color=SERIES[1], label=f"Gibbs, {counts.sum()} samples")
    ax.set(xlabel="state (sorted by exact probability)", ylabel="P(s)",
           title=f"L=3, q=3 coupled Potts: total variation = {tv:.4f}")
    ax.set_xticks(x, ["".join(map(str, s)) for s in states[order]], rotation=90, fontsize=6)
    ax.legend()
    save(fig, outdir / "state_distribution.png")


def test_energy_trace_equilibrates(rng, outdir):
    """From random starts the mean energy decreases then plateaus (burn-in visible)."""
    L, q = 20, 4
    m = Potts(rng.normal(0, 0.5, (L, q)), symmetrize(rng.normal(0, 0.3, (L, L, q, q))))
    res = gibbs_sample(m, n_chains=200, n_burnin=60, rng=rng)
    e = res.energy_trace.mean(axis=1)
    assert e[-10:].mean() < e[0]                             # it relaxed from the random start
    assert abs(e[-10:].mean() - e[-30:-20].mean()) < 0.5     # and has stopped drifting

    fig, ax = plt.subplots(figsize=(5, 3))
    for c in range(5):
        ax.plot(res.energy_trace[:, c], color=SERIES[0], lw=0.8, alpha=0.35)
    ax.plot(e, color=SERIES[0], lw=2, label="mean over 200 chains")
    ax.set(xlabel="sweep", ylabel="energy E(s)", title="Energy vs sweeps (L=20, q=4)")
    ax.legend()
    save(fig, outdir / "energy_trace.png")
