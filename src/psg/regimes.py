"""Dynamical regimes of masked diffusion: speciation and collapse (theory: docs/modelization.pdf, Part IV).

After Biroli, Bonnaire, de Bortoli & Mezard, "Dynamical regimes of diffusion models" (Nat. Commun. 2024), transposed
from Gaussian noise to masking. Everything is written in terms of k = number of revealed sites (k = L at t = 0,
k = 0 at t = 1). Given k, the revealed set U is a uniformly random k-subset, so any observable f(x_t) averaged at
time t is sum_k Binom(k; L, 1 - t) f_k  (`to_time`).

Speciation (cloning experiment). From a partially revealed x_t, complete it C times independently ("clones"); give
every completion a class label. The commitment
    phi_k = P(two clones get the same label) = E[ sum_c P(c | x_U)^2 ]           (exact denoiser)
goes from sum_c pi_c^2 (nothing revealed) to 1 (everything revealed). Normalised:
    psi = (phi - phi_0) / (1 - phi_0),  speciation k_S where psi = 1/2.
Theory for two classes: if the log-likelihood ratio Lambda of the classes given x_t is Gaussian with mean mu and
variance 2 mu (exact for a Gaussian mixture under Gaussian noise, and for many weakly informative masked sites),
psi = E[tanh^2(Lambda / 2)]: a universal function of the class signal-to-noise mu (`gaussian_llr_commitment`).

Collapse (memorisation) with the EMPIRICAL denoiser (the exact optimum of the loss on n training sequences): its
posterior is uniform over the N training sequences that agree with x_t on the revealed sites, so two clones end at
the same training sequence with probability E[1/N]. For a trajectory aimed at training point a (planted),
N = 1 + Binomial(n - 1, P(a_U)) ~ 1 + Poisson(lambda), lambda = (n - 1) P(a_U), and
    E[1 / N | lambda] = (1 - exp(-lambda)) / lambda                                  (`poisson_commitment`).
Collapse when typical lambda ~ 1, i.e. when the entropy of the revealed residues reaches log n.
"""
from __future__ import annotations

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------- masks and time
def random_orders(B: int, L: int, generator: torch.Generator, device="cpu") -> torch.Tensor:
    """(B, L) rank of every site in a uniformly random reveal order: site i is revealed among the first k iff
    rank[b, i] < k."""
    return torch.argsort(torch.rand((B, L), generator=generator, device=device), dim=1).argsort(dim=1)


def reveal_first_k(x0: torch.Tensor, rank: torch.Tensor, k: int, mask_id: int) -> torch.Tensor:
    """x0 with every site whose rank >= k replaced by [MASK]."""
    return torch.where(rank < k, x0, torch.full_like(x0, mask_id))


def to_time(k_values, f_k, t_values, L: int) -> np.ndarray:
    """Convert an observable known at k revealed sites (interpolated linearly between the given k) into its average
    at masking time t: sum_k Binom(k; L, 1 - t) f(k)."""
    ks = np.arange(L + 1)
    f_all = np.interp(ks, np.asarray(k_values, float), np.asarray(f_k, float))
    return np.array([float((binomial_weights(L, 1.0 - t) * f_all).sum()) for t in np.atleast_1d(t_values)])


def binomial_weights(L: int, p: float) -> np.ndarray:
    """Binom(k; L, p) for k = 0..L."""
    if p <= 0.0 or p >= 1.0:
        w = np.zeros(L + 1)
        w[L if p >= 1.0 else 0] = 1.0
        return w
    ks = np.arange(L + 1)
    logw = np.array([math.lgamma(L + 1) - math.lgamma(k + 1) - math.lgamma(L - k + 1) for k in ks]) \
        + ks * math.log(p) + (L - ks) * math.log(1 - p)
    return np.exp(logw)


def crossing(x, y, level: float) -> float:
    """First x at which y crosses `level` (linear interpolation). NaN if it never does."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = np.sign(y - level)
    for i in range(len(x) - 1):
        if s[i] == 0:
            return float(x[i])
        if s[i] != s[i + 1]:
            return float(x[i] + (level - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return float(x[-1]) if s[-1] == 0 else float("nan")


# ---------------------------------------------------------------------------- cloning
@torch.no_grad()
def complete(denoiser, xt: torch.Tensor, n_clones: int, q: int, generator: torch.Generator,
             batch_size: int = 8192) -> torch.Tensor:
    """Complete every partially masked row of xt (B, L) n_clones times, independently. The masked sites are revealed
    one at a time in a uniformly random order, each drawn from denoiser(x, t) with t = current fraction masked
    (the exact reverse process for a t-independent denoiser). Returns (B, n_clones, L)."""
    B, L = xt.shape
    x_all = xt.repeat_interleave(n_clones, dim=0)
    out = []
    for k0 in range(0, len(x_all), batch_size):
        x = x_all[k0:k0 + batch_size].clone()
        b = len(x)
        rows = torch.arange(b, device=x.device)
        # random order among masked sites: masked sites get a random key in [0,1), revealed sites key 2
        key = torch.where(x == q, torch.rand(x.shape, generator=generator, device=x.device), torch.full(x.shape, 2.0, device=x.device))
        order = key.argsort(dim=1)
        n_masked = (x == q).sum(1)
        for j in range(int(n_masked.max().item()) if b else 0):
            active = j < n_masked
            t = (n_masked - j).float() / L
            probs = torch.softmax(denoiser(x, t).float(), dim=-1)
            site = order[:, j]
            draw = torch.multinomial(probs[rows, site], 1, generator=generator).squeeze(1)
            x[rows[active], site[active]] = draw[active]
        out.append(x)
    return torch.cat(out).reshape(B, n_clones, L)


def pair_agreement(labels: np.ndarray) -> np.ndarray:
    """labels (B, C): for every row, the fraction of the C(C-1)/2 pairs of clones with the same label."""
    labels = np.asarray(labels)
    B, C = labels.shape
    same = (labels[:, :, None] == labels[:, None, :]).sum((1, 2)) - C
    return same / (C * (C - 1))


def normalised_commitment(phi, phi0: float):
    """(phi - phi0) / (1 - phi0): 0 when nothing is decided, 1 when the class is fixed."""
    return (np.asarray(phi, float) - phi0) / (1.0 - phi0)


# ---------------------------------------------------------------------------- theory: class posterior
def profile_log_posterior(S: np.ndarray, revealed: np.ndarray, log_f: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    """Class posterior when every class is an independent-site (profile) model f_c,i(a):
    log P(c | x_U) = log pi_c + sum_{i in U} log f_c,i(x_i) - normaliser.
    S (B, L) ints, revealed (B, L) bool, log_f (C, L, q), log_prior (C,). Returns (B, C), normalised."""
    L = S.shape[1]
    ll = log_f[:, np.arange(L)[None, :], S]                     # (C, B, L)
    lp = (ll * revealed[None]).sum(-1).T + log_prior[None]      # (B, C)
    return lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)


def gaussian_llr_commitment(mu, n_nodes: int = 80) -> np.ndarray:
    """psi(mu) = E[tanh^2(Lambda/2)], Lambda ~ N(mu, 2 mu): the normalised two-class commitment when the class
    log-likelihood ratio is Gaussian with the 'consistent' variance 2 mu (Gauss-Hermite quadrature)."""
    z, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / w.sum()
    mu = np.atleast_1d(np.asarray(mu, float))
    lam = mu[:, None] + np.sqrt(2 * np.maximum(mu, 0))[:, None] * z[None, :]
    return (np.tanh(lam / 2) ** 2 * w[None]).sum(1)


def solve_mu_star(level: float = 0.5) -> float:
    """The class signal-to-noise mu* at which psi(mu*) = level (bisection)."""
    lo, hi = 0.0, 100.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gaussian_llr_commitment(mid)[0] < level else (lo, mid)
    return 0.5 * (lo + hi)


def sibling_kl(freqs: dict, parent_of: dict) -> float:
    """Mean over ordered pairs of sibling classes (same parent) of sum_i KL(f_c,i || f_c',i)."""
    kl = [float((freqs[a] * np.log(freqs[a] / freqs[b])).sum())
          for a in freqs for b in freqs if a != b and parent_of[a] == parent_of[b]]
    return float(np.mean(kl))


# ---------------------------------------------------------------------------- collapse: counting training matches
@torch.no_grad()
def consistent_counts(xt: torch.Tensor, train: torch.Tensor, mask_id: int, labels: torch.Tensor | None = None,
                      n_classes: int = 0, chunk: int = 16) -> torch.Tensor:
    """Number of training rows that agree with xt (B, L) on every revealed site. With `labels` (n,), returns the
    counts per class (B, n_classes) instead of (B,)."""
    out = []
    oh = None if labels is None else torch.nn.functional.one_hot(labels, n_classes).float()
    for k0 in range(0, len(xt), chunk):
        x = xt[k0:k0 + chunk]
        ok = ((train[None] == x[:, None]) | (x[:, None] == mask_id)).all(-1)      # (b, n)
        out.append(ok.sum(1) if oh is None else ok.float() @ oh)
    return torch.cat(out)


def poisson_commitment(lam) -> np.ndarray:
    """E[1 / (1 + N)] for N ~ Poisson(lam): (1 - exp(-lam)) / lam, = 1 at lam = 0."""
    lam = np.asarray(lam, float)
    small = lam < 1e-8
    return np.where(small, 1.0 - lam / 2, -np.expm1(-np.where(small, 1.0, lam)) / np.where(small, 1.0, lam))


def subset_agreement(D: np.ndarray, k: int, L: int) -> float:
    """Mean over pairs of P(a uniformly random k-subset of sites avoids all D mismatches) = C(L-D, k) / C(L, k):
    the collision probability of the revealed residues (annealed count / n)."""
    D = np.asarray(D)
    ok = D <= L - k
    lg = np.full(len(D), -np.inf)
    d = D[ok]
    lg[ok] = (np.array([math.lgamma(L - x + 1) - math.lgamma(L - x - k + 1) for x in d])
              - (math.lgamma(L + 1) - math.lgamma(L - k + 1)))
    return float(np.exp(lg).mean())


# ---------------------------------------------------------------------------- entropy of the revealed residues
@torch.no_grad()
def revealed_log_prob(denoiser, x0: torch.Tensor, rank: torch.Tensor, q: int) -> torch.Tensor:
    """Chain rule along the reveal order: returns (B, L+1), entry k = log P_theta(x_{U_k}) where U_k is the set of
    the first k revealed sites (entry 0 = 0). Uses L denoiser calls."""
    B, L = x0.shape
    rows = torch.arange(B, device=x0.device)
    order = rank.argsort(dim=1)
    x = torch.full_like(x0, q)
    out = torch.zeros(B, L + 1, device=x0.device)
    for j in range(L):
        logp = torch.log_softmax(denoiser(x, torch.full((B,), (L - j) / L, device=x0.device)).float(), -1)
        site = order[:, j]
        out[:, j + 1] = out[:, j] + logp[rows, site, x0[rows, site]]
        x[rows, site] = x0[rows, site]
    return out


# ---------------------------------------------------------------------------- exact class posteriors (needs Z)
def _flat(model, device):
    L, q = model.L, model.q
    h = torch.as_tensor(model.h.reshape(-1), dtype=torch.float32, device=device)
    J = torch.as_tensor(model.J.transpose(0, 2, 1, 3).reshape(L * q, L * q), dtype=torch.float32, device=device)
    return h, J


@torch.no_grad()
def thermodynamic_log_z(model, n_lambda: int = 41, n_chains: int = 2000, n_equil: int = 40, n_measure: int = 20,
                        n_burnin: int = 200, seed: int = 0, device="cpu") -> dict:
    """log Z of a Potts model by thermodynamic integration over the coupling strength lambda in [0, 1]:
        E_lambda = -sum_i h_i - lambda sum_{i<j} J_ij,   d log Z_lambda / d lambda = < sum_{i<j} J_ij >_lambda,
        log Z_0 = sum_i log sum_a exp h_i(a)   (independent sites, exact).
    Gibbs chains are warm-started from one lambda to the next and the integral uses Simpson's rule. The run is done
    upwards and downwards; half their difference is returned as an error estimate (hysteresis)."""
    L, q = model.L, model.q
    gen = torch.Generator(device=device).manual_seed(seed)
    h = torch.as_tensor(model.h, dtype=torch.float32, device=device)
    Jcol = torch.as_tensor(model.J, dtype=torch.float32, device=device).permute(0, 1, 3, 2).reshape(L, L * q, q)
    _, Jf = _flat(model, device)
    lams = np.linspace(0.0, 1.0, n_lambda)

    def sweep(X, lam):
        for i in range(L):
            field = h[i] + lam * (X.reshape(len(X), L * q) @ Jcol[i])
            new = torch.multinomial(torch.softmax(field, 1), 1, generator=gen).squeeze(1)
            X[:, i] = torch.nn.functional.one_hot(new, q).float()

    def pair_sum(X):
        Xf = X.reshape(len(X), L * q)
        return 0.5 * ((Xf @ Jf.T) * Xf).sum(1)

    def run(order):
        X = torch.nn.functional.one_hot(torch.randint(0, q, (n_chains, L), generator=gen, device=device), q).float()
        means = {}
        for j, lam in enumerate(order):
            for _ in range(n_burnin if j == 0 else n_equil):
                sweep(X, float(lam))
            acc = []
            for _ in range(n_measure):
                sweep(X, float(lam))
                acc.append(pair_sum(X).mean().item())
            means[float(lam)] = float(np.mean(acc))
        return np.array([means[float(lam)] for lam in lams])

    up, down = run(lams), run(lams[::-1])
    log_z0 = float(np.log(np.exp(model.h).sum(1)).sum())
    simpson = lambda y: float((lams[1] - lams[0]) / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum()))  # noqa: E731
    I_up, I_down = simpson(up), simpson(down)
    return {"log_z": log_z0 + 0.5 * (I_up + I_down), "error": 0.5 * abs(I_up - I_down), "log_z0": log_z0,
            "lambdas": lams.tolist(), "mean_pair_up": up.tolist(), "mean_pair_down": down.tolist()}


@torch.no_grad()
def exact_log_likelihoods(S: np.ndarray, models, log_z, device="cpu", batch_size: int = 16384) -> np.ndarray:
    """log P_c(x) = -E_c(x) - log Z_c for every sequence and model: (M, C)."""
    L, q = models[0].L, models[0].q
    flats = [_flat(m, device) for m in models]
    out = np.empty((len(S), len(models)))
    for k0 in range(0, len(S), batch_size):
        x = torch.as_tensor(S[k0:k0 + batch_size], dtype=torch.long, device=device)
        oh = torch.nn.functional.one_hot(x, q).float().reshape(len(x), L * q)
        for c, (h, J) in enumerate(flats):
            minus_e = oh @ h + 0.5 * ((oh @ J.T) * oh).sum(1)
            out[k0:k0 + len(x), c] = minus_e.double().cpu().numpy() - log_z[c]
    return out


def soft_pair_agreement(R: np.ndarray) -> np.ndarray:
    """R (B, C, K): class probabilities of each of C clones. For every row, the mean over pairs of clones a != b of
    sum_y R_a(y) R_b(y), an unbiased estimate of sum_y P(y | x_U)^2 with the exact denoiser."""
    s = R.sum(1)                                                        # (B, K)
    return ((s ** 2).sum(-1) - (R ** 2).sum((1, 2))) / (R.shape[1] * (R.shape[1] - 1))


# ---------------------------------------------------------------------------- correlation-weighted profile
def within_class_mutual_information(S: np.ndarray, labels: np.ndarray, q: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mutual information between every pair of sites inside each class, averaged over classes (weights = class
    sizes), minus the same quantity after shuffling every column independently (finite-sample bias).
    Returns (MI corrected (L, L), null MI (L, L))."""
    rng = np.random.default_rng(seed)
    L = S.shape[1]

    def mi(X):
        oh = np.eye(q, dtype=np.float32)[X].reshape(len(X), L * q)
        Fij = (oh.T @ oh / len(X)).astype(np.float64).reshape(L, q, L, q).transpose(0, 2, 1, 3)   # (L, L, q, q)
        fi = oh.mean(0, dtype=np.float64).reshape(L, q)
        ratio = Fij / (fi[:, None, :, None] * fi[None, :, None, :] + 1e-300)
        out = (Fij * np.log(np.where(Fij > 0, ratio, 1.0))).sum((2, 3))
        np.fill_diagonal(out, 0.0)
        return out

    tot, null, n = np.zeros((L, L)), np.zeros((L, L)), 0
    for c in np.unique(labels):
        X = S[labels == c]
        Xs = np.stack([rng.permutation(X[:, i]) for i in range(L)], 1)
        tot += len(X) * mi(X)
        null += len(X) * mi(Xs)
        n += len(X)
    return np.clip((tot - null) / n, 0.0, None), null / n


def information_correlation(mi: np.ndarray) -> np.ndarray:
    """Linfoot's informational coefficient of correlation, rho = sqrt(1 - exp(-2 MI)) in [0, 1): equals |r| for
    Gaussian variables, 0 for independent sites, -> 1 for copies."""
    return np.sqrt(1.0 - np.exp(-2.0 * mi))


def correlation_weights(revealed: np.ndarray, rho: np.ndarray, partners: np.ndarray, mode: str) -> np.ndarray:
    """Weight w_i of the evidence of every revealed site, given the other revealed sites j (revealed (B, L) bool,
    rho (L, L), partners (L, L) bool = pairs counted as correlated):
      'geo'  : w_i = [ prod_{j in U, j != i, partner} (1 - rho_ij) ]^(1/n_i),  n_i = number of such partners
      'prod' : w_i =   prod_{j in U, j != i, partner} (1 - rho_ij)
      'eff'  : w_i = 1 / (1 + sum_{j in U, j != i, partner} rho_ij)          (effective number of sites)
      'none' : w_i = 1."""
    m = revealed.astype(float)
    A = partners & ~np.eye(len(rho), dtype=bool)
    if mode == "none":
        return np.ones_like(m)
    if mode == "eff":
        return 1.0 / (1.0 + m @ (A * rho).T)
    logs = m @ (A * np.log1p(-np.minimum(rho, 1 - 1e-9))).T                 # (B, L): sum over revealed partners
    if mode == "prod":
        return np.exp(logs)
    n = m @ A.T.astype(float)
    return np.exp(logs / np.maximum(n, 1.0))


def weighted_profile_log_posterior(S, revealed, log_f, log_prior, weights) -> np.ndarray:
    """profile_log_posterior with every revealed site's log-likelihood multiplied by its weight (B, L)."""
    L = S.shape[1]
    ll = log_f[:, np.arange(L)[None, :], S]                                   # (C, B, L)
    lp = (ll * (revealed * weights)[None]).sum(-1).T + log_prior[None]
    return lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)
