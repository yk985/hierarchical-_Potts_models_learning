"""Masked (absorbing-state) discrete diffusion for sequences, MDLM-style (theory: docs/modelization.pdf).

Tokens 0..q-1 are amino acids, token q = [MASK] (the absorbing state).

Forward (corruption) process, linear schedule alpha_t = 1 - t, t in [0, 1]:
    each site is independently replaced by [MASK] with probability 1 - alpha_t = t, and then stays masked.

Denoiser: a bidirectional transformer that sees the partially masked sequence x_t and the time t and outputs,
for every site, a distribution over the q amino acids:  p_theta(x_0^i = a | x_t).
("SUBS" parametrisation: no probability on [MASK]; unmasked sites are copied unchanged.)

Loss: continuous-time negative ELBO (an upper bound on -log p_theta(x_0)):
    NELBO(x_0) = E_{t ~ U(0,1)} [ (1/t) * sum_{i : x_t^i = MASK} -log p_theta(x_0^i | x_t) ].

Sampler (ancestral): start from the fully masked sequence at t = 1, go down a grid t_0=1 > t_1 > ... > t_T = 0;
from t to s every still-masked site is revealed with probability (t - s)/t, its residue drawn from p_theta(. | x_t).
"""
from __future__ import annotations

import math

import torch
from torch import nn


# ---------------------------------------------------------------------------- forward process and loss
def mask_tokens(x0: torch.Tensor, t: torch.Tensor, mask_id: int, generator: torch.Generator | None = None):
    """Mask each site of x0 (B, L) independently with probability t (B,). Returns (x_t, is_masked).
    `generator` must live on x0.device."""
    u = torch.rand(x0.shape, generator=generator, device=x0.device)
    masked = u < t[:, None]
    return torch.where(masked, torch.full_like(x0, mask_id), x0), masked


def sample_times(batch: int, eps: float = 1e-3, generator: torch.Generator | None = None,
                 device: str | torch.device = "cpu") -> torch.Tensor:
    """Antithetic (stratified) times: one uniform offset shifted by k/B, then mapped to [eps, 1].
    Covers [0, 1] evenly within each batch and reduces the variance of the loss estimate."""
    u0 = torch.rand(1, generator=generator).item()
    t = (u0 + torch.arange(batch, dtype=torch.float32) / batch) % 1.0
    return (eps + (1 - eps) * t).to(device)


def nelbo_per_sequence(denoiser, x0: torch.Tensor, t: torch.Tensor, mask_id: int,
                       generator: torch.Generator | None = None) -> torch.Tensor:
    """One-sample Monte-Carlo estimate of the NELBO of each sequence (B,), in nats."""
    xt, masked = mask_tokens(x0, t, mask_id, generator)
    logits = denoiser(xt, t)                                          # (B, L, q)
    ce = nn.functional.cross_entropy(logits.transpose(1, 2), x0, reduction="none")
    return (ce * masked).sum(dim=1) / t


# ---------------------------------------------------------------------------- sampler
@torch.no_grad()
def ancestral_sample(denoiser, n: int, L: int, q: int, n_steps: int, seed: int = 0,
                     device: str | torch.device = "cpu", batch_size: int = 2048) -> torch.Tensor:
    """Generate n sequences (n, L) with `n_steps` reverse steps. `denoiser(x_t, t) -> logits (B, L, q)`."""
    generator = torch.Generator(device=device).manual_seed(seed)
    out = []
    ts = torch.linspace(1.0, 0.0, n_steps + 1)
    for k0 in range(0, n, batch_size):
        B = min(batch_size, n - k0)
        x = torch.full((B, L), q, dtype=torch.long, device=device)
        for k in range(n_steps):
            t, s = ts[k].item(), ts[k + 1].item()
            masked = x == q
            if not masked.any():
                break
            probs = torch.softmax(denoiser(x, torch.full((B,), t, device=device)).float(), dim=-1)
            draw = torch.multinomial(probs.reshape(-1, q), 1, generator=generator).reshape(B, L)
            reveal_p = 1.0 if s == 0.0 else (t - s) / t
            reveal = torch.rand((B, L), generator=generator, device=device) < reveal_p
            x = torch.where(masked & reveal, draw, x)
        out.append(x.cpu())
    return torch.cat(out)


# ---------------------------------------------------------------------------- the denoiser network
class TimeEmbedding(nn.Module):
    """t in [0,1] -> Fourier features [sin(2 pi 2^k t), cos(2 pi 2^k t)]_k -> MLP -> R^d."""

    def __init__(self, d_model: int, n_freq: int = 8):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq) * 2 * math.pi)
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        a = t[:, None] * self.freqs[None, :]
        return self.mlp(torch.cat([torch.sin(a), torch.cos(a)], dim=1))


class MaskedDiffusionTransformer(nn.Module):
    """Time-conditioned bidirectional transformer denoiser.
    input  : x_t (B, L) in {0..q-1, q=[MASK]},  t (B,)
    output : logits (B, L, q) of p_theta(x_0^i = a | x_t)."""

    def __init__(self, L: int, q: int, d_model: int = 256, n_layers: int = 6, n_heads: int = 8,
                 d_ff: int = 1024, dropout: float = 0.0, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.config = dict(L=L, q=q, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
                           dropout=dropout, seed=seed)
        self.L, self.q, self.mask_id = L, q, q
        self.tok = nn.Embedding(q + 1, d_model)
        self.pos = nn.Parameter(0.02 * torch.randn(L, d_model))
        self.time = TimeEmbedding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, q)

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.tok(xt) + self.pos[None] + self.time(t)[:, None, :]
        return self.head(self.norm(self.encoder(h)))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        c = self.config
        n = {k: sum(p.numel() for p in m.parameters()) for k, m in
             [("tok", self.tok), ("time", self.time), ("encoder", self.encoder), ("head", self.head), ("norm", self.norm)]}
        return "\n".join([
            f"MaskedDiffusionTransformer  (L={c['L']}, q={c['q']}, d_model={c['d_model']}, layers={c['n_layers']}, "
            f"heads={c['n_heads']}, d_ff={c['d_ff']})",
            f"  input   : x_t (batch, L) tokens in {{0..{c['q'] - 1}}} + [MASK]={c['q']};  t (batch,) in [0, 1]",
            f"  embed   : token emb ({c['q'] + 1} x {c['d_model']}) + position emb ({c['L']} x {c['d_model']}) "
            f"+ time emb (Fourier(t) -> MLP)   [{n['tok'] + c['L'] * c['d_model'] + n['time']:,} params]",
            f"  encoder : {c['n_layers']} x [LayerNorm -> multi-head self-attention ({c['n_heads']} heads) -> "
            f"LayerNorm -> MLP {c['d_model']}->{c['d_ff']}->{c['d_model']}]   [{n['encoder']:,} params]",
            f"  head    : LayerNorm -> Linear {c['d_model']} -> {c['q']}   [{n['head'] + n['norm']:,} params]",
            f"  output  : logits (batch, L, q) = log p_theta(x_0^i = a | x_t) up to a constant, for every site i",
            f"  loss    : NELBO = E_t [ (1/t) sum_(masked i) -log p_theta(x_0^i | x_t) ]   (MDLM, alpha_t = 1 - t)",
            f"  sampler : ancestral unmasking from t=1 (all [MASK]) to t=0",
            f"  total trainable parameters: {self.n_parameters():,}",
        ])


def save_diffusion(model: MaskedDiffusionTransformer, path, **meta) -> None:
    torch.save({"class": type(model).__name__, "config": model.config, "state_dict": model.state_dict(),
                "meta": meta}, path)


def load_diffusion(path, device="cpu") -> MaskedDiffusionTransformer:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = MaskedDiffusionTransformer(**ck["config"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device)


@torch.no_grad()
def site_conditionals(denoiser, S: torch.Tensor, q: int, t: float | None = None, batch_size: int = 1024) -> torch.Tensor:
    """P_theta(x^i = a | all other sites) for every sequence and site: mask site i only and read the denoiser's
    output at i. Returns (M, L, q) probabilities. t defaults to 1/L (the noise level at which one site is masked)."""
    M, L = S.shape
    t = 1.0 / L if t is None else t
    out = torch.empty(M, L, q)
    eye = torch.eye(L, dtype=torch.bool, device=S.device)
    for k in range(0, M, max(1, batch_size // L)):
        x = S[k:k + max(1, batch_size // L)]                          # (b, L)
        b = len(x)
        xm = x[:, None, :].expand(b, L, L).clone()                    # copy l masks site l
        xm[:, eye] = q
        logits = denoiser(xm.reshape(b * L, L), torch.full((b * L,), t, device=S.device))
        probs = torch.softmax(logits.float(), -1).reshape(b, L, L, q)
        out[k:k + b] = probs[:, torch.arange(L), torch.arange(L)].cpu()   # site l of copy l
    return out


@torch.no_grad()
def continuous_time_sample(denoiser, n: int, L: int, q: int, seed: int = 0, device: str | torch.device = "cpu",
                           batch_size: int = 2048) -> torch.Tensor:
    """Exact T -> infinity limit of the ancestral sampler, with only L network calls per sequence.

    In the forward process each site is masked at an independent Uniform(0,1) time. Running it backwards, the sites
    are revealed one at a time, in decreasing order of those times: a uniformly random order, the k-th reveal
    happening at the k-th largest of L uniform times. At each reveal the site is drawn from p_theta(. | x_t) at the
    current time t. (With T steps, several sites can be revealed in the same step and are then drawn independently;
    here that never happens.)"""
    generator = torch.Generator(device=device).manual_seed(seed)
    out = []
    for k0 in range(0, n, batch_size):
        B = min(batch_size, n - k0)
        x = torch.full((B, L), q, dtype=torch.long, device=device)
        times, order = torch.sort(torch.rand((B, L), generator=generator, device=device), dim=1, descending=True)
        rows = torch.arange(B, device=device)
        for k in range(L):
            t = times[:, k]
            probs = torch.softmax(denoiser(x, t).float(), dim=-1)                  # (B, L, q)
            site = order[:, k]
            draw = torch.multinomial(probs[rows, site], 1, generator=generator).squeeze(1)
            x[rows, site] = draw
        out.append(x.cpu())
    return torch.cat(out)
