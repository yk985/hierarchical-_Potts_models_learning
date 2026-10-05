"""Factored-attention Potts model (one attention layer whose attention depends only on positions).

For each head h = 1..H:
    Q^h, K^h in R^{L x d}   learned per position (they do not see the residues)
    A^h_ij = softmax_j( Q^h_i . K^h_j / sqrt(d) ),  with j = i masked out     (each row sums to 1)
    V^h    in R^{q x q}     residue-residue "value" matrix of the head

    J~_ij(a, b) = sum_h A^h_ij V^h(a, b)
    J_ij(a, b)  = ( J~_ij(a, b) + J~_ji(b, a) ) / 2          (symmetric, J_ii = 0)

Because A does not depend on the sequence, (h, J) is an ordinary Potts model: the network *is* the tensor J.
It is trained with the pseudo-likelihood:

    logits_i(a | s) = h_i(a) + sum_{j != i} J_ij(a, s_j)          ( = local field of psg.potts )
    loss            = (1/M) sum_m sum_i  -log softmax(logits_i)(s_i^m)  +  lam_h ||h||^2  +  lam_J ||J||^2

The conditional version (hierarchical tags family / subfamily / clade) is ConditionalFactoredAttentionPotts.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..potts import Potts


def assemble_couplings(A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """A (H, L, L), V (H, q, q) -> symmetric J (L, L, q, q)."""
    Jt = torch.einsum("hij,hab->ijab", A, V)
    return 0.5 * (Jt + Jt.permute(1, 0, 3, 2))


def coupling_norm2(A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """||J||^2 (sum over all i, j, a, b) of the symmetric J built from (A, V), without building J.

    With J = (J~ + J~^T)/2 :  ||J||^2 = 1/2 ||J~||^2 + 1/2 <J~, J~^T>, and both terms are traces of
    products of small H x H Gram matrices:
        ||J~||^2      = sum_{h,h'} [sum_ij A^h_ij A^h'_ij] [sum_ab V^h_ab V^h'_ab]
        <J~, J~^T>    = sum_{h,h'} [sum_ij A^h_ij A^h'_ji] [sum_ab V^h_ab V^h'_ba]
    V may carry leading batch dimensions (..., H, q, q); the result then has shape (...).
    """
    H = A.shape[0]
    Af = A.reshape(H, -1)
    G_A = Af @ Af.T
    G_At = Af @ A.transpose(1, 2).reshape(H, -1).T
    Vf = V.reshape(*V.shape[:-2], -1)
    G_V = Vf @ Vf.transpose(-1, -2)
    G_Vt = Vf @ V.transpose(-1, -2).reshape(*V.shape[:-2], -1).transpose(-1, -2)
    return 0.5 * (G_A * G_V).sum(dim=(-1, -2)) + 0.5 * (G_At * G_Vt).sum(dim=(-1, -2))


def potts_logits(S: torch.Tensor, h: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """S (M, L) long, h (L, q), J (L, L, q, q) -> logits (M, L, q) with logits[m,i,a] = h_i(a) + sum_j J_ij(a, s_j)."""
    L, q = h.shape
    X = nn.functional.one_hot(S, q).to(h.dtype).reshape(S.shape[0], L * q)
    Jmat = J.permute(0, 2, 1, 3).reshape(L * q, L * q)          # row (i, a), column (j, b)
    return (X @ Jmat.T).reshape(-1, L, q) + h


def pseudo_nll(logits: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Per-sequence negative log pseudo-likelihood  sum_i -log P(s_i | s_-i).  Returns (M,)."""
    return nn.functional.cross_entropy(logits.transpose(1, 2), S, reduction="none").sum(dim=1)


class FactoredAttentionPotts(nn.Module):
    def __init__(self, L: int, q: int, n_heads: int = 256, d_k: int = 32, init_v: float = 0.01, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.config = dict(L=L, q=q, n_heads=n_heads, d_k=d_k, init_v=init_v, seed=seed)
        self.L, self.q, self.n_heads, self.d_k = L, q, n_heads, d_k
        self.Q = nn.Parameter(torch.randn(n_heads, L, d_k, generator=g))
        self.K = nn.Parameter(torch.randn(n_heads, L, d_k, generator=g))
        self.V = nn.Parameter(init_v * torch.randn(n_heads, q, q, generator=g))
        self.h = nn.Parameter(torch.zeros(L, q))
        self.register_buffer("diag_mask", torch.eye(L, dtype=torch.bool))

    # ---------------------------------------------------------------- the Potts parameters
    def attention(self) -> torch.Tensor:
        """A (H, L, L): row-softmax of Q K^T / sqrt(d), self-attention masked (A_ii = 0)."""
        s = torch.einsum("hid,hjd->hij", self.Q, self.K) / math.sqrt(self.d_k)
        return torch.softmax(s.masked_fill(self.diag_mask, float("-inf")), dim=-1)

    def couplings(self) -> torch.Tensor:
        return assemble_couplings(self.attention(), self.V)

    def fields(self) -> torch.Tensor:
        return self.h

    # ---------------------------------------------------------------- training objective
    def logits(self, S: torch.Tensor) -> torch.Tensor:
        return potts_logits(S, self.fields(), self.couplings())

    def regularization(self, lam_h: float, lam_J: float) -> torch.Tensor:
        return lam_h * (self.h ** 2).sum() + lam_J * coupling_norm2(self.attention(), self.V)

    def loss(self, S: torch.Tensor, lam_h: float = 0.01, lam_J: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (total loss, mean pseudo-NLL per sequence)."""
        pnll = pseudo_nll(self.logits(S), S).mean()
        return pnll + self.regularization(lam_h, lam_J), pnll

    # ---------------------------------------------------------------- export
    @torch.no_grad()
    def to_potts(self) -> Potts:
        return Potts(self.fields().double().cpu().numpy(), self.couplings().double().cpu().numpy())

    def n_parameters(self) -> dict:
        return {n: p.numel() for n, p in self.named_parameters()} | {"total": sum(p.numel() for p in self.parameters())}

    def describe(self) -> str:
        L, q, H, d = self.L, self.q, self.n_heads, self.d_k
        n = self.n_parameters()
        return "\n".join([
            f"{type(self).__name__}  (L={L}, q={q}, heads H={H}, key dim d={d})",
            f"  input   : S (batch, L) integers in [0, q)  -> one-hot (batch, L*q)",
            f"  Q, K    : (H, L, d) = {tuple(self.Q.shape)}   positional, residue-independent   [{n['Q'] + n['K']:,} params]",
            f"  A       : softmax_j(Q K^T / sqrt(d)), A_ii = 0  -> (H, L, L)",
            f"  V       : (H, q, q) = {tuple(self.V.shape)}   [{n['V']:,} params]",
            f"  h       : (L, q) = {tuple(self.h.shape)}   [{n['h']:,} params]",
            f"  J       : sym( sum_h A^h (x) V^h ) -> (L, L, q, q) = ({L}, {L}, {q}, {q})",
            f"  output  : logits (batch, L, q),  logits_i(a) = h_i(a) + sum_j J_ij(a, s_j)",
            f"  loss    : pseudo-NLL  sum_i -log softmax(logits_i)(s_i)  + lam_h||h||^2 + lam_J||J||^2",
            f"  total trainable parameters: {n['total']:,}",
        ])


class ConditionalFactoredAttentionPotts(FactoredAttentionPotts):
    """Same model, with hierarchical label tags (family, subfamily, clade).

    tags: (M, 3) long tensor of GLOBAL indices [family, subfamily, clade]; -1 means "unspecified".
    For a tag c:
        h(c) = h + dh_fam[f] + dh_sub[s] + dh_clade[c]          (terms with tag -1 are omitted)
        V(c) = V + dV_fam[f] + dV_sub[s] + dV_clade[c]
        J(c) = sym( sum_h A^h (x) V^h(c) )                        (attention A is shared by all labels)
    The attention (Q, K) is shared: it encodes WHICH pairs interact (the common fold); the labels change
    HOW they interact (values V) and the conservation (fields h). Tags (-1, -1, -1) give the base model.
    """

    def __init__(self, L: int, q: int, n_families: int, n_subfamilies: int, n_clades: int,
                 n_heads: int = 256, d_k: int = 32, init_v: float = 0.01, seed: int = 0):
        super().__init__(L, q, n_heads, d_k, init_v, seed)
        self.config.update(n_families=n_families, n_subfamilies=n_subfamilies, n_clades=n_clades)
        sizes = (n_families, n_subfamilies, n_clades)
        self.dh = nn.ParameterList([nn.Parameter(torch.zeros(n, L, q)) for n in sizes])
        self.dV = nn.ParameterList([nn.Parameter(torch.zeros(n, n_heads, q, q)) for n in sizes])

    def label_parameters(self, tag) -> tuple[torch.Tensor, torch.Tensor]:
        """tag: 3 ints (global indices, -1 = unspecified) -> (h(c) (L, q), V(c) (H, q, q))."""
        h, V = self.h, self.V
        for lvl, t in enumerate([int(x) for x in tag]):
            if t >= 0:
                h = h + self.dh[lvl][t]
                V = V + self.dV[lvl][t]
        return h, V

    def logits(self, S: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        A = self.attention()
        uniq, inv = torch.unique(tags, dim=0, return_inverse=True)
        out = torch.empty(S.shape[0], self.L, self.q, device=S.device, dtype=self.h.dtype)
        for u in range(len(uniq)):
            idx = torch.nonzero(inv == u).squeeze(1)
            h, V = self.label_parameters(uniq[u].tolist())
            out[idx] = potts_logits(S[idx], h, assemble_couplings(A, V))
        return out

    def regularization(self, lam_h: float, lam_J: float, lam_label: float = 0.01) -> torch.Tensor:
        A = self.attention()
        reg = lam_h * (self.h ** 2).sum() + lam_J * coupling_norm2(A, self.V)
        for dh, dV in zip(self.dh, self.dV):
            reg = reg + lam_label * ((dh ** 2).sum() + coupling_norm2(A, dV).sum())
        return reg

    def loss(self, S, tags, lam_h: float = 0.01, lam_J: float = 0.01, lam_label: float = 0.01):
        pnll = pseudo_nll(self.logits(S, tags), S).mean()
        return pnll + self.regularization(lam_h, lam_J, lam_label), pnll

    @torch.no_grad()
    def to_potts(self, tag=(-1, -1, -1)) -> Potts:
        h, V = self.label_parameters(tag)
        return Potts(h.double().cpu().numpy(), assemble_couplings(self.attention(), V).double().cpu().numpy())

    def describe(self) -> str:
        nF, nS, nC = self.config["n_families"], self.config["n_subfamilies"], self.config["n_clades"]
        n_lab = sum(p.numel() for p in list(self.dh) + list(self.dV))
        base = super().describe().splitlines()
        return "\n".join(base[:-3] + [
            f"  tags    : (batch, 3) = [family, subfamily, clade] global ids, -1 = unspecified",
            f"  labels  : dh (L, q) and dV (H, q, q) per family ({nF}), subfamily ({nS}), clade ({nC})  [{n_lab:,} params]",
            f"  h(c), V(c) = base + sum over specified levels;  J(c) = sym( sum_h A^h (x) V^h(c) )",
            base[-3],
            base[-2] + " + lam_label (||dh||^2 + ||dJ||^2)",
            f"  total trainable parameters: {sum(p.numel() for p in self.parameters()):,}",
        ])


def drop_tags(tags: torch.Tensor, p_family: float, p_subfamily: float, p_clade: float,
              generator: torch.Generator | None = None) -> torch.Tensor:
    """Hierarchical tag dropout: dropping a level also drops every level below it (so a clade tag is never
    given without its family). Teaches the model the family-level and subfamily-level (and unconditioned) models."""
    M = tags.shape[0]
    r = torch.rand(M, 3, generator=generator, device=tags.device if generator is None else "cpu").to(tags.device)
    drop_f = r[:, 0] < p_family
    drop_s = drop_f | (r[:, 1] < p_subfamily)
    drop_c = drop_s | (r[:, 2] < p_clade)
    out = tags.clone()
    out[drop_f, 0] = -1
    out[drop_s, 1] = -1
    out[drop_c, 2] = -1
    return out


MODEL_CLASSES = {"FactoredAttentionPotts": FactoredAttentionPotts,
                 "ConditionalFactoredAttentionPotts": ConditionalFactoredAttentionPotts}


def save_model(model: FactoredAttentionPotts, path, **meta) -> None:
    torch.save({"class": type(model).__name__, "config": model.config, "state_dict": model.state_dict(),
                "meta": meta}, path)


def load_model(path, device="cpu") -> FactoredAttentionPotts:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = MODEL_CLASSES[ck["class"]](**ck["config"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device)
