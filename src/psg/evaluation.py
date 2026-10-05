"""Evaluation of a learned Potts model against the ground truth.

Two families of checks:
  * parameters : is the learned J close to the true J?  (both in the zero-sum gauge)
                 does the learned J find the true contacts?  (top-scoring pairs vs contact map)
  * samples    : do sequences Gibbs-sampled from the learned model reproduce the statistics of the held-out
                 test sequences?  (psg.stats.compare_statistics + sequence identity)
"""
from __future__ import annotations

import numpy as np

from .potts import Potts
from .stats import compare_statistics, mean_pairwise_identity, pearson


def contact_scores(model: Potts, apc: bool = True) -> np.ndarray:
    """F_ij = ||J^0_ij||_F (zero-sum gauge), optionally with the average-product correction
    F_ij - F_i. F_.j / F_.. that removes site-wise background (standard in DCA contact prediction)."""
    J0 = model.zero_sum_gauge().J
    F = np.sqrt((J0 ** 2).sum(axis=(2, 3)))
    if apc:
        L = F.shape[0]
        off = ~np.eye(L, dtype=bool)
        Fi = (F * off).sum(1) / (L - 1)
        Fm = F[off].mean()
        F = F - np.outer(Fi, Fi) / Fm
    np.fill_diagonal(F, -np.inf)
    return F


def contact_precision(scores: np.ndarray, contacts: np.ndarray, min_sep: int = 4, k: int | None = None) -> float:
    """Fraction of true contacts among the k highest-scoring pairs with |i-j| >= min_sep.
    k defaults to the number of true contacts at that separation."""
    L = scores.shape[0]
    i, j = np.triu_indices(L, k=min_sep)
    true = contacts[i, j]
    k = int(true.sum()) if k is None else k
    top = np.argsort(-scores[i, j])[:k]
    return float(true[top].mean())


def coupling_recovery(learned: Potts, true: Potts, contacts: np.ndarray | None = None) -> dict:
    """Pearson correlations between learned and true parameters, both in the zero-sum gauge."""
    a, b = learned.zero_sum_gauge(), true.zero_sum_gauge()
    iu = np.triu_indices(a.L, k=1)
    out = {"pearson_J_all_pairs": pearson(a.J[iu].ravel(), b.J[iu].ravel()),
           "pearson_h": pearson(a.h.ravel(), b.h.ravel()),
           "rel_error_J": float(np.linalg.norm(a.J[iu] - b.J[iu]) / np.linalg.norm(b.J[iu]))}
    if contacts is not None:
        c = contacts[iu]
        x, y = b.J[iu][c].ravel(), a.J[iu][c].ravel()
        out["pearson_J_contacts"] = pearson(y, x)
        out["slope_J_contacts"] = float((x * y).sum() / (x * x).sum())     # learned vs true, through origin
        out["mean_abs_J_non_contacts"] = float(np.abs(a.J[iu][~c]).mean())
    return out


def compare_to_nodes(learned: Potts, nodes: dict, contacts: np.ndarray) -> dict:
    """coupling_recovery of `learned` against EVERY node of the ground-truth tree (root, families, subfamilies,
    clades), plus the Frobenius distances d_J, d_h (zero-sum gauge, as in Part I)."""
    from .hierarchy import coupling_distance, field_distance

    out = {}
    for name, node in nodes.items():
        r = coupling_recovery(learned, node.model, contacts)
        r["d_J"] = coupling_distance(learned, node.model)
        r["d_h"] = field_distance(learned, node.model)
        r["level"] = node.level
        r["family"] = node.path[0] if node.path else -1
        out[name] = r
    return out


def family_assignment(Z: np.ndarray, Z_ref: np.ndarray, family_ref: np.ndarray, quantile: float = 0.99) -> dict:
    """Assign sequences (PCA coordinates Z) to the nearest family centroid of a labelled reference set.
    A sequence farther from every centroid than `quantile` of the reference sequences are from their own
    centroid is counted as 'between families'. Returns the fractions."""
    fams = np.unique(family_ref)
    cent = np.stack([Z_ref[family_ref == f].mean(0) for f in fams])
    own = np.linalg.norm(Z_ref - cent[np.searchsorted(fams, family_ref)], axis=1)
    thr = np.quantile(own, quantile)
    d = np.linalg.norm(Z[:, None, :] - cent[None], axis=2)
    nearest, far = d.argmin(1), d.min(1) > thr
    out = {f"F{int(f)}": float(((nearest == k) & ~far).mean()) for k, f in enumerate(fams)}
    out["between"] = float(far.mean())
    return out


def compare_samples(generated: np.ndarray, reference: np.ndarray, q: int) -> dict:
    """compare_statistics + identities: within generated, within reference, generated vs reference, and the
    fraction of generated sequences that are exact copies of a reference sequence (memorisation check)."""
    out = compare_statistics(reference, generated, q)
    out["identity_gen_gen"] = mean_pairwise_identity(generated, max_pairs=50_000)
    out["identity_ref_ref"] = mean_pairwise_identity(reference, max_pairs=50_000)
    out["identity_gen_ref"] = mean_pairwise_identity(generated, reference, max_pairs=50_000)
    ref_set = {r.tobytes() for r in np.ascontiguousarray(reference)}
    out["fraction_copies_of_reference"] = float(np.mean([g.tobytes() in ref_set for g in np.ascontiguousarray(generated)]))
    return out


def pseudo_loglik_scores(S: np.ndarray, models: list[Potts]) -> np.ndarray:
    """score[m, c] = sum_i log P_c(s_i^m | s_-i^m): the pseudo-log-likelihood of each sequence under each model.
    It needs no partition function, so it can compare models with different Z."""
    out = np.empty((len(S), len(models)))
    for c, mdl in enumerate(models):
        nb = mdl.neighbors()
        tot = np.zeros(len(S))
        for i in range(S.shape[1]):
            f = mdl.local_field(S, i, nb[i])
            f = f - f.max(axis=1, keepdims=True)
            tot += f[np.arange(len(S)), S[:, i]] - np.log(np.exp(f).sum(axis=1))
        out[:, c] = tot
    return out


def pseudo_loglik_scores_torch(S: np.ndarray, models: list[Potts], device="cpu", batch_size: int = 16384) -> np.ndarray:
    """Same as pseudo_loglik_scores, on the GPU: every local field at once, H = h + onehot(S) @ J_flat^T with
    J_flat[(i,a), (j,b)] = J_ij(a, b)."""
    import torch

    M, L = S.shape
    q = models[0].q
    hs = [torch.as_tensor(m.h.reshape(-1), dtype=torch.float32, device=device) for m in models]
    Js = [torch.as_tensor(m.J.transpose(0, 2, 1, 3).reshape(L * q, L * q), dtype=torch.float32, device=device)
          for m in models]
    out = np.empty((M, len(models)))
    for k0 in range(0, M, batch_size):
        x = torch.as_tensor(S[k0:k0 + batch_size], dtype=torch.long, device=device)
        oh = torch.nn.functional.one_hot(x, q).float().reshape(len(x), L * q)
        for c in range(len(models)):
            logp = torch.log_softmax((hs[c] + oh @ Js[c].T).reshape(len(x), L, q), -1)
            out[k0:k0 + len(x), c] = logp.gather(2, x[..., None]).sum((1, 2)).double().cpu().numpy()
    return out


def classify_by_pseudo_likelihood(S: np.ndarray, models: list[Potts]) -> np.ndarray:
    """Index of the model with the highest pseudo-log-likelihood, for every sequence."""
    return pseudo_loglik_scores(S, models).argmax(axis=1)
