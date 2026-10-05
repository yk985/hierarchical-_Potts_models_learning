"""Training data: top up each clade with extra Gibbs samples, split train / val / test per clade.

Labels are stored as (family, subfamily, clade, leaf), with subfamily / clade indices *within* their parent
(as in all_sequences.npz). `global_tags` converts them to the global indices used by the conditional model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hierarchy import Node
from .torch_sampling import gibbs_sample_torch


@dataclass
class LabelledSequences:
    S: np.ndarray          # (M, L) int
    family: np.ndarray     # (M,)
    subfamily: np.ndarray  # (M,) index within family
    clade: np.ndarray      # (M,) index within subfamily
    leaf: np.ndarray       # (M,) global clade index 0..n_leaves-1

    def __len__(self):
        return len(self.S)

    def subset(self, idx: np.ndarray) -> "LabelledSequences":
        return LabelledSequences(self.S[idx], self.family[idx], self.subfamily[idx], self.clade[idx], self.leaf[idx])

    def save(self, path: str | Path, **extra) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, S=self.S, family=self.family, subfamily=self.subfamily, clade=self.clade,
                            leaf=self.leaf, **extra)

    @classmethod
    def load(cls, path: str | Path) -> "LabelledSequences":
        d = np.load(path)
        return cls(d["S"], d["family"], d["subfamily"], d["clade"], d["leaf"])


def concat(parts: list[LabelledSequences]) -> LabelledSequences:
    return LabelledSequences(*[np.concatenate([getattr(p, k) for p in parts])
                               for k in ["S", "family", "subfamily", "clade", "leaf"]])


def top_up(data: LabelledSequences, leaves: list[Node], n_per_clade: int, n_burnin: int, seed: int,
           verbose: bool = False) -> LabelledSequences:
    """Add independent Gibbs samples (GPU sampler, new seed per clade) so every clade has n_per_clade sequences.
    Existing sequences are kept first."""
    parts = []
    for k, node in enumerate(leaves):
        have = data.subset(np.flatnonzero(data.leaf == k))
        parts.append(have)
        n_new = n_per_clade - len(have)
        if n_new > 0:
            S = gibbs_sample_torch(node.model, n_new, n_burnin, seed=seed + k, energy_every=0).samples
            f, s, c = node.path
            parts.append(LabelledSequences(S, np.full(n_new, f), np.full(n_new, s), np.full(n_new, c), np.full(n_new, k)))
            if verbose:
                print(f"{node.name}: +{n_new} sequences", flush=True)
    return concat(parts)


def split_per_clade(leaf: np.ndarray, fractions=(0.8, 0.1, 0.1), seed: int = 0) -> dict[str, np.ndarray]:
    """Stratified random split: each clade is split with the same fractions. Returns index arrays."""
    assert abs(sum(fractions) - 1) < 1e-9
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for k in np.unique(leaf):
        idx = rng.permutation(np.flatnonzero(leaf == k))
        n_tr = int(round(fractions[0] * len(idx)))
        n_va = int(round(fractions[1] * len(idx)))
        out["train"].append(idx[:n_tr])
        out["val"].append(idx[n_tr:n_tr + n_va])
        out["test"].append(idx[n_tr + n_va:])
    return {k: np.sort(np.concatenate(v)) for k, v in out.items()}


def global_tags(data: LabelledSequences, n_subfamilies: int) -> np.ndarray:
    """(M, 3) global indices [family, family*nS + subfamily, leaf] used by the conditional model."""
    return np.stack([data.family, data.family * n_subfamilies + data.subfamily, data.leaf], axis=1)
