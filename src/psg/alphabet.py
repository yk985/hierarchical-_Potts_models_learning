"""Amino-acid alphabet, integer encoding and FASTA I/O.

A sequence of length L is stored as an integer array with values in {0, ..., q-1}.
An alignment of M sequences is an (M, L) integer array.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard residues, alphabetical by 1-letter code
Q = len(AMINO_ACIDS)
_AA_TO_INT = {aa: k for k, aa in enumerate(AMINO_ACIDS)}


def encode(seq: str) -> np.ndarray:
    """'ACD' -> array([0, 1, 2])."""
    return np.array([_AA_TO_INT[c] for c in seq.upper()], dtype=np.int64)


def decode(x: np.ndarray) -> str:
    """array([0, 1, 2]) -> 'ACD'."""
    return "".join(AMINO_ACIDS[int(k)] for k in x)


def one_hot(S: np.ndarray, q: int) -> np.ndarray:
    """(M, L) integers -> (M, L, q) float one-hot tensor."""
    S = np.asarray(S)
    return (S[..., None] == np.arange(q)).astype(np.float64)


def write_fasta(path: str | Path, S: np.ndarray, headers: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if headers is None:
        headers = [f"seq{m}" for m in range(len(S))]
    with open(path, "w") as f:
        for h, s in zip(headers, S):
            f.write(f">{h}\n{decode(s)}\n")


def read_fasta(path: str | Path) -> tuple[list[str], np.ndarray]:
    headers, seqs, cur = [], [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
            headers.append(line[1:])
        else:
            cur.append(line)
    if cur:
        seqs.append("".join(cur))
    return headers, np.stack([encode(s) for s in seqs])
