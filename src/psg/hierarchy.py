"""Hierarchical family of Potts models: root -> families -> subfamilies -> clades.

    theta_child = theta_parent + delta,   delta ~ N(0, sigma_level^2) on the allowed entries,

with sigma decreasing with depth. Couplings only live on the (shared) contact map. Every node is
finally put in the zero-sum gauge, so that Frobenius distances between nodes are gauge-independent.
Leaves (clades) are the models we sample sequences from.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .potts import Potts, symmetrize

LEVELS = ["root", "family", "subfamily", "clade"]


@dataclass
class HierarchyConfig:
    L: int = 60
    q: int = 20
    n_families: int = 3
    n_subfamilies: int = 3          # per family
    n_clades: int = 2               # per subfamily
    sigma_h: tuple = (1.5, 0.9, 0.5, 0.3)    # std of field draw at root / family / subfamily / clade
    sigma_J: tuple = (0.8, 0.45, 0.27, 0.15)    # std of coupling draw on contacts, same levels
    n_long_contacts: int = 120      # random long-range contacts, |i-j| >= min_separation
    min_separation: int = 4
    seed: int = 0


@dataclass
class Node:
    name: str                     # "root", "F0", "F0.S1", "F0.S1.C0"
    level: int                    # 0 root, 1 family, 2 subfamily, 3 clade
    parent: str | None
    path: tuple = field(default=())   # (family, subfamily, clade) indices known at this level
    model: Potts | None = None


def make_contact_map(L: int, n_long: int, min_sep: int, rng: np.random.Generator) -> np.ndarray:
    """Symmetric boolean (L, L): backbone neighbours (i, i+1) + n_long random pairs with |i-j| >= min_sep."""
    C = np.zeros((L, L), dtype=bool)
    i = np.arange(L - 1)
    C[i, i + 1] = True
    cand = [(a, b) for a in range(L) for b in range(a + min_sep, L)]
    pick = rng.choice(len(cand), size=min(n_long, len(cand)), replace=False)
    for k in pick:
        a, b = cand[k]
        C[a, b] = True
    return C | C.T


def draw_couplings(contacts: np.ndarray, q: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian J blocks on the contact pairs, zero elsewhere, symmetric J[i,j,a,b] = J[j,i,b,a]."""
    L = contacts.shape[0]
    J = rng.normal(0.0, sigma, (L, L, q, q)) * contacts[:, :, None, None]
    return symmetrize(J)


def build_hierarchy(cfg: HierarchyConfig) -> tuple[dict[str, Node], np.ndarray]:
    """Returns ({name: Node}, contact_map). Nodes are in the zero-sum gauge."""
    rng = np.random.default_rng(cfg.seed)
    L, q = cfg.L, cfg.q
    contacts = make_contact_map(L, cfg.n_long_contacts, cfg.min_separation, rng)

    def child(parent: Potts, level: int) -> Potts:
        h = parent.h + rng.normal(0.0, cfg.sigma_h[level], (L, q))
        J = parent.J + draw_couplings(contacts, q, cfg.sigma_J[level], rng)
        return Potts(h, J)

    raw = {"root": Potts(rng.normal(0.0, cfg.sigma_h[0], (L, q)), draw_couplings(contacts, q, cfg.sigma_J[0], rng))}
    nodes = {"root": Node("root", 0, None, ())}
    for f in range(cfg.n_families):
        fn = f"F{f}"
        raw[fn] = child(raw["root"], 1)
        nodes[fn] = Node(fn, 1, "root", (f,))
        for s in range(cfg.n_subfamilies):
            sn = f"{fn}.S{s}"
            raw[sn] = child(raw[fn], 2)
            nodes[sn] = Node(sn, 2, fn, (f, s))
            for c in range(cfg.n_clades):
                cn = f"{sn}.C{c}"
                raw[cn] = child(raw[sn], 3)
                nodes[cn] = Node(cn, 3, sn, (f, s, c))
    for name, node in nodes.items():
        node.model = raw[name].zero_sum_gauge()
    return nodes, contacts


def leaves(nodes: dict[str, Node]) -> list[Node]:
    return [n for n in nodes.values() if n.level == 3]


# ---------------------------------------------------------------------- distances
def coupling_distance(a: Potts, b: Potts) -> float:
    """||J^a - J^b||_F over pairs i<j, in the zero-sum gauge."""
    D = a.zero_sum_gauge().J - b.zero_sum_gauge().J
    iu = np.triu_indices(a.L, k=1)
    return float(np.sqrt((D[iu] ** 2).sum()))


def field_distance(a: Potts, b: Potts) -> float:
    """||h^a - h^b||_F in the zero-sum gauge."""
    return float(np.linalg.norm(a.zero_sum_gauge().h - b.zero_sum_gauge().h))


def distance_matrix(models: list[Potts], kind: str = "J") -> np.ndarray:
    fn = coupling_distance if kind == "J" else field_distance
    n = len(models)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = fn(models[i], models[j])
    return D


def common_ancestor_level(path_a: tuple, path_b: tuple) -> int:
    """Depth of the lowest common ancestor: 0 root (different families), 1 same family,
    2 same subfamily, 3 same clade."""
    k = 0
    for x, y in zip(path_a, path_b):
        if x != y:
            break
        k += 1
    return k


def expected_distances(cfg: HierarchyConfig, contacts: np.ndarray) -> dict[str, dict[int, float]]:
    """Theoretical root-mean-square distance between two leaves whose common ancestor is at level `lca`
    (derivation in docs/modelization.pdf):

        E||dJ||^2 = 2 * sum_{l > lca} sigma_J[l]^2 * n_contacts * (q-1)^2
        E||dh||^2 = 2 * sum_{l > lca} (q-1) * (L sigma_h[l]^2 + 2 n_contacts sigma_J[l]^2 / q)
    """
    q, L = cfg.q, cfg.L
    nc = int(np.triu(contacts, 1).sum())
    out: dict[str, dict[int, float]] = {"J": {}, "h": {}}
    for lca in range(3):
        below = range(lca + 1, 4)
        out["J"][lca] = float(np.sqrt(2 * sum(cfg.sigma_J[l] ** 2 for l in below) * nc * (q - 1) ** 2))
        out["h"][lca] = float(np.sqrt(2 * sum((q - 1) * (L * cfg.sigma_h[l] ** 2 + 2 * nc * cfg.sigma_J[l] ** 2 / q)
                                              for l in below)))
    return out


RELATION_NAMES = {0: "different\nfamily", 1: "same family,\ndiff. subfamily", 2: "same subfamily,\ndiff. clade"}


def distances_by_relation(leaf_nodes: list[Node], D: np.ndarray) -> dict[int, np.ndarray]:
    """Group the upper-triangular entries of D by the level of the leaves' common ancestor."""
    out: dict[int, list] = {}
    for i in range(len(leaf_nodes)):
        for j in range(i + 1, len(leaf_nodes)):
            lvl = common_ancestor_level(leaf_nodes[i].path, leaf_nodes[j].path)
            out.setdefault(lvl, []).append(D[i, j])
    return {k: np.array(v) for k, v in sorted(out.items())}


# ---------------------------------------------------------------------- I/O
def save_hierarchy(nodes: dict[str, Node], contacts: np.ndarray, cfg: HierarchyConfig, outdir: str | Path) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, n in nodes.items():
        n.model.save(outdir / f"{name}.npz")
    np.save(outdir / "contact_map.npy", contacts)
    tree = {name: {"level": n.level, "parent": n.parent, "path": list(n.path)} for name, n in nodes.items()}
    (outdir / "tree.json").write_text(json.dumps({"config": asdict(cfg), "nodes": tree}, indent=2))


def load_hierarchy(outdir: str | Path) -> tuple[dict[str, Node], np.ndarray, HierarchyConfig]:
    outdir = Path(outdir)
    meta = json.loads((outdir / "tree.json").read_text())
    cfg = HierarchyConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in meta["config"].items()})
    nodes = {
        name: Node(name, d["level"], d["parent"], tuple(d["path"]), Potts.load(outdir / f"{name}.npz"))
        for name, d in meta["nodes"].items()
    }
    return nodes, np.load(outdir / "contact_map.npy"), cfg
