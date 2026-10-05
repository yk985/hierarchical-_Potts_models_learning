"""Step 3 - hierarchical tree of Potts models. Small tree (L=12, q=4, 2 x 2 x 2 leaves)."""
import numpy as np

from psg.hierarchy import (
    RELATION_NAMES,
    HierarchyConfig,
    build_hierarchy,
    common_ancestor_level,
    coupling_distance,
    distance_matrix,
    distances_by_relation,
    leaves,
    load_hierarchy,
    save_hierarchy,
)
from psg.plots import plot_contact_map, plot_distance_by_relation, plot_distance_matrix, save

SMALL = HierarchyConfig(L=12, q=4, n_families=2, n_subfamilies=2, n_clades=2, n_long_contacts=10, seed=1)


def test_tree_structure():
    nodes, C = build_hierarchy(SMALL)
    assert len(nodes) == 1 + 2 + 4 + 8
    assert [n.name for n in leaves(nodes)][:3] == ["F0.S0.C0", "F0.S0.C1", "F0.S1.C0"]
    assert nodes["F1.S0.C1"].parent == "F1.S0" and nodes["F1.S0"].parent == "F1"
    assert C.sum() // 2 == 11 + 10          # 11 backbone pairs + 10 long-range
    assert (C == C.T).all()


def test_models_are_valid_and_sparse():
    nodes, C = build_hierarchy(SMALL)
    for n in nodes.values():
        m = n.model
        m.check_symmetry()
        np.testing.assert_allclose(m.J.sum(axis=2), 0, atol=1e-10)   # zero-sum gauge
        # couplings only on contacts
        assert np.abs(m.J[~C]).max() == 0.0


def test_common_ancestor_level():
    assert common_ancestor_level((0, 1, 0), (0, 1, 1)) == 2
    assert common_ancestor_level((0, 1, 0), (0, 0, 0)) == 1
    assert common_ancestor_level((0, 1, 0), (1, 1, 0)) == 0
    assert common_ancestor_level((0, 1, 0), (0, 1, 0)) == 3


def test_distances_are_hierarchical(outdir):
    """Every same-subfamily pair is closer than every same-family pair, which is closer than every
    different-family pair (strict separation, not only on average)."""
    nodes, C = build_hierarchy(SMALL)
    lv = leaves(nodes)
    D = distance_matrix([n.model for n in lv], kind="J")
    g = distances_by_relation(lv, D)
    assert g[2].max() < g[1].min()
    assert g[1].max() < g[0].min()
    # a model is at distance 0 from itself, distances are symmetric
    assert coupling_distance(lv[0].model, lv[0].model) == 0.0
    np.testing.assert_allclose(D, D.T)

    names = [n.name for n in lv]
    save(plot_distance_matrix(D, names, "||J^a - J^b||_F between clades"), outdir / "J_distance_matrix.png")
    save(plot_distance_by_relation(g, RELATION_NAMES, "Coupling distance by relation"), outdir / "J_distance_by_relation.png")
    save(plot_contact_map(C), outdir / "contact_map.png")


def test_save_load_roundtrip(tmp_path):
    nodes, C = build_hierarchy(SMALL)
    save_hierarchy(nodes, C, SMALL, tmp_path)
    nodes2, C2, cfg2 = load_hierarchy(tmp_path)
    assert cfg2 == SMALL
    assert nodes2.keys() == nodes.keys()
    np.testing.assert_array_equal(nodes2["F1.S1.C0"].model.J, nodes["F1.S1.C0"].model.J)
    assert nodes2["F1.S1.C0"].path == (1, 1, 0)


def test_same_seed_same_tree():
    a, _ = build_hierarchy(SMALL)
    b, _ = build_hierarchy(SMALL)
    np.testing.assert_array_equal(a["F0.S1.C1"].model.h, b["F0.S1.C1"].model.h)


def test_distances_match_theory():
    """Measured mean distances per relation agree with the closed-form expectation (within 10%)."""
    from psg.hierarchy import expected_distances

    cfg = HierarchyConfig(L=30, q=6, n_families=3, n_subfamilies=2, n_clades=2, n_long_contacts=40, seed=3)
    nodes, C = build_hierarchy(cfg)
    lv = leaves(nodes)
    th = expected_distances(cfg, C)
    for kind in ["J", "h"]:
        g = distances_by_relation(lv, distance_matrix([n.model for n in lv], kind))
        for lca, v in g.items():
            rms = np.sqrt((v ** 2).mean())
            assert abs(rms / th[kind][lca] - 1) < 0.10, (kind, lca, rms, th[kind][lca])
