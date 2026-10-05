"""Step 7 - training data: top-up with extra samples, stratified train/val/test split, global tags."""
import numpy as np

from psg.data import LabelledSequences, global_tags, split_per_clade, top_up
from psg.hierarchy import HierarchyConfig, build_hierarchy, leaves

TINY = HierarchyConfig(L=8, q=4, n_families=2, n_subfamilies=2, n_clades=2, n_long_contacts=4, seed=0)


def fake_data(n_per_leaf, lv):
    parts = []
    for k, n in enumerate(lv):
        f, s, c = n.path
        parts.append((np.zeros((n_per_leaf, TINY.L), int), f, s, c, k))
    return LabelledSequences(np.concatenate([p[0] for p in parts]),
                             np.repeat([p[1] for p in parts], n_per_leaf), np.repeat([p[2] for p in parts], n_per_leaf),
                             np.repeat([p[3] for p in parts], n_per_leaf), np.repeat([p[4] for p in parts], n_per_leaf))


def test_top_up_counts_and_labels():
    nodes, _ = build_hierarchy(TINY)
    lv = leaves(nodes)
    d = fake_data(30, lv)                      # 30 existing (all-zero) sequences per clade
    full = top_up(d, lv, n_per_clade=100, n_burnin=20, seed=0)
    assert len(full) == 8 * 100
    assert (np.bincount(full.leaf) == 100).all()
    for k, n in enumerate(lv):                 # labels of every sequence agree with its clade's path
        sel = full.leaf == k
        assert (full.family[sel] == n.path[0]).all() and (full.subfamily[sel] == n.path[1]).all()
        assert (full.clade[sel] == n.path[2]).all()
    first = full.S[full.leaf == 0][:30]        # existing sequences are kept, new ones are real samples
    assert (first == 0).all() and full.S[full.leaf == 0][30:].any()


def test_split_is_disjoint_and_stratified():
    leaf = np.repeat(np.arange(5), 200)
    sp = split_per_clade(leaf, (0.8, 0.1, 0.1), seed=0)
    allidx = np.concatenate(list(sp.values()))
    assert len(allidx) == 1000 and len(np.unique(allidx)) == 1000          # disjoint and complete
    for name, frac in [("train", 0.8), ("val", 0.1), ("test", 0.1)]:
        assert (np.bincount(leaf[sp[name]], minlength=5) == int(frac * 200)).all()
    assert (split_per_clade(leaf, seed=0)["test"] == sp["test"]).all()      # reproducible


def test_global_tags():
    nodes, _ = build_hierarchy(TINY)
    d = fake_data(1, leaves(nodes))
    T = global_tags(d, n_subfamilies=2)
    # leaves in order F0.S0.C0, F0.S0.C1, F0.S1.C0, ... -> subfamily global ids 0,0,1,1,2,2,3,3
    np.testing.assert_array_equal(T[:, 0], [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(T[:, 1], [0, 0, 1, 1, 2, 2, 3, 3])
    np.testing.assert_array_equal(T[:, 2], np.arange(8))


def test_save_load(tmp_path):
    nodes, _ = build_hierarchy(TINY)
    d = fake_data(3, leaves(nodes))
    d.save(tmp_path / "x.npz")
    e = LabelledSequences.load(tmp_path / "x.npz")
    np.testing.assert_array_equal(d.leaf, e.leaf)
