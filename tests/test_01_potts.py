"""Step 1 - Potts model core: energy, conditionals, gauge, I/O. All on systems small enough to check by hand."""
import numpy as np

from psg.alphabet import decode, encode, read_fasta, write_fasta
from psg.potts import Potts, enumerate_states, exact_distribution, symmetrize


def hand_model():
    """L=2, q=2. h_1=(0.5,-0.5), h_2=(1,0), J_12=[[1,0],[0,2]]."""
    h = np.array([[0.5, -0.5], [1.0, 0.0]])
    J = np.zeros((2, 2, 2, 2))
    J[0, 1] = [[1.0, 0.0], [0.0, 2.0]]
    J[1, 0] = J[0, 1].T
    return Potts(h, J)


def random_model(L, q, rng, scale=1.0):
    h = rng.normal(0, scale, (L, q))
    J = symmetrize(rng.normal(0, scale, (L, L, q, q)))
    return Potts(h, J)


def test_energy_by_hand():
    m = hand_model()
    m.check_symmetry()
    S = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # E = -h1(s1) - h2(s2) - J12(s1,s2)
    expected = np.array([-(0.5 + 1.0 + 1.0), -(0.5 + 0.0 + 0.0), -(-0.5 + 1.0 + 0.0), -(-0.5 + 0.0 + 2.0)])
    np.testing.assert_allclose(m.energy(S), expected)


def test_exact_distribution_by_hand():
    m = hand_model()
    states, p = exact_distribution(m)
    w = np.exp([2.5, 0.5, 0.5, 1.5])  # exp(-E) for (00, 01, 10, 11)
    np.testing.assert_allclose(p, w / w.sum())


def test_conditional_matches_enumeration(rng):
    """P(s_i=a | rest) from local fields == ratio of exact Boltzmann weights."""
    m = random_model(3, 3, rng)
    states, p = exact_distribution(m)
    lookup = {tuple(s): pk for s, pk in zip(states, p)}
    for s in states[::5]:
        for i in range(3):
            cond = m.conditional(s[None, :], i)[0]
            w = []
            for a in range(3):
                t = s.copy()
                t[i] = a
                w.append(lookup[tuple(t)])
            np.testing.assert_allclose(cond, np.array(w) / sum(w), rtol=1e-10)


def test_local_field_with_neighbor_list(rng):
    """Restricting the sum to non-zero coupling blocks gives the same field."""
    m = random_model(5, 4, rng)
    m.J[0, 3] = 0.0
    m.J[3, 0] = 0.0
    S = rng.integers(0, 4, (10, 5))
    nbrs = m.neighbors()
    assert 3 not in nbrs[0]
    for i in range(5):
        np.testing.assert_allclose(m.local_field(S, i), m.local_field(S, i, nbrs[i]))


def test_zero_sum_gauge_preserves_distribution(rng):
    m = random_model(4, 3, rng)
    g = m.zero_sum_gauge()
    S = enumerate_states(4, 3)
    dE = g.energy(S) - m.energy(S)
    np.testing.assert_allclose(dE, dE[0], atol=1e-10)  # energies shift by a constant -> same P(s)
    np.testing.assert_allclose(g.J.sum(axis=2), 0, atol=1e-10)
    np.testing.assert_allclose(g.J.sum(axis=3), 0, atol=1e-10)
    np.testing.assert_allclose(g.h.sum(axis=1), 0, atol=1e-10)
    g.check_symmetry()
    # the gauge is a projection: applying it twice changes nothing
    g2 = g.zero_sum_gauge()
    np.testing.assert_allclose(g2.J, g.J, atol=1e-12)
    np.testing.assert_allclose(g2.h, g.h, atol=1e-12)


def test_save_load_and_fasta(tmp_path, rng):
    m = random_model(3, 2, rng)
    m.save(tmp_path / "m.npz")
    m2 = Potts.load(tmp_path / "m.npz")
    np.testing.assert_array_equal(m.J, m2.J)
    assert decode(encode("ACDWY")) == "ACDWY"
    S = rng.integers(0, 20, (7, 12))
    write_fasta(tmp_path / "x.fasta", S)
    headers, S2 = read_fasta(tmp_path / "x.fasta")
    assert len(headers) == 7
    np.testing.assert_array_equal(S, S2)
