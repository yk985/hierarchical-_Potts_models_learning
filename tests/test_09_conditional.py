"""Step 9 - conditional factored-attention model with hierarchical tags."""
import numpy as np
import torch

from psg.models.factored_attention import ConditionalFactoredAttentionPotts, drop_tags, load_model, save_model
from psg.plots import SERIES, plt, save
from psg.potts import Potts
from psg.sampling import gibbs_sample
from psg.stats import site_frequencies
from psg.training import TrainConfig, train

CPU = torch.device("cpu")


def small_model(**kw):
    m = ConditionalFactoredAttentionPotts(L=5, q=3, n_families=2, n_subfamilies=3, n_clades=4, n_heads=2, d_k=3, **kw)
    with torch.no_grad():
        for p in list(m.dh) + list(m.dV):
            p.normal_()
    return m


def test_label_parameters_compose():
    m = small_model()
    h, V = m.label_parameters((1, 2, 3))
    torch.testing.assert_close(h, m.h + m.dh[0][1] + m.dh[1][2] + m.dh[2][3])
    torch.testing.assert_close(V, m.V + m.dV[0][1] + m.dV[1][2] + m.dV[2][3])
    h, V = m.label_parameters((1, -1, -1))               # family only
    torch.testing.assert_close(h, m.h + m.dh[0][1])
    h, V = m.label_parameters((-1, -1, -1))              # unconditioned = base model
    torch.testing.assert_close(V, m.V)


def test_mixed_batch_logits_equal_per_label(rng):
    m = small_model()
    S = torch.as_tensor(rng.integers(0, 3, (12, 5)))
    tags = torch.tensor([[0, 0, 0], [1, 2, 3], [0, -1, -1], [-1, -1, -1]] * 3)
    out = m.logits(S, tags)
    for k in range(12):
        P = m.to_potts(tags[k].tolist())
        for i in range(5):
            np.testing.assert_allclose(out[k, i].detach().numpy(), P.local_field(S[k:k + 1].numpy(), i)[0], atol=1e-5)


def test_drop_tags_is_hierarchical():
    g = torch.Generator().manual_seed(0)
    tags = torch.tensor([[1, 2, 3]]).repeat(20000, 1)
    d = drop_tags(tags, 0.1, 0.2, 0.3, generator=g)
    # a dropped level implies all levels below are dropped
    assert ((d[:, 0] == -1) <= (d[:, 1] == -1)).all() and ((d[:, 1] == -1) <= (d[:, 2] == -1)).all()
    # marginal drop rates: family 0.1, subfamily 1-(0.9*0.8)=0.28, clade 1-(0.9*0.8*0.7)=0.496
    rates = (d == -1).float().mean(0).numpy()
    np.testing.assert_allclose(rates, [0.1, 0.28, 0.496], atol=0.015)


def test_save_load(tmp_path):
    m = small_model()
    save_model(m, tmp_path / "c.pt")
    m2 = load_model(tmp_path / "c.pt")
    torch.testing.assert_close(m.to_potts((1, 2, 3)).J, m2.to_potts((1, 2, 3)).J)


def test_learns_clade_specific_fields(rng, outdir):
    """Two clades share a coupling (0,1) but prefer different residues at site 2 (residue 0 vs residue 2).
    Conditioned on its clade tag, the model must generate the right residue at site 2; conditioned on the
    family only (clade dropped) it must generate a mixture of both."""
    L, q = 5, 3
    J = np.zeros((L, L, q, q))
    J[0, 1] = np.array([[1.5, -1, -0.5], [-1, 1.5, -0.5], [-0.5, -0.5, 1.0]])
    J[1, 0] = J[0, 1].T
    truths = []
    for pref in [0, 2]:
        h = np.zeros((L, q))
        h[2, pref] = 2.5
        truths.append(Potts(h, J))
    S = np.concatenate([gibbs_sample(t, 6000, 30, rng=rng).samples for t in truths])
    leaf = np.repeat([0, 1], 6000)
    tags = np.stack([np.zeros_like(leaf), np.zeros_like(leaf), leaf], axis=1)
    perm = rng.permutation(len(S))
    tr, va = perm[:10000], perm[10000:]

    model = ConditionalFactoredAttentionPotts(L, q, n_families=1, n_subfamilies=1, n_clades=2, n_heads=3, d_k=4)
    cfg = TrainConfig(epochs=60, batch_size=1000, lr=0.05, lam_h=1e-4, lam_J=1e-4, lam_label=1e-4,
                      tag_dropout=(0.0, 0.0, 0.3), patience=15)
    train(model, S[tr], S[va], cfg, tags[tr], tags[va], device=CPU, verbose=False)

    f_site2 = {}
    for name, tag in [("clade C0", (0, 0, 0)), ("clade C1", (0, 0, 1)), ("family only", (0, 0, -1))]:
        gen = gibbs_sample(model.to_potts(tag), 4000, 30, rng=rng).samples
        f_site2[name] = site_frequencies(gen, q)[2]
    true_f = [site_frequencies(S[leaf == k], q)[2] for k in range(2)]
    np.testing.assert_allclose(f_site2["clade C0"], true_f[0], atol=0.04)
    np.testing.assert_allclose(f_site2["clade C1"], true_f[1], atol=0.04)
    assert f_site2["family only"][0] > 0.25 and f_site2["family only"][2] > 0.25     # a mixture of both

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(q)
    for k, (name, f) in enumerate(f_site2.items()):
        ax.bar(x + (k - 1) * 0.26, f, 0.24, color=SERIES[k], label=f"generated, {name}")
    for k in range(2):
        ax.scatter(x + (k - 1) * 0.26, true_f[k], marker="_", s=300, color="#0b0b0b", zorder=5,
                   label="true clade frequency" if k == 0 else None)
    ax.set_xticks(x, ["residue 0", "residue 1", "residue 2"])
    ax.set(ylabel="frequency at site 2", title="Conditioning on the tag selects the clade")
    ax.legend(fontsize=7)
    save(fig, outdir / "site2_by_tag.png")
