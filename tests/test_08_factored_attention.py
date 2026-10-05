"""Step 8 - factored-attention Potts model (unconditioned): structure, consistency with psg.potts, learning a toy model."""
import numpy as np
import torch

from psg.models.factored_attention import (
    FactoredAttentionPotts,
    assemble_couplings,
    coupling_norm2,
    load_model,
    pseudo_nll,
    save_model,
)
from psg.plots import INK_2, SEQ_BLUE, SERIES, plt, save
from psg.potts import Potts
from psg.sampling import gibbs_sample
from psg.stats import pearson
from psg.training import TrainConfig, train

CPU = torch.device("cpu")


def test_attention_and_coupling_structure():
    m = FactoredAttentionPotts(L=7, q=4, n_heads=3, d_k=5, init_v=0.5)
    A = m.attention()
    assert A.shape == (3, 7, 7)
    torch.testing.assert_close(A.sum(-1), torch.ones(3, 7))              # each row is a distribution over j
    assert (A[:, torch.arange(7), torch.arange(7)] == 0).all()             # no self-attention
    J = m.couplings()
    assert J.shape == (7, 7, 4, 4)
    torch.testing.assert_close(J, J.permute(1, 0, 3, 2))                    # J_ij(a,b) = J_ji(b,a)
    assert (J[torch.arange(7), torch.arange(7)] == 0).all()
    m.to_potts().check_symmetry()


def test_logits_equal_potts_local_field(rng):
    """The network's logits ARE the Potts local fields of its exported (h, J)."""
    m = FactoredAttentionPotts(L=6, q=5, n_heads=4, d_k=3, init_v=0.7)
    with torch.no_grad():
        m.h.normal_()
    S = rng.integers(0, 5, (20, 6))
    logits = m.logits(torch.as_tensor(S)).detach().double().numpy()
    P = m.to_potts()
    for i in range(6):
        np.testing.assert_allclose(logits[:, i], P.local_field(S, i), atol=1e-5)
    # and the pseudo-NLL equals -sum_i log P(s_i | s_-i) computed with psg.potts
    pn = pseudo_nll(torch.as_tensor(logits), torch.as_tensor(S)).numpy()
    ref = -sum(np.log(P.conditional(S, i)[np.arange(20), S[:, i]]) for i in range(6))
    np.testing.assert_allclose(pn, ref, rtol=1e-6)


def test_coupling_norm_trick():
    g = torch.Generator().manual_seed(0)
    A = torch.softmax(torch.randn(5, 8, 8, generator=g), -1)
    V = torch.randn(5, 3, 3, generator=g)
    direct = (assemble_couplings(A, V) ** 2).sum()
    torch.testing.assert_close(coupling_norm2(A, V), direct)
    Vb = torch.randn(4, 5, 3, 3, generator=g)                               # batch of value sets
    direct_b = torch.stack([(assemble_couplings(A, v) ** 2).sum() for v in Vb])
    torch.testing.assert_close(coupling_norm2(A, Vb), direct_b)


def test_save_load(tmp_path):
    m = FactoredAttentionPotts(L=5, q=3, n_heads=2, d_k=2)
    save_model(m, tmp_path / "m.pt", note="x")
    m2 = load_model(tmp_path / "m.pt")
    torch.testing.assert_close(m.couplings(), m2.couplings())


def toy_ground_truth(rng):
    """L=6, q=3: weak fields and ONE strong coupling, between sites 0 and 3."""
    L, q = 6, 3
    J = np.zeros((L, L, q, q))
    J[0, 3] = rng.normal(0, 1.5, (q, q))
    J[3, 0] = J[0, 3].T
    return Potts(rng.normal(0, 0.3, (L, q)), J).zero_sum_gauge()


def test_learns_single_contact(rng, outdir):
    true = toy_ground_truth(rng)
    S = gibbs_sample(true, 10000, 30, rng=rng).samples
    model = FactoredAttentionPotts(L=6, q=3, n_heads=4, d_k=4, seed=0)
    cfg = TrainConfig(epochs=80, batch_size=1000, lr=0.05, lam_h=1e-4, lam_J=1e-4, patience=15)
    hist = train(model, S[:8000], S[8000:], cfg, device=CPU, verbose=False)
    assert hist["val_pnll_site"][-1] < hist["val_pnll_site"][0] - 0.01

    learned = model.to_potts().zero_sum_gauge()
    F = np.sqrt((learned.J ** 2).sum(axis=(2, 3)))
    iu = np.triu_indices(6, 1)
    top = (iu[0][np.argmax(F[iu])], iu[1][np.argmax(F[iu])])
    assert top == (0, 3), top
    r = pearson(learned.J[0, 3].ravel(), true.J[0, 3].ravel())
    assert r > 0.9, r
    r_h = pearson(learned.h.ravel(), true.h.ravel())
    assert r_h > 0.8, r_h

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    ax[0].plot(hist["val_pnll_site"], color=SERIES[0], label="validation")
    ax[0].plot(hist["train_pnll_site"], color=SERIES[1], label="train")
    ax[0].set(xlabel="epoch", ylabel="pseudo-NLL per site", title="Training curve")
    ax[0].legend()
    im = ax[1].imshow(F, cmap=SEQ_BLUE)
    ax[1].grid(False)
    ax[1].set(title="learned ||J_ij||_F (true contact: 0-3)", xlabel="j", ylabel="i")
    fig.colorbar(im, ax=ax[1], shrink=0.8).outline.set_visible(False)
    lim = np.abs(true.J[0, 3]).max() * 1.2
    ax[2].plot([-lim, lim], [-lim, lim], color=INK_2, lw=1, ls="--")
    ax[2].scatter(true.J[0, 3].ravel(), learned.J[0, 3].ravel(), s=40, color=SERIES[0], label="J_03(a,b)")
    ax[2].scatter(true.h.ravel(), learned.h.ravel(), s=25, color=SERIES[2], marker="^", label="h_i(a)")
    ax[2].set(xlabel="true (zero-sum gauge)", ylabel="learned (zero-sum gauge)", title=f"J_03: r = {r:.3f}, h: r = {r_h:.3f}")
    ax[2].legend()
    fig.tight_layout()
    save(fig, outdir / "toy_single_contact.png")
