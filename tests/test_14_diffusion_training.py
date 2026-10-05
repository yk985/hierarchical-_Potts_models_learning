"""Step 14 - training the masked-diffusion denoiser on a tiny Potts model with a known distribution."""
import numpy as np
import torch

from psg.diffusion_training import DiffusionTrainConfig, evaluate_nelbo, train_diffusion
from psg.models.masked_diffusion import MaskedDiffusionTransformer, ancestral_sample
from psg.plots import INK_2, SERIES, plt, save
from psg.potts import Potts, exact_distribution, symmetrize

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_learns_tiny_potts(rng, outdir):
    L, q = 4, 3
    model_true = Potts(rng.normal(0, 1.0, (L, q)), symmetrize(rng.normal(0, 1.0, (L, L, q, q))))
    states, p = exact_distribution(model_true)
    H_site = float(-(p * np.log(p)).sum()) / L
    S = states[rng.choice(len(states), size=24000, p=p)]
    net = MaskedDiffusionTransformer(L, q, d_model=64, n_layers=2, n_heads=4, d_ff=128, seed=0)
    cfg = DiffusionTrainConfig(epochs=60, batch_size=256, lr=1e-3, warmup_steps=100, ema_decay=0.99,
                               val_repeats=8, patience=15, seed=0)
    hist, best = train_diffusion(net, S[:20000], S[20000:], cfg, device=DEV, verbose=False)

    # 1. the NELBO is an upper bound on -log p(x) for every x. Reference: the exact -log p(x) averaged over the SAME
    #    validation sequences (a finite sample can sit slightly below the population entropy H).
    idx_val = S[20000:] @ (q ** np.arange(L - 1, -1, -1))
    nll_val = float(-np.log(p[idx_val]).mean()) / L
    val = evaluate_nelbo(best, torch.as_tensor(S[20000:], device=DEV), repeats=32)
    assert val > nll_val - 0.01 and val < nll_val + 0.03, (val, nll_val)

    # 2. the probabilities: fully masked -> marginals; all but site i observed -> Potts conditional of site i
    best.eval()
    with torch.no_grad():
        full_mask = torch.full((1, L), q, device=DEV)
        pm = torch.softmax(best(full_mask, torch.ones(1, device=DEV)), -1)[0].cpu().numpy()
        marg = np.stack([[p[states[:, i] == a].sum() for a in range(q)] for i in range(L)])
        assert np.abs(pm - marg).max() < 0.03
        # error of P(s_i | s_-i) for every state and site, weighted by how often the context occurs (p(s)):
        # rare contexts are seen only a few times in training, so they are judged on the same footing as data noise
        errs, w = [], []
        for s_, ps in zip(states, p):
            for i in range(L):
                x = torch.as_tensor(s_, device=DEV)[None].clone()
                x[0, i] = q
                pc = torch.softmax(best(x, torch.full((1,), 1 / L, device=DEV)), -1)[0, i].cpu().numpy()
                errs.append(np.abs(pc - model_true.conditional(s_[None], i)[0]).max())
                w.append(ps)
        errs, w = np.array(errs), np.array(w)
        weighted = float((errs * w).sum() / w.sum())
        assert weighted < 0.03, weighted
        assert errs[w > 1 / len(p)].max() < 0.1          # frequent contexts: every conditional is right

    # 3. the samples
    G = ancestral_sample(best, 20000, L, q, n_steps=64, seed=1, device=DEV).numpy()
    idx = G @ (q ** np.arange(L - 1, -1, -1))
    emp = np.bincount(idx, minlength=q ** L) / len(G)
    tv = 0.5 * np.abs(emp - p).sum()
    assert tv < 0.05, tv

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].plot(hist["epoch"], hist["train_nelbo_site"], color=SERIES[1], label="train")
    ax[0].plot(hist["epoch"], hist["val_nelbo_site"], color=SERIES[0], label="validation (EMA)")
    ax[0].axhline(nll_val, color=INK_2, ls="--", lw=1, label="exact -log p(x) of the validation set")
    ax[0].set(xlabel="epoch", ylabel="NELBO per site (nats)", title="Diffusion training on a tiny Potts model")
    ax[0].legend(fontsize=7)
    order = np.argsort(-p)
    x = np.arange(len(p))
    ax[1].bar(x - 0.2, p[order], 0.4, color=SERIES[0], label="exact")
    ax[1].bar(x + 0.2, emp[order], 0.4, color=SERIES[1], label="diffusion samples")
    ax[1].set(xlabel="state (sorted by exact probability)", ylabel="P(s)", title=f"81 states: TV = {tv:.3f}")
    ax[1].set_xticks([])
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, outdir / "tiny_potts_diffusion.png")
