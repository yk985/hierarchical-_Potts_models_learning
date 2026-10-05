"""Step 16 - checkpoints during diffusion training: best model, periodic snapshots, exact resume."""
import numpy as np
import torch

from psg.diffusion_training import DiffusionTrainConfig, train_diffusion
from psg.models.masked_diffusion import MaskedDiffusionTransformer, load_diffusion

CPU = torch.device("cpu")


def data(rng):
    S = rng.integers(0, 3, (600, 6))
    S[:, 1] = S[:, 0]                              # one correlated pair, so there is something to learn
    return S[:500], S[500:]


def net():
    return MaskedDiffusionTransformer(6, 3, d_model=16, n_layers=1, n_heads=2, d_ff=32, seed=0)


def cfg(epochs):
    return DiffusionTrainConfig(epochs=epochs, batch_size=100, lr=3e-3, warmup_steps=5, ema_decay=0.9,
                                val_repeats=2, patience=50, save_every=2, seed=0)


def test_checkpoint_files(rng, tmp_path):
    tr, va = data(rng)
    hist, best = train_diffusion(net(), tr, va, cfg(4), device=CPU, verbose=False, checkpoint_dir=tmp_path)
    assert (tmp_path / "model.pt").exists() and (tmp_path / "last.pt").exists()
    assert sorted(p.name for p in (tmp_path / "checkpoints").iterdir()) == ["epoch_001.pt", "epoch_003.pt"]
    # model.pt on disk is the best EMA model
    m = load_diffusion(tmp_path / "model.pt")
    x, t = torch.as_tensor(va[:4]), torch.full((4,), 0.5)
    torch.testing.assert_close(m(x, t), best(x, t))


def test_resume_is_exact(rng, tmp_path):
    """4 epochs, then resume to 6 == 6 epochs in one go (same losses, same final weights)."""
    tr, va = data(rng)
    h_full, m_full = train_diffusion(net(), tr, va, cfg(6), device=CPU, verbose=False, checkpoint_dir=tmp_path / "a")
    train_diffusion(net(), tr, va, cfg(4), device=CPU, verbose=False, checkpoint_dir=tmp_path / "b")
    h_res, m_res = train_diffusion(net(), tr, va, cfg(6), device=CPU, verbose=False, checkpoint_dir=tmp_path / "b",
                                   resume=True)
    assert h_res["epoch"] == list(range(6))
    np.testing.assert_allclose(h_res["val_nelbo_site"], h_full["val_nelbo_site"], rtol=1e-6)
    np.testing.assert_allclose(h_res["train_nelbo_site"], h_full["train_nelbo_site"], rtol=1e-6)
    for a, b in zip(m_res.parameters(), m_full.parameters()):
        torch.testing.assert_close(a, b)
