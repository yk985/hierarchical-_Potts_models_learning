"""Step B - Gibbs-sample sequences from every clade (leaf) Potts model, one FASTA per clade.

Each chain starts from a uniformly random sequence, runs n_burnin sweeps, and contributes ONE sequence
(so all sequences are independent). Outputs (under --root):
    data/sequences/<clade>.fasta
    data/sequences/all_sequences.npz      S (M, L) + family / subfamily / clade / leaf labels
    evaluations/ground_truth/sampling_diagnostics.json, energy_traces.npz
    figures/ground_truth/02_sampling/energy_traces.png

    python scripts/s02_sample_ground_truth.py --config configs/ground_truth.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psg import PROJECT_ROOT  # noqa: E402
from psg.alphabet import write_fasta  # noqa: E402
from psg.hierarchy import leaves, load_hierarchy  # noqa: E402
from psg.plots import SERIES, plt, save  # noqa: E402
from psg.sampling import gibbs_sample  # noqa: E402


def autocorrelation(trace: np.ndarray, lags) -> list[float]:
    """Energy autocorrelation averaged over chains; trace (T, n_chains)."""
    x = trace - trace.mean(axis=0)
    v = (x * x).mean()
    return [float((x[:-k] * x[k:]).mean() / v) for k in lags]


def main(config: dict, root: Path = PROJECT_ROOT, verbose: bool = True) -> dict:
    sc = config["sampling"]
    nodes, _, _ = load_hierarchy(root / "data" / "ground_truth_potts")
    lv = leaves(nodes)
    rng = np.random.default_rng(sc["seed"])
    seq_dir = root / "data" / "sequences"

    S_all, labels, traces, diag = [], [], {}, {}
    for k, node in enumerate(lv):
        t0 = time.time()
        res = gibbs_sample(node.model, n_chains=sc["n_per_clade"], n_burnin=sc["n_burnin"],
                           beta=sc["beta"], rng=rng)
        S = res.samples
        write_fasta(seq_dir / f"{node.name}.fasta", S, [f"{node.name}|seq{m:05d}" for m in range(len(S))])
        S_all.append(S)
        labels += [(*node.path, k)] * len(S)
        mean_trace = res.energy_trace.mean(axis=1)
        traces[node.name] = mean_trace
        half = sc["n_burnin"] // 2
        diag[node.name] = {
            "final_mean_energy": float(mean_trace[-1]),
            "energy_drift_second_half": float(mean_trace[half:half + half // 2].mean() - mean_trace[-(half // 2):].mean()),
            "energy_autocorr_lag_1_5_20": autocorrelation(res.energy_trace[half:], [1, 5, 20]),
            "seconds": round(time.time() - t0, 1),
        }
        if verbose:
            print(f"[{k + 1}/{len(lv)}] {node.name}: {len(S)} seqs, <E>={mean_trace[-1]:.1f}, "
                  f"{diag[node.name]['seconds']}s", flush=True)

    S_all = np.concatenate(S_all)
    lab = np.array(labels)
    np.savez_compressed(seq_dir / "all_sequences.npz", S=S_all, family=lab[:, 0], subfamily=lab[:, 1],
                        clade=lab[:, 2], leaf=lab[:, 3], leaf_names=np.array([n.name for n in lv]))

    ev = root / "evaluations" / "ground_truth"
    ev.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ev / "energy_traces.npz", **traces)
    (ev / "sampling_diagnostics.json").write_text(json.dumps({"config": sc, "per_clade": diag}, indent=2))

    # energy traces, one panel per family (colour = family)
    fams = sorted({n.path[0] for n in lv})
    fig, axes = plt.subplots(1, len(fams), figsize=(3.6 * len(fams), 3), sharey=True, squeeze=False)
    for ax, f in zip(axes[0], fams):
        for n in lv:
            if n.path[0] == f:
                ax.plot(traces[n.name], color=SERIES[f % len(SERIES)], lw=1, alpha=0.7)
        ax.axvline(half, color="#52514e", lw=1, ls="--")
        ax.set(title=f"Family F{f}: mean energy per clade", xlabel="sweep")
    axes[0, 0].set_ylabel("mean E(s) over chains")
    fig.tight_layout()
    save(fig, root / "figures" / "ground_truth" / "02_sampling" / "energy_traces.png")
    return diag


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "ground_truth.json"))
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    a = ap.parse_args()
    main(json.loads(Path(a.config).read_text()), Path(a.root))
