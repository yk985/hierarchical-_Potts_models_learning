"""Shared plotting style and figures. Static PNGs (matplotlib), one visual system for the whole project."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Categorical slots, fixed order. Scatter plots (all pairs visible) use only the first 3.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "^", "s", "D", "v", "P"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f7fafe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIV_BLUE_RED = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#a32424"]
)


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "font.size": 9,
        "lines.linewidth": 2,
        "figure.dpi": 110,
    })


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


style()


# ---------------------------------------------------------------------- hierarchy figures
def plot_contact_map(C, title="Contact map"):
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(C, cmap=SEQ_BLUE, vmin=0, vmax=1, interpolation="nearest")
    ax.grid(False)
    ax.set(xlabel="site j", ylabel="site i", title=f"{title} ({int(C.sum() // 2)} pairs)")
    return fig


def plot_distance_matrix(D, labels, title, cbar_label="distance"):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.32 * n + 2.4, 0.32 * n + 1.6))
    im = ax.imshow(D, cmap=SEQ_BLUE, interpolation="nearest")
    ax.grid(False)
    ax.set_xticks(range(n), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n), labels, fontsize=7)
    # thin surface lines separating families (labels "F<f>...")
    fam = [str(l).split(".")[0] for l in labels]
    for k in range(1, n):
        if fam[k] != fam[k - 1]:
            ax.axhline(k - 0.5, color=SURFACE, lw=2)
            ax.axvline(k - 0.5, color=SURFACE, lw=2)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(cbar_label)
    cb.outline.set_visible(False)
    ax.set_title(title)
    return fig


def plot_distance_by_relation(groups, names, title, ylabel="distance", theory=None):
    """groups: {lca_level: array of distances}. One strip + mean bar per relation, ordered far -> close."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    keys = sorted(groups)
    rng = np.random.default_rng(0)
    for x, k in enumerate(keys):
        v = groups[k]
        ax.scatter(x + rng.uniform(-0.15, 0.15, len(v)), v, s=14, color=SERIES[0], alpha=0.55, lw=0)
        ax.plot([x - 0.28, x + 0.28], [v.mean()] * 2, color=INK, lw=2)
        ax.text(x + 0.32, v.mean(), f"{v.mean():.2f}", va="center", fontsize=8, color=INK)
        if theory is not None and k in theory:
            ax.scatter([x], [theory[k]], s=70, marker="D", facecolor="none", edgecolor=SERIES[1], lw=1.5,
                       zorder=5, label="theory (RMS)" if x == 0 else None)
    ax.set_xticks(range(len(keys)), [f"{names[k]}\n(n={len(groups[k])})" for k in keys])
    ax.set_xlim(-0.5, len(keys) - 0.3)
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    ax.set(ylabel=ylabel, title=title)
    if theory is not None:
        ax.legend(loc="upper right")
    return fig


# ---------------------------------------------------------------------- sequence-statistics figures
def plot_pca(scores, family, subfamily, evr, title="PCA of sequences", pcs=(0, 1), ax=None, s=6, alpha=0.5,
             color_name="family F", marker_name="subfamily S"):
    """Color = `family` (<= 3 groups, the all-pairs-safe slots), marker = `subfamily`."""
    import numpy as np

    fams, subs = np.unique(family), np.unique(subfamily)
    if len(fams) > 3:
        raise ValueError("scatter colour budget is 3 families; facet instead")
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.6, 5))
    for fi, f in enumerate(fams):
        for si, sb in enumerate(subs):
            k = (family == f) & (subfamily == sb)
            if k.any():
                ax.scatter(scores[k, pcs[0]], scores[k, pcs[1]], s=s, alpha=alpha, color=SERIES[fi],
                           marker=MARKERS[si % len(MARKERS)], lw=0, rasterized=True)
    # legend: colour = family, marker = subfamily
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", color=SERIES[i], ms=7, label=f"{color_name}{f}") for i, f in enumerate(fams)]
    h += [Line2D([], [], ls="", marker=MARKERS[i % len(MARKERS)], color=INK_2, ms=6, label=f"{marker_name}{sb}")
          for i, sb in enumerate(subs)]
    ax.legend(handles=h, loc="best", fontsize=7)
    ax.set(xlabel=f"PC{pcs[0] + 1} ({100 * evr[pcs[0]]:.1f}% var.)",
           ylabel=f"PC{pcs[1] + 1} ({100 * evr[pcs[1]]:.1f}% var.)", title=title)
    return fig if fig is not None else ax.figure


def plot_stat_comparison(ref_f1, gen_f1, ref_C, gen_C, metrics, ref_name="reference", gen_name="generated"):
    """Two panels: single-site frequencies and connected correlations, gen vs ref, with y = x."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, x, y, name, r in [
        (axes[0], ref_f1, gen_f1, "f_i(a)", metrics["pearson_f1"]),
        (axes[1], ref_C, gen_C, "C_ij(a,b)", metrics["pearson_C"]),
    ]:
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color=INK_2, lw=1, ls="--")
        ax.scatter(x, y, s=4, color=SERIES[0], alpha=0.35, lw=0, rasterized=True)
        ax.set(xlabel=f"{name}  [{ref_name}]", ylabel=f"{name}  [{gen_name}]", title=f"{name}: Pearson r = {r:.3f}")
    fig.tight_layout()
    return fig


def plot_site_frequencies(f1, alphabet, title="Site frequencies f_i(a)"):
    L, q = f1.shape
    fig, ax = plt.subplots(figsize=(max(6, 0.12 * L + 2), 3.4))
    im = ax.imshow(f1.T, aspect="auto", cmap=SEQ_BLUE, vmin=0, vmax=max(0.5, f1.max()), interpolation="nearest")
    ax.grid(False)
    ax.set_yticks(range(q), list(alphabet[:q]), fontsize=6)
    ax.set(xlabel="site i", ylabel="amino acid a", title=title)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.outline.set_visible(False)
    return fig


def plot_correlation_strength(C, contacts=None, title="Correlation strength ||C_ij||_F"):
    """L x L map of the Frobenius norm of each connected-correlation block; contacts outlined in the upper triangle."""
    import numpy as np

    N = np.sqrt((C ** 2).sum(axis=(2, 3)))
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(N, cmap=SEQ_BLUE, interpolation="nearest")
    ax.grid(False)
    if contacts is not None:
        i, j = np.nonzero(np.triu(contacts, 1))
        ax.scatter(j, i, s=10, facecolor="none", edgecolor=SERIES[1], lw=0.8, label="true contact")
        ax.legend(loc="lower left", fontsize=7)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.outline.set_visible(False)
    ax.set(xlabel="site j", ylabel="site i", title=title)
    return fig


def plot_frequency_maps(f_ref, f_gen, alphabet, ref_name="test", gen_name="generated"):
    """Three panels: f_i(a) reference, generated, and the difference generated - reference (diverging)."""
    import numpy as np

    L, q = f_ref.shape
    vmax = max(f_ref.max(), f_gen.max())
    D = f_gen - f_ref
    dmax = np.abs(D).max()
    fig, axes = plt.subplots(3, 1, figsize=(max(7, 0.13 * L + 2), 8), sharex=True)
    for ax, M, title, cmap, lim in [(axes[0], f_ref, f"f_i(a), {ref_name}", SEQ_BLUE, (0, vmax)),
                                    (axes[1], f_gen, f"f_i(a), {gen_name}", SEQ_BLUE, (0, vmax)),
                                    (axes[2], D, f"difference {gen_name} - {ref_name}", DIV_BLUE_RED, (-dmax, dmax))]:
        im = ax.imshow(M.T, aspect="auto", cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest")
        ax.grid(False)
        ax.set_yticks(range(q), list(alphabet[:q]), fontsize=5)
        ax.set(ylabel="amino acid", title=title)
        cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
        cb.outline.set_visible(False)
    axes[2].set_xlabel("site i")
    fig.tight_layout()
    return fig


def plot_correlation_triangles(C_ref, C_gen, contacts=None, ref_name="test", gen_name="generated"):
    """One L x L map: ||C_ij||_F of the reference below the diagonal, of the generated set above it."""
    import numpy as np

    Nr = np.sqrt((C_ref ** 2).sum(axis=(2, 3)))
    Ng = np.sqrt((C_gen ** 2).sum(axis=(2, 3)))
    M = np.tril(Nr, -1) + np.triu(Ng, 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(M, cmap=SEQ_BLUE, interpolation="nearest")
    ax.grid(False)
    if contacts is not None:
        i, j = np.nonzero(np.triu(contacts, 1))
        ax.scatter(j, i, s=9, facecolor="none", edgecolor=SERIES[1], lw=0.7, label="true contact")
        ax.scatter(i, j, s=9, facecolor="none", edgecolor=SERIES[1], lw=0.7)
        ax.legend(loc="lower left", fontsize=7)
    ax.text(0.98, 0.98, gen_name, transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK)
    ax.text(0.02, 0.02 if contacts is None else 0.12, ref_name, transform=ax.transAxes, fontsize=8, color=INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.outline.set_visible(False)
    ax.set(xlabel="site j", ylabel="site i", title="||C_ij||_F: upper = generated, lower = reference")
    return fig


def plot_distance_distributions(series: dict, title: str, xlabel: str = "normalised Hamming distance", bins=60,
                                step: float | None = None):
    """Overlaid step histograms (densities) of distance samples; series = {label: array}. At most 4 series.
    step: the values are multiples of `step` (e.g. 1/L for Hamming distances); bins are then centred on them."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    lo = min(float(np.min(v)) for v in series.values())
    hi = max(float(np.max(v)) for v in series.values())
    if step is None:
        edges = np.linspace(lo, hi, bins + 1)
    else:
        edges = np.arange(np.floor(lo / step) - 0.5, np.ceil(hi / step) + 1.0) * step
    for k, (lab, v) in enumerate(series.items()):
        ax.hist(v, bins=edges, density=True, histtype="step", lw=2, color=SERIES[k],
                label=f"{lab} (mean {np.mean(v):.3f})")
    ax.set(xlabel=xlabel, ylabel="density", title=title)
    ax.legend(fontsize=7)
    return fig
