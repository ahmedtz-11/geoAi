"""Figures for Section 2 (interpretation & prediction)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

# Geosoft / Oasis montaj rainbow: blue (low) -> cyan -> green -> yellow -> orange -> red -> pink (high).
GEOSOFT_COLORS = ["#0000ff", "#0040ff", "#0080ff", "#00c0ff", "#00ffff", "#00ffc0", "#00ff80", "#00ff40",
                  "#00ff00", "#40ff00", "#80ff00", "#c0ff00", "#ffff00", "#ffd000", "#ffa000", "#ff7000",
                  "#ff4000", "#ff0000", "#ff0080", "#ff00ff", "#ff80ff"]
GEOSOFT = LinearSegmentedColormap.from_list("geosoft", GEOSOFT_COLORS, N=256)
try:
    matplotlib.colormaps.register(GEOSOFT, name="geosoft")
except ValueError:
    pass

# How values are spread over the colour table. Oasis montaj calls these colour methods.
STRETCHES = ["Equal area (histogram)", "Linear"]
ZONES = 32          # colour zones for the equal-area stretch, as in Oasis montaj


def colour_norm(a, stretch=STRETCHES[0], lo=2, hi=98, zones=ZONES):
    """Value -> colour mapping.

    Equal area gives every colour the same number of cells (Geosoft's "histogram equalization"), so a
    field whose variation is a few hundred nT on a 33,000 nT background still shows its full structure.
    A linear stretch over the same data is one flat colour, because a handful of spikes set the range.
    """
    f = np.asarray(a, float)
    f = f[np.isfinite(f)]
    if not f.size:
        return None, (0.0, 1.0)
    if stretch == STRETCHES[0] and f.size > zones:
        edges = np.unique(np.percentile(f, np.linspace(0, 100, zones + 1)))
        if len(edges) > 2:
            return BoundaryNorm(edges, 256), (float(edges[0]), float(edges[-1]))
    vmin, vmax = (float(v) for v in np.percentile(f, [lo, hi]))
    return None, (vmin, vmax)


def _draw(ax, arr, extent, cmap, stretch, lo, hi, smooth=True, **kw):
    """Draw a grid. Oasis montaj and Geosoft resample the grid smoothly when they draw it, so a map of a
    few hundred cells reads as a continuous field rather than as squares; `smooth` does the same here."""
    norm, (vmin, vmax) = colour_norm(arr, stretch, lo, hi)
    args = dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax)
    return ax.imshow(arr, extent=extent, cmap=cmap,
                     interpolation="bilinear" if smooth else "nearest", **args, **kw)


def _ext(transform, shape):
    h, w = shape[:2]
    return [transform.c, transform.c + w * transform.a, transform.f + h * transform.e, transform.f]


def _style(ax, title):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(useOffset=False, style="plain")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_aspect("equal")


def _limits(a, lo=2, hi=98):
    f = np.asarray(a, float)
    f = f[np.isfinite(f)]
    return tuple(np.percentile(f, [lo, hi])) if f.size else (0, 1)


def grid_figure(arr, transform, title="", cmap="viridis", label="", overlay_pts=None, lo=2, hi=98,
                targets=None, figsize=(7, 6), stretch="Linear", smooth=False):
    fig, ax = plt.subplots(figsize=figsize)
    im = _draw(ax, arr, _ext(transform, arr.shape), cmap, stretch, lo, hi, smooth)
    plt.colorbar(im, ax=ax, shrink=0.8, label=label)
    if targets is not None and targets.any():
        ax.imshow(np.ma.masked_where(~targets, targets), extent=_ext(transform, arr.shape), cmap="cool",
                  alpha=0.85, vmin=0, vmax=1, interpolation="nearest")
    if overlay_pts is not None:
        ax.scatter(overlay_pts[0], overlay_pts[1], s=3, c="k", alpha=0.35, linewidths=0)
    _style(ax, title)
    fig.tight_layout()
    return fig


def panels(arrays, titles, transform, cmaps="viridis", ncols=3, lo=2, hi=98, suptitle="", stretch="Linear", smooth=False):
    n = len(arrays)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.9 * nrows), squeeze=False)
    cm = cmaps if isinstance(cmaps, list) else [cmaps] * n
    for ax, a, t, c in zip(axes.ravel(), arrays, titles, cm):
        im = _draw(ax, a, _ext(transform, a.shape), c, stretch, lo, hi, smooth)
        plt.colorbar(im, ax=ax, shrink=0.75)
        ax.set_title(t, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def points_figure(x, y, values=None, classes=None, labels=None, colors=None, title="", cmap="viridis",
                  label="", background=None, log=False, sizes=None, figsize=(7, 6)):
    """Sample / station map: continuous values or categories, optional raster or image background."""
    fig, ax = plt.subplots(figsize=figsize)
    if background is not None:
        arr, tr, bcmap = background
        if arr.ndim == 3:
            ax.imshow(arr, extent=_ext(tr, arr.shape), alpha=0.75)
        else:
            vmin, vmax = _limits(arr)
            ax.imshow(arr, extent=_ext(tr, arr.shape), cmap=bcmap or "gray", vmin=vmin, vmax=vmax, alpha=0.6)
    if classes is not None:
        for i, (lab, col) in enumerate(zip(labels, colors)):
            m = classes == i
            if m.any():
                ax.scatter(x[m], y[m], s=(sizes[i] if sizes else 18), c=col, label=f"{lab} ({m.sum()})",
                           edgecolors="k", linewidths=0.3, zorder=3 + i)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    else:
        v = np.asarray(values, float)
        from matplotlib.colors import LogNorm
        pos = v[np.isfinite(v) & (v > 0)]
        norm = LogNorm(*np.percentile(pos, [2, 99.5])) if log and pos.size else None
        sc = ax.scatter(x, y, c=v, s=16, cmap=cmap, norm=norm, edgecolors="none",
                        **({} if norm else dict(zip(("vmin", "vmax"), _limits(v)))))
        plt.colorbar(sc, ax=ax, shrink=0.8, label=label)
    _style(ax, title)
    fig.tight_layout()
    return fig


def section_figure(xi, zi, G, pts=None, threshold=None, title="", cmap="jet", label="", hi=99.8):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    vmin, vmax = _limits(G, 1, hi)
    im = ax.imshow(G, extent=[xi.min(), xi.max(), zi.max(), zi.min()], aspect="auto", cmap=cmap, vmin=vmin,
                   vmax=vmax, interpolation="bilinear")
    plt.colorbar(im, ax=ax, label=label, shrink=0.9)
    if threshold is not None and np.isfinite(G).any():
        ax.contour(xi, zi, G, levels=[threshold], colors="k", linewidths=1.4)
    if pts is not None:
        ax.scatter(pts[0], pts[1], s=4, c="k", alpha=0.4, linewidths=0)
    ax.set_xlabel("Distance / Easting along line (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title, fontsize=10)
    ax.ticklabel_format(useOffset=False, style="plain")
    fig.tight_layout()
    return fig


def spectrum_figure(k, lnp, fits):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(k, lnp, "o", ms=4, color="#555", label="radially averaged power")
    cols = {"deep": "#c0392b", "shallow": "#2471a3"}
    for name, f in fits.items():
        a, b = f["range"]
        kk = k[a:b]
        ax.plot(kk, f["slope"] * kk + f["intercept"], "-", lw=2.5, color=cols[name],
                label=f"{name}: ≈ {f['depth']:.0f} m")
    ax.set_xlabel("Wavenumber k (rad/m)")
    ax.set_ylabel("ln(power)")
    ax.set_title("Power spectrum depth estimate", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def euler_figure(background, transform, sols, title="Euler deconvolution solutions"):
    fig, ax = plt.subplots(figsize=(7, 6))
    vmin, vmax = _limits(background)
    ax.imshow(background, extent=_ext(transform, background.shape), cmap="gray", vmin=vmin, vmax=vmax,
              interpolation="nearest")
    if len(sols):
        d = sols.depth.to_numpy()
        sc = ax.scatter(sols.x, sols.y, c=d, s=10 + 30 * (d / max(d.max(), 1)), cmap="plasma_r",
                        vmin=np.percentile(d, 2), vmax=np.percentile(d, 98), edgecolors="k", linewidths=0.2)
        plt.colorbar(sc, ax=ax, shrink=0.8, label="Depth below surface (m)")
    _style(ax, title)
    e = _ext(transform, background.shape)
    ax.set_xlim(e[0], e[1])
    ax.set_ylim(e[2], e[3])
    fig.tight_layout()
    return fig


def hist_figure(values, limits=None, title="", unit="", log=True):
    v = np.asarray(values, float)
    v = v[np.isfinite(v) & ((v > 0) if log else True)]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    if v.size:
        bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 40) if log and v.min() > 0 else 40
        ax.hist(v, bins=bins, color="#8c8c8c", edgecolor="white")
        if log:
            ax.set_xscale("log")
    names = ["weak", "anomalous", "strong"]
    for lim, nm, c in zip(limits or [], names, ["#f2c14e", "#e4572e", "#7b0d1e"]):
        if np.isfinite(lim):
            ax.axvline(lim, color=c, lw=2, label=f"{nm} ≥ {lim:.3g}")
    ax.set_xlabel(unit)
    ax.set_ylabel("Samples")
    ax.set_title(title, fontsize=10)
    if limits:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def corr_figure(corr, title="Spearman correlation"):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(1.2 + 0.6 * n, 1.0 + 0.55 * n))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), corr.index, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.values[i, j]) > 0.6 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def stereonet_figure(px, py, title="Poles to planes (equal area, lower hemisphere)"):
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), "k-", lw=1)
    ax.plot([0, 0], [-1, 1], color="#ccc", lw=0.6)
    ax.plot([-1, 1], [0, 0], color="#ccc", lw=0.6)
    ax.scatter(px, py, s=14, c="#c0392b", edgecolors="k", linewidths=0.3)
    ax.text(0, 1.06, "N", ha="center", fontsize=10, weight="bold")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def bar_figure(d, title="", xlabel=""):
    items = sorted(d.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6, 0.5 + 0.38 * len(items)))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#2e86c1")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def image_figure(img, extent, title="", pts=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(np.asarray(img), extent=extent, aspect="equal")
    if pts is not None:
        ax.scatter(pts[0], pts[1], s=8, c="k", alpha=0.5)
    _style(ax, title)
    fig.tight_layout()
    return fig


def section_points_figure(x, depth, v, threshold=None, title="", cmap="jet", label="", hi=99.8):
    """Pseudosection drawn as coloured readings (used when there are too few readings to interpolate)."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x, depth, v = (np.asarray(a, float) for a in (x, depth, v))
    vmin, vmax = _limits(v, 1, hi)
    sc = ax.scatter(x, depth, c=v, cmap=cmap, vmin=vmin, vmax=vmax, s=60, edgecolors="k", linewidths=0.3)
    if threshold is not None:
        hot = v >= threshold
        ax.scatter(x[hot], depth[hot], s=140, facecolors="none", edgecolors="k", linewidths=1.4)
    plt.colorbar(sc, ax=ax, label=label, shrink=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("Distance / Easting along line (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig
