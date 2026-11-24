"""Interpolating scattered sample values onto a grid, by the methods a geochemist expects to choose from.

Each method answers "what is the value between my samples?" differently, and the differences matter:

  * **Inverse distance** is the simple average of nearby samples weighted by 1/distance^p. Predictable and
    fast, but it cannot rise above the highest sample or fall below the lowest, and it leaves a bull's-eye
    around every point.
  * **Natural neighbour** gives each sample the share of ground that would be taken from it if the grid
    node were inserted as a new sample (Sibson, 1981). It has no bull's-eyes, honours the data exactly,
    and never invents values outside their range, but it stops at the edge of the sampled ground.
  * **Ordinary kriging** fits a variogram, measuring how fast the values really decorrelate with distance,
    then weights the samples to minimise the estimation error. It is the only method here that also says
    how uncertain each estimate is, and the variogram itself tells you the size of the anomalies.
  * **Minimum curvature** draws the smoothest surface through every reading; it can overshoot beyond the
    data range in gaps but gives the most natural-looking contours.

No Streamlit inside.
"""
import numpy as np
from scipy.spatial import cKDTree

from . import geophysics

METHODS = ["Natural neighbour", "Inverse distance (IDW)", "Ordinary kriging", "Minimum curvature",
           "Linear triangulation"]
VARIOGRAMS = ["Auto (best fit)", "Spherical", "Exponential", "Gaussian"]


def suggest_cell(x, y, max_cells=400):
    """A sensible grid interval: about a third of the sample spacing, but never an unusable number of cells."""
    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    if len(pts) < 2:
        return 1.0
    d, _ = cKDTree(pts).query(pts, k=2)
    nn = d[:, 1]
    nn = nn[nn > 0]
    cell = float(np.median(nn)) / 3 if nn.size else 1.0
    extent = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1.0)
    return float(max(cell, extent / max_cells))


def _axes(x, y, cell, pad):
    x0, x1 = float(np.min(x)) - pad, float(np.max(x)) + pad
    y0, y1 = float(np.min(y)) - pad, float(np.max(y)) + pad
    nx = max(int(np.ceil((x1 - x0) / cell)) + 1, 2)
    ny = max(int(np.ceil((y1 - y0) / cell)) + 1, 2)
    return x0 + np.arange(nx) * cell, y1 - np.arange(ny) * cell


# ---------------------------------------------------------------- the methods
def idw(x, y, v, XI, YI, power=2.0, neighbours=12, smoothing=0.0):
    """Inverse distance weighting over the nearest `neighbours` samples."""
    pts = np.column_stack([x, y])
    k = int(min(max(neighbours, 1), len(pts)))
    d, idx = cKDTree(pts).query(np.column_stack([XI.ravel(), YI.ravel()]), k=k)
    d = np.atleast_2d(d.T).T if k > 1 else d[:, None]
    idx = np.atleast_2d(idx.T).T if k > 1 else idx[:, None]
    exact = d < 1e-9                                        # sitting on a sample: take that sample
    w = 1.0 / np.maximum(d, 1e-9) ** power if smoothing <= 0 else \
        1.0 / (d ** 2 + smoothing ** 2) ** (power / 2)
    w = np.where(exact.any(1)[:, None], exact.astype(float), w)
    return ((w * np.asarray(v)[idx]).sum(1) / w.sum(1)).reshape(XI.shape)


def natural_neighbour(x, y, v, XI, YI, cell, refine=3, max_radius=None, chunk=64):
    """Sibson natural-neighbour interpolation, computed by stealing area on a fine raster.

    Each grid node's value is the mean of the sample values owned by the fine pixels that the node would
    capture if it were inserted into the Voronoi diagram, which is exactly Sibson's area-stealing weight.
    """
    x, y, v = (np.asarray(a, float) for a in (x, y, v))
    fine = cell / max(int(refine), 1)
    pad = float(max_radius if max_radius else 3 * suggest_cell(x, y) * 3)
    fx = np.arange(np.min(XI) - pad, np.max(XI) + pad + fine, fine)
    fy = np.arange(np.max(YI) + pad, np.min(YI) - pad - fine, -fine)
    FX, FY = np.meshgrid(fx, fy)
    dmin, nearest = cKDTree(np.column_stack([x, y])).query(np.column_stack([FX.ravel(), FY.ravel()]))
    dmin = dmin.reshape(FX.shape)
    owner_value = v[nearest].reshape(FX.shape)

    # how far a node can reach: beyond its own third-nearest sample it captures nothing
    d3, _ = cKDTree(np.column_stack([x, y])).query(np.column_stack([XI.ravel(), YI.ravel()]),
                                                   k=min(3, len(x)))
    reach = float(np.percentile(np.atleast_2d(d3.T).T[:, -1], 99)) if len(x) >= 3 else pad
    R = int(np.ceil(min(reach, pad) / fine)) + 1
    off = np.arange(-R, R + 1)
    OX, OY = np.meshgrid(off * fine, off * fine)
    stencil = np.hypot(OX, OY)                              # distance from the node to each stencil pixel
    inside = stencil <= R * fine

    out = np.full(XI.shape, np.nan)
    ny, nx = XI.shape
    col = np.clip(np.round((XI[0] - fx[0]) / fine).astype(int), R, len(fx) - R - 1)
    row = np.clip(np.round((fy[0] - YI[:, 0]) / fine).astype(int), R, len(fy) - R - 1)
    for a in range(0, ny, chunk):
        b = min(a + chunk, ny)
        rows = row[a:b]
        # windows: (rows, cols, stencil, stencil)
        ri = rows[:, None, None, None] + off[None, None, :, None]
        ci = col[None, :, None, None] + off[None, None, None, :]
        win_d = dmin[ri, ci]
        win_v = owner_value[ri, ci]
        taken = (stencil[None, None] < win_d) & inside[None, None]
        count = taken.sum((2, 3))
        total = np.where(taken, win_v, 0.0).sum((2, 3))
        out[a:b] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out


def experimental_variogram(x, y, v, n_lags=14, max_dist=None, max_pairs=400000, seed=0):
    """Semivariance against separation: how fast the values stop resembling each other."""
    x, y, v = (np.asarray(a, float) for a in (x, y, v))
    n = len(v)
    i, j = np.triu_indices(n, 1)
    if len(i) > max_pairs:
        pick = np.random.default_rng(seed).choice(len(i), max_pairs, replace=False)
        i, j = i[pick], j[pick]
    h = np.hypot(x[i] - x[j], y[i] - y[j])
    g = 0.5 * (v[i] - v[j]) ** 2
    top = float(max_dist if max_dist else np.percentile(h, 50))   # beyond half the field pairs get sparse
    keep = (h > 0) & (h <= top)
    if keep.sum() < 10:
        return np.array([]), np.array([]), np.array([])
    h, g = h[keep], g[keep]
    edges = np.linspace(0, top, n_lags + 1)
    which = np.clip(np.digitize(h, edges) - 1, 0, n_lags - 1)
    lag, semi, cnt = [], [], []
    for b in range(n_lags):
        m = which == b
        if m.sum() >= 5:
            lag.append(h[m].mean())
            semi.append(g[m].mean())
            cnt.append(int(m.sum()))
    return np.array(lag), np.array(semi), np.array(cnt)


def variogram_model(name, h, nugget, sill, rng):
    """Semivariance predicted by a fitted model at separation h."""
    h = np.asarray(h, float)
    partial = max(sill - nugget, 0.0)
    r = max(rng, 1e-9)
    if name == "Exponential":
        shape = 1 - np.exp(-3 * h / r)
    elif name == "Gaussian":
        shape = 1 - np.exp(-3 * (h / r) ** 2)
    else:                                                   # spherical
        t = np.clip(h / r, 0, 1)
        shape = np.where(h < r, 1.5 * t - 0.5 * t ** 3, 1.0)
    return np.where(h <= 0, 0.0, nugget + partial * shape)


def fit_variogram(lag, semi, cnt, name="Spherical"):
    """Least-squares fit of nugget, sill and range, weighted by how many pairs each lag holds."""
    from scipy.optimize import least_squares
    if not len(lag):
        return None
    s0 = float(np.mean(semi[-max(2, len(semi) // 3):]))
    p0 = [max(float(semi[0]) * 0.5, 1e-12), max(s0, 1e-12), float(lag[-1]) * 0.5]
    w = np.sqrt(np.asarray(cnt, float))

    def resid(p):
        nug, sill, rng = p
        return (variogram_model(name, lag, nug, max(sill, nug), rng) - semi) * w

    try:
        r = least_squares(resid, p0, bounds=([0, 1e-14, 1e-9], [max(s0, 1e-12) * 2, s0 * 4 + 1e-12,
                                                                float(lag[-1]) * 3]))
        nug, sill, rng = r.x
    except Exception:
        return None
    pred = variogram_model(name, lag, nug, max(sill, nug), rng)
    ss = float(np.sum((semi - pred) ** 2))
    tot = float(np.sum((semi - semi.mean()) ** 2)) or 1.0
    return {"model": name, "nugget": float(nug), "sill": float(max(sill, nug)), "range": float(rng),
            "fit": 1 - ss / tot, "lag": lag, "semi": semi, "count": cnt,
            "nugget_ratio": float(nug / max(sill, 1e-12))}


def ordinary_kriging(x, y, v, XI, YI, vg, neighbours=16, chunk=4096):
    """Ordinary kriging with a local neighbourhood. Returns (estimate, standard error)."""
    x, y, v = (np.asarray(a, float) for a in (x, y, v))
    pts = np.column_stack([x, y])
    k = int(min(max(neighbours, 2), len(pts)))
    tree = cKDTree(pts)
    targets = np.column_stack([XI.ravel(), YI.ravel()])
    out = np.empty(len(targets))
    err = np.empty(len(targets))
    name, nug, sill, rng = vg["model"], vg["nugget"], vg["sill"], vg["range"]
    for a in range(0, len(targets), chunk):
        b = min(a + chunk, len(targets))
        d0, idx = tree.query(targets[a:b], k=k)
        d0 = np.atleast_2d(d0.T).T
        idx = np.atleast_2d(idx.T).T
        P = pts[idx]                                        # (m, k, 2)
        hij = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)
        A = np.zeros((b - a, k + 1, k + 1))
        A[:, :k, :k] = variogram_model(name, hij, nug, sill, rng)
        A[:, :k, k] = 1.0
        A[:, k, :k] = 1.0
        rhs = np.zeros((b - a, k + 1))
        rhs[:, :k] = variogram_model(name, d0, nug, sill, rng)
        rhs[:, k] = 1.0
        A[:, np.arange(k), np.arange(k)] += 1e-9            # keep the system solvable at duplicate points
        try:
            sol = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:                       # a singular block: fall back node by node
            sol = np.stack([np.linalg.lstsq(A[m], rhs[m], rcond=None)[0] for m in range(b - a)])
        w = sol[:, :k]
        out[a:b] = (w * v[idx]).sum(1)
        var = (w * rhs[:, :k]).sum(1) + sol[:, k]
        err[a:b] = np.sqrt(np.maximum(var, 0))
    return out.reshape(XI.shape), err.reshape(XI.shape)


# ---------------------------------------------------------------- the one entry point
def grid(x, y, v, method=METHODS[0], cell=None, blank=None, power=2.0, neighbours=12,
         variogram="Spherical", clip_to_outline=True):
    """Scattered samples -> a grid, by the chosen method.

    `cell` is the grid interval in metres (None picks about a third of the sample spacing). `blank` is how
    far from a sample the grid is still drawn (None uses 1.5 sample spacings). Returns a dict with z,
    transform, cell, the mask, and for kriging the standard error and the fitted variogram.
    """
    from rasterio.transform import from_origin

    x, y, v = (np.asarray(a, float) for a in (x, y, v))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[ok], y[ok], v[ok]
    if len(v) < 4:
        raise ValueError("at least four samples are needed to interpolate")
    spacing = suggest_cell(x, y) * 3
    cell = float(cell) if cell else suggest_cell(x, y)
    reach = float(blank) if blank else 1.5 * spacing
    xs, ys = _axes(x, y, cell, pad=reach)
    XI, YI = np.meshgrid(xs, ys)

    out = {"cell": cell, "spacing": spacing, "reach": reach, "method": method,
           "transform": from_origin(xs[0] - cell / 2, ys[0] + cell / 2, cell, cell),
           "error": None, "variogram": None}
    if method == METHODS[1]:
        z = idw(x, y, v, XI, YI, power, neighbours)
    elif method == METHODS[0]:
        z = natural_neighbour(x, y, v, XI, YI, cell, max_radius=reach)
    elif method == METHODS[2]:
        lag, semi, cnt = experimental_variogram(x, y, v)
        names = VARIOGRAMS[1:] if variogram == VARIOGRAMS[0] else [variogram]
        tried = [f for f in (fit_variogram(lag, semi, cnt, nm) for nm in names) if f]
        if not tried:
            raise ValueError("the variogram could not be fitted; try another method")
        vg = max(tried, key=lambda f: f["fit"])
        z, err = ordinary_kriging(x, y, v, XI, YI, vg, neighbours)
        out["error"], out["variogram"] = err, vg
    else:
        g = geophysics.grid_points(x, y, v, cell=cell, bounds=(xs[0], xs[-1], ys[-1], ys[0]),
                                   method="Minimum curvature" if method == METHODS[3] else
                                   "Linear interpolation", line_aware=False)
        z = g["zf"]
        if z.shape != XI.shape:                             # geophysics sizes its own grid; resample to ours
            from scipy.ndimage import map_coordinates
            tr = g["transform"]
            cols = (XI - tr.c) / tr.a - 0.5
            rows = (YI - tr.f) / tr.e - 0.5
            z = map_coordinates(np.nan_to_num(z, nan=float(np.nanmean(z))), [rows, cols], order=1,
                                mode="nearest")

    dist, _ = cKDTree(np.column_stack([x, y])).query(np.column_stack([XI.ravel(), YI.ravel()]))
    mask = (dist <= reach).reshape(XI.shape)
    if clip_to_outline:
        mask &= geophysics.survey_outline(x, y, XI, YI, reach)
    out["mask"] = mask
    out["z"] = np.where(mask, z, np.nan)
    if out["error"] is not None:
        out["error"] = np.where(mask, out["error"], np.nan)
    return out


def variogram_figure(vg, unit="", title="Variogram"):
    """The fitted variogram: how fast the assays stop resembling each other with distance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.plot(vg["lag"], vg["semi"], "o", ms=5, color="#333", label="measured")
    h = np.linspace(0, vg["lag"][-1] * 1.05, 300)
    ax.plot(h, variogram_model(vg["model"], h, vg["nugget"], vg["sill"], vg["range"]), "-", lw=2,
            color="#c0392b", label=f"{vg['model']} model (fit {vg['fit']:.2f})")
    ax.axhline(vg["sill"], color="#999", ls=":", lw=1)
    ax.axvline(vg["range"], color="#2471a3", ls="--", lw=1.4, label=f"range {vg['range']:,.0f} m")
    ax.plot(0, vg["nugget"], "s", color="#7d3c98", ms=7, label=f"nugget {vg['nugget']:.3g}")
    ax.set_xlabel("Separation between samples (m)")
    ax.set_ylabel(f"Semivariance{f' ({unit})' if unit else ''}")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
