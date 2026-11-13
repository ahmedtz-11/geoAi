"""2-D inversion of induced-polarisation (chargeability) data along one survey line.

What Res2DInv calls the "inverse model chargeability section": the subsurface under the line is divided
into rectangular blocks and the chargeability of every block is found so that the calculated apparent
chargeabilities match the measured ones, with a smoothness constraint (Loke & Barker, 1995, 1996).

Apparent chargeability is, to first order, the sensitivity-weighted average of the block chargeabilities
(Seigel, 1959; Oldenburg & Li, 1994):   ma_i = sum_j S_ij m_j ,   S_ij = d ln(rho_a,i) / d ln(rho_j).
The sensitivities used here are those of a homogeneous half-space (the Frechet derivative integrated along
strike), the same starting approximation as Loke & Barker's quasi-Newton method. Res2DInv goes further: it
inverts resistivity as well and recomputes the sensitivities for that resistivity model by finite elements,
so its sections differ from these where the ground resistivity is strongly non-uniform.

No Streamlit inside.
"""
import numpy as np
from scipy.optimize import lsq_linear
from scipy.sparse import coo_matrix, vstack

REMOTE = 1e6

# Res2DInv colour scale (17 colours, low -> high)
RES2D_COLORS = ["#000080", "#0000b4", "#0000ff", "#0050ff", "#00a0ff", "#00f0ff", "#00c8a0", "#00e000", "#008000",
                "#80c000", "#ffff00", "#c08000", "#ff8000", "#ff0000", "#c00000", "#800040", "#800080"]


# ---------- sensitivities ----------
def _gl(n, lo, hi):
    t, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (hi - lo) * t + 0.5 * (hi + lo), 0.5 * (hi - lo) * w


_TH, _WTH = _gl(20, 0.0, np.pi / 2)


def frechet_2d(x, z, s):
    """Frechet derivative of the pole-pole potential (current electrode at 0, potential electrode at s, both on
    the surface) for a homogeneous half-space, integrated along strike. x, z: arrays of the same shape."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    r1, r2 = np.hypot(x, z), np.hypot(x - s, z)
    scale = np.maximum(np.minimum(r1, r2), 1e-9)[..., None]         # width of the kernel along strike
    y = scale * np.tan(_TH)
    jac = scale / np.cos(_TH) ** 2
    y2 = y * y
    a = (x * x + z * z)[..., None] + y2
    b = ((x - s) ** 2 + z * z)[..., None] + y2
    f = ((x * (x - s) + z * z)[..., None] + y2) / (a * b) ** 1.5
    return 2 * (f * jac * _WTH).sum(-1) / (4 * np.pi ** 2)


def _template(s, dx, ncol, zedges):
    """Block-integrated kernel for separation s on cells k = -ncol .. ncol-1 (current electrode on the edge
    between cells -1 and 0). Returns array (nlayer, 2*ncol)."""
    out = np.zeros((len(zedges) - 1, 2 * ncol))
    k = np.arange(-ncol, ncol)
    for j in range(len(zedges) - 1):
        nq = 4 if j < 2 else 3 if j < 5 else 2
        ux, wx = _gl(nq, 0.0, dx)
        uz, wz = _gl(nq, zedges[j], zedges[j + 1])
        X = (k[:, None, None] * dx + ux[None, :, None]) + 0 * uz[None, None, :]
        Z = 0 * X + uz[None, None, :]
        out[j] = (frechet_2d(X, Z, s) * wx[None, :, None] * wz[None, None, :]).sum((1, 2))
    return out


def model_grid(c1, c2, p1, p2, pseudo_depth, unit=None, pad=None):
    """Blocks under the line: width = unit electrode spacing, layers thickening with depth."""
    el = np.concatenate([c1, c2, p1, p2])
    el = np.unique(np.round(el[np.isfinite(el) & (np.abs(el) < REMOTE)], 6))
    if unit is None:
        d = np.diff(el)
        d = d[d > 1e-6]
        unit = float(np.round(np.min(d), 6)) if d.size else 1.0
        if np.ptp(el) / unit > 160:                          # keep the model a sensible size
            unit *= np.ceil(np.ptp(el) / unit / 160)
    pz = np.sort(np.asarray(pseudo_depth, float)[np.isfinite(pseudo_depth)])
    zmax = 1.1 * float(pz[-min(3, len(pz))])              # a single deepest reading does not set the depth
    for growth in (1.1, 1.15, 1.25, 1.4):
        t, edges = 0.5 * unit, [0.0]
        while edges[-1] < zmax:
            edges.append(edges[-1] + t)
            t *= growth
        if len(edges) - 1 <= 18:
            break
    nshow = len(edges) - 1
    for _ in range(3):                                   # deeper padding layers (not displayed)
        edges.append(edges[-1] + 1.6 * (edges[-1] - edges[-2]))
    x0, x1 = el.min(), el.max()
    pad = pad if pad is not None else int(max(6, np.ceil(edges[nshow] / unit)))
    ncol = int(round((x1 - x0) / unit)) + 2 * pad
    return {"unit": float(unit), "x0": float(x0 - pad * unit), "ncol": ncol, "pad": pad,
            "zedges": np.array(edges), "nshow": nshow}


def sensitivity(c1, c2, p1, p2, grid):
    """Normalised sensitivity matrix (readings x blocks): each row sums to 1."""
    dx, x0, ncol, zed = grid["unit"], grid["x0"], grid["ncol"], grid["zedges"]
    nl = len(zed) - 1
    J = np.zeros((len(p1), nl * ncol))
    total = np.zeros(len(p1))
    cache = {}
    for cur, pot, sign in ((c1, p1, 1), (c1, p2, -1), (c2, p1, -1), (c2, p2, 1)):
        ok = np.isfinite(cur) & np.isfinite(pot) & (np.abs(cur) < REMOTE) & (np.abs(pot) < REMOTE)
        ia = np.where(ok, np.round((np.where(ok, cur, x0) - x0) / dx), 0).astype(int)
        sep = np.where(ok, np.round((np.where(ok, pot - cur, 0.0)) / dx, 0) * dx, 0.0)
        for i in np.flatnonzero(ok):
            s = float(sep[i])
            if s == 0:
                continue
            if s not in cache:
                cache[s] = _template(s, dx, ncol, zed)
            T = cache[s]
            lo = ncol - ia[i]                              # template column of model column 0
            J[i] += sign * T[:, lo:lo + ncol].ravel()
            total[i] += sign / (2 * np.pi * abs(s))
    total = np.where(np.abs(total) < 1e-12, np.nan, total)
    J /= total[:, None]
    captured = J.sum(1)                                    # share of the half-space inside the model (<= 1)
    J /= np.where(np.abs(captured) > 1e-6, captured, 1.0)[:, None]
    return J, captured


# ---------- inversion ----------
def _roughness(nl, ncol, zedges, dx):
    idx = np.arange(nl * ncol).reshape(nl, ncol)
    rows = []

    def diff(a, b, w):
        n = a.size
        r = np.arange(n)
        return coo_matrix((np.r_[w * np.ones(n), -w * np.ones(n)], (np.r_[r, r], np.r_[a.ravel(), b.ravel()])),
                          shape=(n, nl * ncol))
    lw = 1.05 ** np.arange(nl)                             # damping grows with depth, as in Res2DInv
    rows.append(diff(idx[:, :-1], idx[:, 1:], np.repeat(lw, ncol - 1)))
    rows.append(diff(idx[:-1, :], idx[1:, :], np.repeat(lw[:-1], ncol)))
    return vstack(rows).tocsr()


def invert(c1, c2, p1, p2, m_app, pseudo_x, pseudo_depth, unit=None, damping=0.02, max_iter=6, robust=True,
           log=False):
    """Smoothness-constrained inversion of apparent chargeability, or of apparent resistivity with log=True.

    Resistivity is inverted as logarithms: ln(rho_a) is, to first order, the same sensitivity-weighted average
    of ln(rho) of the blocks (Loke & Barker, 1995), so one solver serves both. Its error is in per cent.

    Returns dict(model (nshow x ncol_shown), xedges, zedges, mask, iteration, abs_error, rms_error, history,
    unit, calculated, captured)."""
    c1, c2, p1, p2, d = (np.asarray(a, float) for a in (c1, c2, p1, p2, m_app))
    if log:
        d = np.log(np.where(d > 0, d, np.nan))
    good = np.isfinite(d) & np.isfinite(p1) & np.isfinite(p2)
    c1, c2, p1, p2, d = c1[good], c2[good], p1[good], p2[good], d[good]
    px, pz = np.asarray(pseudo_x, float)[good], np.asarray(pseudo_depth, float)[good]
    if len(d) < 6:
        raise ValueError("Too few readings to invert.")
    grid = model_grid(c1, c2, p1, p2, pz, unit)
    J, captured = sensitivity(c1, c2, p1, p2, grid)
    use = np.isfinite(J).all(1)
    J, d, px, pz = J[use], d[use], px[use], pz[use]
    nl, ncol = len(grid["zedges"]) - 1, grid["ncol"]
    C = _roughness(nl, ncol, grid["zedges"], grid["unit"])
    CtC_scale = float(C.multiply(C).sum())
    m = np.full(nl * ncol, float(np.median(d)))
    lo_b, hi = (float(d.min()) - 2.3, float(d.max()) + 2.3) if log else \
        (0.0, max(4 * float(np.percentile(d, 99)), 1e-6))
    w = np.ones(len(d))
    history, best = [], None
    for it in range(1, max_iter + 1):
        lam = max(damping / 2 ** (it - 1), damping / 5)
        Jw = J * w[:, None]
        lam_abs = lam * float((J * J).sum()) / CtC_scale * 50
        A = np.vstack([Jw, np.sqrt(lam_abs) * C.toarray()])
        b = np.r_[d * w, np.zeros(C.shape[0])]
        m = lsq_linear(A, b, bounds=(lo_b, hi), method="trf", lsmr_tol="auto", max_iter=60).x
        res = d - J @ m
        err = 100 * (np.exp(-res) - 1) if log else res        # per cent of the measured value for resistivity
        abs_err, rms = float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))
        history.append({"iteration": it, "abs_error": abs_err, "rms_error": rms})
        if best is None or abs_err < best[1]:
            best = (it, abs_err, rms, m.copy(), res.copy())
        if it > 1 and (history[-2]["abs_error"] - abs_err) < 0.05 * history[-2]["abs_error"]:
            break                                           # Res2DInv's 5 % convergence limit
        if robust:                                          # readings that will not fit count less (Huber)
            cut = 2.5 * 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
            w = np.minimum(1.0, np.sqrt(cut / np.maximum(np.abs(res), 1e-12)))
    it, abs_err, rms, m, res = best
    pad, ns = grid["pad"], grid["nshow"]
    model = m.reshape(nl, ncol)[:ns, pad:ncol - pad]
    back = np.exp if log else (lambda a: a)
    model = back(model)
    xedges = grid["x0"] + (pad + np.arange(ncol - 2 * pad + 1)) * grid["unit"]
    zedges = grid["zedges"][:ns + 1]
    # blocks under the data (the trapezoid Res2DInv draws)
    xc, zc = 0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (zedges[1:] + zedges[:-1])
    mask = np.zeros(model.shape, bool)
    lo = hi_x = None
    for j in range(ns):
        deep = pz >= zc[j]
        if j and deep.sum() < 3:                           # too few readings this deep to show a block
            break
        l, r = (px[deep].min(), px[deep].max()) if j else (px.min() - grid["unit"], px.max() + grid["unit"])
        l, r = (l, r) if lo is None else (max(l, lo), min(r, hi_x))
        lo, hi_x = l, r
        mask[j] = (xc >= l - 0.5 * grid["unit"]) & (xc <= r + 0.5 * grid["unit"])
    return {"model": model, "xedges": xedges, "zedges": zedges, "mask": mask, "iteration": it, "abs_error": abs_err,
            "rms_error": rms, "history": history, "unit": grid["unit"], "calculated": back(d - res), "measured": back(d),
            "log": bool(log), "captured": float(np.nanmedian(captured)), "n": int(len(d))}


def anomalies(inv, threshold):
    """Strongest chargeable zone in the model: where, how deep, peak value."""
    m = np.where(inv["mask"], inv["model"], np.nan)
    if not np.isfinite(m).any():
        return None
    xc = 0.5 * (inv["xedges"][1:] + inv["xedges"][:-1])
    j, i = np.unravel_index(np.nanargmax(m), m.shape)
    hot = m >= threshold
    if not hot[j, i]:
        return {"peak": float(m[j, i]), "peak_x": float(xc[i]), "peak_depth": float(0.5 * (inv["zedges"][j] + inv["zedges"][j + 1])),
                "zone": None}
    from scipy.ndimage import label
    lab, _ = label(hot)
    z = lab == lab[j, i]
    rows, cols = np.where(z)
    return {"peak": float(m[j, i]), "peak_x": float(xc[i]),
            "peak_depth": float(0.5 * (inv["zedges"][j] + inv["zedges"][j + 1])),
            "zone": (float(inv["xedges"][cols.min()]), float(inv["xedges"][cols.max() + 1]),
                     float(inv["zedges"][rows.min()]), float(inv["zedges"][rows.max() + 1])),
            "open_at_depth": bool(rows.max() == m.shape[0] - 1)}


# ---------- figure ----------
def levels_for(models, n=16):
    """Contour values shared by several sections. Linear intervals, or logarithmic ones (a Res2DInv option)
    when a few strong values would otherwise leave the rest of the section one colour."""
    v = np.concatenate([np.where(i["mask"], i["model"], np.nan).ravel() for i in models])
    v = v[np.isfinite(v)]
    lo, hi = np.percentile(v, [2, 99.5])
    hi = max(hi, 1e-6)
    lo = max(lo, hi / 100)
    if hi <= lo * 1.05:
        hi = lo * 1.5
    if any(i.get("log") for i in models):                  # resistivity: logarithmic intervals, as in Res2DInv
        return np.geomspace(lo, hi, n)
    return np.geomspace(lo, hi, n) if hi / lo > 8 else np.linspace(lo, hi, n)


def extremes(inv):
    """Most conductive and most resistive block under the line: (value, x, depth) each."""
    m = np.where(inv["mask"], inv["model"], np.nan)
    xc = 0.5 * (inv["xedges"][1:] + inv["xedges"][:-1])
    zc = 0.5 * (inv["zedges"][1:] + inv["zedges"][:-1])
    out = []
    for f in (np.nanargmin, np.nanargmax):
        j, i = np.unravel_index(f(m), m.shape)
        out.append((float(m[j, i]), float(xc[i]), float(zc[j])))
    return out


def figure(inv, title_line="", unit=None, levels=None, threshold=None):
    """The section drawn the way Res2DInv draws it."""
    resist = bool(inv.get("log"))
    what = "Resistivity" if resist else "Chargeability"
    unit = unit or ("ohm.m" if resist else "mV/V")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Polygon, Rectangle

    xe, ze, model, mask = inv["xedges"], inv["zedges"], inv["model"], inv["mask"]
    xc, zc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ze[1:] + ze[:-1])
    levels = levels_for([inv]) if levels is None else np.asarray(levels)
    cmap = ListedColormap(RES2D_COLORS[1:-1]).with_extremes(under=RES2D_COLORS[0], over=RES2D_COLORS[-1])
    norm = BoundaryNorm(levels, cmap.N)

    # contour the block-centre values, padded to the top, bottom and sides of the model
    X = np.r_[xe[0], xc, xe[-1]]
    Z = np.r_[0.0, zc, ze[-1]]
    V = np.pad(model, 1, mode="edge")
    # refine between block centres so the contours curve as they do in Res2DInv
    from scipy.ndimage import gaussian_filter, zoom
    k = 8
    V = gaussian_filter(zoom(V, k, order=1, mode="nearest"), 0.45 * k, mode="nearest")   # no overshoot
    X = np.interp(np.linspace(0, len(X) - 1, V.shape[1]), np.arange(len(X)), X)
    Z = np.interp(np.linspace(0, len(Z) - 1, V.shape[0]), np.arange(len(Z)), Z)
    font = {"family": "monospace", "weight": "bold", "size": 9}
    width = 13.0
    fig = plt.figure(figsize=(width, 3.9))
    ax = fig.add_axes([0.06, 0.34, 0.90, 0.50])
    cs = ax.contourf(X, Z, V, levels=levels, cmap=cmap, norm=norm, extend="both")
    # trapezoid of blocks that have data above them
    left = [(xe[np.flatnonzero(mask[j])[0]], ze[j]) for j in range(len(zc)) if mask[j].any()]
    right = [(xe[np.flatnonzero(mask[j])[-1] + 1], ze[j]) for j in range(len(zc)) if mask[j].any()]
    rows = [j for j in range(len(zc)) if mask[j].any()]
    poly = [left[0]]
    for (x, z), j in zip(left, rows):
        poly += [(x, z), (x, ze[j + 1])]
    for (x, z), j in zip(right[::-1], rows[::-1]):
        poly += [(x, ze[j + 1]), (x, z)]
    # smooth the staircase into the sloping sides Res2DInv shows
    pl = [(left[0][0], 0.0)] + [(x, 0.5 * (ze[j] + ze[j + 1])) for (x, _), j in zip(left, rows)][1:] + \
         [(left[-1][0], ze[rows[-1] + 1])]
    pr = [(right[0][0], 0.0)] + [(x, 0.5 * (ze[j] + ze[j + 1])) for (x, _), j in zip(right, rows)][1:] + \
         [(right[-1][0], ze[rows[-1] + 1])]
    clip = Polygon(pl + pr[::-1], closed=True, transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    cs.set_clip_path(clip)
    if threshold is not None and np.nanmax(np.where(mask, model, np.nan)) >= threshold:
        ct = ax.contour(X, Z, V, levels=[threshold], colors="k", linewidths=1.3)
        ct.set_clip_path(clip)
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ze[rows[-1] + 1], 0)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", labelsize=8, direction="out", length=4)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily("monospace")
        lab.set_fontweight("bold")
    pick = np.unique(np.linspace(0, rows[-1], min(6, rows[-1] + 1)).round().astype(int))
    ax.set_yticks(zc[pick])
    ax.set_yticklabels([f"{v:.3g}" if v < 10 else f"{v:.1f}" for v in zc[pick]])
    ax.set_xticks(np.arange(len(xe))[::max(1, len(xe) // 60)] * inv["unit"] + xe[0], minor=True)
    ax.tick_params(axis="x", which="minor", length=2)
    for sp in ("bottom", "right"):
        ax.spines[sp].set_visible(False)
    fig.text(0.012, 0.93, "Depth", **font)
    fig.text(0.06, 0.93, f"Iteration {inv['iteration']} Abs. error = {inv['abs_error']:.2g}" + (" %" if resist else ""), **font)
    fig.text(0.975, 0.875, "m.", ha="right", **font)
    if title_line:
        fig.text(0.96, 0.93, title_line, ha="right", **font)
    fig.text(0.06, 0.255, f"Inverse Model {what} Section", **font)
    # colour blocks
    lg = fig.add_axes([0.02, 0.13, 0.50, 0.075])
    cols = RES2D_COLORS
    for k, c in enumerate(cols):
        lg.add_patch(Rectangle((k, 0), 0.82, 1, facecolor=c, edgecolor="k", linewidth=0.6))
    lg.set_xlim(-0.1, len(cols))
    lg.set_ylim(0, 1)
    lg.axis("off")
    for k in range(0, len(levels), 2):                      # value at the boundary to the right of block k
        v = levels[k]
        fig.text(0.02 + 0.50 * (k + 0.91 + 0.1) / (len(cols) + 0.1), 0.075,
                 f"{v:.3g}" if abs(v) < 100 else f"{v:.0f}", ha="center", **font)
    fig.text(0.17, 0.015, f"{what} in {unit}.", **font)
    fig.text(0.56, 0.015, f"Unit electrode spacing is {inv['unit']:g} m.", **font)
    return fig
