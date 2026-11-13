"""Magnetic and IP (induced polarisation) processing and interpretation.

Magnetics: gridding, FFT derivatives, reduction filters (residual, analytic signal, tilt, horizontal
gradient, first vertical derivative), structural edges, depth from Euler deconvolution and from the
radially averaged power spectrum.
IP: plan maps of chargeability / resistivity, and pseudosections (station x depth or n-level).
"""
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

SURVEYS = ["Magnetic (TMI, nT)", "IP chargeability (mV/V)", "IP resistivity (ohm·m)"]
STRUCTURAL_INDEX = {"0 · contact / fault (thick)": 0.0, "0.5 · thick step / sill edge": 0.5,
                    "1 · dyke, sill, thin vein": 1.0, "2 · pipe / cylinder": 2.0, "3 · sphere / point": 3.0}


# ---------- gridding ----------
# Minimum-curvature stencils: the surface that satisfies these is the smoothest one through the data.
_K_BIHARMONIC = np.array([[0, 0, -1, 0, 0], [0, -2, 8, -2, 0], [-1, 8, 0, 8, -1],
                          [0, -2, 8, -2, 0], [0, 0, -1, 0, 0]], float) / 20.0
_K_LAPLACE = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float) / 4.0
GRID_METHODS = ["Minimum curvature", "Linear interpolation"]


def survey_geometry(x, y):
    """Layout of a survey: station spacing along the lines, spacing between lines, and line direction.

    Line surveys (ground magnetics, airborne) are sampled densely along lines and sparsely across them,
    so the two spacings differ and the grid cell must follow the LINE spacing, not the station spacing.

    The line direction comes from a local PCA over each reading's few nearest neighbours. Using only
    the single nearest neighbour fails on walked ground surveys, where the GPS scatter (about a metre)
    is as large as the step between readings, so the nearest neighbour points in a random direction.
    The lines are then found as peaks in the density of the across-line coordinate, which still works
    when the operator wandered and the lines are not evenly spaced.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    n = len(pts)
    k = min(6, n - 1)
    d, idx = cKDTree(pts).query(pts, k=k + 1)
    station = float(np.median(d[:, 1])) or 1.0
    out = {"station": station, "line_spacing": station, "azimuth": 0.0, "n_lines": 1,
           "concentration": 0.0, "regularity": 0.0, "is_line_survey": False}
    if n < 20:
        return out

    nb = pts[idx] - pts[idx].mean(1, keepdims=True)         # each reading's neighbourhood, centred
    w, V = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb) / k)
    straight = (w[:, 1] - w[:, 0]) / np.maximum(w[:, 1], 1e-12)   # 1 = neighbours in a straight line
    a2 = 2 * np.arctan2(V[:, 1, 1], V[:, 0, 1])             # doubled angles: directions 180° apart agree
    c = float((np.cos(a2) * straight).sum() / straight.sum())
    s = float((np.sin(a2) * straight).sum() / straight.sum())
    az = np.arctan2(s, c) / 2
    out["azimuth"] = float(np.degrees(az) % 180)
    out["concentration"] = float(np.hypot(c, s))            # 1 = every traverse in the same direction
    if out["concentration"] < 0.3:
        return out                                          # scattered samples or a square grid

    across = pts @ np.array([-np.sin(az), np.cos(az)])
    span = float(np.ptp(across))
    if span < 4 * station:
        return out                                          # a single line
    nbins = 4000
    hist, _ = np.histogram(across, bins=nbins)
    width = span / nbins
    best = None
    for smooth in (0.5, 1.0, 2.0):                          # in units of the station spacing
        sigma = max(smooth * station / width, 1.0)
        # The outermost lines sit at the ends of the range, where a peak finder sees no peak unless
        # there is empty ground beyond them, so pad well past the width of the smoothing.
        pad = int(4 * sigma) + 2
        dens = gaussian_filter1d(np.r_[np.zeros(pad), hist, np.zeros(pad)], sigma)
        peaks, _ = find_peaks(dens, prominence=0.25 * dens.std())
        if len(peaks) >= 3 and (best is None or len(peaks) > len(best)):
            best = float(np.min(across)) + (peaks - pad + 0.5) * width
    if best is None:
        return out
    # refine: give every reading to its nearest peak, then put each line through its own readings
    which = np.abs(across[:, None] - best[None, :]).argmin(1) if len(best) * n < 2e7 else None
    if which is not None:
        centres = np.array([across[which == j].mean() for j in range(len(best)) if (which == j).any()])
        if len(centres) >= 3:
            best = centres
    gaps = np.diff(best)
    spacing = float(np.median(gaps))
    out.update(line_spacing=spacing, n_lines=int(len(best)),
               regularity=float(np.clip(1 - np.std(gaps) / max(spacing, 1e-9), 0, 1)))
    out["is_line_survey"] = bool(spacing > 1.8 * station and 2 < len(best) < n / 3)
    return out


def despike(x, y, v, neighbours=10, tol=6.0):
    """Flag readings that disagree with their own neighbourhood (Geosoft's non-linear filter).

    A magnetometer dropped, swung, or carried past a fence or a piece of iron records jumps of thousands
    of nT between readings a metre apart, which no rock can produce. Each reading is compared with the
    median of its nearest neighbours; anything more than `tol` robust standard deviations away is
    dropped. Minimum curvature has to honour every reading it is given, so one spike pulls a bullseye
    into the grid and its derivatives, and widens the colour range until the real field looks flat.

    Returns (keep mask, robust standard deviation of the residuals).
    """
    x, y, v = (np.asarray(a, float) for a in (x, y, v))
    n = len(v)
    if n < 3 * neighbours or tol <= 0:
        return np.ones(n, bool), 0.0
    k = min(neighbours, n - 1)
    _, idx = cKDTree(np.column_stack([x, y])).query(np.column_stack([x, y]), k=k + 1)
    res = v - np.median(v[idx[:, 1:]], axis=1)          # neighbours only, not the reading itself
    sigma = 1.4826 * float(np.median(np.abs(res - np.median(res))))
    if not sigma > 0:
        return np.ones(n, bool), 0.0
    return np.abs(res - np.median(res)) <= tol * sigma, sigma


def survey_outline(x, y, XI, YI, reach, max_points=60000):
    """Which grid cells lie inside the ground the survey actually walked (a concave hull).

    Delaunay triangulation of the readings, keeping the triangles small enough to be real ground
    (circumradius within `reach`). Long thin triangles spanning the outside of the survey are dropped,
    which leaves the ragged survey boundary instead of a convex box around it.
    """
    from scipy.spatial import Delaunay

    pts = np.column_stack([x, y])
    if len(pts) > max_points:                       # the outline does not need every reading
        pts = pts[np.linspace(0, len(pts) - 1, max_points).astype(int)]
    try:
        tri = Delaunay(pts)
    except Exception:
        return np.ones(XI.shape, bool)
    t = pts[tri.simplices]
    a = np.linalg.norm(t[:, 1] - t[:, 0], axis=1)
    b = np.linalg.norm(t[:, 2] - t[:, 1], axis=1)
    c = np.linalg.norm(t[:, 0] - t[:, 2], axis=1)
    e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    circum = a * b * c / np.maximum(4 * area, 1e-12)
    keep = circum <= reach
    if not keep.any():
        return np.ones(XI.shape, bool)
    simplex = tri.find_simplex(np.column_stack([XI.ravel(), YI.ravel()]))
    inside = (simplex >= 0) & keep[np.clip(simplex, 0, None)]
    return inside.reshape(XI.shape)


def minimum_curvature(x, y, v, x0, y1, cell, nx, ny, tension=0.25, iterations=1500, tol=1e-6, omega=0.5):
    """Briggs (1974) minimum-curvature grid: the smoothest surface that passes through the readings.

    Fills the whole rectangle, including between survey lines, so there are no gaps or stripes.
    Solved by damped Jacobi relaxation of the biharmonic stencil (omega < 0.625 keeps it stable),
    blended with the Laplace stencil by `tension` to stop overshoot beside steep anomalies.
    """
    from scipy import ndimage

    col = np.rint((np.asarray(x, float) - x0) / cell).astype(int)
    row = np.rint((y1 - np.asarray(y, float)) / cell).astype(int)
    v = np.asarray(v, float)
    ok = (col >= 0) & (col < nx) & (row >= 0) & (row < ny) & np.isfinite(v)
    if not ok.any():
        raise ValueError("No readings fall inside the grid.")
    flat = row[ok] * nx + col[ok]
    total = np.bincount(flat, weights=v[ok], minlength=nx * ny)
    count = np.bincount(flat, minlength=nx * ny)
    data = np.full(nx * ny, np.nan)
    nz = count > 0
    data[nz] = total[nz] / count[nz]                 # several readings in one cell: average them
    data = data.reshape(ny, nx)
    known = np.isfinite(data)
    ind = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    z = data[tuple(ind)].astype(float)               # start from nearest-neighbour, then relax
    scale = max(float(np.nanstd(data[known])), 1e-12)
    for _ in range(iterations):
        jac = ((1 - tension) * ndimage.convolve(z, _K_BIHARMONIC, mode="reflect")
               + tension * ndimage.convolve(z, _K_LAPLACE, mode="reflect"))
        new = z + omega * (jac - z)
        new[known] = data[known]
        change = float(np.abs(new - z).max())
        z = new
        if change < tol * scale:
            break
    return z, known


def point_spacing(x, y):
    pts = np.column_stack([x, y])
    if len(pts) < 3:
        return 1.0
    d, _ = cKDTree(pts).query(pts, k=2)
    s = float(np.median(d[:, 1]))
    return s if s > 0 else 1.0


def is_collinear(x, y, tol=1e-3):
    """True when the points are (nearly) on one straight line, e.g. a single survey line."""
    p = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    p = p[np.all(np.isfinite(p), 1)]
    if len(p) < 3:
        return True
    s = np.linalg.svd(p - p.mean(0), compute_uv=False)
    return s[0] == 0 or s[-1] / s[0] < tol


def grid_points(x, y, v, cell=None, max_cells=300, mask_factor=None, bounds=None,
                method=GRID_METHODS[0], coverage_factor=0.75, tension=0.25, line_aware=True):
    """Scattered points -> north-up grid.

    The cell size follows the LINE spacing for a line survey (a quarter of it, the usual 1/3-1/5 rule),
    otherwise the station spacing. Minimum curvature then fills the whole survey smoothly.

    Returns dict(z (NaN outside the survey), zf (filled everywhere, used by the FFT filters),
    valid (inside the survey), coverage (close to an actual reading), transform, cell, geometry).
    """
    from rasterio.transform import from_origin

    x, y, v = (np.asarray(a, float) for a in (x, y, v))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[ok], y[ok], v[ok]
    if len(x) < 4:
        raise ValueError("At least 4 valid points are needed to make a grid.")
    if is_collinear(x, y):
        raise ValueError("The points lie on a single line, so they can't be gridded into a map.")
    geom = survey_geometry(x, y)
    sp, line_sp = geom["station"], geom["line_spacing"]
    x0, x1, y0, y1 = bounds or (x.min(), x.max(), y.min(), y.max())
    extent = max(x1 - x0, y1 - y0, 1e-9)
    # line_aware=False sizes the cell from the station spacing only (the original "Normal" image)
    want = line_sp / 4 if geom["is_line_survey"] and line_aware else sp / 1.5
    cell = cell or max(want, extent / max_cells)
    nx, ny = int(np.ceil((x1 - x0) / cell)) + 1, int(np.ceil((y1 - y0) / cell)) + 1
    xi = x0 + np.arange(nx) * cell
    yi = y1 - np.arange(ny) * cell
    XI, YI = np.meshgrid(xi, yi)

    if method == GRID_METHODS[0]:
        zf, _ = minimum_curvature(x, y, v, x0, y1, cell, nx, ny, tension=tension)
    else:
        lin = griddata((x, y), v, (XI, YI), method="linear")
        near = griddata((x, y), v, (XI, YI), method="nearest")
        zf = np.where(np.isnan(lin), near, lin)

    dist, _ = cKDTree(np.column_stack([x, y])).query(np.column_stack([XI.ravel(), YI.ravel()]))
    dist = dist.reshape(XI.shape)
    # inside the survey: far enough to bridge the gap between lines, not far enough to invent ground.
    # Lines walked by hand are not evenly spaced, so widen the reach when the spacing varies.
    reach = (1.05 + 0.8 * (1 - geom.get("regularity", 1.0))) * line_sp
    blank = mask_factor * max(sp, cell) if mask_factor else \
        (reach if geom["is_line_survey"] else 2.5 * max(sp, cell))
    valid = dist <= blank
    if mask_factor is None:
        # A distance mask alone also reaches OUTSIDE the survey, where there is nothing on the far side
        # to constrain the surface and minimum curvature runs away into false highs and lows. Clipping to
        # the survey outline keeps the fill between the lines and stops it at the edge, as Oasis montaj does.
        valid &= survey_outline(x, y, XI, YI, blank)
    coverage = dist <= coverage_factor * max(cell, sp)   # where readings actually are
    z = np.where(valid, zf, np.nan)
    return {"z": z.astype(np.float32), "zf": zf.astype(np.float64), "valid": valid, "coverage": coverage,
            "transform": from_origin(x0 - cell / 2, y1 + cell / 2, cell, cell), "cell": float(cell),
            "spacing": sp, "geometry": geom, "X": XI, "Y": YI}


def grid_from_raster(arr, transform):
    """Wrap an existing grid (raster / georeferenced image) in the same structure as grid_points."""
    a = np.asarray(arr, float)
    valid = np.isfinite(a)
    if not valid.any():
        raise ValueError("The grid has no valid values.")
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    zf = a[tuple(idx)]
    h, w = a.shape
    cell = abs(transform.a)
    XI, YI = np.meshgrid(transform.c + cell * (np.arange(w) + 0.5), transform.f - cell * (np.arange(h) + 0.5))
    return {"z": np.where(valid, a, np.nan).astype(np.float32), "zf": zf, "valid": valid, "coverage": valid,
            "transform": transform, "cell": float(cell), "spacing": float(cell),
            "geometry": {"is_line_survey": False, "station": float(cell), "line_spacing": float(cell)},
            "X": XI, "Y": YI}


# ---------- FFT tools ----------
def _prep(zf, pad_frac=0.25):
    h, w = zf.shape
    yy, xx = np.mgrid[0:h, 0:w]
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(h * w)])
    coef, *_ = np.linalg.lstsq(A, zf.ravel(), rcond=None)
    plane = (A @ coef).reshape(h, w)
    py, px = int(h * pad_frac) + 2, int(w * pad_frac) + 2
    padded = np.pad(zf - plane, ((py, py), (px, px)), mode="reflect")
    return padded, plane, (py, px), coef


def _k(shape, cell):
    ky = 2 * np.pi * np.fft.fftfreq(shape[0], d=cell)
    kx = 2 * np.pi * np.fft.fftfreq(shape[1], d=cell)
    KX, KY = np.meshgrid(kx, ky)
    return KX, KY, np.hypot(KX, KY)


def _filter(zf, cell, fn):
    padded, plane, (py, px), coef = _prep(zf)
    F = np.fft.fft2(padded)
    KX, KY, K = _k(padded.shape, cell)
    out = np.real(np.fft.ifft2(fn(F, KX, KY, K)))
    return out[py:py + zf.shape[0], px:px + zf.shape[1]], plane, coef


def derivatives(zf, cell):
    """dT/dx (east), dT/dy (north), dT/dz (z positive DOWN, the Euler convention)."""
    dx, _, coef = _filter(zf, cell, lambda F, KX, KY, K: 1j * KX * F)
    drow, _, _ = _filter(zf, cell, lambda F, KX, KY, K: 1j * KY * F)
    dz, _, _ = _filter(zf, cell, lambda F, KX, KY, K: K * F)
    dx = dx + coef[0] / cell  # add back the removed linear trend
    dy = -(drow + coef[1] / cell)  # rows increase southwards
    return dx, dy, dz


def upward(zf, cell, height):
    out, plane, _ = _filter(zf, cell, lambda F, KX, KY, K: F * np.exp(-K * height))
    return out + plane


def microlevel(zf, cell, azimuth, line_spacing, along=12.0, across=4.0):
    """Decorrugation (Minty, 1991): remove the stripes left by line-to-line level differences.

    Each survey line is walked at a different time, so diurnal drift and instrument offsets leave every
    line at a slightly different level. Gridded, that shows as corrugation running along the lines, which
    the derivative filters then amplify into false structure. The corrugation is what is left after
    low-passing ALONG the lines (`along` x the line spacing) and high-passing ACROSS them (`across` x the
    line spacing); subtracting it levels the survey. Returns (levelled field, correction).
    """
    a = np.radians(azimuth)
    ka = 2 * np.pi / max(along * line_spacing, 1e-9)
    kc = 2 * np.pi / max(across * line_spacing, 1e-9)

    def fn(F, KX, KY, K):
        k_along = KX * np.cos(a) - KY * np.sin(a)       # rows increase southwards
        k_across = KX * np.sin(a) + KY * np.cos(a)
        return F * np.exp(-(k_along / ka) ** 2) * (1 - np.exp(-(k_across / kc) ** 2))

    corr, _, _ = _filter(zf, cell, fn)
    return zf - corr, corr


def mean_spacing(x, y):
    """Average data spacing sqrt(area / n): on line surveys this sits between station and line spacing."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    area = max((np.nanmax(x) - np.nanmin(x)) * (np.nanmax(y) - np.nanmin(y)), 1e-9)
    return float(np.sqrt(area / max(np.isfinite(x).sum(), 1)))


def magnetic_products(grid, regional_height=None, smooth_m=0.0):
    """Standard enhancement filters for a magnetic grid.

    Derivatives are taken on the field upward-continued by `smooth_m` metres, which suppresses
    line-noise and gridding artefacts (derivative filters amplify short wavelengths).
    Returns (products, (dx, dy, dz), smoothed field)."""
    zf, cell = grid["zf"], grid["cell"]
    zs = upward(zf, cell, smooth_m) if smooth_m and smooth_m > 0 else zf
    dx, dy, dz = derivatives(zs, cell)
    hgm = np.hypot(dx, dy)
    asig = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    tilt = np.degrees(np.arctan2(dz, hgm))
    h = regional_height or 10 * cell
    reg = upward(zf, cell, h)
    m = grid["valid"]
    nan = lambda a: np.where(m, a, np.nan).astype(np.float32)  # noqa: E731
    return {"Total field": nan(zf), f"Regional (upward {h:.0f} m)": nan(reg), "Residual": nan(zf - reg),
            "Analytic signal": nan(asig), "Tilt derivative (°)": nan(tilt), "Horizontal gradient": nan(hgm),
            "First vertical derivative": nan(dz)}, (dx, dy, dz), zs


# ---------- depth ----------
def radial_spectrum(zf, cell, nbins=40):
    padded, _, _, _ = _prep(zf, 0.0)
    h, w = padded.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    P = np.abs(np.fft.fft2(padded * win)) ** 2
    _, _, K = _k(padded.shape, cell)
    k, p = K.ravel(), P.ravel()
    sel = k > 0
    edges = np.linspace(0, k[sel].max(), nbins + 1)
    idx = np.digitize(k[sel], edges)
    kk, lp = [], []
    for i in range(1, nbins + 1):
        m = idx == i
        if m.sum() > 3:
            kk.append(k[sel][m].mean())
            lp.append(np.log(p[sel][m].mean() + 1e-30))
    return np.array(kk), np.array(lp)


def spectral_depths(k, lnp, deep=(0.02, 0.15), shallow=(0.15, 0.40)):
    """Depths (m) to deep and shallow source ensembles from straight-line fits of ln(P) vs k (rad/m):
    ln P ~ -2 h k, so h = -slope / 2. `deep` / `shallow` are fit ranges as fractions of the k axis."""
    n = len(k)
    out = {}
    for name, (fa, fb) in {"deep": deep, "shallow": shallow}.items():
        a, b = max(int(round(n * fa)), 0), min(int(round(n * fb)), n)
        if b - a < 3:
            continue
        s, c = np.polyfit(k[a:b], lnp[a:b], 1)
        out[name] = {"depth": float(max(-s / 2, 0)), "slope": float(s), "intercept": float(c), "range": (a, b)}
    return out


def euler(grid, field, derivs, si=1.0, window=10, max_rel_err=0.15, min_valid=0.8):
    """Standard 3-D Euler deconvolution in moving windows. Returns DataFrame(x, y, depth, depth_err)."""
    dx, dy, dz = derivs
    T, X, Y, valid = field, grid["X"], grid["Y"], grid["valid"]
    h, w = T.shape
    cell = grid["cell"]
    step = max(window // 2, 1)
    sols = []
    for r in range(0, h - window + 1, step):
        for c in range(0, w - window + 1, step):
            sl = (slice(r, r + window), slice(c, c + window))
            if valid[sl].mean() < min_valid:
                continue
            tx, ty, tz, t = dx[sl].ravel(), dy[sl].ravel(), dz[sl].ravel(), T[sl].ravel()
            x, y = X[sl].ravel(), Y[sl].ravel()
            A = np.column_stack([tx, ty, tz, np.full_like(tx, si)])
            b = x * tx + y * ty + si * t
            try:
                sol, rss, rank, _ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if rank < 4:
                continue
            x0, y0, z0, _ = sol
            res = b - A @ sol
            dof = max(len(b) - 4, 1)
            try:
                cov = (res @ res / dof) * np.linalg.inv(A.T @ A)
            except np.linalg.LinAlgError:
                continue
            zerr = float(np.sqrt(abs(cov[2, 2])))
            margin = window * cell
            if z0 <= 0 or zerr / z0 > max_rel_err:
                continue
            if not (x.min() - margin <= x0 <= x.max() + margin and y.min() - margin <= y0 <= y.max() + margin):
                continue
            sols.append((x0, y0, z0, zerr))
    return pd.DataFrame(sols, columns=["x", "y", "depth", "depth_err"])


# ---------- IP sections ----------
def section_grid(x, depth, v, nx=200, nz=80):
    """Pseudosection grid (distance along line x depth). Depth positive down."""
    x, depth, v = (np.asarray(a, float) for a in (x, depth, v))
    ok = np.isfinite(x) & np.isfinite(depth) & np.isfinite(v)
    x, depth, v = x[ok], depth[ok], v[ok]
    if len(x) < 4:
        raise ValueError("Too few points for a section.")
    xi = np.linspace(x.min(), x.max(), nx)
    zi = np.linspace(depth.min(), depth.max(), nz)
    XI, ZI = np.meshgrid(xi, zi)
    G = griddata((x, depth), v, (XI, ZI), method="linear")
    return xi, zi, G


def section_anomalies(x, depth, v, threshold, min_stations=2):
    """Contiguous runs of stations whose values exceed threshold: x range, depth to top / bottom, peak.
    Runs shorter than `min_stations` stations (isolated noisy readings) are ignored."""
    d = pd.DataFrame({"x": x, "depth": depth, "v": v}).dropna()
    hot = d[d.v >= threshold]
    if hot.empty:
        return pd.DataFrame(columns=["from_x", "to_x", "depth_top", "depth_bottom", "peak", "peak_x"])
    xs = np.sort(hot.x.unique())
    spacing = np.median(np.diff(np.sort(d.x.unique()))) if d.x.nunique() > 1 else 1
    groups, cur = [], [xs[0]]
    for a in xs[1:]:
        if a - cur[-1] <= 1.6 * spacing:
            cur.append(a)
        else:
            groups.append(cur)
            cur = [a]
    groups.append(cur)
    groups = [g for g in groups if len(g) >= min_stations]
    if not groups:
        return pd.DataFrame(columns=["from_x", "to_x", "depth_top", "depth_bottom", "peak", "peak_x"])
    rows = []
    for g in groups:
        sub = hot[hot.x.between(min(g), max(g))]
        pk = sub.loc[sub.v.idxmax()]
        rows.append({"from_x": float(min(g)), "to_x": float(max(g)), "depth_top": float(sub.depth.min()),
                     "depth_bottom": float(sub.depth.max()), "peak": float(pk.v), "peak_x": float(pk.x)})
    return pd.DataFrame(rows).sort_values("peak", ascending=False)


def n_level_to_depth(n, a_spacing, array="Dipole-dipole"):
    """Approximate median depth of investigation (Edwards, 1977) for common arrays."""
    from .ipdata import depth_factor
    arr = "Wenner" if str(array).startswith("Wenner") else array
    return a_spacing * depth_factor(n, arr)
