"""Magnetic data enhancement and interpretation (ground or airborne TMI).

Maps produced: TMI, RTP or RTE, 1VD, 2VD, analytic signal, tilt derivative, total horizontal derivative
(THDR), lineaments, upward continuation, SPI depth, Euler 3D solutions and a magnetic target map.

Conventions: x = east, y = north, z = down. Inclination positive downwards (negative in the southern
hemisphere), declination clockwise from north.
"""
import datetime

import numpy as np
import pandas as pd

from . import geophysics, lineaments
from .geophysics import _filter, derivatives, upward

REDUCTIONS = ["Auto (RTP above ±25°, RTE below)", "RTP (reduction to pole)", "RTE (reduction to equator)"]


# ---------- Earth's field ----------
def field_direction(lon, lat, date=None):
    """(inclination, declination, total field nT) from IGRF (ppigrf); dipole approximation if unavailable."""
    date = date or datetime.datetime.now()
    try:
        import ppigrf
        be, bn, bu = (float(np.squeeze(v)) for v in ppigrf.igrf(lon, lat, 0, date))
        h = np.hypot(be, bn)
        return float(np.degrees(np.arctan2(-bu, h))), float(np.degrees(np.arctan2(be, bn))), float(np.hypot(h, bu)), "IGRF"
    except Exception:
        inc = np.degrees(np.arctan(2 * np.tan(np.radians(lat))))  # geocentric dipole, rough
        return float(inc), 0.0, float("nan"), "dipole approximation"


def auto_method(inc):
    return "RTP" if abs(inc) >= 25 else "RTE"


# ---------- reduction filters ----------
def _theta_terms(KX, KY, inc, dec):
    th = np.arctan2(KX, -KY)  # azimuth of the wavenumber vector, clockwise from north (rows run south)
    c = np.cos(np.radians(dec) - th)
    return np.sin(np.radians(inc)), np.cos(np.radians(inc)), c


def reduce(zf, cell, inc, dec, method="RTP", amp_inc=None):
    """Reduction to the pole (RTP) or to the equator (RTE) for induced magnetisation.

    Low-latitude stability: the amplitude part uses a larger inclination `amp_inc` (MacLeod et al., 1993),
    by default max(|inc|, 50°), so the E-W stripes of a plain low-latitude RTP are suppressed.
    """
    ia = np.radians(np.sign(inc or 1) * max(abs(inc), amp_inc or 50.0))

    def fn(F, KX, KY, K):
        s, co, c = _theta_terms(KX, KY, inc, dec)
        sa, ca = np.sin(ia), np.cos(ia)
        L = (s - 1j * co * c) ** 2 / ((sa ** 2 + ca ** 2 * c ** 2) * (s ** 2 + co ** 2 * c ** 2))
        if method == "RTE":
            # field direction (inc, dec) -> horizontal field with the same declination
            L = L * (1j * c) ** 2
        L[K == 0] = 1.0 if method == "RTP" else 0.0
        return F * L

    out, plane, _ = _filter(zf, cell, fn)
    return out + (plane if method == "RTP" else 0)


def vertical_derivative(zf, cell, order=1):
    out, _, _ = _filter(zf, cell, lambda F, KX, KY, K: F * K ** order)
    return out


def thdr(zf, cell):
    dx, dy, _ = derivatives(zf, cell)
    return np.hypot(dx, dy)


def tilt(zf, cell):
    dx, dy, dz = derivatives(zf, cell)
    return np.degrees(np.arctan2(dz, np.hypot(dx, dy)))


def analytic_signal(zf, cell):
    dx, dy, dz = derivatives(zf, cell)
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def spi_depth(reduced, cell, max_tilt=10.0, smooth_cells=1.0):
    """Source Parameter Imaging (Thurston & Smith, 1997), contact model.

    Local wavenumber k = |grad(tilt)| (radians per metre); depth = 1 / k where the tilt is close to zero,
    i.e. over the contacts. Returns (depth grid with NaN away from contacts, local wavenumber grid)."""
    from scipy.ndimage import gaussian_filter
    t = np.radians(tilt(reduced, cell))
    if smooth_cells:
        t = gaussian_filter(t, smooth_cells)
    gy, gx = np.gradient(t, cell)
    k = np.hypot(gx, gy)
    with np.errstate(divide="ignore"):
        depth = np.where((np.abs(np.degrees(t)) <= max_tilt) & (k > 0), 1.0 / k, np.nan)
    return depth.astype(np.float32), k.astype(np.float32)


# ---------- full analysis ----------
def analyze(grid, inc, dec, method="Auto", smooth_m=0.0, si=1.0, euler_window=10, up_height=None,
            lin_pct=90, lin_len_cells=15):
    """All magnetic products for one grid (see geophysics.grid_points)."""
    zf, cell, valid = grid["zf"], grid["cell"], grid["valid"]
    m = "RTP" if method.startswith("RTP") else "RTE" if method.startswith("RTE") else auto_method(inc)
    zs = upward(zf, cell, smooth_m) if smooth_m and smooth_m > 0 else zf
    red = reduce(zs, cell, inc, dec, m)
    # at the equator induced anomalies are lows; invert the RTE so bodies are highs for the derivatives
    red_pos = red if m == "RTP" else -red
    nan = lambda a: np.where(valid, a, np.nan).astype(np.float32)  # noqa: E731
    up_h = up_height or max(10 * cell, 100.0)
    tilt_red = tilt(red_pos, cell)
    spi, kloc = spi_depth(red_pos, cell)
    maps = {
        "TMI": nan(zf),
        m: nan(red),
        "1VD": nan(vertical_derivative(red_pos, cell, 1)),
        "2VD": nan(vertical_derivative(red_pos, cell, 2)),
        "Analytic signal": nan(analytic_signal(zs, cell)),
        "Tilt derivative": nan(tilt_red),
        "THDR": nan(thdr(red_pos, cell)),
        f"Upward continuation {up_h:.0f} m": nan(upward(zf, cell, up_h)),
        "SPI depth": nan(spi),
    }
    derivs = derivatives(zs, cell)
    eu = geophysics.euler(grid, zs, derivs, si, euler_window)
    eu["depth"] = eu["depth"] - (smooth_m or 0)
    eu = eu[eu.depth > 0].reset_index(drop=True)
    lines, _ = lineaments.detect([tilt_red], valid, cell, sigma=1.5, high_pct=lin_pct,
                                 min_length_m=lin_len_cells * cell, max_gap_m=2 * cell, merge_m=3 * cell)
    k, lp = geophysics.radial_spectrum(zf, cell)
    # keep the keys other modules use
    products = {"Total field": maps["TMI"], "Residual": nan(zf - upward(zf, cell, up_h)),
                "Analytic signal": maps["Analytic signal"], "Tilt derivative (°)": maps["Tilt derivative"],
                "Horizontal gradient": maps["THDR"], "First vertical derivative": maps["1VD"]}
    spi_pts = _spi_points(spi, grid)
    return {"maps": maps, "products": products, "reduction": m, "inc": inc, "dec": dec, "euler": eu,
            "lineaments": lines, "spi_points": spi_pts, "spectrum": (k, lp, geophysics.spectral_depths(k, lp)),
            "up_height": up_h, "smooth_m": float(smooth_m or 0)}


def _spi_points(spi, grid, max_points=4000):
    ok = np.isfinite(spi) & grid["valid"]
    if not ok.any():
        return pd.DataFrame(columns=["x", "y", "depth"])
    d = pd.DataFrame({"x": grid["X"][ok], "y": grid["Y"][ok], "depth": spi[ok]})
    d = d[(d.depth > 0) & (d.depth < np.nanpercentile(d.depth, 99))]
    return d.sample(max_points, random_state=0) if len(d) > max_points else d


# ---------- synthetic test model ----------
def dipole_tmi(x, y, src, depth, inc, dec, moment=1e9):
    """TMI anomaly of an induced point dipole (for tests): field and magnetisation along (inc, dec)."""
    I, D = np.radians(inc), np.radians(dec)
    f = np.array([np.cos(I) * np.sin(D), np.cos(I) * np.cos(D), np.sin(I)])  # east, north, down
    rx, ry, rz = x - src[0], y - src[1], -depth * np.ones_like(x)  # vector source -> observation (z down)
    r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    mdotr = f[0] * rx + f[1] * ry + f[2] * rz
    bx = moment * (3 * mdotr * rx / r ** 2 - f[0]) / r ** 3
    by = moment * (3 * mdotr * ry / r ** 2 - f[1]) / r ** 3
    bz = moment * (3 * mdotr * rz / r ** 2 - f[2]) / r ** 3
    return bx * f[0] + by * f[1] + bz * f[2]
