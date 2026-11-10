"""Automatic geological lineament extraction.

Edges are detected (Canny) on multi-directional hillshades of the DEM and/or on the first principal
component of the imagery, short noisy edges are removed, and straight segments are fitted with a
probabilistic Hough transform. Lineaments often follow faults, shear zones, fractures, dykes and
contacts, which control many gold deposits.
"""
import math

import numpy as np
from scipy import ndimage


def edges_from(image, valid, sigma=2.0, high_pct=90):
    from skimage.feature import canny

    img = np.where(valid & np.isfinite(image), image, np.nan)
    fill = np.nanmean(img) if np.isfinite(img).any() else 0
    img = np.nan_to_num(img, nan=fill)
    rng = np.ptp(img) or 1
    img = (img - img.min()) / rng
    hi = min(max(high_pct, 50), 99.5) / 100
    e = canny(img, sigma=sigma, low_threshold=hi * 0.8, high_threshold=hi, use_quantiles=True)
    # do not keep edges along the data border
    return e & ndimage.binary_erosion(valid, iterations=3)


def merge_parallel(lines, res, merge_m=150, max_angle=10):
    """Drop a lineament that runs alongside a longer one (e.g. the two walls of one valley)."""
    kept = []
    for l in sorted(lines, key=lambda q: q["length_m"], reverse=True):
        mx, my = (l["x0"] + l["x1"]) / 2, (l["y0"] + l["y1"]) / 2
        dup = False
        for k in kept:
            da = abs(l["azimuth"] - k["azimuth"]) % 180
            if min(da, 180 - da) > max_angle:
                continue
            vx, vy = k["x1"] - k["x0"], k["y1"] - k["y0"]
            L = math.hypot(vx, vy) or 1
            ux, uy = vx / L, vy / L
            along = (mx - k["x0"]) * ux + (my - k["y0"]) * uy
            perp = abs((mx - k["x0"]) * uy - (my - k["y0"]) * ux) * res
            if perp <= merge_m and -merge_m / res <= along <= L + merge_m / res:
                dup = True
                break
        if not dup:
            kept.append(l)
    return kept


def detect(images, valid, res, sigma=2.0, high_pct=90, min_length_m=300, max_gap_m=60, min_edge_px=15,
           merge_m=150):
    """images: list of 2-D arrays (e.g. hillshades). Returns (lines list, edge mask).

    Near-parallel lines closer than `merge_m` metres are merged (0 = keep all).

    Each line: dict(x0, y0, x1, y1 in pixels, length_m, azimuth 0-180° clockwise from north).
    """
    from skimage.transform import probabilistic_hough_line

    edges = np.zeros(valid.shape, bool)
    for im in images:
        edges |= edges_from(im, valid, sigma, high_pct)
    # drop short, noisy edge fragments
    lab, n = ndimage.label(edges, structure=np.ones((3, 3)))
    if n:
        sizes = np.bincount(lab.ravel())
        keep = sizes >= min_edge_px
        keep[0] = False
        edges = keep[lab]
    L = max(int(min_length_m / res), 5)
    G = max(int(max_gap_m / res), 1)
    segs = probabilistic_hough_line(edges, threshold=10, line_length=L, line_gap=G, rng=0)
    lines = []
    for (x0, y0), (x1, y1) in segs:
        dx, dn = (x1 - x0) * res, -(y1 - y0) * res
        length = math.hypot(dx, dn)
        az = (math.degrees(math.atan2(dx, dn)) + 180) % 180
        lines.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "length_m": length, "azimuth": az})
    if merge_m:
        lines = merge_parallel(lines, res, merge_m)
    return lines, edges


def line_mask(lines, shape):
    from skimage.draw import line as draw_line

    m = np.zeros(shape, bool)
    for l in lines:
        rr, cc = draw_line(int(l["y0"]), int(l["x0"]), int(l["y1"]), int(l["x1"]))
        ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
        m[rr[ok], cc[ok]] = True
    return m


def density(lines, shape, res, radius_m=500):
    """Lineament length per km² (Gaussian kernel)."""
    m = line_mask(lines, shape).astype(np.float32) * res / 1000.0  # km of line per pixel
    sigma = max(radius_m / res / 2, 1)
    d = ndimage.gaussian_filter(m, sigma)
    px_km2 = (res / 1000.0) ** 2
    return d / px_km2


def distance(lines, shape, res):
    """Distance (m) from each pixel to the nearest lineament."""
    m = line_mask(lines, shape)
    if not m.any():
        return np.full(shape, np.inf, np.float32)
    return (ndimage.distance_transform_edt(~m) * res).astype(np.float32)


def rose_stats(lines, bin_deg=10):
    """Length-weighted histogram of strike (0-180°). Returns (bin_starts, lengths_km)."""
    bins = np.arange(0, 180 + bin_deg, bin_deg)
    if not lines:
        return bins[:-1], np.zeros(len(bins) - 1)
    az = np.array([l["azimuth"] for l in lines])
    ln = np.array([l["length_m"] for l in lines]) / 1000
    h, _ = np.histogram(az, bins=bins, weights=ln)
    return bins[:-1], h


def strike_label(az):
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE"]
    a = az % 180
    i = int(((a + 11.25) % 180) // 22.5)
    opp = ["S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return f"{names[i]}-{opp[i]}"


def to_geojson(lines, transform, epsg):
    """Lineaments as a GeoJSON FeatureCollection in WGS84."""
    import json

    from pyproj import Transformer

    t = Transformer.from_crs(epsg, 4326, always_xy=True)
    feats = []
    for i, l in enumerate(lines, 1):
        pts = []
        for x, y in [(l["x0"], l["y0"]), (l["x1"], l["y1"])]:
            e = transform.c + transform.a * (x + 0.5) + transform.b * (y + 0.5)
            n = transform.f + transform.d * (x + 0.5) + transform.e * (y + 0.5)
            pts.append(list(t.transform(e, n)))
        feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": pts},
                      "properties": {"id": i, "length_m": round(l["length_m"], 1),
                                     "azimuth": round(l["azimuth"], 1), "trend": strike_label(l["azimuth"])}})
    return json.dumps({"type": "FeatureCollection", "features": feats})
