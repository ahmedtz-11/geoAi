"""Sentinel-2 band maths for gold exploration screening.

Gold in greenstone belts (e.g. Lake Victoria Goldfields) and shear-hosted systems is often
surrounded by hydrothermal alteration: sericite/clay (phyllic-argillic), iron oxides from
weathered sulphides (gossans) and chlorite/ferrous minerals (propylitic). Band ratios highlight
these minerals at the surface. They are screening tools: targets still need field checking,
geochemistry and geophysics (Section 2).
"""
import numpy as np
from scipy import ndimage


def ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b
    r[~np.isfinite(r)] = np.nan
    return r


def nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (a - b) / (a + b)
    r[~np.isfinite(r)] = np.nan
    return r


INDICES = {
    "clay_hydroxyl": {
        "label": "Clay / hydroxyl (B11/B12)",
        "fn": lambda b: ratio(b["B11"], b["B12"]),
        "cmap": "magma",
        "info": "High = OH-bearing minerals (sericite, kaolinite, illite, muscovite): phyllic/argillic "
                "alteration that often surrounds gold-bearing quartz veins and shear zones.",
    },
    "iron_oxide": {
        "label": "Iron oxide (B4/B2)",
        "fn": lambda b: ratio(b["B04"], b["B02"]),
        "cmap": "YlOrRd",
        "info": "High = ferric iron (hematite, goethite, jarosite), typical of gossans formed by "
                "weathering of sulphides such as pyrite and arsenopyrite.",
    },
    "ferric_oxides": {
        "label": "Ferric oxides (B11/B8A)",
        "fn": lambda b: ratio(b["B11"], b["B8A"]),
        "cmap": "Oranges",
        "info": "High = ferric oxide minerals and bare oxidised soil/rock.",
    },
    "ferrous": {
        "label": "Ferrous iron (B12/B8 + B3/B4)",
        "fn": lambda b: ratio(b["B12"], b["B08"]) + ratio(b["B03"], b["B04"]),
        "cmap": "PuBuGn",
        "info": "High = Fe2+ minerals (chlorite, amphibole, mafic rocks): propylitic alteration "
                "and mafic/greenstone units.",
    },
    "gossan": {
        "label": "Gossan (B11/B4)",
        "fn": lambda b: ratio(b["B11"], b["B04"]),
        "cmap": "copper",
        "info": "High = iron-rich weathered caps over sulphide zones.",
    },
    "ndvi": {
        "label": "Vegetation NDVI ((B8-B4)/(B8+B4))",
        "fn": lambda b: nd(b["B08"], b["B04"]),
        "cmap": "RdYlGn",
        "info": "High = dense vegetation, which hides the rock signal. Used to mask the target map.",
    },
}

COMPOSITES = {
    "True colour (B4-B3-B2)": ("B04", "B03", "B02"),
    "Geology false colour (B12-B11-B2)": ("B12", "B11", "B02"),
    "Colour infrared (B8-B4-B3)": ("B08", "B04", "B03"),
    "Alteration ratios (R=B11/B12 clay, G=B4/B2 iron, B=B11/B8A ferric)": ("clay_hydroxyl", "iron_oxide", "ferric_oxides"),
}
COMPOSITE_INFO = {
    "True colour (B4-B3-B2)": "What the eye would see.",
    "Geology false colour (B12-B11-B2)": "Separates rock and soil types; bare rock shows in browns/purples, vegetation green.",
    "Colour infrared (B8-B4-B3)": "Vegetation shows red; useful to judge how much rock is exposed.",
    "Alteration ratios (R=B11/B12 clay, G=B4/B2 iron, B=B11/B8A ferric)":
        "Bright/white = clay + iron together (strong alteration); red = clay; green = iron oxide.",
}


def compute_index(bands, key):
    return INDICES[key]["fn"](bands).astype(np.float32)


def stretch(a, valid=None, lo=2, hi=98):
    """Percentile stretch to 0-1 using only valid pixels."""
    a = np.asarray(a, np.float32)
    sel = np.isfinite(a) if valid is None else (valid & np.isfinite(a))
    if not sel.any():
        return np.zeros_like(a)
    p0, p1 = np.percentile(a[sel], [lo, hi])
    out = (a - p0) / ((p1 - p0) or 1)
    return np.clip(np.nan_to_num(out, nan=0), 0, 1)


def composite(bands, name, valid=None, lo=2, hi=98, outside_alpha=0.35):
    """RGB array (H, W, 3) in 0-1. Pixels outside `valid` are dimmed."""
    keys = COMPOSITES[name]
    arrs = [bands[k] if k in bands else compute_index(bands, k) for k in keys]
    if all(k in bands for k in keys) and name.startswith("True colour"):
        # one shared stretch keeps natural colour balance
        stack = np.dstack(arrs)
        sel = (np.ones(stack.shape[:2], bool) if valid is None else valid)[..., None] & np.isfinite(stack)
        p0, p1 = np.percentile(stack[sel], [lo, hi]) if sel.any() else (0, 1)
        rgb = np.clip(np.nan_to_num((stack - p0) / ((p1 - p0) or 1), nan=0), 0, 1).astype(np.float32)
    else:
        rgb = np.dstack([stretch(a, valid, lo, hi) for a in arrs])
    if valid is not None:
        rgb[~valid] *= outside_alpha
    return rgb


def zscore(a, sel):
    m, s = np.nanmean(a[sel]), np.nanstd(a[sel])
    return (a - m) / (s or 1)


def smooth_masked(a, valid, sigma):
    """Gaussian smoothing that ignores invalid pixels (normalised convolution)."""
    if sigma <= 0:
        return a
    w = valid.astype(np.float32)
    num = ndimage.gaussian_filter(np.where(valid, a, 0).astype(np.float32), sigma)
    den = ndimage.gaussian_filter(w, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(valid, num / den, np.nan)


def alteration_targets(bands, clear, weights, ndvi_max=0.4, top_pct=5.0, min_pixels=4, transform=None,
                       smooth=1.0):
    """Weighted alteration score and candidate zones.

    weights: {index_key: weight}. Score = weighted mean of z-scores over clear, low-vegetation pixels,
    optionally smoothed (sigma in pixels) so speckle does not split zones.
    Returns dict(score, usable, targets (bool), labels, zones: list of dict).
    """
    ndvi = compute_index(bands, "ndvi")
    usable = clear & np.isfinite(ndvi) & (ndvi <= ndvi_max)
    out = {"score": np.full(ndvi.shape, np.nan, np.float32), "usable": usable,
           "targets": np.zeros(ndvi.shape, bool), "labels": None, "zones": []}
    if usable.sum() < 20:
        return out
    acc, wsum = np.zeros(ndvi.shape, np.float32), 0.0
    for k, w in weights.items():
        if not w:
            continue
        acc += w * np.nan_to_num(zscore(compute_index(bands, k), usable))
        wsum += abs(w)
    if not wsum:
        return out
    score = smooth_masked(np.where(usable, acc / wsum, np.nan), usable, smooth).astype(np.float32)
    z = extract_zones(score, usable, top_pct, min_pixels, transform)
    out.update(score=score, **z)
    return out


def extract_zones(score, usable, top_pct=5.0, min_pixels=4, transform=None):
    """Top `top_pct` % of usable pixels grouped into connected zones (smaller than min_pixels dropped).

    Returns dict(targets bool, labels, zones list (ranked by strength x size), threshold).
    """
    thr = float(np.nanpercentile(score[usable], 100 - top_pct)) if usable.any() else np.inf
    tgt = usable & (score >= thr)
    labels, n = ndimage.label(tgt, structure=np.ones((3, 3)))
    zones = []
    if n:
        idx = np.arange(1, n + 1)
        sizes = ndimage.sum(tgt, labels, idx)
        small = idx[sizes < min_pixels]
        if len(small):
            drop = np.isin(labels, small)
            tgt[drop] = False
            labels[drop] = 0
        for i in idx[sizes >= min_pixels]:
            m = labels == i
            r, c = np.nonzero(m)
            zz = {"pixels": int(m.sum()), "mean_score": float(score[m].mean()), "max_score": float(score[m].max()),
                  "row": float(r.mean()), "col": float(c.mean())}
            if transform is not None:
                cc, rr = zz["col"] + 0.5, zz["row"] + 0.5
                zz.update(easting=float(transform.c + transform.a * cc + transform.b * rr),
                          northing=float(transform.f + transform.d * cc + transform.e * rr),
                          area_ha=zz["pixels"] * abs(transform.a * transform.e) / 1e4)
            zones.append(zz)
    zones.sort(key=lambda q: q["mean_score"] * np.sqrt(q["pixels"]), reverse=True)
    return {"targets": tgt, "labels": labels, "zones": zones, "threshold": thr}


# ---------- Band ratios ----------
RATIOS = {
    "B4/B2": ("Ferric iron / iron oxide", lambda b: ratio(b["B04"], b["B02"]), "YlOrRd"),
    "B11/B12": ("Clay / hydroxyl (OH)", lambda b: ratio(b["B11"], b["B12"]), "magma"),
    "B11/B8A": ("Ferric oxides", lambda b: ratio(b["B11"], b["B8A"]), "Oranges"),
    "B12/B8": ("Ferrous iron (Fe2+)", lambda b: ratio(b["B12"], b["B08"]), "PuBuGn"),
    "B11/B4": ("Gossan", lambda b: ratio(b["B11"], b["B04"]), "copper"),
    "B8/B4": ("Vegetation", lambda b: ratio(b["B08"], b["B04"]), "Greens"),
    "B4/B3": ("Iron (red/green)", lambda b: ratio(b["B04"], b["B03"]), "Reds"),
    "B12/B11": ("Carbonate / Mg-OH (low = clay)", lambda b: ratio(b["B12"], b["B11"]), "BuPu"),
}
RATIO_PRESETS = {
    "Sabins (R=B11/B12, G=B4/B2, B=B8/B4)": ("B11/B12", "B4/B2", "B8/B4"),
    "Kaufmann (R=B12/B8, G=B8/B4, B=B11/B12)": ("B12/B8", "B8/B4", "B11/B12"),
    "Alteration (R=B11/B12, G=B4/B2, B=B11/B8A)": ("B11/B12", "B4/B2", "B11/B8A"),
}


def compute_ratio(bands, key):
    return RATIOS[key][1](bands).astype(np.float32)


def ratio_rgb(bands, keys, valid=None, lo=2, hi=98):
    rgb = np.dstack([stretch(compute_ratio(bands, k), valid, lo, hi) for k in keys])
    if valid is not None:
        rgb[~valid] = 0
    return rgb


# ---------- Iron oxide (ferric iron) classification ----------
IRON_CLASSES = ["Background", "Weak", "Moderate", "Strong"]
IRON_COLORS = ["#e9e5dc", "#f5c26b", "#e0652a", "#8f1a0b"]
CLAY_COLORS = ["#ece9f2", "#c6a8dc", "#8a55b8", "#43106e"]


def classify_sigma(arr, valid):
    """Classes from the mean and standard deviation of valid pixels.

    0 = below mean (background), 1 = mean to +1σ (weak), 2 = +1σ to +2σ (moderate), 3 = above +2σ (strong).
    Returns (classes int8 array with -1 where invalid, thresholds [mean, +1σ, +2σ]).
    """
    sel = valid & np.isfinite(arr)
    cls = np.full(arr.shape, -1, np.int8)
    if not sel.any():
        return cls, [np.nan] * 3
    m, s = float(np.mean(arr[sel])), float(np.std(arr[sel]))
    thr = [m, m + s, m + 2 * s]
    cls[sel] = np.digitize(arr[sel], thr).astype(np.int8)
    return cls, thr
