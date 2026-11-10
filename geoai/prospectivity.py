"""Knowledge-driven (fuzzy overlay) mineral prospectivity mapping.

Each evidence layer is rescaled to a 0-1 fuzzy membership (1 = most favourable for gold), then layers
are combined by weighted sum or fuzzy gamma. The result ranks ground for follow-up; it is not a
probability of finding gold.
"""
import numpy as np

from .indices import extract_zones


def fuzzy_linear(arr, valid, lo_pct=5, hi_pct=95, invert=False):
    a = np.asarray(arr, np.float32)
    sel = valid & np.isfinite(a)
    out = np.full(a.shape, np.nan, np.float32)
    if not sel.any():
        return out
    lo, hi = np.percentile(a[sel], [lo_pct, hi_pct])
    m = np.clip((a - lo) / ((hi - lo) or 1), 0, 1)
    if invert:
        m = 1 - m
    out[sel] = m[sel]
    return out


def fuzzy_distance(dist_m, valid, half_m=250):
    """1 on the feature, 0.5 at `half_m` metres, approaching 0 further away."""
    m = 1.0 / (1.0 + (np.asarray(dist_m, np.float32) / max(half_m, 1)) ** 2)
    return np.where(valid, m, np.nan).astype(np.float32)


def combine(layers, weights, valid, method="Weighted sum", gamma=0.8):
    """layers: {name: membership 0-1}; weights: {name: w >= 0}."""
    use = [(layers[k], w) for k, w in weights.items() if w > 0 and k in layers]
    if not use:
        return np.full(valid.shape, np.nan, np.float32)
    ok = valid.copy()
    for a, _ in use:
        ok &= np.isfinite(a)
    if method == "Weighted sum":
        tot = sum(w for _, w in use)
        s = sum(np.nan_to_num(a) * w for a, w in use) / tot
    else:  # fuzzy gamma on weighted memberships
        ws = [np.clip(np.nan_to_num(a) * w / max(w for _, w in use), 1e-6, 1) for a, w in use]
        fsum = 1 - np.prod([1 - m for m in ws], axis=0)
        fprod = np.prod(ws, axis=0)
        s = fsum ** gamma * fprod ** (1 - gamma)
        s = (s - s[ok].min()) / ((s[ok].max() - s[ok].min()) or 1) if ok.any() else s
    return np.where(ok, s, np.nan).astype(np.float32)


def targets(score, valid, top_pct=5, min_pixels=4, transform=None):
    usable = valid & np.isfinite(score)
    return extract_zones(score, usable, top_pct, min_pixels, transform)
