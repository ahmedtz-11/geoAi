"""Soil / rock geochemistry for gold exploration.

- Cleans lab values: '<5' and negative codes (-5) = below detection -> half the detection limit,
  '>10000' -> 10000, 'BDL', 'nd', 'is' -> missing.
- Statistics and anomaly thresholds on log-transformed data (assay data are strongly skewed).
- Multi-element gold index from Au and pathfinder elements (As, Sb, Bi, Te, W, Ag, Cu, Mo, Hg).
"""
import re

import numpy as np
import pandas as pd

ELEMENTS = ["Au", "Ag", "As", "Sb", "Bi", "Te", "W", "Mo", "Cu", "Pb", "Zn", "Hg", "Ni", "Co", "Cr", "Fe", "S",
            "Mn", "Ba", "Sn", "Tl", "Se"]
PATHFINDERS = ["As", "Sb", "Bi", "Te", "W", "Ag", "Cu", "Mo", "Hg"]
_EL_RE = re.compile(r"^\s*(" + "|".join(ELEMENTS) + r")(?=$|[\s_\-\(\[]|ppm|ppb|pct|%|g)", re.I)
_UNIT_RE = re.compile(r"(ppb|ppm|pct|%|g/t|gpt)", re.I)
CLASSES = ["Background", "Weak anomaly", "Anomalous", "Strongly anomalous"]
CLASS_COLORS = ["#c9c9c9", "#f2c14e", "#e4572e", "#7b0d1e"]
METHODS = ["Mean + σ on log values", "Percentiles (75 / 90 / 98)", "Median + MAD on log values"]


def element_of(col):
    m = _EL_RE.match(str(col))
    if not m:
        return None, None
    el = m.group(1).capitalize()
    if el not in ELEMENTS:
        el = next((e for e in ELEMENTS if e.lower() == el.lower()), el)
    u = _UNIT_RE.search(str(col))
    return el, (u.group(1).lower().replace("pct", "%") if u else None)


def detect_elements(df):
    out = {}
    for c in df.columns:
        el, unit = element_of(c)
        if el:
            out[c] = (el, unit)
    return out


def clean_values(series):
    """Return (float values, below-detection mask)."""
    vals, bdl = [], []
    for v in series.tolist():
        b = False
        if v is None or (isinstance(v, float) and np.isnan(v)):
            vals.append(np.nan)
            bdl.append(False)
            continue
        if isinstance(v, (int, float, np.number)):
            f = float(v)
            if f < 0:  # common lab convention: -5 means < 5
                f, b = abs(f) / 2, True
            vals.append(f)
            bdl.append(b)
            continue
        s = str(v).strip().replace(",", ".")
        low = s.lower()
        if low in {"", "bdl", "nd", "n.d.", "is", "ins", "lnr", "na", "n/a", "-", "nss"}:
            vals.append(np.nan)
            bdl.append(False)
            continue
        m = re.search(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", s, re.I)
        if not m:
            vals.append(np.nan)
            bdl.append(False)
            continue
        f = float(m.group(0))
        if s.startswith("<") or f < 0:
            f, b = abs(f) / 2, True
        vals.append(f)
        bdl.append(b)
    return np.array(vals, float), np.array(bdl, bool)


def stats(values, bdl=None):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {}
    pos = v[v > 0]
    return {"n": int(v.size), "below DL": int(bdl.sum()) if bdl is not None else 0,
            "min": float(v.min()), "median": float(np.median(v)), "mean": float(v.mean()),
            "geo. mean": float(np.exp(np.log(pos).mean())) if pos.size else np.nan,
            "P90": float(np.percentile(v, 90)), "P98": float(np.percentile(v, 98)), "max": float(v.max())}


def thresholds(values, method=METHODS[0]):
    """Three class limits (weak, anomalous, strongly anomalous) in original units."""
    v = values[np.isfinite(values) & (values > 0)]
    if v.size < 5:
        return [np.nan] * 3
    if method.startswith("Percentiles"):
        return [float(np.percentile(v, p)) for p in (75, 90, 98)]
    lv = np.log10(v)
    if method.startswith("Median"):
        med = np.median(lv)
        mad = 1.4826 * np.median(np.abs(lv - med)) or lv.std()
        return [float(10 ** (med + k * mad)) for k in (1, 2, 3)]
    m, s = lv.mean(), lv.std()
    return [float(10 ** (m + k * s)) for k in (1, 2, 3)]


def classify(values, limits):
    cls = np.full(values.shape, -1, int)
    ok = np.isfinite(values)
    cls[ok] = np.digitize(values[ok], limits)
    return cls


def log_z(values):
    v = np.where(values > 0, values, np.nan)
    lv = np.log10(v)
    return (lv - np.nanmean(lv)) / (np.nanstd(lv) or 1)


def correlation(table):
    """Spearman rank correlation (robust to outliers and skew)."""
    return table.rank().corr()


def gold_index(table, au_col, pathfinders):
    """Weighted mean of log z-scores. Au weight 2; each pathfinder weighted by its positive Spearman
    correlation with Au (pathfinders unrelated to gold get ~0)."""
    cols = [au_col] + [p for p in pathfinders if p != au_col]
    corr = correlation(table[cols])[au_col]
    weights = {au_col: 2.0}
    for p in pathfinders:
        if p != au_col:
            weights[p] = float(max(corr.get(p, 0), 0))
    acc = np.zeros(len(table))
    wsum = np.zeros(len(table))
    for c, w in weights.items():
        if w <= 0:
            continue
        z = log_z(table[c].to_numpy(float))
        ok = np.isfinite(z)
        acc[ok] += w * z[ok]
        wsum[ok] += w
    idx = np.where(wsum > 0, acc / np.where(wsum == 0, 1, wsum), np.nan)
    return idx, weights
