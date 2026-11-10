"""Unsupervised spectral (lithological) classification of Sentinel-2 data with K-means.

Classes are spectral units: areas whose surface reflectance is similar. They usually follow rock
type and regolith, but also vegetation and soil, so each unit must be named from field geology.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .indices import compute_index

FEATURE_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
FEATURE_INDICES = ["clay_hydroxyl", "iron_oxide", "ndvi"]
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#17becf", "#393b79", "#ad494a"]


def classify(bands, valid, k=6, sample=150_000, seed=0):
    """Returns dict(labels (-1 = not classified), means DataFrame, hints list, colors)."""
    feats = [bands[b] for b in FEATURE_BANDS] + [compute_index(bands, i) for i in FEATURE_INDICES]
    F = np.stack(feats).astype(np.float64)
    sel = valid & np.all(np.isfinite(F), axis=0)
    X = F[:, sel].T
    if len(X) < k * 10:
        raise ValueError("Not enough clear pixels to classify.")
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Xs = (X - mu) / sd
    fit = Xs if len(Xs) <= sample else Xs[np.random.default_rng(seed).choice(len(Xs), sample, replace=False)]
    km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(fit)
    lab = km.predict(Xs)
    # order classes by brightness so colours stay stable between runs
    bright = np.array([X[lab == c, :len(FEATURE_BANDS)].mean() if (lab == c).any() else 0 for c in range(k)])
    remap = np.empty(k, int)
    remap[np.argsort(bright)] = np.arange(k)
    lab = remap[lab]
    labels = np.full(valid.shape, -1, np.int16)
    labels[sel] = lab

    cols = FEATURE_BANDS + FEATURE_INDICES
    means = pd.DataFrame([X[lab == c].mean(0) for c in range(k)], columns=cols)
    means.insert(0, "pixels", [(lab == c).sum() for c in range(k)])
    means["hint"] = describe(means)
    means.index = [f"Unit {c + 1}" for c in range(k)]
    return {"labels": labels, "means": means, "colors": PALETTE[:k]}


def describe(means):
    """Rough spectral interpretation of each class (to be confirmed in the field)."""
    z = (means[["clay_hydroxyl", "iron_oxide"]] - means[["clay_hydroxyl", "iron_oxide"]].mean()) / \
        means[["clay_hydroxyl", "iron_oxide"]].std().replace(0, 1)
    bright = means[FEATURE_BANDS].mean(axis=1)
    bz = (bright - bright.mean()) / (bright.std() or 1)
    hints = []
    for i, r in means.iterrows():
        if r["ndvi"] > 0.5:
            h = "Dense vegetation"
        elif r["ndvi"] < 0 and r["B08"] < 0.12:
            h = "Water / wet ground"
        elif z.loc[i, "clay_hydroxyl"] > 0.8 and z.loc[i, "iron_oxide"] > 0.8:
            h = "Clay + iron oxide rich (possible alteration)"
        elif z.loc[i, "clay_hydroxyl"] > 0.8:
            h = "Clay / sericite rich (possible alteration, felsic rock)"
        elif z.loc[i, "iron_oxide"] > 0.8:
            h = "Iron-oxide rich (laterite, ironstone or gossan)"
        elif bz[i] > 0.8:
            h = "Bright bare ground (quartz-rich / granitic, sand)"
        elif bz[i] < -0.8:
            h = "Dark surface (mafic rock, burnt ground or shadow)"
        elif r["ndvi"] > 0.3:
            h = "Sparse vegetation over soil"
        else:
            h = "Mixed soil and rock"
        hints.append(h)
    return hints
