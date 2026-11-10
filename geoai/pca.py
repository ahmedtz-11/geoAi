"""Principal Component Analysis of Sentinel-2 bands.

- Standard PCA: decorrelates the bands; PC1 is mostly brightness/topography, higher PCs show
  subtle spectral (mineral, lithological) differences.
- Crósta technique (Feature-Oriented Principal Component Selection): PCA on a 4-band subset; the PC
  whose loadings are strong and opposite for the diagnostic bands maps the target mineral group.
  Sentinel-2 equivalents of the classic Landsat TM sets are used.
"""
import numpy as np

PCA_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

CROSTA = {
    "hydroxyl": {"bands": ["B02", "B08", "B11", "B12"], "pos": "B11", "neg": "B12",
                 "label": "Hydroxyl / clay (Crósta PCA)",
                 "info": "PCA on B2, B8, B11, B12; the PC with opposite B11 and B12 loadings maps OH-bearing "
                         "minerals (sericite, kaolinite, illite)."},
    "iron": {"bands": ["B02", "B04", "B08", "B11"], "pos": "B04", "neg": "B02",
             "label": "Iron oxide (Crósta PCA)",
             "info": "PCA on B2, B4, B8, B11; the PC with opposite B4 and B2 loadings maps ferric iron oxides."},
}


def pca(bands, names, valid, n_components=None, standardize=True, sample=200_000, seed=0):
    """PCA over valid pixels. Returns dict(pcs (k,H,W), explained (k,), loadings (k, n_bands), names)."""
    stack = np.stack([bands[n] for n in names]).astype(np.float64)  # (b, H, W)
    sel = valid & np.all(np.isfinite(stack), axis=0)
    X = stack[:, sel].T  # (n, b)
    if len(X) < len(names) + 2:
        raise ValueError("Not enough clear pixels for PCA.")
    mean = X.mean(0)
    std = X.std(0) if standardize else np.ones(len(names))
    std[std == 0] = 1
    Xs = (X - mean) / std
    fit = Xs
    if len(Xs) > sample:
        fit = Xs[np.random.default_rng(seed).choice(len(Xs), sample, replace=False)]
    cov = np.cov(fit, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # make the largest loading of each PC positive so signs are stable
    vecs *= np.sign(vecs[np.abs(vecs).argmax(0), range(vecs.shape[1])])
    k = n_components or len(names)
    scores = Xs @ vecs[:, :k]
    pcs = np.full((k,) + valid.shape, np.nan, np.float32)
    pcs[:, sel] = scores.T
    return {"pcs": pcs, "explained": vals[:k] / vals.sum(), "loadings": vecs[:, :k].T, "names": list(names)}


def crosta(bands, valid, kind="hydroxyl"):
    """Feature-oriented PC image (bright = target mineral). Returns dict(image, pc_index, loadings, ...)."""
    cfg = CROSTA[kind]
    res = pca(bands, cfg["bands"], valid, standardize=False)
    ip, ineg = cfg["bands"].index(cfg["pos"]), cfg["bands"].index(cfg["neg"])
    L = res["loadings"]
    best, best_contrast = 1, -1.0
    for i in range(len(L)):  # a brightness PC (all loadings same sign) never qualifies
        a, b = L[i, ip], L[i, ineg]
        if a * b < 0 and abs(a - b) > best_contrast:
            best, best_contrast = i, abs(a - b)
    sign = 1.0 if L[best, ip] > 0 else -1.0
    return {"image": (res["pcs"][best] * sign).astype(np.float32), "pc_index": best,
            "sign": sign, "loadings": L, "explained": res["explained"], "names": cfg["bands"],
            "label": cfg["label"], "info": cfg["info"]}
