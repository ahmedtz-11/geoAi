"""Section 2 pipeline: analyse each dataset, build evidence layers on one grid, predict gold targets.

Datasets (prepared by the interface, all coordinates already in one working UTM system):
  geophysics  {"survey", "kind": "points"|"grid"|"section", x, y, v | array, transform | line, x, y, depth, v}
  geology     {"kind": "points", x, y, lith, strike?, dip?}  or  {"kind": "image_ref", ...}
  remote      {"kind": "grid", layers{name: array}, transform}  or  {"kind": "points", x, y, layers{}}
  geochem     {"kind": "points", x, y, table (clean floats), bdl{}, au, pathfinders[]}
"""
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from . import geochem, geophysics, lineaments, magnetics, prospectivity
from .indices import extract_zones


# ---------- per-dataset analysis ----------
def analyze_geophysics(ds, si=1.0, euler_window=10, regional_height=None, lin_pct=90, lin_len_cells=15,
                       smooth_m=None, inc=None, dec=None, reduction="Auto", up_height=None, epsg=None,
                       cell=None, grid_method=geophysics.GRID_METHODS[0], despike_tol=6.0, microlevel=False, classic=False):
    """classic=True reproduces the original "Normal" image: every reading kept, linear interpolation,
    cell from the station spacing, blank beyond 2.5 spacings (gaps between lines stay visible)."""
    out = {"survey": ds["survey"], "kind": ds["kind"]}
    if ds["kind"] == "section":
        out["lines"] = {}
        for lid, g in pd.DataFrame({"line": ds["line"], "x": ds["x"], "depth": ds["depth"], "v": ds["v"]}).groupby("line"):
            thr = float(np.nanpercentile(ds["v"], 90))
            out["lines"][lid] = {"data": g, "anomalies": geophysics.section_anomalies(g.x, g.depth, g.v, thr),
                                 "threshold": thr}
        if ds.get("y") is not None:  # stations have map positions: use the max value per station for the plan map
            st = pd.DataFrame({"x": ds["x_map"], "y": ds["y"], "v": ds["v"], "depth": ds["depth"]}).dropna()
            agg = st.groupby(["x", "y"]).agg(v=("v", "max")).reset_index()
            out["plan_points"] = agg
            if not geophysics.is_collinear(agg.x, agg.y):  # several lines in one file: plan map
                out["grid"] = geophysics.grid_points(agg.x, agg.y, agg.v)
        return out
    if ds["kind"] == "points":
        x, y, v = ds["x"], ds["y"], ds["v"]
        keep, sigma = geophysics.despike(x, y, v, tol=0.0 if classic else despike_tol)
        out["despiked"] = {"dropped": int((~keep).sum()), "total": int(keep.size), "sigma": float(sigma),
                           "tol": float(despike_tol)}
        if keep.sum() >= 4:
            x, y, v = np.asarray(x)[keep], np.asarray(y)[keep], np.asarray(v)[keep]
        grid = geophysics.grid_points(x, y, v, method=geophysics.GRID_METHODS[1], mask_factor=2.5,
                                      line_aware=False) if classic else \
            geophysics.grid_points(x, y, v, cell=cell, method=grid_method)
        gm = grid["geometry"]
        if not classic and microlevel and gm.get("is_line_survey"):
            lev, corr = geophysics.microlevel(grid["zf"], grid["cell"], gm["azimuth"], gm["line_spacing"])
            grid["zf"] = lev
            grid["z"] = np.where(grid["valid"], lev, np.nan).astype(np.float32)
            out["microlevelled"] = float(np.std(corr))
    else:
        grid = geophysics.grid_from_raster(ds["array"], ds["transform"])
    out["grid"] = grid
    if ds["survey"].startswith("Magnetic"):
        if smooth_m is None:  # auto: about one average data spacing
            smooth_m = geophysics.mean_spacing(ds["x"], ds["y"]) if ds["kind"] == "points" else 2 * grid["cell"]
        out["smooth_m"] = float(smooth_m)
        if inc is None and epsg is not None:  # Earth's field direction from IGRF at the data centre
            from pyproj import Transformer
            lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(
                float(np.nanmean(grid["X"])), float(np.nanmean(grid["Y"])))
            inc, dec, _, _ = magnetics.field_direction(lon, lat)
        if inc is not None:
            out.update(magnetics.analyze(grid, inc, dec or 0.0, reduction, smooth_m, si, euler_window, up_height,
                                         lin_pct, lin_len_cells))
            return out
        prods, derivs, zs = geophysics.magnetic_products(grid, regional_height, smooth_m)
        out["products"] = prods
        eu = geophysics.euler(grid, zs, derivs, si, euler_window)
        eu["depth"] = eu["depth"] - smooth_m  # undo the upward continuation
        out["euler"] = eu[eu.depth > 0].reset_index(drop=True)
        k, lp = geophysics.radial_spectrum(grid["zf"], grid["cell"])
        out["spectrum"] = (k, lp, geophysics.spectral_depths(k, lp))
        lines, _ = lineaments.detect([prods["Tilt derivative (°)"]], grid["valid"], grid["cell"], sigma=1.5,
                                     high_pct=lin_pct, min_length_m=lin_len_cells * grid["cell"],
                                     max_gap_m=2 * grid["cell"], merge_m=3 * grid["cell"])
        out["lineaments"] = lines
    else:
        v = grid["z"][grid["valid"]]
        out["limits"] = [float(np.nanmean(v) + k * np.nanstd(v)) for k in (1, 2, 3)]
    return out


def analyze_geochem(ds, method=geochem.METHODS[0]):
    t = ds["table"]
    au = ds["au"]
    stats = pd.DataFrame({c: geochem.stats(t[c].to_numpy(float), ds["bdl"].get(c)) for c in t.columns}).T
    limits = {c: geochem.thresholds(t[c].to_numpy(float), method) for c in t.columns}
    stats["weak ≥"] = [limits[c][0] for c in stats.index]
    stats["anomalous ≥"] = [limits[c][1] for c in stats.index]
    stats["strong ≥"] = [limits[c][2] for c in stats.index]
    out = {"stats": stats, "limits": limits, "corr": geochem.correlation(t)}
    if au:
        out["au_class"] = geochem.classify(t[au].to_numpy(float), limits[au])
        out["index"], out["index_weights"] = geochem.gold_index(t, au, ds["pathfinders"])
    return out


# ---------- common grid & resampling ----------
def bounds_of(ds):
    if ds.get("kind") in ("points", "section") and ds.get("x") is not None:
        x = ds["x_map"] if ds.get("kind") == "section" else ds["x"]
        y = ds.get("y")
        if y is None:
            return None
        return float(np.nanmin(x)), float(np.nanmax(x)), float(np.nanmin(y)), float(np.nanmax(y))
    if ds.get("transform") is not None:
        a = ds.get("array") if ds.get("array") is not None else next(iter(ds["layers"].values()))
        t = ds["transform"]
        h, w = a.shape
        return t.c, t.c + w * t.a, t.f + h * t.e, t.f
    return None


def common_grid(bounds_list, max_cells=250, min_cell=5.0):
    from rasterio.transform import from_origin
    b = np.array([x for x in bounds_list if x is not None])
    x0, x1, y0, y1 = b[:, 0].min(), b[:, 1].max(), b[:, 2].min(), b[:, 3].max()
    cell = max(max(x1 - x0, y1 - y0) / max_cells, min_cell)
    w, h = int(np.ceil((x1 - x0) / cell)) + 1, int(np.ceil((y1 - y0) / cell)) + 1
    tr = from_origin(x0 - cell / 2, y1 + cell / 2, cell, cell)
    return tr, (h, w), cell


def cell_centres(transform, shape):
    h, w = shape
    xs = transform.c + transform.a * (np.arange(w) + 0.5)
    ys = transform.f + transform.e * (np.arange(h) + 0.5)
    return np.meshgrid(xs, ys)


def resample(arr, src_transform, dst_transform, shape, epsg):
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    dst = np.full(shape, np.nan, np.float32)
    reproject(np.asarray(arr, np.float32), dst, src_transform=src_transform, src_crs=f"EPSG:{epsg}",
              dst_transform=dst_transform, dst_crs=f"EPSG:{epsg}", src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    return dst


def points_to_grid(x, y, v, transform, shape, max_dist=None):
    x, y, v = (np.asarray(a, float) for a in (x, y, v))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[ok], y[ok], v[ok]
    if len(x) < 3:
        return np.full(shape, np.nan, np.float32)
    X, Y = cell_centres(transform, shape)
    z = griddata((x, y), v, (X, Y), method="linear")
    near = griddata((x, y), v, (X, Y), method="nearest")
    z = np.where(np.isnan(z), near, z)
    md = max_dist or 2.5 * geophysics.point_spacing(x, y)
    d, _ = cKDTree(np.column_stack([x, y])).query(np.column_stack([X.ravel(), Y.ravel()]))
    return np.where(d.reshape(shape) <= md, z, np.nan).astype(np.float32)


def distance_to_points(x, y, transform, shape):
    X, Y = cell_centres(transform, shape)
    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    pts = pts[np.all(np.isfinite(pts), 1)]
    if len(pts) == 0:
        return np.full(shape, np.inf, np.float32)
    d, _ = cKDTree(pts).query(np.column_stack([X.ravel(), Y.ravel()]))
    return d.reshape(shape).astype(np.float32)


# ---------- evidence ----------
def build_evidence(datasets, results, transform, shape, epsg, favourable_liths=None):
    """Return {name: dict(raw, default_weight, favourable ('High'|'Low'), source, info)}."""
    ev = {}
    cell = abs(transform.a)
    for i, ds in enumerate(datasets.get("geophysics", [])):
        r = results["geophysics"][i]
        tag = f" ({ds['name']})" if len(datasets["geophysics"]) > 1 else ""
        g = r.get("grid")
        if g is None:
            continue  # single IP lines are combined below
        if ds["survey"].startswith("Magnetic"):
            p = r["products"]
            ev["Magnetic analytic signal" + tag] = dict(
                raw=resample(p["Analytic signal"], g["transform"], transform, shape, epsg), default_weight=0.3,
                favourable="High", source="Magnetic", info="Edges and bodies of magnetic rocks (BIF, mafic units).")
            ev["Magnetic low (magnetite destruction)" + tag] = dict(
                raw=resample(p["Residual"], g["transform"], transform, shape, epsg), default_weight=0.0,
                favourable="Low", source="Magnetic",
                info="Hydrothermal alteration along shears can destroy magnetite, leaving magnetic lows.")
            lines = r.get("lineaments") or []
            if lines:
                gshape = g["z"].shape
                dens = lineaments.density(lines, gshape, g["cell"], radius_m=8 * g["cell"])
                dist = lineaments.distance(lines, gshape, g["cell"])
                ev["Magnetic structure density" + tag] = dict(
                    raw=resample(np.where(g["valid"], dens, np.nan), g["transform"], transform, shape, epsg),
                    default_weight=0.8, favourable="High", source="Magnetic",
                    info="Concentration of faults / contacts traced on the tilt derivative.")
                ev["Proximity to magnetic structures" + tag] = dict(
                    raw=resample(np.where(g["valid"], dist, np.nan), g["transform"], transform, shape, epsg),
                    default_weight=0.6, favourable="Low", source="Magnetic",
                    info="Distance to the nearest structure (closer is better).")
        else:
            charge = "chargeability" in ds["survey"]
            ev[("IP chargeability" if charge else "IP resistivity") + tag] = dict(
                raw=resample(g["z"], g["transform"], transform, shape, epsg),
                default_weight=0.8 if charge else 0.3, favourable="High", source="IP",
                info="High chargeability = disseminated sulphides (pyrite, arsenopyrite) that often host gold."
                if charge else "High resistivity can mark silicification / quartz veining; low can mark clay "
                               "alteration or graphitic shear. Set the direction that fits your geology.")
    # IP lines uploaded one file per line: grid all lines of the same survey type together
    for survey in {ds["survey"] for ds in datasets.get("geophysics", []) if ds["kind"] == "section"}:
        pts = [r["plan_points"] for ds, r in zip(datasets["geophysics"], results["geophysics"])
               if ds["survey"] == survey and r.get("grid") is None and isinstance(r.get("plan_points"), pd.DataFrame)]
        if not pts:
            continue
        p = pd.concat(pts)
        if len(p) < 4 or geophysics.is_collinear(p.x, p.y):
            continue
        charge = "chargeability" in survey
        name = f"IP {'chargeability' if charge else 'resistivity'} ({len(pts)} line{'s' if len(pts) > 1 else ''})"
        g = geophysics.grid_points(p.x, p.y, p.v, mask_factor=1.5)
        ev[name] = dict(raw=resample(g["z"], g["transform"], transform, shape, epsg),
                        default_weight=0.8 if charge else 0.3, favourable="High", source="IP",
                        info="Maximum value at each station, gridded between the survey lines.")
    if favourable_liths is not None:
        lay = lithology_layer(datasets, favourable_liths, transform, shape)
        if lay is not None:
            ev["Proximity to favourable lithology"] = lay
    for ds in datasets.get("geology", []):
        if ds["kind"] != "points":
            continue
        if ds.get("strike") is not None and np.isfinite(ds["strike"]).any():
            ok = np.isfinite(ds["strike"])
            d = distance_to_points(ds["x"][ok], ds["y"][ok], transform, shape)
            ev["Proximity to measured structures"] = dict(raw=d, default_weight=0.3, favourable="Low",
                                                          source="Geology",
                                                          info="Distance to field structural measurements.")
    for ds in datasets.get("remote", []):
        for name, arr in ds["layers"].items():
            raw = resample(arr, ds["transform"], transform, shape, epsg) if ds["kind"] == "grid" else \
                points_to_grid(ds["x"], ds["y"], arr, transform, shape)
            if np.isfinite(raw).sum() < 10:
                continue
            ev[f"Remote sensing: {name}"] = dict(raw=raw, default_weight=0.5, favourable="High",
                                                 source="Remote sensing", info=f"From {ds['name']}.")
    for i, ds in enumerate(datasets.get("geochem", [])):
        r = results["geochem"][i]
        if "index" in r:
            ev["Gold geochemistry index"] = dict(
                raw=points_to_grid(ds["x"], ds["y"], r["index"], transform, shape), default_weight=1.0,
                favourable="High", source="Geochemistry", info="Au and correlated pathfinders (log z-scores).")
    return ev


def lithology_layer(datasets, favourable_liths, transform, shape):
    xs, ys = [], []
    for ds in datasets.get("geology", []):
        if ds["kind"] == "points":
            fav = np.isin(pd.Series(ds["lith"]).astype(str), list(favourable_liths))
            xs.append(ds["x"][fav])
            ys.append(ds["y"][fav])
    if not xs or not sum(len(x) for x in xs):
        return None
    return dict(raw=distance_to_points(np.concatenate(xs), np.concatenate(ys), transform, shape), default_weight=0.6,
                favourable="Low", source="Geology", info="Distance to mapped gold-favourable host rocks.")


def run(specs, settings=None, progress=None):
    """Full Section 2 run. specs: list of upload specs (see geoai.datasets.build)."""
    from . import datasets as dsm
    from .geology import favourable_default

    settings = settings or {}
    say = progress or (lambda p, t: None)
    work = dsm.working_epsg(specs)
    data = {k: [] for k in dsm.SLOTS}
    errors = []
    for i, sp in enumerate(specs):
        say(0.05 + 0.25 * i / max(len(specs), 1), f"Reading {sp['name']}…")
        try:
            data[sp["slot"]].append(dsm.build(sp, work))
        except Exception as e:
            errors.append(f"{sp['name']}: {e}")
    results = {"geophysics": [], "geochem": []}
    for i, ds in enumerate(data["geophysics"]):
        say(0.35 + 0.3 * i / max(len(data["geophysics"]), 1), f"Processing {ds['name']}…")
        try:
            results["geophysics"].append(analyze_geophysics(
                ds, si=settings.get("si", 1.0), euler_window=settings.get("euler_window", 10),
                smooth_m=settings.get("smooth_m"), inc=settings.get("inc"), dec=settings.get("dec"),
                reduction=settings.get("reduction", "Auto"), up_height=settings.get("up_height"), epsg=work,
                cell=settings.get("cell"),
                grid_method=settings.get("grid_method", geophysics.GRID_METHODS[0]),
                despike_tol=settings.get("despike_tol", 6.0),
                microlevel=settings.get("microlevel", False)))
        except Exception as e:
            errors.append(f"{ds['name']}: {e}")
            results["geophysics"].append({"survey": ds.get("survey", ""), "kind": ds["kind"], "error": str(e)})
    keep = [i for i, r in enumerate(results["geophysics"]) if "error" not in r]
    data["geophysics"] = [data["geophysics"][i] for i in keep]
    results["geophysics"] = [results["geophysics"][i] for i in keep]
    if settings.get("normal_view"):      # second image type: the original, unfilled processing
        results["geophysics_normal"] = []
        for ds in data["geophysics"]:
            say(0.66, f"Normal image for {ds['name']}…")
            try:
                results["geophysics_normal"].append(analyze_geophysics(
                    ds, si=settings.get("si", 1.0), euler_window=settings.get("euler_window", 10),
                    smooth_m=settings.get("smooth_m"), inc=settings.get("inc"), dec=settings.get("dec"),
                    reduction=settings.get("reduction", "Auto"), up_height=settings.get("up_height"),
                    epsg=work, classic=True) if ds["kind"] == "points" else None)
            except Exception:
                results["geophysics_normal"].append(None)
    for ds in data["geochem"]:
        say(0.7, f"Analysing {ds['name']}…")
        results["geochem"].append(analyze_geochem(ds, settings.get("geochem_method", geochem.METHODS[0])))
    liths = sorted({str(l) for ds in data["geology"] if ds["kind"] == "points" for l in ds["lith"]})
    fav = favourable_default(liths)
    say(0.8, "Building evidence layers…")
    spatial = [d for k in data for d in data[k] if d["kind"] != "image_ref"]
    bl = [bounds_of(d) for d in spatial]
    grid, evidence = None, {}
    if any(b is not None for b in bl):
        tr, shape, cell = common_grid(bl)
        grid = (tr, shape, cell)
        evidence = build_evidence(data, results, tr, shape, work, fav)
    say(1.0, "Done")
    return {"epsg": work, "datasets": data, "results": results, "grid": grid, "evidence": evidence,
            "lithologies": liths, "fav_default": fav, "errors": errors, "settings": settings}


def membership(raw, favourable="High", valid=None):
    """Rescale an evidence layer to 0-1 (1 = most favourable)."""
    v = np.isfinite(raw) if valid is None else valid & np.isfinite(raw)
    return prospectivity.fuzzy_linear(raw, v, 2, 98, invert=favourable == "Low")


def knowledge_driven(members, weights, min_weight_frac=0.5):
    """Weighted mean of the available memberships per cell; cells need >= min_weight_frac of total weight."""
    tot = sum(w for w in weights.values() if w > 0)
    if not tot:
        return None, None
    shape = next(iter(members.values())).shape
    acc, wsum = np.zeros(shape), np.zeros(shape)
    for k, m in members.items():
        w = weights.get(k, 0)
        if w <= 0:
            continue
        ok = np.isfinite(m)
        acc[ok] += w * m[ok]
        wsum[ok] += w
    cover = wsum / tot
    score = np.where(cover >= min_weight_frac, acc / np.where(wsum == 0, 1, wsum), np.nan)
    return score.astype(np.float32), cover


def data_driven(members, sample_xy, labels, transform, shape, n_trees=300, seed=0):
    """Random Forest trained on gold-anomalous vs background samples using the non-geochemical evidence.
    Returns dict(prob grid, importances, auc (spatial block cross-validation), n_pos, n_neg) or reason."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    names = list(members)
    if not names:
        return {"error": "No non-geochemical evidence layers to learn from."}
    stack = np.stack([np.nan_to_num(members[n], nan=0.5) for n in names], -1)
    h, w = shape
    cols = np.clip(((sample_xy[:, 0] - transform.c) / transform.a).astype(int), 0, w - 1)
    rows = np.clip(((sample_xy[:, 1] - transform.f) / transform.e).astype(int), 0, h - 1)
    X = stack[rows, cols]
    y = np.asarray(labels, int)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos < 5 or nneg < 5:
        return {"error": f"Need at least 5 gold-anomalous and 5 background samples (have {npos} / {nneg})."}
    blocks = (rows // max(h // 5, 1)) * 10 + cols // max(w // 5, 1)
    aucs = []
    nfold = min(5, len(np.unique(blocks)))
    if nfold >= 2:
        for tr_i, te_i in GroupKFold(nfold).split(X, y, blocks):
            if len(np.unique(y[te_i])) < 2 or len(np.unique(y[tr_i])) < 2:
                continue
            m = RandomForestClassifier(n_trees, min_samples_leaf=3, class_weight="balanced", random_state=seed,
                                       n_jobs=-1).fit(X[tr_i], y[tr_i])
            aucs.append(roc_auc_score(y[te_i], m.predict_proba(X[te_i])[:, 1]))
    model = RandomForestClassifier(n_trees, min_samples_leaf=3, class_weight="balanced", random_state=seed,
                                   n_jobs=-1).fit(X, y)
    prob = model.predict_proba(stack.reshape(-1, len(names)))[:, 1].reshape(shape).astype(np.float32)
    covered = np.zeros(shape, bool)
    for n in names:
        covered |= np.isfinite(members[n])
    return {"prob": np.where(covered, prob, np.nan), "importances": dict(zip(names, model.feature_importances_)),
            "auc": float(np.mean(aucs)) if aucs else None, "n_pos": npos, "n_neg": nneg}


# ---------- targets ----------
def describe_targets(score, valid, transform, top_pct, min_pixels, members, results, datasets, radius_m=None):
    z = extract_zones(score, valid & np.isfinite(score), top_pct, min_pixels, transform)
    cell = abs(transform.a)
    radius = radius_m or max(15 * cell, 200)
    euler_pts = [r["euler"] for r in results.get("geophysics", []) if isinstance(r.get("euler"), pd.DataFrame)
                 and len(r["euler"])]
    euler_all = pd.concat(euler_pts) if euler_pts else None
    spi_pts = [r["spi_points"] for r in results.get("geophysics", []) if isinstance(r.get("spi_points"), pd.DataFrame)
               and len(r["spi_points"])]
    spi_all = pd.concat(spi_pts) if spi_pts else None
    spec = next((r["spectrum"][2] for r in results.get("geophysics", []) if r.get("spectrum")), {})
    lines_all = []
    for r in results.get("geophysics", []):
        g = r.get("grid")
        for l in r.get("lineaments") or []:
            t = g["transform"]
            lines_all.append((t.c + t.a * (l["x0"] + l["x1"]) / 2, t.f + t.e * (l["y0"] + l["y1"]) / 2,
                              l["azimuth"], l["length_m"]))
    geo_strikes = []
    for ds in datasets.get("geology", []):
        if ds.get("strike") is not None:
            ok = np.isfinite(ds["strike"])
            geo_strikes += list(zip(ds["x"][ok], ds["y"][ok], ds["strike"][ok]))
    ip_secs = []
    for i, r in enumerate(results.get("geophysics", [])):
        ds = datasets["geophysics"][i]
        if r["kind"] == "section" and ds.get("y") is not None:
            d = pd.DataFrame({"x": ds["x_map"], "y": ds["y"], "depth": ds["depth"], "v": ds["v"]}).dropna()
            thr = np.nanpercentile(d.v, 90)
            ip_secs.append(d[d.v >= thr])
    ip_hot = pd.concat(ip_secs) if ip_secs else None

    rows = []
    for k, zz in enumerate(z["zones"], 1):
        e, n = zz["easting"], zz["northing"]
        depth, depth_src = None, ""
        if euler_all is not None:
            near = euler_all[np.hypot(euler_all.x - e, euler_all.y - n) <= radius]
            if len(near):
                depth, depth_src = float(near.depth.median()), f"Euler ({len(near)} solutions)"
        if depth is None and spi_all is not None:
            near = spi_all[np.hypot(spi_all.x - e, spi_all.y - n) <= radius]
            if len(near) >= 3:
                depth, depth_src = float(near.depth.median()), f"SPI ({len(near)} points)"
        if depth is None and ip_hot is not None and len(ip_hot):
            near = ip_hot[np.hypot(ip_hot.x - e, ip_hot.y - n) <= radius]
            if len(near):
                depth, depth_src = float(near.depth.min()), "IP section (top of high chargeability)"
        if depth is None and spec.get("shallow"):
            depth, depth_src = spec["shallow"]["depth"], "Power spectrum (area average)"
        trend, trend_src = None, ""
        near_l = [(a, L) for x, y, a, L in lines_all if np.hypot(x - e, y - n) <= radius * 1.5]
        if near_l:
            trend, trend_src = _dominant([a for a, _ in near_l], [L for _, L in near_l]), "Magnetic structures"
        elif geo_strikes:
            near_g = [s for x, y, s in geo_strikes if np.hypot(x - e, y - n) <= radius * 3]
            if near_g:
                trend, trend_src = _dominant(near_g), "Field measurements"
        r0, c0 = int(round(zz["row"])), int(round(zz["col"]))
        strong = [nm for nm, mem in members.items()
                  if np.isfinite(mem[r0, c0]) and mem[r0, c0] >= 0.7]
        rows.append({"Target": f"T{k}", "Easting": round(e, 1), "Northing": round(n, 1),
                     "Area (ha)": round(zz["area_ha"], 2), "Score": round(zz["mean_score"], 3),
                     "Est. depth (m)": None if depth is None else round(depth, 0), "Depth from": depth_src,
                     "Trend": None if trend is None else f"{lineaments.strike_label(trend)} ({trend:.0f}°)",
                     "Trend from": trend_src, "Supporting evidence": ", ".join(strong) or "—"})
    return z, pd.DataFrame(rows)


def _dominant(az, weights=None, bin_deg=10):
    az = np.asarray(az, float) % 180
    h, edges = np.histogram(az, bins=np.arange(0, 180 + bin_deg, bin_deg), weights=weights)
    i = int(np.argmax(h))
    sel = (az >= edges[i]) & (az < edges[i + 1])
    return float(np.mean(az[sel])) if sel.any() else float(edges[i] + bin_deg / 2)
