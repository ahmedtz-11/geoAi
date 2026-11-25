"""1 · Remote sensing: four corner coordinates in, nine interpreted satellite maps out.

The area is cheap to rebuild from its corners, so it is kept as a small file and recomputed each
request. The Sentinel-2 scene is not: fetching it is a download of tens of megabytes, so it runs as a
background job and is stored with the run, and every map below is drawn from that stored scene.
"""
import json
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
from flask import abort, flash, redirect, render_template, request, url_for
from flask_login import current_user

from geoai import coords, export, indices, lineaments, lithology, pca, prospectivity, terrain, terrain3d
from geoai.imagery import basemap
from geoai.imagery import dem as dem_mod
from geoai.imagery import sentinel
from geoai.io import read_raw_table

from .. import store, uploads
from ..render import plotly_div, safe_name, table_html
from . import base

KEY = "remote-sensing"
VEG_NDVI = 0.4
CORNERS_FILE = "corners.csv"
SAMPLE_CSV = ("corner,latitude,longitude\n"
              "C1,-2.8500,32.1500\n"
              "C2,-2.8500,32.1900\n"
              "C3,-2.8900,32.1900\n"
              "C4,-2.8900,32.1500\n")

TABS = [("High-resolution image", "image"), ("Iron oxide/ferric iron", "iron"),
        ("Clay/hydroxyl alteration", "clay"), ("Band ratios", "ratios"), ("PCA images", "pca"),
        ("Lithological classification", "lithology"), ("Geological lineaments", "lineaments"),
        ("DEM / hillshade", "dem"), ("3-D terrain", "terrain3d"),
        ("⭐ Integrated prospectivity", "prospectivity")]

DRAPES = {
    "Shaded relief": ("picture", None),
    "Satellite true colour": ("picture", "True colour (B4-B3-B2)"),
    "Geology false colour": ("picture", "Geology false colour (B12-B11-B2)"),
    "Alteration ratios (clay / iron / ferric)":
        ("picture", "Alteration ratios (R=B11/B12 clay, G=B4/B2 iron, B=B11/B8A ferric)"),
    "Elevation": ("value", "elevation"),
    "Slope": ("value", "slope"),
    "Clay / hydroxyl alteration": ("value", "clay_hydroxyl"),
    "Iron oxide": ("value", "iron_oxide"),
}
SCALES = {"elevation": ("Earth", "Elevation (m)", True), "slope": ("YlOrBr", "Slope (°)", False),
          "clay_hydroxyl": ("YlOrRd", "Clay / hydroxyl", False),
          "iron_oxide": ("YlOrRd", "Iron oxide", False)}
DETAIL = {"Standard (fastest)": 250, "High": 400, "Very high (slower)": 600}
PROSPECT_LAYERS = {
    "clay": ("Clay / hydroxyl alteration", 1.0, "Purples"),
    "iron": ("Iron oxide (gossan)", 0.7, "YlOrRd"),
    "lden": ("Lineament density", 0.8, "hot_r"),
    "lprox": ("Proximity to lineaments", 0.6, "Reds"),
    "lith": ("Favourable spectral units", 0.4, "Greens"),
}


def attach(bp):
    from .. import section
    bp.add_url_rule("/remote-sensing", "remote_sensing", section(KEY)(page), methods=["GET", "POST"])
    bp.add_url_rule("/remote-sensing/panel/<name>", "remote_panel", section(KEY)(panel))
    bp.add_url_rule("/remote-sensing/scenes", "remote_scenes", section(KEY)(scenes), methods=["POST"])
    bp.add_url_rule("/remote-sensing/aoi.<kind>", "remote_aoi_file", section(KEY)(aoi_file))


# ------------------------------------------------------------------ the area
def _corners_path():
    for p in store.draft_files(current_user.id, KEY):
        return p
    return None


def _read_corners():
    p = _corners_path()
    if p is None:
        return None, None
    try:
        with open(p, "rb") as fh:
            buf = uploads._named(fh.read(), p.name)
        return read_raw_table(buf), None
    except Exception as e:
        return None, f"Could not read {p.name}: {e}"


def _aoi(values):
    """The area of interest, rebuilt from the stored corners and the chosen columns."""
    df, err = _read_corners()
    if df is None:
        return None, None, err
    if df.empty:
        return None, df, "That file has no rows."
    cols = [str(c) for c in df.columns]
    gx, gy = coords.guess_columns(df)
    xcol = values.get("xcol") or (str(gx) if gx is not None else cols[0])
    ycol = values.get("ycol") or (str(gy) if gy is not None else cols[min(1, len(cols) - 1)])
    if xcol not in df.columns or ycol not in df.columns:
        return None, df, "Choose the longitude and latitude columns."
    # with nothing chosen yet, fall back to what the numbers themselves look like
    epsg = uploads.resolve_crs(values, "crs",
                               uploads.default_crs(coords.parse_column(df[xcol]).dropna(),
                                                   coords.parse_column(df[ycol]).dropna()))
    try:
        return coords.aoi_from_table(df, xcol, ycol, epsg), df, None
    except Exception as e:
        return None, df, str(e)


def _corner_fields(df, values):
    if df is None:
        return []
    cols = [str(c) for c in df.columns]
    gx, gy = coords.guess_columns(df)
    f = uploads.field
    xs = coords.parse_column(df[str(gx)]).dropna() if gx is not None and str(gx) in df else pd.Series(dtype=float)
    ys = coords.parse_column(df[str(gy)]).dropna() if gy is not None and str(gy) in df else pd.Series(dtype=float)
    return [
        f("xcol", "Longitude / Easting column", "select", cols,
          values.get("xcol") or (str(gx) if gx is not None else cols[0]), width="third"),
        f("ycol", "Latitude / Northing column", "select", cols,
          values.get("ycol") or (str(gy) if gy is not None else cols[min(1, len(cols) - 1)]),
          width="third"),
        uploads.crs_field("crs", "Coordinate system", xs, ys,
                          "Detected from the values. Tanzania spans UTM zones 35S–37S; older maps "
                          "often use Arc 1960.", width="third"),
    ]


def _aoi_map(aoi, height=440):
    import plotly.graph_objects as go
    lon, lat = list(aoi.lon) + [aoi.lon[0]], list(aoi.lat) + [aoi.lat[0]]
    fig = go.Figure()
    fig.add_trace(go.Scattermap(lon=lon, lat=lat, mode="lines", fill="toself",
                                fillcolor="rgba(255,212,0,0.12)", line=dict(color="#ffd400", width=3),
                                hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scattermap(lon=list(aoi.lon), lat=list(aoi.lat), mode="markers+text",
                                marker=dict(size=9, color="#ffd400"),
                                text=[f"C{i + 1}" for i in range(len(aoi.lon))],
                                textposition="top right", textfont=dict(color="white", size=13),
                                hovertemplate="%{text}<br>Lat %{lat:.6f}<br>Lon %{lon:.6f}<extra></extra>",
                                showlegend=False))
    cx, cy = aoi.centroid
    span = max(aoi.bbox[2] - aoi.bbox[0], (aoi.bbox[3] - aoi.bbox[1]) * 1.3, 1e-6)
    zoom = float(np.clip(math.log2(360 / span) - 0.8, 2, 17))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                      map=dict(style="white-bg", center=dict(lon=cx, lat=cy), zoom=zoom,
                               layers=[dict(below="traces", sourcetype="raster",
                                            source=[basemap.ESRI_URL],
                                            sourceattribution=basemap.ESRI_ATTRIBUTION)]))
    return plotly_div(fig, "aoi-map", height=height)


# ------------------------------------------------------------------ the page
def page():
    if request.method == "POST":
        action = request.form.get("do")
        if action == "upload":
            store.clear_draft(current_user.id, KEY)
            base.take_upload(KEY)
            return redirect(url_for("app.remote_sensing"))
        if action == "typed":
            text = request.form.get("corners", "").strip() or SAMPLE_CSV
            store.clear_draft(current_user.id, KEY)
            store.add_draft(current_user.id, KEY, CORNERS_FILE, text.encode("utf-8"))
            return redirect(url_for("app.remote_sensing"))
        if action in ("remove", "clear"):
            store.clear_draft(current_user.id, KEY)
            return redirect(url_for("app.remote_sensing"))
        if action == "load":
            return _load_scene()

    aoi, df, err = _aoi(request.values)
    run = base.current_run(KEY)
    scene = _scene(run) if run is not None else None
    if scene is not None and aoi is not None and scene["key"][0] != aoi.key:
        scene = None                        # the stored scene belongs to a different area
    typed = _corners_path()
    clear_pct = 0.0
    if scene is not None:
        d = scene["data"]
        clear_pct = 100 * d["clear"].sum() / max(d["inside"].sum(), 1)
    return base.page(KEY, "app/remote.html", aoi=aoi, corners_df=df, aoi_error=err,
                     fields=_corner_fields(df, request.values), run=run, scene=scene,
                     tabs=TABS, sample=SAMPLE_CSV, clear_pct=clear_pct,
                     aoi_map=_aoi_map(aoi) if aoi is not None else None,
                     aoi_q=_aoi_query(), hidden_aoi=_hidden_aoi(),
                     resolutions=[r for r in sentinel.RESOLUTIONS
                                  if aoi is None or r >= sentinel.auto_resolution(aoi)],
                     corners_text=(typed.read_text(encoding="utf-8", errors="replace")
                                   if typed and typed.name == CORNERS_FILE else SAMPLE_CSV),
                     have_file=typed is not None and typed.name != CORNERS_FILE,
                     filename=typed.name if typed else None,
                     groups=_remembered_scenes(), search=_search_defaults())


def _aoi_query():
    """The column and CRS choices, carried in every URL so each request rebuilds the same area."""
    v = request.values
    q = {k: v.get(k) for k in ("xcol", "ycol", "crs", "crs_epsg") if v.get(k)}
    return q


def _hidden_aoi():
    """The same choices as hidden form inputs."""
    from markupsafe import Markup, escape
    return Markup("".join(f'<input type="hidden" name="{escape(k)}" value="{escape(v)}">'
                          for k, v in _aoi_query().items()))


def _search_defaults():
    v = request.values
    return {"start": v.get("start") or str(date.today() - timedelta(days=120)),
            "end": v.get("end") or str(date.today()),
            "cloud": int(v.get("cloud", 20) or 20),
            "res": v.get("res", "")}


def _remembered_scenes():
    """The last search's results, kept in the Flask session so the list survives a page load."""
    from flask import session
    got = session.get("rs_scenes")
    aoi, _df, _err = _aoi(request.values)
    if not got or aoi is None or got.get("aoi") != aoi.key:
        return None
    return got.get("groups")


def scenes():
    """Search the Sentinel-2 archive and remember what came back."""
    from flask import session
    aoi, _df, err = _aoi(request.form)
    if aoi is None:
        flash(err or "Set the area first.", "error")
        return redirect(url_for("app.remote_sensing"))
    s = _search_defaults()
    try:
        groups = sentinel.search_scenes(json.loads(aoi.geometry_json()), s["start"], s["end"],
                                        s["cloud"])
    except Exception as e:
        flash(f"The scene search failed: {e}", "error")
        return redirect(url_for("app.remote_sensing") + "#scene")
    session["rs_scenes"] = {
        "aoi": aoi.key,
        "groups": [{k: g[k] for k in ("date", "cloud", "coverage", "tiles", "platform")}
                   | {"i": i} for i, g in enumerate(groups)]}
    session["rs_scene_raw"] = None               # the items themselves are re-fetched on load
    if not groups:
        flash("No scenes found. Widen the date range or raise the cloud limit.", "error")
    return redirect(url_for("app.remote_sensing") + "#scene")


def _load_scene():
    aoi, _df, err = _aoi(request.form)
    if aoi is None:
        flash(err or "Set the area first.", "error")
        return redirect(url_for("app.remote_sensing"))
    s = _search_defaults()
    pick = request.form.get("pick")
    auto = request.form.get("auto") == "1"
    try:
        res = int(float(request.form.get("res") or sentinel.auto_resolution(aoi)))
    except (TypeError, ValueError):
        res = sentinel.auto_resolution(aoi)
    geometry = json.loads(aoi.geometry_json())
    aoi_key = aoi.key

    def work(run, job):
        if auto:
            job.say(0.05, "Searching the last 6 months for a scene with under 20% cloud…")
            found = sentinel.search_scenes(geometry, str(date.today() - timedelta(days=180)),
                                           str(date.today()), 20)
            found = [g for g in found if g["coverage"] > 0.99] or found
            if not found:
                raise ValueError("No scene with under 20% cloud in the last 6 months. "
                                 "Choose a date range instead.")
            g, use_res = found[0], sentinel.auto_resolution(aoi)
        else:
            job.say(0.05, "Finding that scene again…")
            groups = sentinel.search_scenes(geometry, s["start"], s["end"], s["cloud"])
            try:
                g = groups[int(pick)]
            except (TypeError, ValueError, IndexError):
                raise ValueError("That scene is no longer in the search results. Search again.")
            use_res = res
        job.say(0.1, f"Loading the Sentinel-2 scene of {g['date']}…")
        data = sentinel.load_scene(g["items"], aoi, use_res,
                                   progress=lambda p, t: job.say(0.1 + 0.75 * min(p, 1.0), t))
        job.say(0.88, "Loading elevation (Copernicus DEM 30 m)…")
        dem, dem_error = None, None
        try:
            h, w = data["inside"].shape
            dem = dem_mod.load_dem(aoi, data["epsg"], data["transform"], w, h)
        except Exception as e:                   # the imagery is still useful without elevation
            dem_error = str(e)
        job.say(0.96, "Saving the scene…")
        run.save_result({"scene": {"key": (aoi_key, g["date"], g["orbit"], use_res), "data": data,
                                   "dem": dem, "dem_error": dem_error, "cloud": g["cloud"],
                                   "tiles": g["tiles"]},
                         "aoi_key": aoi_key})

    return base.submit(KEY, work, label="Fetching satellite data", keep_files=False,
                       settings={"auto": auto, "pick": pick, "res": res})


def aoi_file(kind):
    """The area itself, for QGIS or Google Earth."""
    aoi, _df, err = _aoi(request.values)
    if aoi is None:
        abort(404)
    if kind == "geojson":
        return base.send_bytes(aoi.geojson().encode("utf-8"), "aoi.geojson", "application/geo+json")
    if kind == "kml":
        return base.send_bytes(export.aoi_kml(aoi), "aoi.kml",
                               "application/vnd.google-earth.kml+xml")
    if kind == "csv":
        return base.send_csv(aoi.corners_table(), "aoi_corners.csv")
    abort(404)


# ------------------------------------------------------------------ the scene and what comes from it
def _scene(run):
    if run is None:
        return None
    try:
        return run.result().get("scene")
    except Exception:
        return None


def _need(run):
    scene = _scene(run)
    if scene is None:
        abort(404)
    return scene


def _header(scene, aoi):
    d = scene["data"]
    return d["bands"], d["transform"], d["epsg"], d["inside"], d["clear"], aoi.to_utm(d["epsg"])


def _cache(run, name, fn, *params):
    return run.cached((name,) + tuple(params), fn)


def _usable(run, scene, hide_veg):
    clear = scene["data"]["clear"]
    if not hide_veg:
        return clear
    return clear & (indices.compute_index(scene["data"]["bands"], "ndvi") <= VEG_NDVI)


def _std_pca(run, scene):
    d = scene["data"]
    return _cache(run, "pca", lambda: pca.pca(d["bands"], pca.PCA_BANDS, d["clear"]))


def _crosta(run, scene, kind):
    d = scene["data"]
    return _cache(run, "crosta", lambda: pca.crosta(d["bands"], d["clear"], kind), kind)


def _index_layer(run, scene, kind, method):
    if str(method).startswith("Crósta"):
        return _crosta(run, scene, "hydroxyl" if kind == "clay" else "iron")["image"]
    return indices.compute_index(scene["data"]["bands"],
                                 "clay_hydroxyl" if kind == "clay" else "iron_oxide")


def _lith(run, scene, k):
    d = scene["data"]
    return _cache(run, "lith", lambda: lithology.classify(d["bands"], d["clear"], k), k)


def _hillshades(run, scene):
    dem, res = scene.get("dem"), scene["data"]["res"]
    if dem is None:
        return None
    return _cache(run, "hs4", lambda: [terrain.hillshade(dem, res, a, 30) for a in (0, 45, 90, 135)])


def _lineaments(run, scene, source, pct, sigma, length, gap):
    d = scene["data"]
    valid = d["inside"] & np.isfinite(d["bands"]["B04"])

    def go():
        imgs = []
        valid_ = valid
        if source in ("Elevation (hillshade)", "Both") and scene.get("dem") is not None:
            imgs += _hillshades(run, scene)
            valid_ = valid & np.isfinite(scene["dem"])
        if source in ("Satellite (PC1)", "Both") or not imgs:
            imgs.append(_std_pca(run, scene)["pcs"][0])
        return lineaments.detect(imgs, valid_, d["res"], sigma, pct, length, gap)

    return _cache(run, "lin", go, source, pct, sigma, length, gap)


# ------------------------------------------------------------------ tab settings
def _i(name, default, lo, hi, args=None):
    a = request.values if args is None else args
    try:
        return max(lo, min(int(float(a.get(name, default))), hi))
    except (TypeError, ValueError):
        return default


def _f(name, default, lo, hi, args=None):
    a = request.values if args is None else args
    try:
        return max(lo, min(float(a.get(name, default)), hi))
    except (TypeError, ValueError):
        return default


def _s(name, default, options, args=None):
    a = request.values if args is None else args
    v = a.get(name)
    return v if v in options else default


def _b(name, default, args=None):
    a = request.values if args is None else args
    return bool(a.get(name)) if a.get("_t") else default


def _lin_opts(scene, args=None):
    has_dem = scene.get("dem") is not None
    sources = ["Elevation (hillshade)", "Satellite (PC1)", "Both"] if has_dem else ["Satellite (PC1)"]
    return {"sources": sources, "source": _s("source", sources[0], sources, args),
            "pct": _i("pct", 90, 80, 98, args), "sigma": _f("sigma", 2.0, 1.0, 4.0, args),
            "length": _i("length", 300, 100, 3000, args), "gap": _i("gap", 60, 10, 500, args),
            "radius": _i("radius", 500, 100, 3000, args)}


def _panel_ctx(run, aoi, scene, name):
    """Everything a panel template needs. fig_url and dl always carry the area's own settings, so a
    picture route rebuilds exactly the area the panel was drawn for."""
    aq = _aoi_query()
    ctx = {"run": run, "aoi": aoi, "scene": scene, "name": name, "aoi_q": aq,
           "fig_url": lambda f, **kw: url_for("app.figure", section_key=KEY, run_id=run.id, name=f,
                                              **{**aq, **kw}),
           "dl": lambda kind, f, **kw: url_for("app.download", section_key=KEY, run_id=run.id,
                                               kind=kind, name=f, **{**aq, **kw})}
    if scene is not None:
        ctx.update(date=scene["data"]["date"], res_m=scene["data"]["res"])
    return ctx


def panel(name):
    run = base.need_run(KEY, request.args.get("run"))
    aoi, _df, err = _aoi(request.values)
    if aoi is None:
        return f'<p class="note err">{err or "Set the area first."}</p>'
    scene = _scene(run)
    if scene is None and name != "image":
        return ('<p class="note warn">Load satellite data first with <strong>Get the latest scene</strong> '
                'above.</p>')
    ctx = _panel_ctx(run, aoi, scene, name)

    if name == "image":
        return _panel_image(ctx)
    if name in ("iron", "clay"):
        return _panel_index(ctx, name)
    if name == "ratios":
        return _panel_ratios(ctx)
    if name == "pca":
        return _panel_pca(ctx)
    if name == "lithology":
        return _panel_lithology(ctx)
    if name == "lineaments":
        return _panel_lineaments(ctx)
    if name == "dem":
        return _panel_dem(ctx)
    if name == "terrain3d":
        return _panel_terrain3d(ctx)
    if name == "prospectivity":
        return _panel_prospectivity(ctx)
    abort(404)


DETAIL_TILES = {"Standard": 36, "High": 100, "Very high": 196}


def _panel_image(ctx):
    detail = _s("detail", "High", list(DETAIL_TILES))
    aoi = ctx["aoi"]
    bbox = basemap.padded_bbox(aoi.bbox, 0.15)
    try:
        img, zoom, failed = ctx["run"].cached(("basemap", tuple(np.round(bbox, 6)),
                                               DETAIL_TILES[detail]),
                                              lambda: basemap.satellite_image(
                                                  bbox, max_tiles=DETAIL_TILES[detail]))
    except Exception as e:
        return render_template("app/remote/_image.html", error=str(e), detail=detail,
                               details=list(DETAIL_TILES), **ctx)
    return render_template("app/remote/_image.html", error=None, detail=detail,
                           details=list(DETAIL_TILES), zoom=zoom, failed=failed,
                           size=(img.width, img.height), attribution=basemap.ESRI_ATTRIBUTION, **ctx)


INDEX_CFG = {
    "iron": {"name": "Iron oxide", "ratio": "B4/B2", "cmap": "YlOrRd", "colors": indices.IRON_COLORS,
             "class_title": "Ferric iron anomaly classes", "file": "iron_oxide",
             "caption": "Iron Oxide Index = Red / Blue (Sentinel-2 B4/B2). High values mark ferric "
                        "iron minerals (hematite, goethite, limonite, jarosite), typical of gossans "
                        "and oxidised sulphide zones that can accompany gold mineralisation.",
             "note": "Moderate and Strong classes are the ones to follow up. Bare red laterite soils "
                     "and ironstone also give high values, so compare with geology and the "
                     "clay/hydroxyl map."},
    "clay": {"name": "Clay / hydroxyl", "ratio": "B11/B12", "cmap": "Purples",
             "colors": indices.CLAY_COLORS, "class_title": "Clay / hydroxyl alteration classes",
             "file": "clay_hydroxyl",
             "caption": "Clay / Hydroxyl Index = SWIR-1 / SWIR-2 (Sentinel-2 B11/B12). Al-OH and "
                        "Mg-OH minerals (sericite/muscovite, kaolinite, illite, smectite, chlorite) "
                        "absorb near 2.2 µm (B12), so high values mark phyllic and argillic "
                        "alteration around gold-bearing quartz veins and shear zones.",
             "note": "Moderate and Strong classes outline possible hydrothermal alteration. "
                     "Weathered clay-rich soils, dry grass and some granites also score high, so "
                     "check against geology and the iron oxide map."},
}


def _panel_index(ctx, kind):
    run, scene, aoi = ctx["run"], ctx["scene"], ctx["aoi"]
    cfg = INDEX_CFG[kind]
    method = _s("method", "Band ratio", ["Band ratio", "Crósta PCA"])
    hide_veg = _b("veg", True)
    bands, tr, epsg, inside, clear, _aoi_utm = _header(scene, aoi)
    arr = _index_layer(run, scene, kind, method)
    valid = _usable(run, scene, hide_veg)
    cls, thr = indices.classify_sigma(arr, valid)
    px_ha = scene["data"]["res"] ** 2 / 1e4
    n_valid = int((cls >= 0).sum())
    if n_valid < 20:
        return render_template("app/remote/_index.html", cfg=cfg, kind=kind, method=method,
                               hide_veg=hide_veg, too_few=True, **ctx)
    unit = cfg["ratio"] if method == "Band ratio" else "PC score"
    cr = None if method == "Band ratio" else _crosta(run, scene,
                                                     "hydroxyl" if kind == "clay" else "iron")
    ranges = [f"< {thr[0]:.3f}", f"{thr[0]:.3f} – {thr[1]:.3f}", f"{thr[1]:.3f} – {thr[2]:.3f}",
              f"> {thr[2]:.3f}"]
    tbl = pd.DataFrame({"Class": indices.IRON_CLASSES,
                        "Rule": ["below mean", "mean to +1σ", "+1σ to +2σ", "above +2σ"],
                        f"Range ({unit})": ranges,
                        "Area (ha)": [round(float((cls == i).sum() * px_ha), 2) for i in range(4)],
                        "% of mapped area": [round(100 * float((cls == i).sum()) / n_valid, 1)
                                             for i in range(4)]})
    q = {"method": method, "veg": 1 if hide_veg else "", "_t": 1}
    return render_template("app/remote/_index.html", cfg=cfg, kind=kind, method=method,
                           hide_veg=hide_veg, too_few=False, unit=unit, thr=thr, cr=cr, q=q,
                           strong_ha=(cls == 3).sum() * px_ha,
                           mapped_pct=100 * n_valid / max(inside.sum(), 1),
                           table=table_html(tbl), **ctx)


def _panel_ratios(ctx):
    presets = list(indices.RATIO_PRESETS) + ["Custom"]
    keys = list(indices.RATIOS)
    preset = _s("preset", presets[0], presets)
    rgb_keys = ((_s("r", keys[0], keys), _s("g", keys[min(1, len(keys) - 1)], keys),
                 _s("b", keys[min(2, len(keys) - 1)], keys)) if preset == "Custom"
                else indices.RATIO_PRESETS[preset])
    q = {"preset": preset, "r": rgb_keys[0], "g": rgb_keys[1], "b": rgb_keys[2], "_t": 1}
    return render_template("app/remote/_ratios.html", presets=presets, preset=preset, keys=keys,
                           rgb_keys=rgb_keys, q=q,
                           rgb_map={"r": rgb_keys[0], "g": rgb_keys[1], "b": rgb_keys[2]}, **ctx)


def _panel_pca(ctx):
    run, scene = ctx["run"], ctx["scene"]
    combos = ["PC1-PC2-PC3", "PC2-PC3-PC4", "PC3-PC4-PC5", "PC4-PC5-PC6"]
    combo = _s("combo", combos[0], combos)
    try:
        res = _std_pca(run, scene)
    except ValueError as e:
        return f'<p class="note warn">{e}</p>'
    crostas = {}
    for kind in ("hydroxyl", "iron"):
        try:
            cr = _crosta(run, scene, kind)
            crostas[kind] = {"cr": cr, "table": table_html(
                pd.DataFrame(np.round(cr["loadings"], 3), columns=cr["names"],
                             index=[f"PC{i + 1} ({100 * e:.1f}%)"
                                    for i, e in enumerate(cr["explained"])]), index=True)}
        except ValueError as e:
            crostas[kind] = {"error": str(e)}
    return render_template("app/remote/_pca.html", combos=combos, combo=combo, res=res,
                           n=len(res["explained"]), crostas=crostas, q={"combo": combo, "_t": 1},
                           **ctx)


def _panel_lithology(ctx):
    run, scene, aoi = ctx["run"], ctx["scene"], ctx["aoi"]
    k = _i("k", 6, 3, 12)
    try:
        res = _lith(run, scene, k)
    except ValueError as e:
        return f'<p class="note warn">{e}</p>'
    bands, tr, epsg, inside, clear, _u = _header(scene, aoi)
    labels, means = res["labels"], res["means"]
    px_ha = scene["data"]["res"] ** 2 / 1e4
    inside_lab = np.where(inside, labels, -1)
    n_in = max(int((inside_lab >= 0).sum()), 1)
    tbl = means[["hint", "clay_hydroxyl", "iron_oxide", "ndvi"]].copy()
    tbl.insert(1, "Area (ha)", [round(float((inside_lab == i).sum() * px_ha), 2) for i in range(k)])
    tbl.insert(2, "% of AOI", [round(100 * float((inside_lab == i).sum()) / n_in, 1)
                               for i in range(k)])
    tbl.columns = ["Spectral interpretation", "Area (ha)", "% of AOI", "Clay index (B11/B12)",
                   "Iron index (B4/B2)", "NDVI"]
    return render_template("app/remote/_lithology.html", k=k, table=table_html(tbl.round(3),
                                                                               index=True),
                           q={"k": k, "_t": 1}, **ctx)


def _panel_lineaments(ctx):
    run, scene, aoi = ctx["run"], ctx["scene"], ctx["aoi"]
    o = _lin_opts(scene)
    bands, tr, epsg, inside, clear, _u = _header(scene, aoi)
    lines, _edges = _lineaments(run, scene, o["source"], o["pct"], o["sigma"], o["length"], o["gap"])
    starts, lens = lineaments.rose_stats(lines)
    order = np.argsort(lens)[::-1] if len(lens) else []
    total_km = sum(l["length_m"] for l in lines) / 1000
    area_km2 = aoi.area_m2 / 1e6
    trends = pd.DataFrame([{"Trend": lineaments.strike_label(starts[i] + 5),
                            "Strike range": f"{starts[i]:.0f}–{starts[i] + 10:.0f}°",
                            "Length (km)": round(float(lens[i]), 2),
                            "Share": f"{100 * lens[i] / max(lens.sum(), 1e-9):.0f}%"}
                           for i in order[:4] if lens[i] > 0])
    q = {**{k: o[k] for k in ("source", "pct", "sigma", "length", "gap", "radius")}, "_t": 1}
    return render_template("app/remote/_lineaments.html", o=o, lines=lines, total_km=total_km,
                           density=total_km / max(area_km2, 1e-9),
                           main=lineaments.strike_label(starts[order[0]] + 5) if len(lines) else "–",
                           trends=table_html(trends), q=q, **ctx)


def _panel_dem(ctx):
    run, scene, aoi = ctx["run"], ctx["scene"], ctx["aoi"]
    if scene.get("dem") is None:
        return (f'<p class="note warn">Elevation data is not available for this scene: '
                f'{scene.get("dem_error")}</p>')
    bands, tr, epsg, inside, clear, _u = _header(scene, aoi)
    dem, res_m = scene["dem"], scene["data"]["res"]
    o = {"az": _i("az", 315, 0, 360), "alt": _i("alt", 45, 5, 85), "z": _f("z", 1.0, 1.0, 5.0),
         "contours": _b("contours", True)}
    v = dem[inside & np.isfinite(dem)]
    if v.size == 0:
        return '<p class="note warn">No elevation values inside the area.</p>'
    slope = terrain.slope_deg(dem, res_m)
    step = terrain.contour_interval(dem) if o["contours"] else None
    q = {**o, "contours": 1 if o["contours"] else "", "_t": 1}
    return render_template("app/remote/_dem.html", o=o, q=q, step=step,
                           low=v.min(), high=v.max(), relief=v.max() - v.min(),
                           mean_slope=float(np.nanmean(slope[inside])),
                           attribution=dem_mod.ATTRIBUTION, **ctx)


def _panel_terrain3d(ctx):
    run, scene, aoi = ctx["run"], ctx["scene"], ctx["aoi"]
    if scene.get("dem") is None:
        return (f'<p class="note warn">Elevation data is not available for this scene: '
                f'{scene.get("dem_error")}</p>')
    bands, tr, epsg, inside, clear, aoi_utm = _header(scene, aoi)
    dem, res_m = scene["dem"], scene["data"]["res"]
    if not np.isfinite(dem[inside]).any():
        return '<p class="note warn">No elevation values inside the area.</p>'
    o = {"drape": _s("drape", list(DRAPES)[0], list(DRAPES)),
         "exag": _f("exag", 6.0, 1.0, 30.0), "detail": _s("detail", "High", list(DETAIL)),
         "az": _i("az", 315, 0, 360), "alt": _i("alt", 45, 5, 85)}
    try:
        fig, step, cell = _terrain_figure(run, scene, aoi, o)
    except Exception as e:
        return f'<p class="note err">Could not build the 3-D view: {e}</p>'
    q = {**o, "_t": 1}
    return render_template("app/remote/_terrain3d.html", o=o, q=q, drapes=list(DRAPES),
                           details=list(DETAIL), plot=plotly_div(fig, "terrain3d", height=620),
                           cell=cell, step=step, attribution=dem_mod.ATTRIBUTION, **ctx)


def _terrain_figure(run, scene, aoi, o):
    bands, tr, epsg, inside, clear, aoi_utm = _header(scene, aoi)
    dem, res_m = scene["dem"], scene["data"]["res"]
    hs = terrain.hillshade(dem, res_m, o["az"], o["alt"], 1.0)
    kind, which = DRAPES[o["drape"]]
    if kind == "picture":
        rgb = (terrain3d.shaded_rgb(dem, hs) if which is None
               else terrain3d.shade_rgb(indices.composite(bands, which, clear), hs))
        return terrain3d.figure(dem, res_m, tr, rgb, exaggeration=o["exag"],
                                max_side=DETAIL[o["detail"]], aoi_utm=aoi_utm,
                                title=f"{o['drape']} in 3-D")
    grid = {"elevation": dem, "slope": terrain.slope_deg(dem, res_m)}.get(which)
    if grid is None:
        grid = np.where(clear, indices.compute_index(bands, which), np.nan)
    scale, label, rev = SCALES[which]
    return terrain3d.figure(dem, res_m, tr, None, exaggeration=o["exag"],
                            max_side=DETAIL[o["detail"]], aoi_utm=aoi_utm,
                            title=f"{o['drape']} in 3-D", scalar=grid, scalar_label=label,
                            colourscale=scale, reverse=rev)


def _prospect(run, scene, aoi, o):
    """The evidence layers, the combined score and the target zones."""
    bands, tr, epsg, inside, clear, _u = _header(scene, aoi)
    res_m = scene["data"]["res"]
    w = o["w"]
    valid = (_usable(run, scene, o["veg"]) if (w["clay"] or w["iron"])
             else inside & np.isfinite(bands["B04"]))
    valid = valid & inside
    layers, notes = {}, []
    if w["clay"]:
        layers["clay"] = prospectivity.fuzzy_linear(
            _index_layer(run, scene, "clay", o["clay_method"]), valid)
    if w["iron"]:
        layers["iron"] = prospectivity.fuzzy_linear(
            _index_layer(run, scene, "iron", o["iron_method"]), valid)
    if w["lden"] or w["lprox"]:
        lin = _lin_opts(scene)
        lines, _e = _lineaments(run, scene, lin["source"], lin["pct"], lin["sigma"], lin["length"],
                                lin["gap"])
        if lines:
            if w["lden"]:
                layers["lden"] = prospectivity.fuzzy_linear(
                    lineaments.density(lines, inside.shape, res_m, lin["radius"]), valid, 0, 98)
            if w["lprox"]:
                layers["lprox"] = prospectivity.fuzzy_distance(
                    lineaments.distance(lines, inside.shape, res_m), valid,
                    half_m=max(2 * res_m, 150))
        else:
            notes.append("No lineaments with the current settings, so the structure layers were "
                         "skipped.")
    units = fav = []
    if w["lith"]:
        lres = _lith(run, scene, o["k"])
        units = list(lres["means"].index)
        default = [u for u, h in zip(units, lres["means"]["hint"])
                   if "alteration" in h or "Iron" in h]
        fav = [u for u in (o["units"] or default) if u in units]
        idx = [units.index(u) for u in fav]
        layers["lith"] = np.where(valid, np.isin(lres["labels"], idx).astype(np.float32), np.nan)
    if not layers:
        return None, notes, units, fav
    score = prospectivity.combine(layers, w, valid, o["method"], o["gamma"])
    min_px = max(1, int(round(o["min_ha"] * 1e4 / res_m ** 2)))
    z = prospectivity.targets(score, valid, o["top"], min_px, tr)
    return {"layers": layers, "score": score, "zones": z, "tr": tr, "epsg": epsg,
            "inside": inside, "res_m": res_m}, notes, units, fav


def _prospect_opts(scene, args=None):
    a = request.values if args is None else args
    w = {k: _f(f"w_{k}", d[1], 0.0, 1.0, a) for k, d in PROSPECT_LAYERS.items()}
    return {"w": w, "method": _s("method", "Weighted sum", ["Weighted sum", "Fuzzy gamma"], a),
            "gamma": _f("gamma", 0.8, 0.5, 0.95, a), "top": _f("top", 5.0, 1.0, 20.0, a),
            "min_ha": _f("min_ha", 0.5, 0.0, 5.0, a), "veg": _b("veg", True, a),
            "k": _i("k", 6, 3, 12, a), "units": a.getlist("units") if hasattr(a, "getlist") else [],
            "clay_method": _s("clay_method", "Band ratio", ["Band ratio", "Crósta PCA"], a),
            "iron_method": _s("iron_method", "Band ratio", ["Band ratio", "Crósta PCA"], a)}


def _panel_prospectivity(ctx):
    run, scene, aoi = ctx["run"], ctx["scene"], ctx["aoi"]
    o = _prospect_opts(scene)
    out, notes, units, fav = _prospect(run, scene, aoi, o)
    q = {f"w_{k}": v for k, v in o["w"].items()}
    q.update({"method": o["method"], "gamma": o["gamma"], "top": o["top"], "min_ha": o["min_ha"],
              "veg": 1 if o["veg"] else "", "k": o["k"], "_t": 1})
    if out is None:
        return render_template("app/remote/_prospectivity.html", o=o, layers=PROSPECT_LAYERS,
                               empty=True, notes=notes, units=units, fav=fav, q=q, **ctx)
    zt = _zones_table(out["zones"]["zones"], out["epsg"])
    return render_template("app/remote/_prospectivity.html", o=o, layers=PROSPECT_LAYERS,
                           empty=False, notes=notes, units=units, fav=fav, q=q,
                           n_zones=len(zt), used=list(out["layers"]),
                           target_ha=out["zones"]["targets"].sum() * out["res_m"] ** 2 / 1e4,
                           evaluated=100 * np.isfinite(out["score"]).sum()
                           / max(out["inside"].sum(), 1),
                           table=table_html(zt.drop(columns=["Easting", "Northing"]))
                           if len(zt) else None, **ctx)


def _zones_table(zones, epsg):
    if not zones:
        return pd.DataFrame()
    from pyproj import Transformer
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    rows = []
    for i, z in enumerate(zones, 1):
        lon, lat = to_ll.transform(z["easting"], z["northing"])
        rows.append({"Zone": f"T{i}", "Latitude": round(lat, 6), "Longitude": round(lon, 6),
                     "Easting": round(z["easting"], 1), "Northing": round(z["northing"], 1),
                     "Area (ha)": round(z["area_ha"], 2), "Mean score": round(z["mean_score"], 3),
                     "Max score": round(z["max_score"], 3)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ pictures
def redraw(run, name, args):
    from ..render import save_fig
    aoi, _df, err = _aoi(args)
    if aoi is None:
        return None
    scene = _scene(run)
    fig = _build_figure(run, scene, aoi, name, args)
    if fig is None:
        return None
    return save_fig(fig, run.fig_path(name))


def _build_figure(run, scene, aoi, name, args):      # noqa: C901 - one branch per map, kept flat
    root = name.split("--")[0]
    if root == "basemap":
        detail = _s("detail", "High", list(DETAIL_TILES), args)
        bbox = basemap.padded_bbox(aoi.bbox, 0.15)
        img, _zoom, _failed = run.cached(("basemap", tuple(np.round(bbox, 6)),
                                          DETAIL_TILES[detail]),
                                         lambda: basemap.satellite_image(
                                             bbox, max_tiles=DETAIL_TILES[detail]))
        return basemap.annotated_figure(img, bbox, aoi,
                                        title="Satellite image of the area of interest")
    if scene is None:
        return None
    bands, tr, epsg, inside, clear, aoi_utm = _header(scene, aoi)
    date_s = scene["data"]["date"]

    if root in ("iron-index", "iron-classes", "clay-index", "clay-classes"):
        kind = root.split("-")[0]
        cfg = INDEX_CFG[kind]
        method = _s("method", "Band ratio", ["Band ratio", "Crósta PCA"], args)
        valid = _usable(run, scene, _b("veg", True, args))
        arr = _index_layer(run, scene, kind, method)
        cls, _thr = indices.classify_sigma(arr, valid)
        if root.endswith("index"):
            unit = cfg["ratio"] if method == "Band ratio" else "PC score"
            label = (f"{cfg['name']} Index ({cfg['ratio']})" if method == "Band ratio"
                     else _crosta(run, scene, "hydroxyl" if kind == "clay" else "iron")["label"])
            return export.raster_figure(arr, tr, aoi_utm, f"{label} · {date_s}", cfg["cmap"],
                                        valid=valid, cbar_label=unit)
        return export.class_figure(cls, tr, indices.IRON_CLASSES, cfg["colors"], aoi_utm,
                                   f"{cfg['class_title']} · {date_s}")

    if root in ("ratio-rgb", "ratio-all"):
        keys = list(indices.RATIOS)
        if root == "ratio-rgb":
            rgb_keys = (_s("r", keys[0], keys, args), _s("g", keys[1], keys, args),
                        _s("b", keys[2], keys, args))
            preset = args.get("preset")
            if preset in indices.RATIO_PRESETS:
                rgb_keys = indices.RATIO_PRESETS[preset]
            rgb = indices.ratio_rgb(bands, rgb_keys, valid=clear)
            return export.rgb_figure(rgb, tr, aoi_utm,
                                     f"R={rgb_keys[0]}  G={rgb_keys[1]}  B={rgb_keys[2]} · {date_s}")
        arrays = [indices.compute_ratio(bands, k) for k in keys]
        return export.panels_figure(arrays, [f"{k} · {indices.RATIOS[k][0]}" for k in keys],
                                    [indices.RATIOS[k][2] for k in keys], tr, valid=clear,
                                    aoi_utm=aoi_utm, ncols=4,
                                    suptitle=f"Sentinel-2 band ratios · {date_s}")

    if root.startswith("pca"):
        res = _std_pca(run, scene)
        pcs, ex = res["pcs"], res["explained"]
        if root == "pca-rgb":
            combo = _s("combo", "PC1-PC2-PC3", ["PC1-PC2-PC3", "PC2-PC3-PC4", "PC3-PC4-PC5",
                                                "PC4-PC5-PC6"], args)
            idx = [int(p[2:]) - 1 for p in combo.split("-")]
            rgb = np.dstack([indices.stretch(pcs[i], clear) for i in idx])
            rgb[~clear] = 0
            return export.rgb_figure(rgb, tr, aoi_utm, f"PCA composite {combo} · {date_s}")
        if root == "pca-loadings":
            return export.loadings_figure(res["loadings"], res["names"], ex,
                                          "Loadings (eigenvectors)")
        if root == "pca-all":
            n = len(ex)
            return export.panels_figure(
                [pcs[i] for i in range(n)],
                [f"PC{i + 1} ({100 * ex[i]:.1f}% of variance)" for i in range(n)],
                ["gray"] * n, tr, valid=clear, aoi_utm=aoi_utm, ncols=3,
                suptitle=f"Standardised PCA of B2, B3, B4, B8, B11, B12 · {date_s}")
        if root in ("crosta-hydroxyl", "crosta-iron"):
            kind = root.split("-")[1]
            cr = _crosta(run, scene, kind)
            cmap = "magma" if kind == "hydroxyl" else "YlOrRd"
            return export.raster_figure(cr["image"], tr, aoi_utm,
                                        f"{cr['label']} = PC{cr['pc_index'] + 1}", cmap,
                                        valid=clear, cbar_label="PC score")

    if root in ("crosta-hydroxyl", "crosta-iron"):
        kind = root.split("-")[1]
        cr = _crosta(run, scene, kind)
        cmap = "magma" if kind == "hydroxyl" else "YlOrRd"
        return export.raster_figure(cr["image"], tr, aoi_utm,
                                    f"{cr['label']} = PC{cr['pc_index'] + 1}", cmap, valid=clear,
                                    cbar_label="PC score")

    if root in ("lith-map", "lith-spectra"):
        k = _i("k", 6, 3, 12, args)
        res = _lith(run, scene, k)
        if root == "lith-map":
            legend = [f"{u}: {h}" for u, h in zip(res["means"].index, res["means"]["hint"])]
            return export.class_figure(np.where(inside, res["labels"], -1), tr, legend,
                                       res["colors"], aoi_utm,
                                       f"Lithological (spectral) classification · {date_s}")
        return _spectra_figure(res["means"], res["colors"])

    if root in ("lin-map", "lin-rose", "lin-density"):
        o = _lin_opts(scene, args)
        lines, _e = _lineaments(run, scene, o["source"], o["pct"], o["sigma"], o["length"], o["gap"])
        if root == "lin-rose":
            starts, lens = lineaments.rose_stats(lines)
            return export.rose_figure(starts, lens, title="Strike rose (length-weighted)")
        if root == "lin-density":
            dens = lineaments.density(lines, inside.shape, scene["data"]["res"], o["radius"])
            return export.raster_figure(dens, tr, aoi_utm, "Lineament density", "hot_r",
                                        valid=inside, cbar_label="km per km²", lo=0, hi=99.5)
        if scene.get("dem") is not None and o["source"] != "Satellite (PC1)":
            bg = terrain.multi_hillshade(scene["dem"], scene["data"]["res"])
            bg_name = "multi-directional hillshade"
        else:
            bg = _std_pca(run, scene)["pcs"][0]
            bg_name = "PC1"
        return export.lines_figure(bg, lines, tr, aoi_utm, f"Lineaments on {bg_name} · {date_s}")

    if root in ("dem-relief", "dem-hillshade", "dem-slope"):
        dem, res_m = scene["dem"], scene["data"]["res"]
        if dem is None:
            return None
        az, alt = _i("az", 315, 0, 360, args), _i("alt", 45, 5, 85, args)
        z = _f("z", 1.0, 1.0, 5.0, args)
        hs = terrain.hillshade(dem, res_m, az, alt, z)
        if root == "dem-hillshade":
            return export.raster_figure(hs, tr, aoi_utm, f"Hillshade (sun {az}°, {alt}° high)",
                                        "gray", cbar_label="illumination", lo=0.5, hi=99.5)
        if root == "dem-slope":
            return export.raster_figure(terrain.slope_deg(dem, res_m), tr, aoi_utm, "Slope",
                                        "YlOrBr", cbar_label="degrees", lo=0, hi=99)
        step = terrain.contour_interval(dem) if _b("contours", True, args) else None
        return export.relief_figure(dem, hs, tr, aoi_utm,
                                    "Shaded relief (elevation + hillshade)"
                                    + (f", contours every {step} m" if step else ""), step)

    if root in ("prospect-map", "prospect-evidence"):
        o = _prospect_opts(scene, args)
        out, _notes, _units, _fav = _prospect(run, scene, aoi, o)
        if out is None:
            return None
        if root == "prospect-map":
            return export.raster_figure(out["score"], tr, aoi_utm,
                                        f"Integrated gold prospectivity · {date_s}", "inferno",
                                        overlay=out["zones"]["targets"],
                                        cbar_label="prospectivity (0–1)", lo=1, hi=99)
        keys = list(out["layers"])
        return export.panels_figure([out["layers"][k] for k in keys],
                                    [f"{PROSPECT_LAYERS[k][0]} (w={o['w'][k]:.1f})" for k in keys],
                                    [PROSPECT_LAYERS[k][2] for k in keys], tr, aoi_utm=aoi_utm,
                                    ncols=min(len(keys), 5), lo=0, hi=100)
    return None


WAVELENGTH = {"B02": 490, "B03": 560, "B04": 665, "B08": 842, "B11": 1610, "B12": 2190}


def _spectra_figure(means, colors):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [WAVELENGTH[b] for b in lithology.FEATURE_BANDS]
    for (name, r), c in zip(means.iterrows(), colors):
        ax.plot(x, [r[b] for b in lithology.FEATURE_BANDS], "-o", color=c, label=name, ms=4)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean reflectance")
    ax.set_title("Average spectrum of each unit", fontsize=10)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ downloads
def deliver(run, kind, name, args):                  # noqa: C901 - a branch per download
    aoi, _df, _err = _aoi(args)
    scene = _scene(run)
    if kind == "png":
        path = run.fig_path(name)
        if not path.is_file():
            path = redraw(run, name, args)
        return base.send_png(path, f"{name}.png")

    if aoi is None:
        abort(404)
    if name == "satellite-image":
        detail = _s("detail", "High", list(DETAIL_TILES), args)
        bbox = basemap.padded_bbox(aoi.bbox, 0.15)
        img, _zoom, _failed = run.cached(("basemap", tuple(np.round(bbox, 6)), DETAIL_TILES[detail]),
                                  lambda: basemap.satellite_image(bbox,
                                                                  max_tiles=DETAIL_TILES[detail]))
        if kind == "img":
            return base.send_bytes(export.image_to_png(img), "aoi_satellite.png", "image/png")
        if kind == "tif":
            from pyproj import Transformer
            from rasterio.transform import from_bounds
            to3857 = Transformer.from_crs(4326, 3857, always_xy=True)
            x0, y0 = to3857.transform(bbox[0], bbox[1])
            x1, y1 = to3857.transform(bbox[2], bbox[3])
            tr3857 = from_bounds(x0, y0, x1, y1, img.width, img.height)
            return base.send_bytes(
                export.rgb_geotiff_bytes(np.asarray(img) / 255.0, tr3857, 3857),
                "aoi_satellite_3857.tif", "image/tiff")

    if scene is None:
        abort(404)
    bands, tr, epsg, inside, clear, _u = _header(scene, aoi)
    date_s = scene["data"]["date"]

    if kind == "tif":
        if name in ("iron", "clay"):
            cfg = INDEX_CFG[name]
            method = _s("method", "Band ratio", ["Band ratio", "Crósta PCA"], args)
            valid = _usable(run, scene, _b("veg", True, args))
            arr = _index_layer(run, scene, name, method)
            cls, _t = indices.classify_sigma(arr, valid)
            return base.send_bytes(export.geotiff_bytes(
                {f"{cfg['file']}_index": np.where(valid, arr, np.nan),
                 f"{cfg['file']}_class": np.where(cls >= 0, cls, np.nan).astype(np.float32)},
                tr, epsg), f"{cfg['file']}_{date_s}.tif", "image/tiff")
        if name == "ratio-rgb":
            keys = list(indices.RATIOS)
            rgb_keys = (_s("r", keys[0], keys, args), _s("g", keys[1], keys, args),
                        _s("b", keys[2], keys, args))
            preset = args.get("preset")
            if preset in indices.RATIO_PRESETS:
                rgb_keys = indices.RATIO_PRESETS[preset]
            return base.send_bytes(
                export.rgb_geotiff_bytes(indices.ratio_rgb(bands, rgb_keys, valid=clear), tr, epsg),
                "ratio_composite.tif", "image/tiff")
        if name == "ratio-all":
            keys = list(indices.RATIOS)
            return base.send_bytes(export.geotiff_bytes(
                {k.replace("/", "_"): np.where(clear, indices.compute_ratio(bands, k), np.nan)
                 for k in keys}, tr, epsg), f"band_ratios_{date_s}.tif", "image/tiff")
        if name == "pca-rgb":
            res = _std_pca(run, scene)
            combo = _s("combo", "PC1-PC2-PC3", ["PC1-PC2-PC3", "PC2-PC3-PC4", "PC3-PC4-PC5",
                                                "PC4-PC5-PC6"], args)
            idx = [int(p[2:]) - 1 for p in combo.split("-")]
            rgb = np.dstack([indices.stretch(res["pcs"][i], clear) for i in idx])
            rgb[~clear] = 0
            return base.send_bytes(export.rgb_geotiff_bytes(rgb, tr, epsg), "pca_composite.tif",
                                   "image/tiff")
        if name == "pca-all":
            res = _std_pca(run, scene)
            return base.send_bytes(export.geotiff_bytes(
                {f"PC{i + 1}": res["pcs"][i] for i in range(len(res["explained"]))}, tr, epsg),
                f"pca_{date_s}.tif", "image/tiff")
        if name == "lithology":
            res = _lith(run, scene, _i("k", 6, 3, 12, args))
            return base.send_bytes(export.geotiff_bytes(
                {"unit": np.where(res["labels"] >= 0, res["labels"] + 1, np.nan).astype(np.float32)},
                tr, epsg), f"lithology_{date_s}.tif", "image/tiff")
        if name == "lineament-density":
            o = _lin_opts(scene, args)
            lines, _e = _lineaments(run, scene, o["source"], o["pct"], o["sigma"], o["length"],
                                    o["gap"])
            dens = lineaments.density(lines, inside.shape, scene["data"]["res"], o["radius"])
            return base.send_bytes(
                export.geotiff_bytes({"lineament_density_km_km2": dens}, tr, epsg),
                "lineament_density.tif", "image/tiff")
        if name == "dem":
            dem, res_m = scene["dem"], scene["data"]["res"]
            az, alt = _i("az", 315, 0, 360, args), _i("alt", 45, 5, 85, args)
            hs = terrain.hillshade(dem, res_m, az, alt, _f("z", 1.0, 1.0, 5.0, args))
            return base.send_bytes(export.geotiff_bytes(
                {"elevation_m": dem, "hillshade": hs, "slope_deg": terrain.slope_deg(dem, res_m),
                 "aspect_deg": terrain.aspect_deg(dem, res_m)}, tr, epsg),
                "dem_products.tif", "image/tiff")
        if name == "prospectivity":
            o = _prospect_opts(scene, args)
            out, _notes, _units, _fav = _prospect(run, scene, aoi, o)
            if out is None:
                abort(404)
            return base.send_bytes(export.geotiff_bytes(
                {"prospectivity": out["score"],
                 "target": out["zones"]["targets"].astype(np.float32),
                 **{f"evidence_{k}": a for k, a in out["layers"].items()}}, tr, epsg),
                "prospectivity.tif", "image/tiff")

    if kind == "csv":
        if name == "lithology":
            res = _lith(run, scene, _i("k", 6, 3, 12, args))
            return base.send_csv(res["means"], "lithology_units.csv", index=True)
        if name == "lineaments":
            o = _lin_opts(scene, args)
            lines, _e = _lineaments(run, scene, o["source"], o["pct"], o["sigma"], o["length"],
                                    o["gap"])
            rows = pd.DataFrame([{"id": i + 1, "length_m": round(l["length_m"], 1),
                                  "azimuth": round(l["azimuth"], 1),
                                  "trend": lineaments.strike_label(l["azimuth"])}
                                 for i, l in enumerate(lines)])
            return base.send_csv(rows, "lineaments.csv")
        if name == "prospectivity":
            o = _prospect_opts(scene, args)
            out, _notes, _units, _fav = _prospect(run, scene, aoi, o)
            if out is None:
                abort(404)
            return base.send_csv(_zones_table(out["zones"]["zones"], out["epsg"]),
                                 "prospectivity_targets.csv")

    if kind == "geojson" and name == "lineaments":
        o = _lin_opts(scene, args)
        lines, _e = _lineaments(run, scene, o["source"], o["pct"], o["sigma"], o["length"],
                                o["gap"])
        return base.send_bytes(lineaments.to_geojson(lines, tr, epsg).encode("utf-8"),
                               "lineaments.geojson", "application/geo+json")

    if kind == "html" and name == "terrain3d":
        o = {"drape": _s("drape", list(DRAPES)[0], list(DRAPES), args),
             "exag": _f("exag", 6.0, 1.0, 30.0, args),
             "detail": _s("detail", "High", list(DETAIL), args),
             "az": _i("az", 315, 0, 360, args), "alt": _i("alt", 45, 5, 85, args)}
        fig, _step, _cell = _terrain_figure(run, scene, aoi, o)
        return base.send_bytes(fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8"),
                               "terrain_3d.html", "text/html")
    abort(404)
