"""5 · Geochemistry: soil samples with a gold assay, read and interpreted for targeting.

Works out where background stops, groups the anomalous samples into zones with their trend and
extent, says which are worth following up and which are probably a single lucky grain, and keeps the
interpretation with the run so section 6 and its report can use it unchanged.
"""
import numpy as np
import pandas as pd
from flask import abort, flash, redirect, render_template, request, url_for
from flask_login import current_user

from geoai import coords, export, geochem, interpolate, interpret, plots, readers, soilgold

from .. import store, uploads
from ..render import safe_name, table_html
from . import base

KEY = "geochemistry"
SLOT = "geochem"

TABS = [("Map", "map"), ("Anomaly zones", "zones"), ("Distribution", "distribution"),
        ("Interpolated", "interpolated"), ("Statistics", "statistics"), ("All samples", "samples")]


def attach(bp):
    from .. import section
    bp.add_url_rule("/geochemistry", "geochemistry", section(KEY)(page), methods=["GET", "POST"])
    bp.add_url_rule("/geochemistry/panel/<name>", "geochemistry_panel", section(KEY)(panel))


# ------------------------------------------------------------------ the form
def _fields(item):
    """What the person chooses about one file of soil samples."""
    if not item or not item["info"]:
        return None
    if item["info"]["kind"] != "table":
        return {"error": "Soil samples must be a table: a CSV or an Excel sheet, not a picture or a grid."}
    df = item["info"]["result"]["df"]
    if len(df.columns) < 3 or len(df) < 3:
        return {"error": "The file needs at least three columns (position and assay) and a few rows."}
    cols = [str(c) for c in df.columns]
    num = [str(c) for c in readers.numeric_columns(df)]
    gx, gy = coords.guess_columns(df)
    gx = str(gx) if gx is not None else cols[0]
    gy = str(gy) if gy is not None else cols[min(1, len(cols) - 1)]
    rest = [c for c in num if c not in (gx, gy)]
    if not rest:
        return {"error": "No numeric column is left for the gold assay."}
    au_guess = next((k for k in rest if any(w in k.lower()
                                            for w in ("au", "gold", "assay", "grade"))), rest[0])
    vals, _ = geochem.clean_values(df[au_guess])
    id_opts = ["(row number)"] + [c for c in cols if c not in (gx, gy, au_guess)]
    id_guess = next((k for k in id_opts[1:] if any(w in k.lower()
                                                   for w in ("sample", "id", "no", "point"))),
                    id_opts[0])
    f = uploads.field
    good = int(np.isfinite(vals).sum())
    return {
        "error": None,
        "good": good,
        "fields": [
            f("xcol", "Easting / longitude", "select", cols, gx, width="third"),
            f("ycol", "Northing / latitude", "select", cols, gy, width="third"),
            uploads.crs_field("crs", "Coordinate system",
                              coords.parse_column(df[gx]).dropna().head(200),
                              coords.parse_column(df[gy]).dropna().head(200), width="third"),
            f("au", "Gold assay column", "select", rest, au_guess, width="half"),
            f("unit", "Assay unit", "select", list(soilgold.UNITS), soilgold.guess_unit(vals),
              "Check this against your lab certificate: it decides every threshold and every number "
              "below. Soil gold is nearly always reported in ppm, which is the same thing as g/t; "
              "per cent is for ore-grade material, where 1 % is 10,000 ppm. Results are given in "
              "your unit and in ppb.", width="half"),
            f("idcol", "Sample ID column", "select", id_opts, id_guess, width="full"),
        ],
        "note": (f"{good:,} of {len(df):,} rows have a usable gold value, from "
                 f"{np.nanmin(vals):,.4g} to {np.nanmax(vals):,.4g}." if good else None),
    }


def page():
    if request.method == "POST":
        action = request.form.get("do")
        if action == "upload":
            base.take_upload(KEY)
            return redirect(url_for("app.geochemistry") + "#files")
        if action in ("remove", "clear"):
            if action == "clear":
                store.clear_draft(current_user.id, KEY)
            else:
                store.drop_draft(current_user.id, KEY, request.form.get("name", ""))
            return redirect(url_for("app.geochemistry"))
        if action == "submit":
            return _submit()

    items = base.drafts(KEY)
    for it in items:
        it["prefix"] = ""
        it["fields"] = []
    form = _fields(items[0]) if items else None
    run = base.current_run(KEY)
    return base.page(KEY, "app/geochemistry.html", items=items, form=form, run=run, tabs=TABS,
                     methods=soilgold.METHODS, method=_method(),
                     analysis=analyse(run) if run is not None else None)


def _spec(item, values):
    info = item["info"]
    df = info["result"]["df"]
    xcol, ycol, aucol = values.get("xcol"), values.get("ycol"), values.get("au")
    if not (xcol and ycol and aucol):
        raise ValueError("Choose the position columns and the gold assay column.")
    vals, _ = geochem.clean_values(df[aucol])
    if int(np.isfinite(vals).sum()) < 5:
        raise ValueError("Fewer than five usable gold values in that column.")
    idcol = values.get("idcol")
    return {"slot": SLOT, "name": info["name"], "kind": "table",
            "data": {"kind": "table", "df": df}, "xcol": xcol, "ycol": ycol,
            "epsg": uploads.resolve_crs(values, "crs"), "au": aucol, "pathfinders": [],
            "idcol": None if idcol in (None, "", "(row number)") else idcol,
            "unit": values.get("unit") or list(soilgold.UNITS)[0]}


def _submit():
    items = base.drafts(KEY)
    if not items:
        flash("Choose a file of soil samples first.", "error")
        return redirect(url_for("app.geochemistry"))
    try:
        spec = _spec(items[0], request.form)
    except (ValueError, KeyError) as e:
        flash(str(e), "error")
        return redirect(url_for("app.geochemistry"))
    unit = spec["unit"]

    def work(run, job):
        job.say(0.2, "Reading the samples…")
        out = interpret.run([spec], {}, progress=lambda p, t: job.say(0.2 + 0.6 * p, t))
        if out["errors"] or not out["datasets"]["geochem"]:
            raise ValueError("; ".join(out["errors"]) or "no usable samples were found")
        out["unit"] = unit
        job.say(0.9, "Working out where background stops…")
        run.save_result(out)

    return base.submit(KEY, work, label=f"Interpreting {spec['name']}", settings={"unit": unit})


# ------------------------------------------------------------------ the analysis
def _method(args=None):
    m = (request.values if args is None else args).get("method")
    return m if m in soilgold.METHODS else soilgold.METHODS[0]


def analyse(run, method=None):
    """Everything derived from the assays, cached beside the run so each tab is a file read."""
    data = run.result()
    ds = data["datasets"]["geochem"][0]
    unit_name = data.get("unit", list(soilgold.UNITS)[0])
    mult = soilgold.UNITS[unit_name]
    x, y = ds["x"], ds["y"]
    v = ds["table"][ds["au"]].to_numpy(float)
    method = method or _method()

    def build():
        lay = soilgold.layout(x, y)
        pops = soilgold.populations(v)
        limits = soilgold.thresholds(v, method, pops)
        return {"layout": lay, "stats": soilgold.summary(v, mult), "pops": pops, "limits": limits,
                "classes": geochem.classify(v, limits),
                "zones": soilgold.zones(x, y, v, limits[1], lay["spacing"]),
                "floor": soilgold.detection_floor(v)}

    A = dict(run.cached(("geochem", method), build))
    A.update({"data": data, "ds": ds, "unit_name": unit_name, "u": unit_name.split(" ")[0],
              "mult": mult, "x": x, "y": y, "v": v, "method": method})
    A["narrative"] = soilgold.narrative(A["stats"], A["layout"], A["pops"], A["limits"], A["zones"],
                                        x, y, mult, unit_name, A["floor"])
    A["fallback"] = (method == soilgold.METHODS[0]
                     and not (A["pops"].get("ok") and A["pops"]["n_pop"] == 2
                              and A["pops"].get("limits")))
    A["n_anomalous"] = int((A["classes"] >= 2).sum())
    A["n_zones"] = int((A["zones"].samples >= 3).sum()) if len(A["zones"]) else 0
    return A


def panel(name):
    run = base.need_run(KEY, request.args.get("run"))
    A = analyse(run)
    ctx = {"run": run, "A": A, "u": A["u"], "method": A["method"],
           "fig_url": lambda f, **kw: url_for("app.figure", section_key=KEY, run_id=run.id, name=f,
                                              method=A["method"], **kw),
           "dl": lambda kind, f, **kw: url_for("app.download", section_key=KEY, run_id=run.id,
                                               kind=kind, name=f, method=A["method"], **kw)}
    if name == "map":
        return render_template("app/geochemistry/_map.html", limits=A["limits"], **ctx)
    if name == "zones":
        t = _zone_table(A)
        return render_template("app/geochemistry/_zones.html",
                               table=None if t is None else table_html(t), **ctx)
    if name == "distribution":
        return render_template("app/geochemistry/_distribution.html", **ctx)
    if name == "interpolated":
        return _panel_interpolated(run, A, ctx)
    if name == "statistics":
        return render_template("app/geochemistry/_statistics.html", rows=_stat_rows(A), **ctx)
    if name == "samples":
        t = _sample_table(A).sort_values("Au", ascending=False)
        return render_template("app/geochemistry/_samples.html", table=table_html(t, max_rows=500),
                               **ctx)
    abort(404)


def _lonlat(A, xs, ys):
    from pyproj import Transformer
    return Transformer.from_crs(A["data"]["epsg"], 4326, always_xy=True).transform(
        np.asarray(xs, float), np.asarray(ys, float))


def _zone_table(A):
    z, x, y, u, mult = A["zones"], A["x"], A["y"], A["u"], A["mult"]
    if not len(z):
        return None
    lon, lat = _lonlat(A, z.peak_x.to_numpy(), z.peak_y.to_numpy())
    return pd.DataFrame({
        "Zone": z.zone, "Samples": z.samples, f"Peak ({u})": z.peak.round(4),
        "Peak (ppb)": (z.peak * mult).round(0), f"Mean ({u})": z["mean"].round(4),
        "Peak easting": z.peak_x.round(0), "Peak northing": z.peak_y.round(0),
        "Latitude": np.round(lat, 6), "Longitude": np.round(lon, 6),
        "Length (m)": z.length.round(0),
        "Trend": [f"{a:.0f}° {soilgold.strike_label(a)}" if np.isfinite(a) else "" for a in z.azimuth],
        "Open towards": [", ".join(soilgold.open_edges(r, x, y, A["layout"]["spacing"]))
                         for _, r in z.iterrows()],
        "Follow up": np.where(z.samples >= 3, "Yes, infill", "Re-sample first")})


def _sample_table(A):
    z, v, x, y = A["zones"], A["v"], A["x"], A["y"]
    zone_of = np.array([""] * len(v), dtype=object)
    for r in z.itertuples():
        zone_of[r.indices] = r.zone
    lon, lat = _lonlat(A, x, y)
    return soilgold.report_table(A["ds"]["ids"], x, y, v, A["classes"], lon, lat, A["mult"], zone_of)


def _stat_rows(A):
    s, lay, limits, u = A["stats"], A["layout"], A["limits"], A["u"]
    return [("Samples", f"{s['n']:,}"), ("Area covered", f"{lay['area_ha']:,.1f} ha"),
            ("Sample spacing", f"{lay['spacing']:,.0f} m"),
            ("Duplicate positions", f"{lay['duplicates']}"),
            ("Minimum", f"{s['min']:,.4g} {u}"), ("25th percentile", f"{s['p25']:,.4g} {u}"),
            ("Median", f"{s['median']:,.4g} {u}"), ("Mean", f"{s['mean']:,.4g} {u}"),
            ("Geometric mean", f"{s['geomean']:,.4g} {u}"),
            ("75th percentile", f"{s['p75']:,.4g} {u}"), ("90th percentile", f"{s['p90']:,.4g} {u}"),
            ("95th percentile", f"{s['p95']:,.4g} {u}"), ("98th percentile", f"{s['p98']:,.4g} {u}"),
            ("Maximum", f"{s['max']:,.4g} {u}"),
            ("Peak / median contrast", f"{s['contrast']:,.1f}×"),
            ("Coefficient of variation", f"{s['cv']:,.2f}"),
            ("Spread of the logs (σ)", f"{s['log_sd']:,.2f}"),
            ("Weak anomaly threshold", f"{limits[0]:,.4g} {u}"),
            ("Anomalous threshold", f"{limits[1]:,.4g} {u}"),
            ("Strongly anomalous threshold", f"{limits[2]:,.4g} {u}")]


# ------------------------------------------------------------------ interpolation
def _interp_args(args=None):
    a = request.values if args is None else args
    sent = "_i" in a                    # the panel's own form has been submitted at least once
    method = a.get("imethod")
    method = method if method in interpolate.METHODS else interpolate.METHODS[0]
    auto = bool(a.get("auto_cell")) if sent else True
    try:
        cell = None if auto else (float(a.get("cell") or 0) or None)
    except (TypeError, ValueError):
        cell = None
    try:
        power = min(4.0, max(1.0, float(a.get("power", 2.0))))
    except (TypeError, ValueError):
        power = 2.0
    vmodel = a.get("vmodel")
    vmodel = vmodel if vmodel in interpolate.VARIOGRAMS else interpolate.VARIOGRAMS[0]
    log_scale = bool(a.get("log")) if sent else True
    return {"imethod": method, "auto_cell": auto, "cell": cell, "power": power, "vmodel": vmodel,
            "log": log_scale}


def _grid(run, A, p):
    v = A["v"]
    vals = np.log10(np.where(v > 0, v, np.nan)) if p["log"] else np.asarray(v, float)
    return run.cached(("interp", p["imethod"], p["cell"], p["power"], p["vmodel"], p["log"]),
                      lambda: interpolate.grid(A["x"], A["y"], vals, method=p["imethod"],
                                               cell=p["cell"], power=p["power"],
                                               variogram=p["vmodel"]))


def interp_query(p):
    """The interpolation settings as URL parameters, so a picture route can rebuild the same grid.
    Without these the server would redraw with the defaults and quietly serve the wrong map."""
    return {"_i": 1, "imethod": p["imethod"], "power": p["power"], "vmodel": p["vmodel"],
            "auto_cell": 1 if p["auto_cell"] else "", "cell": "" if p["cell"] is None else p["cell"],
            "log": 1 if p["log"] else ""}


def _interp_figname(p, prefix="interp"):
    return prefix + "-" + safe_name(
        f"{p['imethod']}-{p['cell']}-{p['power']}-{p['vmodel']}-{p['log']}")


def _panel_interpolated(run, A, ctx):
    p = _interp_args()
    suggested = interpolate.suggest_cell(A["x"], A["y"])
    common = dict(p=p, suggested=suggested, methods=interpolate.METHODS,
                  variograms=interpolate.VARIOGRAMS, kriging=interpolate.METHODS[2],
                  idw=interpolate.METHODS[1], iq=interp_query(p), **ctx)
    try:
        g = _grid(run, A, p)
    except Exception as e:
        return render_template("app/geochemistry/_interpolated.html", g=None, error=str(e),
                               figname=None, vgname=None, errname=None, vg=None, **common)
    return render_template("app/geochemistry/_interpolated.html", g=g, error=None,
                           figname=_interp_figname(p), vgname=_interp_figname(p, "variogram"),
                           errname=_interp_figname(p, "krigerror"), vg=g.get("variogram"),
                           **common)


# ------------------------------------------------------------------ pictures
def redraw(run, name, args):
    from ..render import save_fig
    A = analyse(run, _method(args))
    u = A["u"]
    fig = None
    if name == "map":
        fig = soilgold.zone_map_figure(A["x"], A["y"], A["v"], A["classes"], A["zones"], u)
    elif name == "histogram":
        fig = plots.hist_figure(A["v"], A["limits"], f"Gold distribution ({u})", u)
    elif name == "probability":
        fig = soilgold.probability_figure(A["v"], A["pops"], A["limits"], u)
    elif name.startswith(("interp-", "variogram-", "krigerror-")):
        p = _interp_args(args)
        g = _grid(run, A, p)
        if name.startswith("interp-"):
            shown = 10 ** g["z"] if p["log"] else g["z"]
            fig = plots.grid_figure(shown, g["transform"],
                                    f"Gold in soil · {p['imethod']} · {g['cell']:,.1f} m grid",
                                    "YlOrRd", u, overlay_pts=(A["x"], A["y"]))
        elif name.startswith("variogram-"):
            if g.get("variogram") is None:
                return None
            fig = interpolate.variogram_figure(g["variogram"], title="Fitted variogram")
        else:
            if g.get("error") is None:
                return None
            fig = plots.grid_figure(g["error"], g["transform"], "Kriging standard error",
                                    "viridis_r", "log10 error" if p["log"] else u,
                                    overlay_pts=(A["x"], A["y"]))
    if fig is None:
        return None
    return save_fig(fig, run.fig_path(name))


def deliver(run, kind, name, args):
    A = analyse(run, _method(args))
    if kind == "png":
        path = run.fig_path(name)
        if not path.is_file():
            path = redraw(run, name, args)
        return base.send_png(path, f"soil_gold_{name}.png")
    if kind == "csv":
        if name == "zones":
            t = _zone_table(A)
            if t is None:
                abort(404)
            return base.send_csv(t, "gold_zones.csv")
        if name == "samples":
            return base.send_csv(_sample_table(A), "soil_gold_classified.csv")
        abort(404)
    if kind == "tif":
        p = _interp_args(args)
        g = _grid(run, A, p)
        shown = 10 ** g["z"] if p["log"] else g["z"]
        return base.send_bytes(
            export.geotiff_bytes({"Au": shown.astype("float32")}, g["transform"], A["data"]["epsg"]),
            "soil_gold_interpolated.tif", "image/tiff")
    abort(404)
