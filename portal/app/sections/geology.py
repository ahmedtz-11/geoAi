"""4 · Geology: structural measurements and / or a QDS geological map sheet.

Both uploads are optional and either on its own is enough, so this page takes two separate slots: a
table of readings and a sheet. It settles which way each plane dips, separates the readings into
structural sets, works out where those sets cross — the line an ore shoot usually follows — and keeps
the interpretation with the run for section 6.
"""
import numpy as np
import pandas as pd
from flask import abort, flash, redirect, render_template, request, url_for
from flask_login import current_user

from geoai import coords, export, interpret, plots, readers, structure

from .. import store, uploads
from ..render import table_html
from . import base

KEY = "geology"
NONE = uploads.NONE
TABLE_TYPES = ("csv", "xls", "xlsx")
MAP_TYPES = ("png", "jpg", "jpeg", "tif", "tiff")

TABS = [("Map", "map"), ("Stereonet", "stereonet"), ("Strike rose", "rose"), ("Sets", "sets"),
        ("Readings", "readings"), ("QDS sheet", "sheet")]


def attach(bp):
    from .. import section
    bp.add_url_rule("/geology", "geology", section(KEY)(page), methods=["GET", "POST"])
    bp.add_url_rule("/geology/panel/<name>", "geology_panel", section(KEY)(panel))


# ------------------------------------------------------------------ the two slots
def _sort(items):
    """Split what was uploaded into the measurements and the sheet."""
    table = sheet = None
    for it in items:
        ext = it["name"].rsplit(".", 1)[-1].lower()
        if ext in TABLE_TYPES and table is None:
            table = it
        elif ext in MAP_TYPES and sheet is None:
            sheet = it
    return table, sheet


def _table_fields(item):
    if not item or not item["info"]:
        return {"error": item["error"] if item else None}
    if item["info"]["kind"] != "table":
        return {"error": "The measurements must be a CSV or an Excel sheet."}
    df = item["info"]["result"]["df"]
    if len(df.columns) < 3:
        return {"error": "The measurements need at least three columns."}
    cols = [str(c) for c in df.columns]
    opt = [NONE] + cols
    gx, gy = coords.guess_columns(df)
    gx = str(gx) if gx is not None else cols[0]
    gy = str(gy) if gy is not None else cols[min(1, len(cols) - 1)]

    def guess(words, allow_none=True):
        g = structure.guess(df, words)
        g = str(g) if g is not None else None
        pool = opt if allow_none else cols
        return g if g in pool else pool[0]

    f = uploads.field
    s_guess = guess(["strike", "trend"], False)
    d_guess = guess(["dip"], False)
    sv = coords.parse_column(df[s_guess]).to_numpy(float)
    dv = coords.parse_column(df[d_guess]).to_numpy(float)
    usable = int((np.isfinite(sv) & np.isfinite(dv)).sum())
    return {
        "error": None, "usable": usable, "rows": len(df),
        "fields": [
            f("xcol", "Easting / longitude", "select", cols, gx, width="third"),
            f("ycol", "Northing / latitude", "select", cols, gy, width="third"),
            uploads.crs_field("crs", "Coordinate system",
                              coords.parse_column(df[gx]).dropna().head(200),
                              coords.parse_column(df[gy]).dropna().head(200), width="third"),
            f("strikecol", "Strike column", "select", cols, s_guess, width="third"),
            f("dipcol", "Dip column", "select", cols, d_guess, width="third"),
            f("ddcol", "Dip direction (optional)", "select", opt, guess(structure.DIPDIR_COLS),
              "With no dip direction the right-hand rule is assumed: the plane dips 90° clockwise of "
              "the strike. If your strikes were not recorded that way, the dips will face the wrong "
              "side.", width="third"),
            f("typecol", "Structure type (optional)", "select", opt, guess(structure.STRUCT_COLS),
              "Shear, fault, vein… These names are what section 6 ranks as gold-favourable hosts.",
              width="half"),
            f("idcol", "Station ID (optional)", "select", opt, guess(structure.STOP_COLS),
              width="half"),
        ],
    }


def _sheet_fields(item, table_item, values):
    """A GeoTIFF brings its own coordinates; a picture needs its edges typed in."""
    if not item or not item["info"]:
        return {"error": item["error"] if item else None}
    res = item["info"]["result"]
    if res["kind"] == "raster":
        return {"error": None, "kind": "raster", "epsg": item["info"].get("epsg"),
                "shape": item["info"].get("shape"), "layers": item["info"].get("layers", []),
                "fields": []}
    if res["kind"] != "image":
        return {"error": "That file is neither a picture nor a GeoTIFF."}

    d0, why = (0.0, 0.0, 0.0, 0.0), ("needed before the sheet can be used: a QDS sheet's corners are "
                                     "printed on it, or take them from the sheet index")
    if table_item and table_item["info"] and table_item["info"]["kind"] == "table":
        df = table_item["info"]["result"]["df"]
        xc, yc = values.get("xcol"), values.get("ycol")
        if xc in df.columns and yc in df.columns:
            xs = coords.parse_column(df[xc]).dropna()
            ys = coords.parse_column(df[yc]).dropna()
            if len(xs) and len(ys):
                d0 = (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))
                why = "pre-filled from the area your measurements cover"
    f = uploads.field
    return {"error": None, "kind": "image", "why": why, "fields": [
        f("mx0", "West", "number", value=round(d0[0], 2), width="quarter", step="any"),
        f("mx1", "East", "number", value=round(d0[1], 2), width="quarter", step="any"),
        f("my0", "South", "number", value=round(d0[2], 2), width="quarter", step="any"),
        f("my1", "North", "number", value=round(d0[3], 2), width="quarter", step="any"),
        uploads.crs_field("mcrs", "Coordinate system of the edges", width="full"),
    ]}


def page():
    if request.method == "POST":
        action = request.form.get("do")
        if action == "upload":
            base.take_upload(KEY, multiple=True)
            return redirect(url_for("app.geology") + "#files")
        if action in ("remove", "clear"):
            if action == "clear":
                store.clear_draft(current_user.id, KEY)
            else:
                store.drop_draft(current_user.id, KEY, request.form.get("name", ""))
            return redirect(url_for("app.geology"))
        if action == "submit":
            return _submit()

    items = base.drafts(KEY)
    for it in items:
        it["prefix"] = ""
        it["fields"] = []
    table_item, sheet_item = _sort(items)
    tform = _table_fields(table_item) if table_item else None
    values = _values(tform, request.values)
    sform = _sheet_fields(sheet_item, table_item, values) if sheet_item else None
    run = base.current_run(KEY)
    return base.page(KEY, "app/geology.html", items=items, table_item=table_item,
                     sheet_item=sheet_item, tform=tform, sform=sform, run=run, tabs=TABS,
                     analysis=analyse(run) if run is not None else None,
                     max_sets=_max_sets())


def _values(tform, submitted):
    from werkzeug.datastructures import MultiDict
    md = MultiDict()
    if tform and not tform.get("error"):
        for f in tform["fields"]:
            if f["value"] is not None:
                md.add(f["name"], f["value"])
    for k in submitted:
        md.setlist(k, submitted.getlist(k))
    return md


def _specs(items, values):
    """Both uploads turned into engine specs. Either on its own is enough."""
    table_item, sheet_item = _sort(items)
    specs, cfg = [], None

    if table_item:
        tform = _table_fields(table_item)
        if tform.get("error"):
            raise ValueError(tform["error"])
        df = table_item["info"]["result"]["df"]
        scol, dcol = values.get("strikecol"), values.get("dipcol")
        if not scol or not dcol:
            raise ValueError("Choose the strike and dip columns.")
        sv = coords.parse_column(df[scol]).to_numpy(float)
        dv = coords.parse_column(df[dcol]).to_numpy(float)
        keep = np.isfinite(sv) & np.isfinite(dv)
        if int(keep.sum()) < 3:
            raise ValueError("Fewer than three rows have both a strike and a dip.")
        # Only the rows that carry a strike and a dip go to the engine, so the point positions it
        # returns line up one-for-one with the readings the stereonet and the sets are built from.
        df = df[keep].reset_index(drop=True)

        def opt(name):
            v = values.get(name)
            return None if v in (None, "", NONE) else v

        cfg = {"xcol": values.get("xcol"), "ycol": values.get("ycol"),
               "epsg": uploads.resolve_crs(values, "crs"), "strikecol": scol, "dipcol": dcol,
               "ddcol": opt("ddcol"), "typecol": opt("typecol"), "idcol": opt("idcol"),
               "dropped": int((~keep).sum())}
        specs.append({"slot": KEY, "name": table_item["name"], "kind": "table",
                      "data": {"kind": "table", "df": df}, "xcol": cfg["xcol"], "ycol": cfg["ycol"],
                      "epsg": cfg["epsg"], "strikecol": scol, "dipcol": dcol,
                      "strike_is_dipdir": False,
                      # the structure name doubles as the "lithology" section 6 ranks, so SHEAR and
                      # VEIN are picked up there as gold-favourable hosts
                      "lithcol": cfg["typecol"]})

    if sheet_item:
        res = sheet_item["info"]["result"]
        if res["kind"] == "raster":
            specs.append({"slot": KEY, "name": sheet_item["name"], "kind": "raster", "data": res})
        elif res["kind"] == "image":
            def num(name):
                try:
                    return float(values.get(name, 0) or 0)
                except (TypeError, ValueError):
                    return 0.0
            x0, x1, y0, y1 = num("mx0"), num("mx1"), num("my0"), num("my1")
            if x1 <= x0 or y1 <= y0:
                raise ValueError("Enter the sheet's edges: east above west, north above south.")
            mepsg = uploads.resolve_crs(values, "mcrs")
            looks_ll = coords.looks_geographic(pd.Series([x0, x1]), pd.Series([y0, y1]))
            if mepsg == 4326 and not looks_ll:
                raise ValueError("Those edges are far too large for degrees of latitude and "
                                 "longitude. They look like UTM metres, so pick the matching UTM zone.")
            if mepsg != 4326 and looks_ll:
                raise ValueError("Those edges look like degrees of latitude and longitude, not UTM "
                                 "metres. Pick WGS84 latitude / longitude.")
            specs.append({"slot": KEY, "name": sheet_item["name"], "kind": "image", "data": res,
                          "extent": (x0, x1, y0, y1), "epsg": mepsg})
        else:
            raise ValueError("That sheet is neither a picture nor a GeoTIFF.")
    if not specs:
        raise ValueError("Upload the measurements, the map sheet, or both.")
    return specs, cfg


def _submit():
    items = base.drafts(KEY)
    table_item, _sheet = _sort(items)
    tform = _table_fields(table_item) if table_item else None
    values = _values(tform, request.form)
    try:
        specs, cfg = _specs(items, values)
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("app.geology"))

    table_df = next((s["data"]["df"] for s in specs if s["kind"] == "table"), None)

    def work(run, job):
        out = interpret.run(specs, {}, progress=lambda p, t: job.say(0.1 + 0.8 * p, t))
        if not out["datasets"]["geology"]:
            raise ValueError("; ".join(out["errors"]) or "nothing usable in these files")
        out["cfg"] = cfg
        out["readings"] = table_df          # the rows the measurements were read from
        job.say(0.95, "Saving the results…")
        run.save_result(out)

    return base.submit(KEY, work, label="Interpreting " + ", ".join(s["name"] for s in specs),
                       settings={"inputs": [s["name"] for s in specs]})


# ------------------------------------------------------------------ the analysis
def _max_sets(args=None):
    a = request.values if args is None else args
    try:
        return max(1, min(int(float(a.get("sets", 3))), 4))
    except (TypeError, ValueError):
        return 3


def analyse(run, max_sets=None):
    data = run.result()
    cfg = data.get("cfg")
    bg = next((d for d in data["datasets"]["geology"] if d["kind"] == "image_ref"), None)
    pts = next((d for d in data["datasets"]["geology"] if d["kind"] == "points"), None)
    out = {"data": data, "cfg": cfg, "bg": bg, "pts": pts, "errors": data.get("errors") or [],
           "sheet_only": pts is None or cfg is None,
           "background": (bg["image"], tuple(bg["extent"])) if bg is not None else None}
    if out["sheet_only"]:
        return out

    max_sets = _max_sets() if max_sets is None else max_sets
    df = data.get("readings")
    if df is None:
        out["sheet_only"] = True
        return out
    strike = coords.parse_column(df[cfg["strikecol"]]).to_numpy(float)
    dip = coords.parse_column(df[cfg["dipcol"]]).to_numpy(float)
    dipdir = coords.parse_column(df[cfg["ddcol"]]).to_numpy(float) if cfg["ddcol"] else None
    types = df[cfg["typecol"]].astype(str).to_numpy() if cfg["typecol"] else None
    ids = (df[cfg["idcol"]].astype(str).to_numpy() if cfg["idcol"]
           else np.arange(1, len(df) + 1).astype(str))

    def build():
        res = structure.resolve(strike, dip, dipdir)
        ok = res.ok.to_numpy()
        sets, labels = structure.find_sets(res.strike[ok], res.dip[ok], max_sets=max_sets)
        return {"res": res, "sets": sets, "labels": labels,
                "inters": structure.intersections(sets),
                "spread": structure.spread(pts["x"], pts["y"], ids),
                "types": structure.classify_types(types) if types is not None else None}

    A = dict(run.cached(("structure", max_sets), build))
    res, sets, labels = A["res"], A["sets"], A["labels"]
    ok = res.ok.to_numpy()
    set_of = np.array([""] * len(res), dtype=object)
    set_of[np.flatnonzero(ok)] = [sets[j]["set"] if 0 <= j < len(sets) else "" for j in labels]
    A.update({"ok": ok, "ids": ids, "set_of": set_of, "max_sets": max_sets,
              "narrative": structure.narrative(res, A["types"], sets, labels, A["inters"],
                                               A["spread"]),
              "n_used": int(ok.sum()), "n_total": int(len(res))})
    out.update(A)
    return out


def panel(name):
    run = base.need_run(KEY, request.args.get("run"))
    A = analyse(run)
    ctx = {"run": run, "A": A,
           "fig_url": lambda f, **kw: url_for("app.figure", section_key=KEY, run_id=run.id, name=f,
                                              sets=A.get("max_sets", 3), **kw),
           "dl": lambda kind, f, **kw: url_for("app.download", section_key=KEY, run_id=run.id,
                                               kind=kind, name=f, sets=A.get("max_sets", 3), **kw)}
    if name == "sheet":
        return render_template("app/geology/_sheet.html", **ctx)
    if A["sheet_only"]:
        return ('<p class="note warn">Only the map sheet was uploaded, so there is nothing to '
                'interpret from measurements.</p>')
    if name == "map":
        return render_template("app/geology/_map.html", **ctx)
    if name == "stereonet":
        return render_template("app/geology/_stereonet.html", **ctx)
    if name == "rose":
        return render_template("app/geology/_rose.html", **ctx)
    if name == "sets":
        return render_template("app/geology/_sets.html", sets_table=_sets_table(A),
                               cross_table=_cross_table(A), **ctx)
    if name == "readings":
        return render_template("app/geology/_readings.html",
                               table=table_html(_readings(A), max_rows=500), **ctx)
    abort(404)


def _sets_table(A):
    if not A["sets"]:
        return None
    return pd.DataFrame([{
        "Set": s["set"], "Readings": s["n"],
        "Mean strike": f"{s['strike']:.0f}° ({structure.strike_label(s['strike'])})",
        "Mean dip": f"{s['dip']:.0f}° {structure.dip_word(s['dip'])}",
        "Dips towards": f"{s['dipdir']:.0f}°",
        "Tightness κ": round(s["kappa"], 1) if np.isfinite(s["kappa"]) else "",
        "Mean within ±": f"{s['a95']:.0f}°" if np.isfinite(s["a95"]) else "",
        "Spread": f"{s['scatter']:.0f}°"} for s in A["sets"]])


def _cross_table(A):
    if not A["inters"]:
        return None
    return pd.DataFrame([{"Sets": i["sets"], "Plunge": f"{i['plunge']:.0f}°",
                          "Towards": f"{i['trend']:.0f}° ({structure.strike_label(i['trend'])})",
                          "Angle between planes": f"{i['angle']:.0f}°"} for i in A["inters"]])


def _readings(A):
    from pyproj import Transformer
    pts, res = A["pts"], A["res"]
    lon, lat = Transformer.from_crs(A["data"]["epsg"], 4326, always_xy=True).transform(
        np.asarray(pts["x"], float), np.asarray(pts["y"], float))
    out = pd.DataFrame({"Station": A["ids"], "Easting": np.round(pts["x"], 1),
                        "Northing": np.round(pts["y"], 1), "Latitude": np.round(lat, 6),
                        "Longitude": np.round(lon, 6),
                        "Strike as recorded": res.strike_in.round(0), "Dip": res.dip.round(0),
                        "Strike (right-hand rule)": res.strike.round(0),
                        "Dips towards": res.dipdir.round(0),
                        "Dip direction off by": res.offset.round(0)})
    if A["types"] is not None:
        out["Structure"] = A["types"]
    out["Set"] = A["set_of"]
    out["Used"] = np.where(A["ok"], "yes", "set aside")
    return out


# ------------------------------------------------------------------ pictures
def redraw(run, name, args):
    from ..render import save_fig
    A = analyse(run, _max_sets(args))
    fig = None
    if name == "sheet":
        if A["bg"] is None:
            return None
        pts = A.get("pts")
        fig = plots.image_figure(A["bg"]["image"], list(A["bg"]["extent"]),
                                 f"QDS sheet · {A['bg']['name']}",
                                 pts=(pts["x"], pts["y"]) if pts is not None else None)
    elif A["sheet_only"]:
        return None
    elif name == "map":
        fig = structure.map_figure(A["pts"]["x"], A["pts"]["y"], A["res"], A["labels"], A["types"],
                                   A["sets"], A["background"], A["ids"],
                                   "Structural measurements"
                                   + (" on the QDS sheet" if A["background"] else ""))
    elif name == "stereonet":
        fig = structure.stereonet_figure(A["res"], A["labels"], A["sets"], A["inters"])
    elif name == "rose":
        starts = np.arange(0, 180, 10.0)
        h, _ = np.histogram(np.asarray(A["res"].strike[A["ok"]]) % 180, bins=np.arange(0, 190, 10))
        fig = export.rose_figure(starts, h.astype(float), title="Strike rose (field measurements)")
    if fig is None:
        return None
    return save_fig(fig, run.fig_path(name))


def deliver(run, kind, name, args):
    A = analyse(run, _max_sets(args))
    if kind == "png":
        path = run.fig_path(name)
        if not path.is_file():
            path = redraw(run, name, args)
        return base.send_png(path, f"geology_{name}.png")
    if kind == "csv":
        if name == "sets":
            t = _sets_table(A)
            if t is None:
                abort(404)
            return base.send_csv(t, "structural_sets.csv")
        if name == "readings":
            return base.send_csv(_readings(A), "structural_readings.csv")
    abort(404)
