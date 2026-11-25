"""Pit mapping: pit coordinates, plus an optional picture and notes.

Gives the mineralisation trend, how much potential each pit still has, where to sink the next pits,
and a register that keeps every coordinate for next time. The register is per account, because it is
the one thing here meant to build up across runs.
"""
import numpy as np
import pandas as pd
from flask import abort, flash, redirect, render_template, request, url_for
from flask_login import current_user

from geoai import coords, pits as pitlib, readers

from .. import store, uploads
from ..render import table_html
from . import base

KEY = "pit-mapping"
TABLE_TYPES = ("xls", "xlsx", "csv", "txt")
IMAGE_TYPES = ("png", "jpg", "jpeg")
NOTE_TYPES = ("txt",)
NONE = uploads.NONE

TABS = [("Map", "map"), ("Pits: what is left", "pits"), ("Proposed pits", "targets"),
        ("Long section", "section"), ("Picture & notes", "extras"), ("Pit register", "register")]

LABELS = {"pit": "Pit name / ID", "depth": "Pit depth (m)", "grade": "Gold grade",
          "status": "Status / remarks", "strike": "Vein strike (°)"}


def attach(bp):
    from .. import section
    bp.add_url_rule("/pit-mapping", "pit_mapping", section(KEY)(page), methods=["GET", "POST"])
    bp.add_url_rule("/pit-mapping/panel/<name>", "pit_panel", section(KEY)(panel))


def _register_path():
    """One register per account, kept beside that person's runs."""
    return store.home(current_user.id) / "pit_register.csv"


# ------------------------------------------------------------------ the uploads
def _sort(items):
    table = picture = notes = None
    for it in items:
        ext = it["name"].rsplit(".", 1)[-1].lower()
        if ext in IMAGE_TYPES and picture is None:
            picture = it
        elif ext == "txt" and table is not None and notes is None:
            notes = it
        elif ext in TABLE_TYPES and table is None:
            table = it
        elif ext == "txt" and notes is None:
            notes = it
    return table, picture, notes


def _fields(item):
    if not item or not item["info"]:
        return {"error": item["error"] if item else None}
    if item["info"]["kind"] != "table":
        return {"error": "The pit coordinates must be a table: Excel, CSV or text."}
    df = item["info"]["result"]["df"]
    if len(df.columns) < 2:
        return {"error": "The file needs at least two columns: easting and northing."}
    cols = [str(c) for c in df.columns]
    gx, gy = coords.guess_columns(df)
    gx = str(gx) if gx is not None else cols[0]
    gy = str(gy) if gy is not None else cols[min(1, len(cols) - 1)]
    guess = pitlib.guess_fields(df, used=(gx, gy))
    opt = [NONE] + [c for c in cols if c not in (gx, gy)]
    f = uploads.field
    out = [f("xcol", "Easting / longitude column", "select", cols, gx, width="third"),
           f("ycol", "Northing / latitude column", "select", cols, gy, width="third"),
           uploads.crs_field("crs", "Coordinate system",
                             coords.parse_column(df[gx]).dropna().head(200),
                             coords.parse_column(df[gy]).dropna().head(200), width="third")]
    for k, label in LABELS.items():
        g = str(guess.get(k)) if guess.get(k) is not None else None
        out.append(f(f"fld_{k}", label, "select", opt, g if g in opt else NONE, width="quarter"))
    return {"error": None, "fields": out, "rows": len(df)}


def page():
    if request.method == "POST":
        action = request.form.get("do")
        if action == "upload":
            base.take_upload(KEY, multiple=True)
            return redirect(url_for("app.pit_mapping") + "#files")
        if action in ("remove", "clear"):
            if action == "clear":
                store.clear_draft(current_user.id, KEY)
            else:
                store.drop_draft(current_user.id, KEY, request.form.get("name", ""))
            return redirect(url_for("app.pit_mapping"))
        if action == "submit":
            return _submit()
        if action == "save-register":
            return _save_register()
        if action == "drop-project":
            return _drop_project()

    items = base.drafts(KEY)
    for it in items:
        it["prefix"] = ""
        it["fields"] = []
    table_item, picture, notes = _sort(items)
    form = _fields(table_item) if table_item else None
    run = base.current_run(KEY)
    return base.page(KEY, "app/pits.html", items=items, form=form, picture=picture, notes=notes,
                     run=run, tabs=TABS, analysis=_analysis(run) if run is not None else None,
                     register=_register_view(), project=request.values.get("project", "My site"),
                     max_t=_i("max_t", 8, 2, 20), outs=_i("outs", 2, 0, 4))


def _i(name, default, lo, hi):
    try:
        return max(lo, min(int(float(request.values.get(name, default))), hi))
    except (TypeError, ValueError):
        return default


def _submit():
    items = base.drafts(KEY)
    table_item, picture, notes = _sort(items)
    if table_item is None:
        flash("Upload the pit coordinates first.", "error")
        return redirect(url_for("app.pit_mapping"))
    form = _fields(table_item)
    if form.get("error"):
        flash(form["error"], "error")
        return redirect(url_for("app.pit_mapping"))

    f = request.form
    df = table_item["info"]["result"]["df"]
    xcol, ycol = f.get("xcol"), f.get("ycol")
    epsg = uploads.resolve_crs(f, "crs")
    fields = {k: (None if f.get(f"fld_{k}") in (None, "", NONE) else f.get(f"fld_{k}"))
              for k in LABELS}
    project = (f.get("project") or "My site").strip() or "My site"
    max_t, outs = _i("max_t", 8, 2, 20), _i("outs", 2, 0, 4)

    image = None
    if picture is not None:
        try:
            image = picture["info"]["result"].get("image")
        except Exception:
            image = None
    extent = None
    if image is not None and f.get("img_is_map"):
        def num(n):
            try:
                return float(f.get(n) or 0)
            except (TypeError, ValueError):
                return 0.0
        e = (num("ix0"), num("ix1"), num("iy0"), num("iy1"))
        if e[1] > e[0] and e[3] > e[2]:
            extent = e
    note_text = ""
    if notes is not None:
        try:
            note_text = notes["path"].read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            note_text = ""

    def work(run, job):
        job.say(0.2, "Reading the pits…")
        p, work_epsg = pitlib.build(df, xcol, ycol, epsg, fields)
        job.say(0.5, "Finding the mineralisation trend…")
        tr = pitlib.trend(p)
        p = pitlib.potential(p, tr)
        job.say(0.8, "Proposing new pits…")
        t = pitlib.propose(p, tr, work_epsg, max_t, outs)
        run.save_result({"pits": p, "trend": tr, "targets": t, "epsg": work_epsg,
                         "project": project, "image": image, "extent": extent,
                         "notes": note_text})

    return base.submit(KEY, work, label=f"Interpreting {len(df)} pits",
                       settings={"project": project, "max_t": max_t, "outs": outs})


def _analysis(run):
    try:
        d = run.result()
    except Exception:
        return None
    p, tr, t = d["pits"], d["trend"], d["targets"]
    return {**d, "n_pits": len(p), "n_targets": len(t),
            "n_high": int((p.potential == "High").sum()),
            "trend_label": pitlib.strike_label(tr["azimuth"]),
            "summary": pitlib.summary(p, tr, t),
            "has_depth": int(np.isfinite(p.depth).sum()) >= 2}


# ------------------------------------------------------------------ tabs
def panel(name):
    run = base.need_run(KEY, request.args.get("run"))
    A = _analysis(run)
    if A is None:
        abort(404)
    ctx = {"run": run, "A": A,
           "fig_url": lambda f: url_for("app.figure", section_key=KEY, run_id=run.id, name=f),
           "dl": lambda kind, f: url_for("app.download", section_key=KEY, run_id=run.id, kind=kind,
                                         name=f)}
    if name == "map":
        return render_template("app/pits/_map.html", **ctx)
    if name == "pits":
        return render_template("app/pits/_pits.html", table=table_html(_pit_table(A)), **ctx)
    if name == "targets":
        t = _target_table(A)
        return render_template("app/pits/_targets.html",
                               table=None if t is None else table_html(t), **ctx)
    if name == "section":
        return render_template("app/pits/_section.html", **ctx)
    if name == "extras":
        return render_template("app/pits/_extras.html",
                               has_image=A.get("image") is not None, **ctx)
    if name == "register":
        return render_template("app/pits/_register.html", **ctx)
    abort(404)


def _pit_table(A):
    p = A["pits"]
    return p[["pit", "potential", "score", "grade", "depth", "status", "why", "x", "y", "lat",
              "lon"]].rename(columns={
                  "pit": "Pit", "potential": "Remaining potential", "score": "Score",
                  "grade": "Grade", "depth": "Depth (m)", "status": "Status", "why": "Why",
                  "x": "Easting", "y": "Northing", "lat": "Latitude", "lon": "Longitude"
              }).sort_values("Score", ascending=False).round(
                  {"Easting": 1, "Northing": 1, "Latitude": 6, "Longitude": 6})


def _target_table(A):
    t = A["targets"]
    if not len(t):
        return None
    return t[["target", "kind", "priority", "score", "x", "y", "lat", "lon", "expected_depth",
              "nearest_pit", "why"]].rename(columns={
                  "target": "Target", "kind": "Type", "priority": "Priority", "score": "Score",
                  "x": "Easting", "y": "Northing", "lat": "Latitude", "lon": "Longitude",
                  "expected_depth": "Expected depth (m)", "nearest_pit": "Nearest pit",
                  "why": "Why here"}).round(
                  {"Easting": 1, "Northing": 1, "Latitude": 6, "Longitude": 6})


# ------------------------------------------------------------------ the register
def _register_view():
    try:
        reg = pitlib.load_register(str(_register_path()))
    except Exception:
        return {"rows": None, "projects": [], "table": None}
    if not len(reg):
        return {"rows": 0, "projects": [], "table": None}
    projects = sorted(reg.project.dropna().unique().tolist())
    chosen = request.values.getlist("projects") or projects
    view = reg[reg.project.isin(chosen)]
    return {"rows": len(reg), "projects": projects, "chosen": chosen,
            "table": table_html(view, max_rows=300)}


def _save_register():
    run = base.current_run(KEY)
    A = _analysis(run) if run else None
    if A is None:
        flash("Interpret a pit file first.", "error")
        return redirect(url_for("app.pit_mapping"))
    try:
        pitlib.save_register(str(_register_path()),
                             pitlib.register_rows(A["project"], A["pits"], A["targets"],
                                                  A["epsg"]))
        flash(f"Saved under “{A['project']}”.", "ok")
    except Exception as e:
        flash(f"Could not save the register: {e}", "error")
    return redirect(url_for("app.pit_mapping") + "#register")


def _drop_project():
    name = request.form.get("project", "")
    path = _register_path()
    try:
        reg = pitlib.load_register(str(path))
        reg[reg.project != name].to_csv(path, index=False)
        flash(f"Removed “{name}” from the register.", "ok")
    except Exception as e:
        flash(f"Could not change the register: {e}", "error")
    return redirect(url_for("app.pit_mapping") + "#register")


# ------------------------------------------------------------------ pictures and downloads
def redraw(run, name, args):
    from ..render import save_fig
    A = _analysis(run)
    if A is None:
        return None
    p, tr, t = A["pits"], A["trend"], A["targets"]
    if name == "map":
        bg = (A["image"], A["extent"]) if A.get("image") is not None and A.get("extent") else None
        fig = pitlib.map_figure(p, tr, t, f"Pit map · {A['project']} · EPSG:{A['epsg']}", bg)
    elif name == "section":
        if not A["has_depth"]:
            return None
        fig = pitlib.section_figure(p, tr, t, f"Long section along the trend · {A['project']}")
    elif name == "picture":
        if A.get("image") is None:
            return None
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(A["image"])
        ax.axis("off")
        fig.tight_layout()
    else:
        return None
    return save_fig(fig, run.fig_path(name))


def deliver(run, kind, name, args):
    A = _analysis(run)
    if A is None:
        abort(404)
    if kind == "png":
        path = run.fig_path(name)
        if not path.is_file():
            path = redraw(run, name, args)
        return base.send_png(path, f"pit_{name}.png")
    if kind == "csv":
        if name == "pits":
            return base.send_csv(_pit_table(A), "pit_ratings.csv")
        if name == "targets":
            t = _target_table(A)
            if t is None:
                abort(404)
            return base.send_csv(t, "proposed_pits.csv")
        if name == "register":
            reg = pitlib.load_register(str(_register_path()))
            return base.send_csv(reg, "pit_register.csv")
    if kind == "kml":
        return base.send_bytes(pitlib.to_kml(A["project"], A["pits"], A["targets"]), "pits.kml",
                               "application/vnd.google-earth.kml+xml")
    abort(404)
