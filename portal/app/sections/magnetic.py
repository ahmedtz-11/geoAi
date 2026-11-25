"""2 · Magnetic: one survey file, twelve interpreted maps.

The twelve maps are drawn on demand and kept as PNGs beside the run, so moving between tabs costs a
file read rather than a regrid. Every picture's name carries the settings that made it, which is both
the cache key and what the download route looks up.
"""
import io
import zipfile
from datetime import date, datetime

import numpy as np
import pandas as pd
from flask import abort, flash, redirect, render_template, request, url_for
from flask_login import current_user

from geoai import datasets, export, geophysics, interpret, lineaments, magnetics, plots, prospectivity
from geoai.geophysics import upward

from .. import store, uploads
from ..render import plotly_div, safe_name, table_html
from . import base

KEY = "magnetic"
SLOT = "geophysics"
SURVEY = geophysics.SURVEYS[0]                      # this page is magnetics only

INFO = {
    "TMI": ("turbo", "nT", "Total magnetic intensity as measured (gridded). Anomalies are shifted and "
                           "asymmetric because the Earth's field is inclined."),
    "RTP": ("turbo", "nT", "Reduced to the pole: anomalies moved over their sources and made symmetric, "
                           "as if measured at the magnetic pole. A low-latitude amplitude correction "
                           "keeps it stable."),
    "RTE": ("turbo", "nT", "Reduced to the equator: used at low magnetic latitudes, where RTP is "
                           "unstable. Magnetic bodies appear as lows elongated E–W; derivative maps use "
                           "the inverted RTE."),
    "1VD": ("RdBu_r", "nT/m", "First vertical derivative: sharpens shallow sources and narrows anomalies, "
                              "outlining near-surface magnetic units, dykes and BIF."),
    "2VD": ("RdBu_r", "nT/m²", "Second vertical derivative: even sharper; its zero line approximates the "
                               "edges of shallow bodies. Noisy data can make it speckled."),
    "Analytic signal": ("magma", "nT/m", "3-D analytic signal: peaks over the edges of magnetic bodies "
                                         "whatever the field direction, which makes it robust near the "
                                         "equator."),
    "Tilt derivative": ("RdBu_r", "degrees", "Tilt angle: balances strong and weak anomalies. The 0° line "
                                             "(black) follows contacts and faults; positive over sources."),
    "THDR": ("inferno", "nT/m", "Total horizontal derivative: ridges over contacts and faults (steepest "
                                "lateral change of the reduced field)."),
    "SPI depth": ("viridis_r", "m", "Depth to the top of magnetic contacts from Source Parameter Imaging."),
}

TARGET_LAYERS = {
    "Structure density": ("High", 1.0, "Concentration of lineaments (faults, contacts)."),
    "Proximity to structures": ("Low", 0.8, "Distance to the nearest lineament (closer is better)."),
    "Magnetic low (alteration)": ("Low", 0.6, "Magnetite destroyed by hydrothermal alteration along shears."),
    "Analytic signal": ("High", 0.3, "Edges of magnetic units such as BIF and mafic rocks."),
}

VIEWS = ["Normal", "Filled"]


# ------------------------------------------------------------------ picking the image type
def _view(args=None):
    v = (args or request.args).get("view", "").lower()
    return "filled" if v == "filled" else "normal"


def _result(run, want):
    """The processed result for the chosen image type, and the type actually used.

    "Normal" is the original processing: every reading, linear interpolation, nothing filled between
    the lines. It does not exist for a file that arrived already gridded, so that falls back to Filled.
    """
    data = run.result()
    items = data["datasets"]["geophysics"]
    if not items:
        return None, None, None, "filled"
    ds = items[0]
    normal = (data["results"].get("geophysics_normal") or [None])[0]
    use = want
    if normal is None or "maps" not in normal:
        use = "filled"
    r = data["results"]["geophysics"][0] if use == "filled" else normal
    if "maps" not in r:
        return data, ds, None, use
    r["_name"] = ds["name"]
    return data, ds, r, use


def _style(use):
    """Filled uses Oasis montaj's equal-area colours on the Geosoft rainbow; Normal keeps the
    original linear stretch and each map's own colours."""
    filled = use == "filled"
    return {"stretch": plots.STRETCHES[0] if filled else plots.STRETCHES[1], "smooth": filled,
            "palette": (lambda own: "geosoft" if filled else own)}


def figkey(*parts):
    return "-".join(safe_name(p) for p in parts if p not in (None, ""))


# ------------------------------------------------------------------ the page
def attach(bp):
    from .. import section
    bp.add_url_rule("/magnetic", "magnetic", section(KEY)(page), methods=["GET", "POST"])
    bp.add_url_rule("/magnetic/panel/<name>", "magnetic_panel", section(KEY)(panel))


def page():
    if request.method == "POST":
        action = request.form.get("do")
        if action == "upload":
            base.take_upload(KEY)
            return redirect(url_for("app.magnetic") + "#files")
        if action == "remove":
            store.drop_draft(current_user.id, KEY, request.form.get("name", ""))
            return redirect(url_for("app.magnetic"))
        if action == "clear":
            store.clear_draft(current_user.id, KEY)
            return redirect(url_for("app.magnetic"))
        if action == "submit":
            return _submit()

    items = base.draft_fields(KEY, SLOT, survey=SURVEY)
    field_info = _field_direction(items)
    run = base.current_run(KEY)
    extra = _results_context(run) if run is not None else {}
    return base.page(KEY, "app/magnetic.html", items=items, field_info=field_info,
                     settings_fields=_settings_fields(), views=VIEWS, survey=SURVEY,
                     run=run, **{"use": _view(), **extra})


TABS = [("Overview", "overview"), ("TMI", "tmi"), (None, "reduction"), ("1VD", "1vd"), ("2VD", "2vd"),
        ("Analytic signal", "analytic-signal"), ("Tilt derivative", "tilt"), ("THDR", "thdr"),
        ("Lineaments", "lineaments"), ("Upward continuation", "upward"), ("SPI depth", "spi"),
        ("Euler 3D", "euler"), ("Final target map", "targets")]


def _results_context(run):
    """What the results half of the page needs: which image types exist, the tabs, and the line
    describing how the data were processed."""
    want = _view()
    data, ds, r, use = _result(run, want)
    if r is None:
        return {"broken": True}
    both = (data["results"].get("geophysics_normal") or [None])[0] is not None
    tabs = [(label or r["reduction"], name) for label, name in TABS]
    return {"broken": False, "ds": ds, "r": r, "use": use, "both_views": both, "tabs": tabs,
            "summary": _summary(r), "from_image": bool(ds.get("from_image"))}


def _summary(r):
    gm = r["grid"].get("geometry", {})
    dsp = r.get("despiked") or {}
    bits = [f"Earth's field used: inclination {r['inc']:.1f}°, declination {r['dec']:.1f}° "
            f"\u2192 {r['reduction']}",
            f"grid cell {r['grid']['cell']:.1f} m",
            f"noise smoothing {r['smooth_m']:.0f} m (derivatives)"]
    if gm.get("is_line_survey"):
        bits.append(f"{gm['n_lines']} survey lines {gm['line_spacing']:.1f} m apart, "
                    f"stations {gm['station']:.1f} m")
    if dsp.get("dropped"):
        bits.append(f"{dsp['dropped']:,} of {dsp['total']:,} readings removed as spikes "
                    f"(> {dsp['tol']:.0f}\u03c3, \u03c3 = {dsp['sigma']:,.0f} nT)")
    if r.get("microlevelled"):
        bits.append(f"microlevelled, correction {r['microlevelled']:,.0f} nT rms")
    return " \u00b7 ".join(bits) + "."


def _field_direction(items):
    """The IGRF inclination and declination at the survey, which RTP/RTE need."""
    if not items or not items[0]["info"]:
        return None
    try:
        spec = uploads.build_spec(SLOT, items[0]["path"], base.effective(items), survey=SURVEY,
                                  prefix="f0_")
        lon, lat = datasets._centre_lonlat(spec)
        when = request.values.get("survey_date") or date.today().isoformat()
        d = datetime.fromisoformat(when)
        inc, dec, total, src = magnetics.field_direction(lon, lat, d)
        return {"inc": round(float(inc), 1), "dec": round(float(dec), 1),
                "total": None if total != total else float(total),
                "source": src, "lat": float(lat), "lon": float(lon), "date": when}
    except Exception:
        return {"inc": -28.0, "dec": 0.5, "total": None, "source": None, "lat": None, "lon": None,
                "date": request.values.get("survey_date") or date.today().isoformat()}


def _settings_fields():
    f = uploads.field
    return {
        "reduction": [f("reduction", "Reduction", "radio", magnetics.REDUCTIONS, magnetics.REDUCTIONS[0],
                        "RTP is unstable close to the magnetic equator; Auto switches to RTE when the "
                        "inclination is within 25° of horizontal.", width="full")],
        "grid": [
            f("auto_cell", "Choose the grid cell size for me", "check", value=True,
              help="A quarter of the survey line spacing (the usual 1/3–1/5 rule). Smaller cells than "
                   "that cannot add real detail across the lines; they only add gaps and stripes.",
              width="full"),
            f("cell", "Grid cell size (m)", "number", value=25.0, min=0.5, max=5000.0, step=0.5,
              needs_not="auto_cell"),
            f("grid_method", "Gridding", "select", geophysics.GRID_METHODS, geophysics.GRID_METHODS[0],
              "Minimum curvature draws the smoothest surface through the readings and fills between the "
              "survey lines. Linear interpolation is simpler and leaves the data pattern more visible."),
        ],
        "clean": [
            f("despike", "Remove spikes", "check", value=True,
              help="Drops readings that disagree with their neighbours by more than the tolerance below. "
                   "A magnetometer swung, dropped or carried past a fence records jumps of thousands of "
                   "nT over a metre, which no rock produces; the grid has to honour every reading, so one "
                   "spike becomes a bullseye and flattens the colours everywhere else.", width="full"),
            f("despike_tol", "Spike tolerance (robust σ)", "range", value=6.0, min=3.0, max=20.0,
              step=0.5, needs="despike",
              help="Lower removes more. The count is reported above the maps."),
            f("microlevel", "Microlevelling (decorrugation)", "check", value=False,
              help="Removes stripes running along the survey lines, left by diurnal drift and instrument "
                   "offsets between lines (Minty, 1991). Helps a survey walked over several days; on a "
                   "short, clean survey it can remove real structure that happens to trend along the "
                   "lines, so check both."),
        ],
        "depth": [
            f("si", "Euler structural index", "select", list(geophysics.STRUCTURAL_INDEX), "Pipes",
              "0 contacts and faults, 1 dykes and veins, 2 pipes, 3 compact bodies.", width="half"),
            f("euler_window", "Euler window (cells)", "range", value=10, min=5, max=25, step=1, width="half"),
            f("auto_smooth", "Choose the noise smoothing for me", "check", value=True,
              help="Upward-continues the field by about one data spacing before taking derivatives.",
              width="half"),
            f("smooth_m", "Smoothing height (m)", "number", value=20.0, min=0.0, max=2000.0,
              width="half", needs_not="auto_smooth"),
            f("auto_up", "Choose the upward-continuation height for me", "check", value=True, width="half"),
            f("up_height", "Continuation height (m)", "number", value=200.0, min=10.0, max=5000.0,
              width="half", needs_not="auto_up"),
        ],
    }


def _settings_from_form():
    form = request.form

    def num(name, default):
        try:
            return float(form.get(name, default))
        except (TypeError, ValueError):
            return float(default)

    manual = bool(form.get("manual_field"))
    inc = num("inc", -28.0) if manual else num("auto_inc", -28.0)
    dec = num("dec", 0.5) if manual else num("auto_dec", 0.5)
    si_label = form.get("si", "Pipes")
    return {
        "inc": inc, "dec": dec,
        "reduction": form.get("reduction", magnetics.REDUCTIONS[0]),
        "si": geophysics.STRUCTURAL_INDEX.get(si_label, 1.0),
        "si_label": si_label,
        "euler_window": int(num("euler_window", 10)),
        "smooth_m": None if form.get("auto_smooth") else num("smooth_m", 20.0),
        "up_height": None if form.get("auto_up") else num("up_height", 200.0),
        "cell": None if form.get("auto_cell") else num("cell", 25.0),
        "grid_method": form.get("grid_method", geophysics.GRID_METHODS[0]),
        "despike_tol": num("despike_tol", 6.0) if form.get("despike") else 0.0,
        "microlevel": bool(form.get("microlevel")),
        "normal_view": True,
        "survey_date": form.get("survey_date", ""),
    }


def _submit():
    try:
        specs = base.build_specs(KEY, SLOT, survey=SURVEY)
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("app.magnetic"))
    settings = _settings_from_form()

    def work(run, job):
        out = interpret.run(specs, settings, progress=job.say)
        if out["errors"] or not out["datasets"]["geophysics"]:
            raise ValueError("; ".join(out["errors"]) or "no magnetic data was found in that file")
        job.say(0.95, "Saving the results…")
        run.save_result(out)

    return base.submit(KEY, work, label=f"Interpreting {specs[0]['name']}", settings=settings)


# ------------------------------------------------------------------ the tabs
def panel(name):
    run = base.need_run(KEY, request.args.get("run"))
    want = _view()
    data, ds, r, use = _result(run, want)
    if r is None:
        return '<p class="note err">This dataset could not be processed as magnetics.</p>'
    ctx = {"run": run, "data": data, "ds": ds, "r": r, "use": use, "name": name,
           "fig_url": lambda fname: url_for("app.figure", section_key=KEY, run_id=run.id,
                                            name=fname, view=use),
           "dl": lambda kind, fname, **kw: url_for("app.download", section_key=KEY, run_id=run.id,
                                                   kind=kind, name=fname, view=use, **kw)}
    maps = r["maps"]
    red = r["reduction"]

    if name == "overview":
        return render_template("app/magnetic/_overview.html", **ctx)
    if name == "lineaments":
        return _panel_lineaments(ctx)
    if name == "upward":
        return _panel_upward(ctx)
    if name == "spi":
        return _panel_spi(ctx)
    if name == "euler":
        return _panel_euler(ctx)
    if name == "targets":
        return _panel_targets(ctx)

    key = {"tmi": "TMI", "reduction": red, "1vd": "1VD", "2vd": "2VD",
           "analytic-signal": "Analytic signal", "tilt": "Tilt derivative", "thdr": "THDR"}.get(name)
    if key is None or key not in maps:
        abort(404)
    cm, unit, info = INFO.get(key, ("turbo", "", ""))
    return render_template("app/magnetic/_map.html", title=key, unit=unit, info=info,
                           arr=maps[key], figname=figkey("map", key), **ctx)


def _panel_lineaments(ctx):
    run, r = ctx["run"], ctx["r"]
    pct = _clamp(request.args.get("pct"), 90, 80, 98)
    lcells = _clamp(request.args.get("len"), 15, 5, 40)
    lines = _lines(run, r, ctx["use"], pct, lcells)
    starts, lens = lineaments.rose_stats(lines)
    order = np.argsort(lens)[::-1] if len(lens) else []
    table = pd.DataFrame([{"Trend": lineaments.strike_label(starts[j] + 5),
                           "Strike": f"{starts[j]:.0f}–{starts[j] + 10:.0f}°",
                           "Length (km)": round(float(lens[j]), 2)}
                          for j in order[:4] if lens[j] > 0])
    return render_template("app/magnetic/_lineaments.html", pct=pct, lcells=lcells, lines=lines,
                           total_km=sum(l["length_m"] for l in lines) / 1000,
                           main=lineaments.strike_label(starts[order[0]] + 5) if len(lines) else "–",
                           rose_table=table_html(table), **ctx)


def _lines(run, r, use, pct, lcells):
    g = r["grid"]
    tilt = r["maps"]["Tilt derivative"]
    return run.cached(("maglin", use, pct, lcells), lambda: lineaments.detect(
        [tilt], g["valid"], g["cell"], 1.5, pct, lcells * g["cell"], 2 * g["cell"],
        merge_m=3 * g["cell"])[0])


def _panel_upward(ctx):
    r = ctx["r"]
    h = _clamp(request.args.get("h"), int(round(r["up_height"])), 10, 2000)
    arr = _upward_grid(ctx["run"], r, h)
    return render_template("app/magnetic/_map.html", title="Upward continuation", unit="nT",
                           info=f"The field as it would be measured {h:,} m higher. Use it to separate "
                                "deep (regional) from shallow sources; the difference from the TMI is "
                                "the residual field.",
                           arr=arr, figname=figkey("up", h), height=h, **ctx)


def _upward_grid(run, r, h):
    g = r["grid"]
    return run.cached(("up", r["reduction"], h),
                      lambda: np.where(g["valid"], upward(g["zf"], g["cell"], h), np.nan).astype(np.float32))


def _panel_spi(ctx):
    pts = ctx["r"]["spi_points"]
    q = pts.depth.quantile([0.25, 0.5, 0.75]) if len(pts) else None
    return render_template("app/magnetic/_spi.html", pts=pts, q=q, **ctx)


def _panel_euler(ctx):
    eu = ctx["r"]["euler"]
    q = eu.depth.quantile([0.1, 0.25, 0.5, 0.75]) if len(eu) else None
    plot = None
    if len(eu):
        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter3d(
            x=eu.x, y=eu.y, z=-eu.depth, mode="markers",
            marker=dict(size=3, color=eu.depth, colorscale="Plasma_r",
                        colorbar=dict(title="Depth (m)"), opacity=0.85),
            hovertemplate="E %{x:.0f}<br>N %{y:.0f}<br>depth %{marker.color:.0f} m<extra></extra>"))
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                          scene=dict(xaxis_title="Easting (m)", yaxis_title="Northing (m)",
                                     zaxis_title="Elevation (m)", aspectmode="manual",
                                     aspectratio=dict(x=1, y=1, z=0.4)))
        plot = plotly_div(fig, "euler3d", height=560)
    return render_template("app/magnetic/_euler.html", eu=eu, q=q, plot=plot,
                           si=ctx["data"].get("settings", {}).get("si", 1.0), **ctx)


def _panel_targets(ctx):
    run, ds, r, use = ctx["run"], ctx["ds"], ctx["r"], ctx["use"]
    weights = {}
    for i, (name, d) in enumerate(TARGET_LAYERS.items()):
        weights[name] = _clampf(request.args.get(f"w{i}"), d[1], 0.0, 1.0)
    top = _clampf(request.args.get("top"), 5.0, 1.0, 20.0)
    min_ha = _clampf(request.args.get("minha"), 1.0, 0.0, 20.0)

    score, zones, table = _targets(run, ds, r, use, weights, top, min_ha)
    if score is None:
        return render_template("app/magnetic/_targets.html", weights=weights, top=top, min_ha=min_ha,
                               layers=TARGET_LAYERS, table=None, empty=True, **ctx)
    return render_template("app/magnetic/_targets.html", weights=weights, top=top, min_ha=min_ha,
                           layers=TARGET_LAYERS, table=table, empty=False,
                           figname=figkey("targets", *[f"{k}{v}" for k, v in weights.items()],
                                          f"top{top}", f"ha{min_ha}"), **ctx)


def _members(run, r, use):
    g = r["grid"]
    valid = g["valid"]
    lines = r["lineaments"]
    red = r["maps"]["RTP"] if "RTP" in r["maps"] else -r["maps"]["RTE"]   # magnetic bodies positive

    def build():
        m = {"Analytic signal": prospectivity.fuzzy_linear(r["maps"]["Analytic signal"], valid),
             "Magnetic low (alteration)": prospectivity.fuzzy_linear(red, valid, invert=True)}
        if lines:
            m["Structure density"] = prospectivity.fuzzy_linear(
                lineaments.density(lines, valid.shape, g["cell"], 8 * g["cell"]), valid, 0, 98)
            m["Proximity to structures"] = prospectivity.fuzzy_distance(
                lineaments.distance(lines, valid.shape, g["cell"]), valid, max(3 * g["cell"], 100))
        return m
    return run.cached(("mag_members", use), build)


def _targets(run, ds, r, use, weights, top, min_ha):
    g = r["grid"]
    mem = _members(run, r, use)
    used = {k: v for k, v in weights.items() if v > 0 and k in mem}
    if not used:
        return None, None, None
    score, _ = interpret.knowledge_driven({k: mem[k] for k in used}, used)
    if score is None:
        return None, None, None
    min_px = max(1, int(round(min_ha * 1e4 / g["cell"] ** 2)))
    zones, t = interpret.describe_targets(score, np.isfinite(score), g["transform"], top, min_px,
                                          {k: mem[k] for k in used}, {"geophysics": [r]},
                                          {"geophysics": [ds]})
    if len(t):
        from pyproj import Transformer
        epsg = run.result()["epsg"]
        lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(
            np.asarray(t["Easting"], float), np.asarray(t["Northing"], float))
        t.insert(1, "Latitude", np.round(lat, 6))
        t.insert(2, "Longitude", np.round(lon, 6))
    return score, zones, t


def _clamp(value, default, lo, hi):
    try:
        return max(lo, min(int(float(value)), hi))
    except (TypeError, ValueError):
        return default


def _clampf(value, default, lo, hi):
    try:
        return max(lo, min(float(value), hi))
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ pictures
def redraw(run, name, args):
    """Draw a picture that is not on disk yet. The name says which map and which settings."""
    want = _view(args)
    data, ds, r, use = _result(run, want)
    if r is None:
        return None
    st = _style(use)
    g = r["grid"]
    path = run.fig_path(name)
    parts = name.split("-")
    fig = None

    if name.startswith("map-"):
        key = _map_key(name, r)
        if key is None:
            return None
        cm, unit, _ = INFO.get(key, ("turbo", ""))
        pts = None
        if key == "TMI" and ds["kind"] == "points" and len(ds["x"]) < 20000:
            pts = (ds["x"], ds["y"])
        fig = plots.grid_figure(r["maps"][key], g["transform"], f"{key} · {r['_name']}",
                                st["palette"](cm), unit, overlay_pts=pts,
                                stretch=st["stretch"], smooth=st["smooth"])
        if key == "Tilt derivative":
            _zero_contour(fig, r["maps"][key], g)
    elif name.startswith("up-"):
        h = _clamp(parts[-1], int(round(r["up_height"])), 10, 2000)
        fig = plots.grid_figure(_upward_grid(run, r, h), g["transform"],
                                f"Upward continuation {h:,} m · {r['_name']}", st["palette"]("turbo"),
                                "nT", stretch=st["stretch"], smooth=st["smooth"])
    elif name == "overview":
        maps = r["maps"]
        show = [k for k in maps if k != "SPI depth"]
        cms = [st["palette"](INFO.get(k, ("turbo",))[0] if not k.startswith("Upward") else "turbo")
               for k in show]
        fig = plots.panels([maps[k] for k in show], show, g["transform"], cms, ncols=4,
                           suptitle=f"Magnetic maps · {r['_name']}", stretch=st["stretch"],
                           smooth=st["smooth"])
    elif name.startswith("lin-"):
        pct, lcells = _clamp(parts[1], 90, 80, 98), _clamp(parts[2], 15, 5, 40)
        lines = _lines(run, r, use, pct, lcells)
        fig = export.lines_figure(r["maps"]["Tilt derivative"], lines, g["transform"], None,
                                  f"Magnetic lineaments on tilt derivative · {r['_name']}", "RdBu_r")
    elif name.startswith("rose-"):
        pct, lcells = _clamp(parts[1], 90, 80, 98), _clamp(parts[2], 15, 5, 40)
        starts, lens = lineaments.rose_stats(_lines(run, r, use, pct, lcells))
        fig = export.rose_figure(starts, lens, title="Strike of magnetic lineaments")
    elif name == "spi-map":
        fig = plots.euler_figure(r["maps"]["Tilt derivative"], g["transform"], r["spi_points"],
                                 f"SPI depth to contacts · {r['_name']}")
    elif name == "spi-hist":
        fig = plots.hist_figure(r["spi_points"].depth, None, "SPI depths", "Depth (m)", log=False)
    elif name == "euler-map":
        bg = r["maps"].get("RTP", r["maps"].get("RTE"))
        fig = plots.euler_figure(bg, g["transform"], r["euler"],
                                 f"Euler solutions (plan) · {r['_name']}")
    elif name == "euler-hist":
        fig = plots.hist_figure(r["euler"].depth, None, "Euler depths", "Depth (m)", log=False)
    elif name.startswith("targets-"):
        score, zones, _t = _targets_from_name(run, ds, r, use, name)
        if score is None:
            return None
        fig = plots.grid_figure(score, g["transform"],
                                f"Magnetic target map (pink = targets) · {r['_name']}", "inferno",
                                "score", targets=zones["targets"], lo=1, hi=99)
    if fig is None:
        return None
    from ..render import save_fig
    return save_fig(fig, path)


def _map_key(name, r):
    want = name[len("map-"):]
    for key in list(r["maps"]) + [r["reduction"]]:
        if safe_name(key) == want:
            return key if key in r["maps"] else None
    return None


def _zero_contour(fig, arr, g):
    ax = fig.axes[0]
    h, w = arr.shape
    t = g["transform"]
    xs = t.c + t.a * (np.arange(w) + 0.5)
    ys = t.f + t.e * (np.arange(h) + 0.5)
    ax.contour(xs, ys, np.where(np.isfinite(arr), arr, np.nan), levels=[0], colors="k", linewidths=0.8)


def _targets_from_name(run, ds, r, use, name):
    """Recover the weights a target picture was asked for from its own name."""
    args = request.args
    weights = {}
    for i, (layer, d) in enumerate(TARGET_LAYERS.items()):
        weights[layer] = _clampf(args.get(f"w{i}"), d[1], 0.0, 1.0)
    top = _clampf(args.get("top"), 5.0, 1.0, 20.0)
    min_ha = _clampf(args.get("minha"), 1.0, 0.0, 20.0)
    return _targets(run, ds, r, use, weights, top, min_ha)


# ------------------------------------------------------------------ downloads
def deliver(run, kind, name, args):
    want = _view(args)
    data, ds, r, use = _result(run, want)
    if r is None:
        abort(404)
    g = r["grid"]
    epsg = data["epsg"]

    if kind == "png":
        path = run.fig_path(name)
        if not path.is_file():
            path = redraw(run, name, args)
        return base.send_png(path, f"{name}.png")

    if kind == "tif":
        key = _map_key("map-" + name, r) if not name.startswith("map-") else _map_key(name, r)
        if key is None and name == "spi":
            key = "SPI depth"
        if key is None:
            abort(404)
        return base.send_bytes(export.geotiff_bytes({safe_name(key): r["maps"][key]}, g["transform"], epsg),
                               f"{safe_name(key)}.tif", "image/tiff")

    if kind == "csv":
        if name == "spi":
            return base.send_csv(r["spi_points"].round(1), "spi_points.csv")
        if name == "euler":
            return base.send_csv(r["euler"].round(1), "euler_solutions.csv")
        if name == "targets":
            _s, _z, t = _targets_from_name(run, ds, r, use, name)
            if t is None:
                abort(404)
            return base.send_csv(t, "magnetic_targets.csv")
        if name == "lineaments":
            pct = _clamp(args.get("pct"), 90, 80, 98)
            lcells = _clamp(args.get("len"), 15, 5, 40)
            starts, lens = lineaments.rose_stats(_lines(run, r, use, pct, lcells))
            return base.send_csv(pd.DataFrame({"strike_from": starts, "strike_to": starts + 10,
                                               "length_km": np.round(lens, 3)}), "lineament_strikes.csv")
        abort(404)

    if kind == "geojson":
        pct = _clamp(args.get("pct"), 90, 80, 98)
        lcells = _clamp(args.get("len"), 15, 5, 40)
        lines = _lines(run, r, use, pct, lcells)
        return base.send_bytes(lineaments.to_geojson(lines, g["transform"], epsg).encode("utf-8"),
                               "magnetic_lineaments.geojson", "application/geo+json")

    if kind == "zip":
        return base.send_bytes(_all_maps_zip(run, r, use, epsg), "magnetic_maps.zip", "application/zip")

    if kind == "targets-tif":
        score, zones, _t = _targets_from_name(run, ds, r, use, name)
        if score is None:
            abort(404)
        return base.send_bytes(
            export.geotiff_bytes({"magnetic_target_score": score,
                                  "target": zones["targets"].astype(np.float32)}, g["transform"], epsg),
            "magnetic_targets.tif", "image/tiff")
    abort(404)


def _all_maps_zip(run, r, use, epsg):
    st = _style(use)
    from ..render import fig_to_png_bytes
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, arr in r["maps"].items():
            base_name = name.split(" ")[0] if name.startswith("Upward") else name
            cm, unit, _ = INFO.get(base_name, ("turbo", "", ""))
            fig = plots.grid_figure(arr, r["grid"]["transform"], f"{name} · {r['_name']}",
                                    st["palette"](cm), unit, stretch=st["stretch"], smooth=st["smooth"])
            z.writestr(f"{safe_name(name)}.png", fig_to_png_bytes(fig, dpi=150))
        z.writestr("magnetic_maps.tif",
                   export.geotiff_bytes({safe_name(k): v for k, v in r["maps"].items()},
                                        r["grid"]["transform"], epsg))
    return buf.getvalue()
