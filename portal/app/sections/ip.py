"""3 · IP (induced polarisation): one file per survey line, one pair of inverse model sections each.

Handles plain tables (station, n-level or depth, value) and instrument exports carrying electrode
positions (IRIS Prosys / ELREC Pro: Tx1, Tx2, Rx1, Rx2, Rho, M…) including their information lines,
remote electrodes and data-quality columns, as well as Res2DInv .dat files.

Where electrode positions exist the section is a real 2-D inversion; where they do not, the measured
pseudosection is drawn instead, because pseudo-depth is all the geometry there is.
"""
import io
import zipfile

import numpy as np
import pandas as pd
from flask import abort, flash, redirect, render_template, request, url_for
from flask_login import current_user

from geoai import coords, geophysics, interpret, ipdata, ipinv, plots, readers
from geoai.coords import parse_column

from .. import store, uploads
from ..render import safe_name, table_html
from . import base

KEY = "ip"
NONE = uploads.NONE
MIN_READINGS = 6
ARRAYS = ["Pole-dipole", "Dipole-dipole", "Wenner", "Schlumberger", "Pole-pole"]
SIMPLE_ARRAYS = ["Dipole-dipole", "Pole-dipole", "Wenner / Schlumberger"]
ORDER = ["resistivity", "chargeability"]


def attach(bp):
    from .. import section
    bp.add_url_rule("/ip", "ip", section(KEY)(page), methods=["GET", "POST"])
    bp.add_url_rule("/ip/sections", "ip_sections", section(KEY)(sections_panel))


# ------------------------------------------------------------------ reading the files
def _first(items):
    """The first readable line file. Its columns decide the layout for all of them."""
    for it in items:
        if it["info"] and it["info"]["kind"] == "table":
            return it
    return None


def _guess(cols, words, exclude=()):
    for w in words:
        for c in cols:
            name = str(c).strip().lower()
            if c in exclude:
                continue
            if (name == w) if len(w) <= 2 else (w in name):
                return c
    return None


def _num(text):
    text = str(text or "").strip().replace(",", "")
    try:
        return float(text) if text else None
    except ValueError:
        return None


def _layout(items):
    """What the person still has to choose, worked out from the first file."""
    first = _first(items)
    if first is None:
        return None
    df = first["info"]["result"]["df"]
    meta = first["info"]["result"].get("meta") or {}
    ip_unit = (meta.get("ip_unit") or "mV/V").rstrip(".")
    cols = [str(c) for c in df.columns]
    electrodes = ipdata.has_electrodes(df)
    f = uploads.field
    out = {"name": first["name"], "columns": cols, "preview": first["info"]["preview"],
           "electrodes": electrodes, "ip_unit": ip_unit, "meta": meta, "fields": [], "res2d": None,
           "detected": None, "ec": None, "value_labels": {}}

    if electrodes:
        ec = ipdata.electrode_columns(df)
        _g, arr0, a0 = ipdata.geometry(df)
        out["detected"] = {"cols": [str(ec[k]) for k in ("c1", "c2", "p1", "p2")],
                           "array": arr0, "a": a0}
        if meta.get("format") == "Res2DInv":
            out["res2d"] = meta
        others = [c for c in readers.numeric_columns(df) if c not in (ec["m"], ec["rho"])]
        values = [c for c in (ec["m"], ec["rho"]) if c is not None] + others
        labels = {}
        if ec["m"] is not None:
            labels[str(ec["m"])] = f"Chargeability {ec['m']} ({ip_unit})"
        if ec["rho"] is not None:
            labels[str(ec["rho"])] = f"Apparent resistivity {ec['rho']} (ohm·m)"
        out["value_labels"] = {str(c): labels.get(str(c), str(c)) for c in values}
        out["fields"] = [
            f("vcol", "Value to show", "select", [str(c) for c in values],
              str(values[0]) if values else None),
            f("array", "Array", "select", ARRAYS, arr0 if arr0 in ARRAYS else ARRAYS[0],
              "Detected from the electrode positions; used for the pseudo-depths."),
        ]
        out["ec"] = {k: (str(v) if v is not None else None) for k, v in ec.items()}
    else:
        xguess = _guess(cols, ["station", "east", "dist", "x"]) or cols[0]
        vguess = (_guess(cols, ["charg", "mv/v", "msec", "ip", "m"], exclude=(xguess,))
                  or _guess(cols, ["resist", "ohm", "rho"], exclude=(xguess,)))
        has_n = any(str(c).lower() in ("n", "n_level", "nlevel", "level") for c in cols)
        rest = [c for c in cols if c != xguess]
        out["fields"] = [
            f("xcol", "Station / distance column", "select", cols, xguess, width="third"),
            f("vcol", "Value to show", "select", rest, vguess or (rest[0] if rest else None),
              width="third"),
            f("ycol", "Northing column in the file (optional)", "select", [NONE] + cols,
              _guess(cols, ["north", "y"]) or NONE, width="third"),
            f("dmode", "Depth given as", "radio", ["n-level", "Depth column"],
              "n-level" if has_n else "Depth column", width="full"),
            f("ncol", "n-level column", "select", cols,
              next((c for c in cols if str(c).lower() in ("n", "n_level", "nlevel", "level")), cols[0]),
              width="third", only="n-level"),
            f("array", "Array", "select", SIMPLE_ARRAYS, SIMPLE_ARRAYS[0], width="third",
              only="n-level"),
            f("a_spacing", "Dipole spacing a (m)", "number", value=50.0, min=1.0, max=1000.0,
              width="third", only="n-level"),
            f("dcol", "Depth column (m)", "select", cols, _guess(cols, ["depth", "z"]) or cols[0],
              width="third", only="depth"),
        ]
    out["quality"] = [
        f("q_neg", "Remove negative / zero values", "check", value=True,
          help="Negative chargeability usually means coupling or noise.", width="half"),
        f("q_contact", "Remove bad electrode contacts", "check", value=electrodes,
          help="Contact values such as INFINI or ------.", width="half"),
        f("q_err", "Largest chargeability error to keep (ErrM)", "number", value=10.0, min=0.0,
          max=1000.0, help="Readings whose error is larger than this are removed.", width="half"),
        f("q_n1", "Exclude n = 1", "check", value=False,
          help="The shallowest level, next to the current electrode, is often spoiled by electrode "
               "effects and poor contact.", width="half"),
    ]
    return out


def _defaults(layout):
    """Every control's default value, so the preview can be built before anything is submitted."""
    d = {}
    for f in (layout["fields"] + layout["quality"]):
        d[f["name"]] = f["value"]
    return d


def _values(layout):
    """The defaults, overridden by whatever the person has actually chosen."""
    from werkzeug.datastructures import MultiDict
    md = MultiDict()
    if layout:
        for name, value in _defaults(layout).items():
            if isinstance(value, bool):
                if value:
                    md.add(name, "1")
            elif value is not None:
                md.add(name, value)
    if request.values.get("_seen"):          # a submitted form says exactly which boxes are ticked
        md = MultiDict()
    for key in request.values:
        md.setlist(key, request.values.getlist(key))
    return md


def _line_rows(items, values):
    """Per-file name and position, so a line whose stations are distances can be placed on a map."""
    rows = []
    for it in items:
        i = it["index"]
        rows.append({
            "index": i, "file": it["name"], "error": it["error"],
            "name": values.get(f"line{i}_name") or it["name"].rsplit(".", 1)[0],
            "e0": values.get(f"line{i}_e0", ""),
            "north": values.get(f"line{i}_north", ""),
            "bearing": values.get(f"line{i}_bearing", "90"),
        })
    return rows


# ------------------------------------------------------------------ the page
def page():
    if request.method == "POST":
        action = request.form.get("do")
        if action == "upload":
            base.take_upload(KEY, multiple=True)
            return redirect(url_for("app.ip") + "#files")
        if action == "remove":
            store.drop_draft(current_user.id, KEY, request.form.get("name", ""))
            return redirect(url_for("app.ip"))
        if action == "clear":
            store.clear_draft(current_user.id, KEY)
            return redirect(url_for("app.ip"))
        if action == "submit":
            return _submit()

    items = base.drafts(KEY)
    layout = _layout(items)
    values = _values(layout) if layout else request.values
    lines = _line_rows(items, values)
    summary = _preview(items, layout, lines, values) if layout else None
    crs_needed = bool(layout) and (any(r["north"] for r in lines)
                                   or (not layout["electrodes"]
                                       and values.get("ycol", NONE) not in (NONE, "", None)))
    return base.page(KEY, "app/ip.html", items=items, layout=layout, lines=lines, summary=summary,
                     values=values, crs_needed=crs_needed,
                     crs_options=[k for k in coords.CRS_OPTIONS if "4326" not in k])


def _preview(items, layout, lines, values):
    """The per-file table shown before submitting: how many readings survive the filters."""
    try:
        specs, rows = _build(items, layout, lines, values)
    except ValueError as e:
        return {"error": str(e), "rows": None, "ready": 0, "total": len(items)}
    return {"error": None, "rows": table_html(pd.DataFrame(rows)), "ready": len(specs),
            "total": len(items)}


# ------------------------------------------------------------------ turning files into specs
def _build(items, layout, lines, values):
    """Every line file prepared for the engine, plus a row per file saying what happened to it.

    One function serves both the preview table and the submitted job, so the two can never disagree
    about which readings were kept.
    """
    if layout is None:
        raise ValueError("Upload at least one line file.")

    electrodes = layout["electrodes"]
    vcol = values.get("vcol")
    drop_neg = bool(values.get("q_neg"))
    drop_contact = bool(values.get("q_contact"))
    drop_n1 = bool(values.get("q_n1"))
    try:
        max_err = float(values.get("q_err") or 10.0)
    except (TypeError, ValueError):
        max_err = 10.0

    ycol = None
    if electrodes:
        is_charge = vcol == layout["ec"]["m"]
        array = values.get("array") or layout["detected"]["array"]
        xcol = depth_cfg = None
    else:
        xcol = values.get("xcol")
        ycol = values.get("ycol")
        ycol = None if ycol in (NONE, "", None) else ycol
        dmode = values.get("dmode", "n-level")
        depth_cfg = ({"depth_mode": "n-level", "ncol": values.get("ncol"),
                      "array": values.get("array", SIMPLE_ARRAYS[0]),
                      "a_spacing": float(values.get("a_spacing") or 50.0)}
                     if dmode == "n-level" else
                     {"depth_mode": "Depth column", "depthcol": values.get("dcol")})
        is_charge = not any(w in str(vcol).lower() for w in ["res", "ohm", "rho"])
    if not vcol:
        raise ValueError("Choose the column holding the readings.")

    survey = geophysics.SURVEYS[1] if is_charge else geophysics.SURVEYS[2]
    epsg = None
    if any(r["north"] for r in lines) or (not electrodes and ycol):
        choice = values.get("crs") or "WGS84 / UTM zone 36S (EPSG:32736)"
        epsg = coords.CRS_OPTIONS.get(choice)

    specs, rows = [], []
    for it, row in zip(items, lines):
        if it["error"]:
            rows.append({"Line": row["name"], "File": it["name"], "Readings": 0, "Kept": 0,
                         "Removed": "—", "Status": f"Not usable: {it['error']}"})
            continue
        try:
            df = it["info"]["result"]["df"].copy()
            elec = None
            if electrodes:
                if not ipdata.has_electrodes(df):
                    raise ValueError("no electrode columns (Tx1, Tx2, Rx1, Rx2) like the first file")
                ec = ipdata.electrode_columns(df)
                g, _arr, _a = ipdata.geometry(df, array)
                df["_station_x"] = g.station_x.to_numpy()
                df["_pseudo_depth"] = g.pseudo_depth.to_numpy()
                keep, why = ipdata.quality_mask(df, vcol, max_err if is_charge else None,
                                                drop_neg, drop_contact)
                if drop_n1:
                    bad = keep & (g.n_level.to_numpy() == 1)
                    why["n = 1 excluded"] = int(bad.sum())
                    keep &= ~bad
                cfg = {"xcol": "_station_x", "depth_mode": "Depth column",
                       "depthcol": "_pseudo_depth", "ycol": None}
                pos = {k: parse_column(df[ec[k]]).to_numpy(float) for k in ("c1", "c2", "p1", "p2")}
                pos = {k: np.where(np.abs(v) >= ipdata.REMOTE, np.nan, v) for k, v in pos.items()}
                elec = {}
                for kind, col in (("chargeability", ec["m"]), ("resistivity", ec["rho"])):
                    if col is None:
                        continue
                    ok, _ = ipdata.quality_mask(df, col, max_err if kind == "chargeability" else None,
                                                drop_neg or kind == "resistivity", drop_contact)
                    if drop_n1:
                        ok &= g.n_level.to_numpy() != 1
                    elec[kind] = {**{k: v[ok] for k, v in pos.items()},
                                  "v": parse_column(df[col]).to_numpy(float)[ok],
                                  "px": g.station_x.to_numpy()[ok],
                                  "pz": g.pseudo_depth.to_numpy()[ok]}
            else:
                missing = [c for c in [xcol, vcol, depth_cfg.get("ncol"), depth_cfg.get("depthcol"),
                                       ycol] if c and c not in df.columns]
                if missing:
                    raise ValueError(f"missing column(s) {', '.join(map(str, missing))}")
                v = parse_column(df[vcol]).to_numpy(float)
                keep = np.isfinite(v) & np.isfinite(parse_column(df[xcol]).to_numpy(float))
                why = {"missing value": int((~keep).sum())}
                if drop_neg:
                    bad = keep & (v <= 0)
                    why["negative or zero"] = int(bad.sum())
                    keep &= ~bad
                if drop_n1 and depth_cfg.get("ncol"):
                    bad = keep & (parse_column(df[depth_cfg["ncol"]]).to_numpy(float) == 1)
                    why["n = 1 excluded"] = int(bad.sum())
                    keep &= ~bad
                cfg = {"xcol": xcol, "ycol": ycol, **depth_cfg}

            total = len(keep)
            df = df[keep].reset_index(drop=True)
            stations = df[cfg["xcol"]].nunique() if len(df) else 0
            ok = len(df) >= MIN_READINGS and stations >= 2
            rows.append({"Line": row["name"], "File": it["name"], "Readings": int(total),
                         "Kept": int(len(df)),
                         "Removed": ", ".join(f"{k}: {v}" for k, v in why.items() if v) or "—",
                         "Status": "OK" if ok else
                                   f"Too few valid readings (at least {MIN_READINGS} over 2 stations)"})
            if not ok:
                continue
            e0, n0, brg = _num(row["e0"]), _num(row["north"]), _num(row["bearing"])
            spec = {"slot": "geophysics", "name": f"{row['name']} · {it['name']}", "kind": "table",
                    "data": {"kind": "table", "df": df, "electrodes": elec},
                    "survey": survey, "geometry": "section", "vcol": vcol,
                    "linecol": None, "line_name": row["name"], "epsg": epsg, **cfg,
                    "line_northing": None, "line_origin": None}
            if not cfg.get("ycol") and n0 is not None:
                if e0 is not None:
                    spec["line_origin"] = (e0, n0, 90.0 if brg is None else brg)
                else:
                    spec["line_northing"] = n0
            specs.append(spec)
        except ValueError as e:
            rows.append({"Line": row["name"], "File": it["name"], "Readings": 0, "Kept": 0,
                         "Removed": "—", "Status": f"Not usable: {e}"})
    return specs, rows


def _submit():
    items = base.drafts(KEY)
    layout = _layout(items)
    values = _values(layout) if layout else request.form
    lines = _line_rows(items, values)
    try:
        specs, _rows = _build(items, layout, lines, values)
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("app.ip"))
    if not specs:
        flash("No line file has enough valid readings. Check the columns and the quality filters.",
              "error")
        return redirect(url_for("app.ip"))

    ip_unit = layout["ip_unit"]
    settings = {"ip_value": str(values.get("vcol")), "ip_unit": ip_unit,
                "lines": [r["name"] for r in lines]}

    def work(run, job):
        out = interpret.run(specs, settings, progress=lambda p, t: job.say(p * 0.85, t))
        inversions = {}
        todo = [s for s in specs if s["data"].get("electrodes")]
        for k, s in enumerate(todo):
            job.say(0.85 + 0.13 * k / max(len(todo), 1), f"Inverting {s['line_name']}…")
            for kind, e in s["data"]["electrodes"].items():
                try:
                    inversions.setdefault(kind, {})[s["name"]] = ipinv.invert(
                        e["c1"], e["c2"], e["p1"], e["p2"], e["v"], e["px"], e["pz"],
                        log=kind == "resistivity")
                except Exception as err:               # one bad line must not cost the others
                    out["errors"].append(f"{s['line_name']}: could not invert {kind} ({err}).")
        out["inversions"] = inversions
        out["ip_unit"] = ip_unit
        job.say(0.98, "Saving the results…")
        run.save_result(out)

    return base.submit(KEY, work, label=f"Inverting {len(specs)} line(s)", settings=settings)


# ------------------------------------------------------------------ results
def _pct(args=None):
    a = request.args if args is None else args
    try:
        return max(50, min(int(float(a.get("pct", 95))), 99))
    except (TypeError, ValueError):
        return 95


def _stretch(args=None):
    a = request.args if args is None else args
    try:
        return max(80.0, min(float(a.get("stretch", 98.0)), 100.0))
    except (TypeError, ValueError):
        return 98.0


def _context(run, pct=None, stretch=None):
    data = run.result()
    inv = data.get("inversions") or {}
    unit = data.get("ip_unit", "mV/V")
    items = list(zip(data["datasets"]["geophysics"], data["results"]["geophysics"]))
    pct = _pct() if pct is None else pct
    stretch = _stretch() if stretch is None else stretch
    thr = thr_model = None
    if items:
        allv = np.concatenate([ds["v"] for ds, _ in items])
        finite = allv[np.isfinite(allv)]
        if finite.size:
            thr = float(np.nanpercentile(finite, pct))
    if inv.get("chargeability"):
        thr_model = float(np.percentile(
            np.concatenate([i["model"][i["mask"]] for i in inv["chargeability"].values()]), pct))
    return {"data": data, "inv": inv, "unit": unit, "items": items, "pct": pct, "stretch": stretch,
            "thr": thr, "thr_model": thr_model,
            "levels": {k: ipinv.levels_for(list(inv[k].values())) for k in ORDER if inv.get(k)}}


def _describe(kind, inv, unit, thr_model):
    if kind == "resistivity":
        (lo, lx, lz), (hi, hx, hz) = ipinv.extremes(inv)
        return (f"<strong>Resistivity</strong>: most conductive ground <strong>{lo:,.0f} ohm.m</strong> "
                f"at {lx:,.0f} m, {lz:.0f} m deep; most resistive <strong>{hi:,.0f} ohm.m</strong> at "
                f"{hx:,.0f} m, {hz:.0f} m deep · {inv['n']} readings · iteration {inv['iteration']}, "
                f"absolute error {inv['abs_error']:.2g} %.")
    an = ipinv.anomalies(inv, thr_model)
    if an and an["zone"]:
        x0, x1, z0, z1 = an["zone"]
        return (f"<strong>Chargeability</strong>: strongest zone from <strong>{x0:,.0f} to "
                f"{x1:,.0f} m</strong>, <strong>{z0:.0f}–{z1:.0f} m</strong> deep"
                + (" and still open at the bottom of the model" if an["open_at_depth"] else "")
                + f", peak {an['peak']:.1f} {unit} at {an['peak_x']:,.0f} m, {an['peak_depth']:.0f} m "
                  f"deep (black outline at or above {thr_model:.1f} {unit}) · {inv['n']} readings · "
                  f"iteration {inv['iteration']}, absolute error {inv['abs_error']:.2g} {unit}.")
    return (f"<strong>Chargeability</strong>: nothing above the outline value "
            f"({thr_model:.1f} {unit}) · {inv['n']} readings · absolute error "
            f"{inv['abs_error']:.2g} {unit}.")


def sections_panel():
    run = base.need_run(KEY, request.args.get("run"))
    return render_template("app/ip/_sections.html", run=run, **_blocks(run))


def _blocks(run, pct=None, stretch=None):
    """One block per survey line: either the two inverse model sections, or the pseudosection."""
    c = _context(run, pct, stretch)
    blocks = []
    for ds, r in c["items"]:
        for lid, L in r["lines"].items():
            found = [(k, c["inv"][k][ds["name"]]) for k in ORDER if ds["name"] in c["inv"].get(k, {})]
            if found:
                blocks.append({
                    "line": lid, "kind": "inverse",
                    "images": [{"kind": k, "fig": f"inv-{k}-{safe_name(lid)}-{c['pct']}",
                                "text": _describe(k, i, c["unit"], c["thr_model"])}
                               for k, i in found]})
            else:
                d = L["data"].dropna()
                unit = c["unit"] if "charge" in ds["survey"] else ds["survey"].split("(")[-1].strip(")")
                an = geophysics.section_anomalies(d.x, d.depth, d.v, c["thr"])
                if len(an):
                    a = an.iloc[0]
                    text = (f"<strong>Line {lid}</strong>: strongest anomaly from "
                            f"<strong>{a.from_x:,.0f} to {a.to_x:,.0f} m</strong>, about "
                            f"<strong>{a.depth_top:.0f}–{a.depth_bottom:.0f} m</strong> deep, peak "
                            f"{a.peak:.1f} {unit} at {a.peak_x:,.0f} m"
                            + (f" · {len(an) - 1} smaller anomaly(ies)" if len(an) > 1 else "")
                            + f" · {len(d)} readings.")
                else:
                    text = (f"<strong>Line {lid}</strong>: no continuous anomaly above the outline "
                            f"threshold · {len(d)} readings.")
                blocks.append({"line": lid, "kind": "pseudo",
                               "images": [{"kind": "pseudosection",
                                           "fig": f"pseudo-{safe_name(lid)}-{c['pct']}-{c['stretch']}",
                                           "text": text}]})
    return {**c, "blocks": blocks, "errors": c["data"].get("errors") or [],
            "has_inversions": bool(c["inv"])}


# ------------------------------------------------------------------ pictures
def _figure_for(run, name):
    """Work out which line and quantity a picture name refers to, and rebuild its figure."""
    parts = name.split("-")
    if parts[0] == "inv" and len(parts) >= 4:
        kind = parts[1]
        try:
            pct = int(parts[-1])
        except ValueError:
            return None
        lid_s = "-".join(parts[2:-1])
        c = _context(run, pct=pct)
        for ds, r in c["items"]:
            for lid in r["lines"]:
                if safe_name(lid) == lid_s and ds["name"] in c["inv"].get(kind, {}):
                    return ipinv.figure(c["inv"][kind][ds["name"]], f"Line {lid}",
                                        "ohm.m" if kind == "resistivity" else c["unit"],
                                        c["levels"][kind],
                                        None if kind == "resistivity" else c["thr_model"])
        return None

    if parts[0] == "pseudo" and len(parts) >= 4:
        try:
            stretch, pct = float(parts[-1]), int(parts[-2])
        except ValueError:
            return None
        lid_s = "-".join(parts[1:-2])
        c = _context(run, pct=pct, stretch=stretch)
        for ds, r in c["items"]:
            for lid, L in r["lines"].items():
                if safe_name(lid) != lid_s:
                    continue
                d = L["data"].dropna()
                param = "Chargeability" if "charge" in ds["survey"] else "Resistivity"
                unit = c["unit"] if param == "Chargeability" else ds["survey"].split("(")[-1].strip(")")
                cmap = "jet" if param == "Chargeability" else "viridis"
                title = (f"{param} pseudosection · line {lid}  "
                         f"(black outline at or above {c['thr']:.1f} {unit})")
                try:
                    xi, zi, G = geophysics.section_grid(d.x, d.depth, d.v)
                    return plots.section_figure(xi, zi, G, (d.x, d.depth), c["thr"], title, cmap,
                                                f"{param} ({unit})", stretch)
                except Exception:   # too few or collinear readings to interpolate: draw the readings
                    return plots.section_points_figure(d.x, d.depth, d.v, c["thr"], title, cmap,
                                                       f"{param} ({unit})", stretch)
    return None


def redraw(run, name, args):
    fig = _figure_for(run, name)
    if fig is None:
        return None
    from ..render import save_fig
    return save_fig(fig, run.fig_path(name))


def deliver(run, kind, name, args):
    if kind == "png":
        path = run.fig_path(name)
        if not path.is_file():
            path = redraw(run, name, args)
        return base.send_png(path, f"ip_{name}.png")
    if kind == "zip":
        from ..render import fig_to_png_bytes
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for block in _blocks(run, _pct(args), _stretch(args))["blocks"]:
                for img in block["images"]:
                    fig = _figure_for(run, img["fig"])
                    if fig is not None:
                        z.writestr(f"ip_{img['kind']}_{safe_name(block['line'])}.png",
                                   fig_to_png_bytes(fig, dpi=150))
        return base.send_bytes(buf.getvalue(), "ip_sections.zip", "application/zip")
    abort(404)
