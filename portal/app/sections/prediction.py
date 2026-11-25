"""6 · Integrated interpretation and gold prediction.

Brings together what every other section has already produced and predicts where gold is most likely,
with an estimated depth and structural trend for each target.

Each section's datasets and results are read back from its stored run, so a prediction can combine
work done on different days — and combining costs a regrid, not a re-analysis, because the heavy
work was already done and saved by the section that did it.
"""
import base64
from datetime import datetime

import numpy as np
import pandas as pd
from flask import abort, flash, redirect, render_template, request, url_for
from flask_login import current_user

from geoai import datasets as dsm
from geoai import export, indices, interpret, lineaments, plots

from .. import store
from ..render import safe_name, table_html
from . import base

KEY = "prediction"
SOURCES = [("remote-sensing", "1 · Remote sensing", "app.remote_sensing"),
           ("magnetic", "2 · Magnetic", "app.magnetic"),
           ("ip", "3 · IP", "app.ip"),
           ("geology", "4 · Geology", "app.geology"),
           ("geochemistry", "5 · Geochemistry", "app.geochemistry")]
METHODS = ["Knowledge-driven (expert weights)", "Data-driven (Random Forest on gold samples)",
           "Average of both"]
TABS = [("Summary", "summary"), ("⭐ Gold prediction", "gold"), ("Remote sensing layers", "remote")]


def attach(bp):
    from .. import section
    bp.add_url_rule("/prediction", "prediction", section(KEY)(page), methods=["GET", "POST"])
    bp.add_url_rule("/prediction/panel/<name>", "prediction_panel", section(KEY)(panel))


# ------------------------------------------------------------------ what is available
def _available():
    out = []
    for key, label, endpoint in SOURCES:
        run = next((r for r in store.Run.list(current_user.id, key) if r.ok), None)
        meta = run.meta() if run else {}
        out.append({"key": key, "label": label, "endpoint": endpoint, "run": run,
                    "files": meta.get("files", []), "when": (run.id[:8] if run else None),
                    "open": key in current_user.allowed_sections()})
    return out


def _chosen(available):
    """Which sections to combine: everything with results, unless the person unticked some."""
    if request.values.get("_pick"):
        want = set(request.values.getlist("use"))
    else:
        want = {a["key"] for a in available if a["run"]}
    return [a for a in available if a["run"] and a["key"] in want]


# ------------------------------------------------------------------ combining
def _section1_layers(run):
    """Clay / hydroxyl and iron oxide from a stored Section 1 scene, as an evidence raster."""
    try:
        scene = run.result().get("scene")
    except Exception:
        return None
    if not scene:
        return None
    d = scene["data"]
    b = d["bands"]
    layers = {"Section 1 clay / hydroxyl (B11/B12)":
              np.where(d["clear"], indices.compute_index(b, "clay_hydroxyl"), np.nan),
              "Section 1 iron oxide (B4/B2)":
              np.where(d["clear"], indices.compute_index(b, "iron_oxide"), np.nan)}
    return {"slot": "remote", "name": f"Section 1 · Sentinel-2 {d['date']}", "kind": "grid",
            "layers": layers, "transform": d["transform"], "epsg": d["epsg"]}


def combine(chosen, fav=None):
    """One combined view of every chosen section, without redoing any section's analysis."""
    data = {k: [] for k in dsm.SLOTS}
    results = {"geophysics": [], "geochem": []}
    errors, epsgs, used = [], [], []

    for a in chosen:
        run = a["run"]
        if a["key"] == "remote-sensing":
            layer = _section1_layers(run)
            if layer is not None:
                data["remote"].append(layer)
                epsgs.append(layer["epsg"])
                used.append(f"{a['label']}: {layer['name']}")
            continue
        try:
            d = run.result()
        except Exception as e:
            errors.append(f"{a['label']}: its stored results could not be read ({e}).")
            continue
        epsgs.append(d["epsg"])
        for slot, items in d["datasets"].items():
            data.setdefault(slot, []).extend(items)
        for k in ("geophysics", "geochem"):
            results[k].extend(d["results"].get(k, []))
        errors.extend(d.get("errors") or [])
        used.append(f"{a['label']}: " + ", ".join(x["name"] for xs in d["datasets"].values()
                                                  for x in xs) if any(d["datasets"].values())
                    else a["label"])

    if not epsgs:
        return None, errors + ["Nothing could be read from the chosen sections."]
    work = max(set(epsgs), key=epsgs.count)
    if len(set(epsgs)) > 1:
        errors.append(f"The sections do not all use the same coordinate system "
                      f"(EPSG {', '.join(str(e) for e in sorted(set(epsgs)))}). "
                      f"EPSG:{work} was used; check that the areas really overlap.")

    liths = sorted({str(l) for ds in data["geology"] if ds["kind"] == "points" for l in ds["lith"]})
    from geoai.geology import favourable_default
    fav_default = favourable_default(liths)
    fav = list(fav) if fav else fav_default

    spatial = [d for k in data for d in data[k] if d["kind"] != "image_ref"]
    bl = [interpret.bounds_of(d) for d in spatial]
    grid, evidence = None, {}
    if any(b is not None for b in bl):
        tr, shape, cell = interpret.common_grid(bl)
        grid = (tr, shape, cell)
        evidence = interpret.build_evidence(data, results, tr, shape, work, fav)
    return {"epsg": work, "datasets": data, "results": results, "grid": grid, "evidence": evidence,
            "lithologies": liths, "fav_default": fav_default, "fav": fav, "errors": errors,
            "used": used}, errors


# ------------------------------------------------------------------ the page
def page():
    available = _available()
    chosen = _chosen(available)
    if not chosen:
        return base.page(KEY, "app/prediction.html", available=available, chosen=[], run=None,
                         combined=None, tabs=TABS, methods=METHODS)
    combined, errors = _cached_combine(chosen)
    return base.page(KEY, "app/prediction.html", available=available, chosen=chosen, run=None,
                     combined=combined, errors=errors, tabs=TABS, methods=METHODS,
                     chosen_keys=[a["key"] for a in chosen], opts=_opts(combined))


def _cache_run():
    """A run folder of this section's own, used only to cache the combination and its figures."""
    run = next(iter(store.Run.list(current_user.id, KEY)), None)
    if run is None:
        run = store.Run.create(current_user.id, KEY)
        run.write_meta({"state": "done", "label": "integrated prediction"})
    return run


def _signature(chosen, fav=None):
    return tuple(sorted((a["key"], a["run"].id) for a in chosen)) + (tuple(fav or ()),)


def _cached_combine(chosen, fav=None):
    run = _cache_run()
    sig = _signature(chosen, fav)
    out = run.cached(("combined", sig), lambda: combine(chosen, fav))
    return out


# ------------------------------------------------------------------ prediction settings
def _wkey(name, field):
    return f"ev_{field}_{abs(hash(name)) % 10 ** 8}"


def _opts(combined):
    """The weights, directions and method currently chosen."""
    if combined is None:
        return None
    ev = combined["evidence"]
    v = request.values
    sent = bool(v.get("_p"))

    def num(name, default):
        try:
            return float(v.get(name, default))
        except (TypeError, ValueError):
            return float(default)

    weights, favour = {}, {}
    for name, e in ev.items():
        weights[name] = (num(_wkey(name, "w"), e["default_weight"]) if sent
                         else float(e["default_weight"]))
        d = v.get(_wkey(name, "dir")) if sent else None
        favour[name] = d if d in ("High", "Low") else e["favourable"]
    method = v.get("method")
    return {"weights": weights, "favourable": favour,
            "method": method if method in METHODS else METHODS[0],
            "top": max(1.0, min(num("top", 5.0), 20.0)),
            "min_ha": max(0.0, min(num("min_ha", 1.0), 20.0)),
            "fav": v.getlist("fav") if sent and v.getlist("fav") else list(combined["fav"])}


def _query(opts):
    q = {"_p": 1, "method": opts["method"], "top": opts["top"], "min_ha": opts["min_ha"]}
    for name, w in opts["weights"].items():
        q[_wkey(name, "w")] = w
        q[_wkey(name, "dir")] = opts["favourable"][name]
    return q


def _predict(combined, opts):
    """The prediction itself, cached under the settings that produced it."""
    run = _cache_run()
    key = ("prediction", tuple(sorted(opts["weights"].items())),
           tuple(sorted(opts["favourable"].items())), opts["method"], opts["top"], opts["min_ha"],
           tuple(opts["fav"]))

    def go():
        return _compute(combined, opts)

    return run.cached(key, go)


def _compute(combined, s):
    if not combined["grid"]:
        return None
    tr, shape, cell = combined["grid"]
    ev = dict(combined["evidence"])
    if combined["datasets"]["geology"]:
        lay = interpret.lithology_layer(combined["datasets"], tuple(s["fav"]), tr, shape)
        ev.pop("Proximity to favourable lithology", None)
        if lay is not None:
            ev["Proximity to favourable lithology"] = lay
    if not ev:
        return None
    members = {k: interpret.membership(e["raw"], s["favourable"].get(k, e["favourable"]))
               for k, e in ev.items()}
    out = {"members": members, "rf": None, "knowledge": None, "evidence": ev}
    used = {k: v for k, v in s["weights"].items() if v > 0 and k in members}
    if s["method"] != METHODS[1]:
        out["knowledge"], out["cover"] = interpret.knowledge_driven(
            {k: members[k] for k in used}, used)
    if s["method"] != METHODS[0]:
        xy, lab, src = _gold_labels(combined)
        if xy is None:
            out["rf"] = {"error": "Needs geochemical data with a gold column."}
        else:
            feats = {k: members[k] for k in members
                     if ev[k]["source"] != "Geochemistry" and s["weights"].get(k, 0) > 0}
            out["rf"] = interpret.data_driven(feats, xy, lab, tr, shape)
            out["rf_source"] = src
    rf_prob = out["rf"].get("prob") if out["rf"] and "prob" in out["rf"] else None
    if s["method"] == METHODS[0]:
        score = out["knowledge"]
    elif s["method"] == METHODS[1]:
        score = rf_prob
    else:
        score = (out["knowledge"] if rf_prob is None
                 else rf_prob if out["knowledge"] is None
                 else np.nanmean([out["knowledge"], rf_prob], axis=0))
    out["score"] = score
    if score is None:
        return out
    min_px = max(1, int(round(s["min_ha"] * 1e4 / cell ** 2)))
    out["zones"], out["targets"] = interpret.describe_targets(
        score, np.isfinite(score), tr, s["top"], min_px, {k: members[k] for k in used},
        combined["results"], combined["datasets"])
    return out


def _gold_labels(combined):
    for ds, r in zip(combined["datasets"]["geochem"], combined["results"]["geochem"]):
        if "au_class" in r:
            return np.column_stack([ds["x"], ds["y"]]), (r["au_class"] >= 2).astype(int), ds["name"]
    return None, None, None


def _with_lonlat(combined, t):
    if t is None or not len(t):
        return t
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(combined["epsg"], 4326, always_xy=True).transform(
        np.asarray(t["Easting"], float), np.asarray(t["Northing"], float))
    t = t.copy()
    t.insert(1, "Latitude", np.round(lat, 6))
    t.insert(2, "Longitude", np.round(lon, 6))
    return t


# ------------------------------------------------------------------ findings
def findings(combined):
    out = []
    for ds, r in zip(combined["datasets"]["geophysics"], combined["results"]["geophysics"]):
        if ds.get("survey", "").startswith("Magnetic") and r.get("products"):
            starts, lens = lineaments.rose_stats(r.get("lineaments") or [])
            if lens.sum():
                o = np.argsort(lens)[::-1]
                tr = [f"{lineaments.strike_label(starts[j] + 5)} "
                      f"({starts[j]:.0f}–{starts[j] + 10:.0f}°)"
                      for j in o[:2] if lens[j] > 0.15 * lens.sum()]
                out.append(f"Magnetics ({ds['name']}): main structural trend"
                           f"{'s' if len(tr) > 1 else ''} {' and '.join(tr)}.")
            eu = r.get("euler")
            if eu is not None and len(eu):
                q = eu.depth.quantile([0.25, 0.5, 0.75])
                out.append(f"Magnetic source depths (Euler, {len(eu)} solutions): median "
                           f"{q[0.5]:.0f} m, middle half {q[0.25]:.0f}–{q[0.75]:.0f} m.")
        elif r.get("lines"):
            for lid, L in r["lines"].items():
                an = L["anomalies"]
                if len(an):
                    a = an.iloc[0]
                    out.append(f"IP line {lid}: strongest anomaly {a.from_x:.0f}–{a.to_x:.0f} m, "
                               f"from about {a.depth_top:.0f} to {a.depth_bottom:.0f} m deep "
                               f"(peak {a.peak:.1f}).")
        elif r.get("limits"):
            out.append(f"{ds.get('survey', 'Geophysics')} ({ds['name']}): strong anomalies above "
                       f"{r['limits'][1]:.2f}.")
    for ds, r in zip(combined["datasets"]["geochem"], combined["results"]["geochem"]):
        au = ds.get("au")
        if au and "au_class" in r:
            lim = r["limits"][au]
            n = int((r["au_class"] >= 2).sum())
            mx = np.nanmax(ds["table"][au])
            corr = r["corr"][au].drop(au)
            pf = [c for c, v in corr.items() if v >= 0.3]
            out.append(f"Geochemistry ({ds['name']}): {n} samples anomalous in gold "
                       f"(at or above {lim[1]:.3g}), maximum {mx:.3g}"
                       + (f"; pathfinders correlated with Au: {', '.join(pf)}." if pf else "."))
    pts = [d for d in combined["datasets"]["geology"] if d["kind"] == "points"]
    if pts:
        s = (np.concatenate([d["strike"] for d in pts if d.get("strike") is not None])
             if any(d.get("strike") is not None for d in pts) else np.array([]))
        s = s[np.isfinite(s)]
        txt = (f"Geology: {sum(len(d['x']) for d in pts)} mapping points; favourable hosts: "
               f"{', '.join(combined['fav']) or 'none chosen'}")
        if s.size:
            h, e = np.histogram(s % 180, bins=np.arange(0, 190, 10))
            txt += f"; measured structures strike mainly {lineaments.strike_label(e[np.argmax(h)] + 5)}"
        out.append(txt + ".")
    return out


# ------------------------------------------------------------------ tabs
def panel(name):
    available = _available()
    chosen = _chosen(available)
    if not chosen:
        return '<p class="note warn">Choose at least one section with results.</p>'
    combined, _errors = _cached_combine(chosen)
    if combined is None:
        return '<p class="note err">Nothing could be combined from those sections.</p>'
    opts = _opts(combined)
    run = _cache_run()
    q = _query(opts)
    ctx = {"run": run, "c": combined, "opts": opts, "q": q, "methods": METHODS,
           "fig_url": lambda f, **kw: url_for("app.figure", section_key=KEY, run_id=run.id, name=f,
                                              **{**q, **kw}),
           "dl": lambda kind, f, **kw: url_for("app.download", section_key=KEY, run_id=run.id,
                                               kind=kind, name=f, **{**q, **kw})}
    if name == "summary":
        pred = _predict(combined, opts) if combined["grid"] else None
        t = _with_lonlat(combined, (pred or {}).get("targets"))
        return render_template("app/prediction/_summary.html", pred=pred, targets=t,
                               findings=_summary_findings(combined, t),
                               counts={k: len(v) for k, v in combined["datasets"].items()}, **ctx)
    if name == "gold":
        return _panel_gold(ctx)
    if name == "remote":
        return _panel_remote(ctx)
    abort(404)


def _summary_findings(combined, t):
    f = findings(combined)
    if t is not None and len(t):
        for j in range(min(3, len(t))):
            row = t.iloc[j]
            f.append(f"Target {row.Target}: {row.Latitude:.5f}, {row.Longitude:.5f} · score "
                     f"{row.Score:.2f} · {row['Area (ha)']} ha"
                     + (f" · estimated depth about {row['Est. depth (m)']:.0f} m"
                        if pd.notna(row["Est. depth (m)"]) else "")
                     + (f" · trend {row.Trend}" if row.Trend else "") + ".")
    return f


def _panel_gold(ctx):
    combined, opts = ctx["c"], ctx["opts"]
    if not combined["grid"]:
        return ('<p class="note warn">The data have no map coordinates, so a prediction map cannot '
                'be made.</p>')
    pred = _predict(combined, opts)
    if pred is None:
        return '<p class="note warn">No evidence layers could be built from the submitted data.</p>'
    ev = pred["evidence"]
    rows = [{"name": n, "source": e["source"], "info": e["info"],
             "w": opts["weights"].get(n, e["default_weight"]),
             "dir": opts["favourable"].get(n, e["favourable"]),
             "wkey": _wkey(n, "w"), "dkey": _wkey(n, "dir")} for n, e in ev.items()]
    if pred.get("score") is None:
        msg = (pred.get("rf") or {}).get("error")
        return render_template("app/prediction/_gold.html", pred=pred, rows=rows, empty=True,
                               message=msg or "Give at least one evidence layer a weight above 0.",
                               **ctx)
    t = _with_lonlat(combined, pred["targets"])
    cell = combined["grid"][2]
    used = [k for k, w in opts["weights"].items() if w > 0 and k in pred["members"]]
    return render_template("app/prediction/_gold.html", pred=pred, rows=rows, empty=False,
                           targets=t, used=used,
                           target_ha=pred["zones"]["targets"].sum() * cell ** 2 / 1e4,
                           short=table_html(t[["Target", "Latitude", "Longitude", "Area (ha)",
                                               "Score", "Est. depth (m)", "Trend"]])
                           if len(t) else None,
                           full=table_html(t) if len(t) else None, **ctx)


def _panel_remote(ctx):
    items = ctx["c"]["datasets"]["remote"]
    if not items:
        return ('<p class="note warn">No remote sensing layers. Load a scene in section 1, or upload '
                'GeoTIFFs there, and they will appear here.</p>')
    try:
        i = max(0, min(int(request.values.get("ds", 0)), len(items) - 1))
    except (TypeError, ValueError):
        i = 0
    ds = items[i]
    layers = []
    if ds["kind"] != "image_ref":
        names = list(ds["layers"])
        for n in names:
            a = (ds["layers"][n] if ds["kind"] == "grid"
                 else interpret.points_to_grid(ds["x"], ds["y"], ds["layers"][n],
                                               ctx["c"]["grid"][0], ctx["c"]["grid"][1]))
            v = a[np.isfinite(a)]
            layers.append({"name": n, "fig": f"rslayer-{i}-{safe_name(n)}",
                           "median": float(np.median(v)) if v.size else None,
                           "p95": float(np.percentile(v, 95)) if v.size else None,
                           "max": float(v.max()) if v.size else None})
    return render_template("app/prediction/_remote.html", items=items, i=i, ds=ds, layers=layers,
                           **ctx)


# ------------------------------------------------------------------ pictures
def _state():
    available = _available()
    chosen = _chosen(available)
    if not chosen:
        return None, None
    combined, _e = _cached_combine(chosen)
    return combined, (_opts(combined) if combined else None)


def redraw(run, name, args):
    from ..render import save_fig
    combined, opts = _state()
    if combined is None:
        return None
    fig = None
    if name == "overview":
        fig = _overview_figure(combined, opts)
    elif name == "prospectivity":
        pred = _predict(combined, opts)
        if not pred or pred.get("score") is None:
            return None
        fig = plots.grid_figure(pred["score"], combined["grid"][0],
                                "Gold prospectivity (pink = targets)", "inferno", "score",
                                targets=pred["zones"]["targets"], lo=1, hi=99)
    elif name == "rf-probability":
        pred = _predict(combined, opts)
        rf = (pred or {}).get("rf") or {}
        if "prob" not in rf:
            return None
        fig = plots.grid_figure(rf["prob"], combined["grid"][0],
                                "Random Forest probability of a gold anomaly", "viridis",
                                "probability", lo=0, hi=100)
    elif name == "rf-importance":
        pred = _predict(combined, opts)
        rf = (pred or {}).get("rf") or {}
        if "importances" not in rf:
            return None
        fig = plots.bar_figure(rf["importances"], "What the model relies on", "importance")
    elif name == "evidence":
        pred = _predict(combined, opts)
        if not pred:
            return None
        used = [k for k, w in opts["weights"].items() if w > 0 and k in pred["members"]]
        if not used:
            return None
        fig = plots.panels([pred["members"][k] for k in used], used, combined["grid"][0],
                           "viridis", ncols=4, lo=0, hi=100)
    elif name.startswith("rslayer-"):
        fig = _remote_layer_figure(combined, name)
    if fig is None:
        return None
    return save_fig(fig, run.fig_path(name))


def _remote_layer_figure(combined, name):
    parts = name.split("-", 2)
    try:
        i = int(parts[1])
    except (ValueError, IndexError):
        return None
    items = combined["datasets"]["remote"]
    if i >= len(items):
        return None
    ds = items[i]
    if ds["kind"] == "image_ref":
        x0, x1, y0, y1 = ds["extent"]
        return plots.image_figure(ds["image"], [x0, x1, y0, y1], ds["name"])
    for n in ds["layers"]:
        if safe_name(n) == parts[2]:
            a = (ds["layers"][n] if ds["kind"] == "grid"
                 else interpret.points_to_grid(ds["x"], ds["y"], ds["layers"][n],
                                               combined["grid"][0], combined["grid"][1]))
            tr = ds["transform"] if ds["kind"] == "grid" else combined["grid"][0]
            return plots.grid_figure(a, tr, f"{n} · {ds['name']}", "magma")
    return None


def _overview_figure(combined, opts):
    import matplotlib.pyplot as plt
    if not combined["grid"]:
        return None
    tr, shape, _cell = combined["grid"]
    pred = _predict(combined, opts)
    t = _with_lonlat(combined, (pred or {}).get("targets"))
    fig, ax = plt.subplots(figsize=(8, 7))
    if pred and pred.get("score") is not None:
        ext = [tr.c, tr.c + shape[1] * tr.a, tr.f + shape[0] * tr.e, tr.f]
        im = ax.imshow(pred["score"], extent=ext, cmap="inferno", alpha=0.85)
        plt.colorbar(im, ax=ax, shrink=0.75, label="gold prospectivity")
    styles = {"geophysics": ("#1f77b4", 2, "Geophysics"), "geology": ("#2ca02c", 14, "Geology"),
              "geochem": ("#ff7f0e", 5, "Geochemistry"), "remote": ("#9467bd", 3, "Remote sensing")}
    for slot, (col, size, lab) in styles.items():
        xs, ys = [], []
        for d in combined["datasets"].get(slot, []):
            if d["kind"] == "points" or (d["kind"] == "section" and d.get("y") is not None):
                xs.append(d["x_map"] if d["kind"] == "section" else d["x"])
                ys.append(d["y"])
        if xs:
            ax.scatter(np.concatenate(xs), np.concatenate(ys), s=size, c=col, alpha=0.45,
                       label=lab, linewidths=0)
    if t is not None and len(t):
        ax.scatter(t.Easting, t.Northing, marker="*", s=260, c="#00e5ff", edgecolors="k",
                   label="Targets", zorder=5)
        for _, row in t.head(10).iterrows():
            ax.annotate(row.Target, (row.Easting, row.Northing), xytext=(6, 6),
                        textcoords="offset points", color="white", weight="bold", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6, lw=0))
    ax.legend(loc="upper left", bbox_to_anchor=(1.25, 1), frameon=False, fontsize=8)
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title("Overview: data and gold targets", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ downloads
def deliver(run, kind, name, args):
    combined, opts = _state()
    if combined is None:
        abort(404)
    if kind == "png":
        path = run.fig_path(name)
        if not path.is_file():
            path = redraw(run, name, args)
        return base.send_png(path, f"{name}.png")
    pred = _predict(combined, opts)
    if kind == "csv" and name == "targets":
        t = _with_lonlat(combined, (pred or {}).get("targets"))
        if t is None or not len(t):
            abort(404)
        return base.send_csv(t, "gold_targets.csv")
    if kind == "tif" and name == "prediction":
        if not pred or pred.get("score") is None:
            abort(404)
        return base.send_bytes(export.geotiff_bytes(
            {"gold_prospectivity": pred["score"],
             "target": pred["zones"]["targets"].astype(np.float32),
             **{f"ev_{i + 1}": pred["members"][k] for i, k in enumerate(pred["members"])}},
            combined["grid"][0], combined["epsg"]), "gold_prediction.tif", "image/tiff")
    if kind == "html" and name == "report":
        return base.send_bytes(_report(combined, opts, pred).encode("utf-8"),
                               "geoai_gold_report.html", "text/html")
    abort(404)


def _report(combined, opts, pred):
    from ..render import fig_to_png_bytes
    tr = combined["grid"][0] if combined["grid"] else None
    img = ""
    if pred and pred.get("score") is not None and tr is not None:
        fig = plots.grid_figure(pred["score"], tr, "Gold prospectivity", "inferno", "score",
                                targets=pred["zones"]["targets"] if pred.get("zones") else None)
        png = base64.b64encode(fig_to_png_bytes(fig, dpi=150)).decode()
        img = f'<img src="data:image/png;base64,{png}" style="width:100%">'
    t = _with_lonlat(combined, (pred or {}).get("targets"))
    used = "".join(f"<li>{k}: weight {w:.1f}, {opts['favourable'][k].lower()} values favourable</li>"
                   for k, w in opts["weights"].items() if w > 0)
    files = "".join(f"<li>{d['name']} ({slot})</li>"
                    for slot, ds in combined["datasets"].items() for d in ds)
    rows = t.to_html(index=False) if t is not None and len(t) else "<p>No targets.</p>"
    return f"""<html><head><meta charset="utf-8"><title>GeoAI gold prediction report</title></head>
<body style="font-family:Arial,sans-serif;max-width:960px;margin:24px auto;color:#222">
<h1>GeoAI · Gold interpretation &amp; prediction</h1>
<p>Generated {datetime.now():%Y-%m-%d %H:%M} · coordinates EPSG:{combined['epsg']}</p>
<h2>Data</h2><ul>{files}</ul>
<h2>Key findings</h2><ul>{''.join(f'<li>{f}</li>' for f in findings(combined))}</ul>
<h2>Prediction</h2><p>Method: {opts['method']}. Top {opts['top']:.1f}% of the area, zones of at
least {opts['min_ha']:.1f} ha.</p><ul>{used}</ul>{img}
<h2>Ranked targets</h2>{rows}
<p style="color:#777;font-size:12px">Targets rank ground for follow-up (mapping, soil infill,
trenching, drilling). They are not proof of mineralisation. Depths are estimates from Euler
deconvolution, IP pseudo-depths or the power spectrum.</p></body></html>"""
