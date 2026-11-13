"""Pit mapping: mineralisation trend through a set of pits, remaining potential of each pit,
proposed neighbour pits, and a pit register kept on disk for future use. No Streamlit inside."""
import os
from datetime import date

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from . import coords

FIELDS = {
    "pit": ["pit", "pit id", "pit_id", "pit name", "name", "id", "station", "point", "label", "no"],
    "depth": ["depth", "pit depth", "depth_m", "depth (m)", "total depth"],
    "grade": ["au", "au_gpt", "au g/t", "au (g/t)", "au_ppm", "gold", "grade", "assay"],
    "status": ["status", "state", "condition", "remarks", "remark", "comment", "comments", "description"],
    "strike": ["strike", "vein strike", "trend", "azimuth"],
    "dip": ["dip", "vein dip"],
    "width": ["width", "vein width", "thickness", "reef width"],
}
BARREN = ("barren", "exhaust", "waste", "no gold", "no ore", "worked out", "mined out", "depleted", "sterile")
HINDERED = ("abandon", "flood", "collaps", "closed", "unsafe", "water")
REGISTER_COLUMNS = ["project", "pit", "type", "easting", "northing", "epsg", "latitude", "longitude", "depth_m",
                    "grade", "status", "potential", "priority", "note", "saved"]


def guess_fields(df, used=()):
    """Column for each optional pit attribute (None if the file has none)."""
    out = {}
    cols = [c for c in df.columns if c not in used]
    for field, names in FIELDS.items():
        out[field] = coords._match(cols, names)
    return out


def build(df, xcol, ycol, epsg, fields):
    """Pit table in a working UTM system. Returns (pits DataFrame, working epsg)."""
    x, y = coords.parse_column(df[xcol]), coords.parse_column(df[ycol])
    ok = x.notna() & y.notna()
    if ok.sum() < 1:
        raise ValueError("No valid coordinates found in the chosen columns.")
    d = df[ok].reset_index(drop=True)
    x, y = x[ok].to_numpy(float), y[ok].to_numpy(float)
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(x, y)
    work = coords.utm_epsg_for(float(np.mean(lon)), float(np.mean(lat))) if epsg == 4326 else epsg
    if work != epsg:
        x, y = Transformer.from_crs(epsg, work, always_xy=True).transform(x, y)

    def num(field):
        c = fields.get(field)
        return pd.to_numeric(d[c].astype(str).str.replace(r"[<>]", "", regex=True), errors="coerce").to_numpy(float) \
            if c else np.full(len(d), np.nan)

    pit = d[fields["pit"]].astype(str).str.strip() if fields.get("pit") else pd.Series(
        [f"Pit {i + 1}" for i in range(len(d))])
    status = d[fields["status"]].fillna("").astype(str).str.strip() if fields.get("status") else pd.Series([""] * len(d))
    pits = pd.DataFrame({"pit": pit.to_numpy(), "x": x, "y": y, "lat": lat, "lon": lon, "depth": np.abs(num("depth")),
                         "grade": num("grade"), "status": status.to_numpy(), "strike": num("strike") % 180,
                         "dip": num("dip"), "width": num("width")})
    low = pits.status.str.lower()
    pits["barren"] = low.apply(lambda s: any(k in s for k in BARREN))
    pits["hindered"] = low.apply(lambda s: any(k in s for k in HINDERED))
    return pits, int(work)


def _axial_mean(deg):
    a = 2 * np.radians(np.asarray(deg, float))
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean()) / 2) % 180)


def strike_label(az):
    names = ["N-S", "NNE-SSW", "NE-SW", "ENE-WSW", "E-W", "ESE-WNW", "SE-NW", "SSE-NNW"]
    return names[int(((az % 180) + 11.25) // 22.5) % 8]


def trend(pits):
    """Mineralisation trend: the line that best fits the mineralised pits, weighted by gold grade.

    Returns dict(azimuth (compass strike 0-180), centre, length, half_width, linearity, along, across,
    weights, basis, measured_strike, plunge)."""
    n = len(pits)
    if n < 3:
        raise ValueError("At least 3 pits are needed to find a trend.")
    g = pits.grade.to_numpy(float)
    has_grade = np.isfinite(g).sum() >= 3 and np.nanmax(g) > 0
    if has_grade:
        cap = np.nanpercentile(g, 95)
        w = np.where(np.isfinite(g), np.clip(g, 0, cap), np.nanmedian(g) * 0.5)
        basis = "pit positions weighted by gold grade"
    else:
        w = np.ones(n)
        basis = "pit positions (no grades in the file, so every pit counts equally)"
    w = np.where(pits.barren.to_numpy(), 0.0, w)
    if (w > 0).sum() < 3:
        w = np.ones(n)
        basis = "pit positions (too few mineralised pits to weight)"
    xy = pits[["x", "y"]].to_numpy(float)
    c = (xy * w[:, None]).sum(0) / w.sum()
    d = xy - c
    cov = (d * w[:, None]).T @ d / w.sum()
    val, vec = np.linalg.eigh(cov)
    u = vec[:, 1]                                   # unit vector along the trend (east, north)
    if u[1] < 0 or (u[1] == 0 and u[0] < 0):
        u = -u                                      # point it northwards
    v = np.array([u[1], -u[0]])
    along, across = d @ u, d @ v
    on = w > 0
    out = {"azimuth": float(np.degrees(np.arctan2(u[0], u[1])) % 180), "centre": c, "u": u, "v": v,
           "length": float(np.ptp(along[on])), "half_width": float(max(2 * np.sqrt(max(val[0], 0)), 1e-6)),
           "linearity": float(1 - val[0] / max(val[1], 1e-12)), "along": along, "across": across, "weights": w,
           "basis": basis, "has_grade": bool(has_grade), "measured_strike": None, "plunge": None}
    s = pits.strike.to_numpy(float)
    if np.isfinite(s).sum() >= 2:
        out["measured_strike"] = _axial_mean(s[np.isfinite(s)])
    dep = pits.depth.to_numpy(float)
    m = on & np.isfinite(dep) & (dep > 0)
    if m.sum() >= 5 and np.ptp(along[m]) > 0:
        r = np.corrcoef(along[m], dep[m])[0, 1]
        slope = np.polyfit(along[m], dep[m], 1)[0]
        if abs(r) >= 0.5:
            az = out["azimuth"] if slope > 0 else (out["azimuth"] + 180) % 360
            out["plunge"] = {"angle": float(np.degrees(np.arctan(abs(slope)))), "toward": float(az), "r": float(r)}
    return out


def spacing(pits, tr):
    """Typical step between neighbouring mineralised pits along the trend."""
    a = np.sort(tr["along"][tr["weights"] > 0])
    gaps = np.diff(a)
    gaps = gaps[gaps > 0]
    return float(np.median(gaps)) if gaps.size else 20.0


def potential(pits, tr):
    """Remaining potential of each pit (screening, not a resource estimate): High / Moderate / Low + reasons."""
    n = len(pits)
    xy = pits[["x", "y"]].to_numpy(float)
    g, dep = pits.grade.to_numpy(float), pits.depth.to_numpy(float)
    sp = spacing(pits, tr)
    k = min(5, n)
    dist, idx = cKDTree(xy).query(xy, k=k)
    grade_rank = pd.Series(g).rank(pct=True).to_numpy()
    on_trend = np.exp(-(tr["across"] / max(tr["half_width"], 0.5 * sp)) ** 2)
    scores, labels, reasons = [], [], []
    for i in range(n):
        parts, why = [], []
        nb = idx[i][1:] if k > 1 else []
        parts.append((on_trend[i], 1.0))
        why.append("on the trend" if on_trend[i] > 0.6 else "off the trend" if on_trend[i] < 0.25 else "edge of the trend")
        if np.isfinite(g[i]):
            parts.append((grade_rank[i], 1.5))
            why.append("high grade" if grade_rank[i] > 0.66 else "low grade" if grade_rank[i] < 0.34 else "moderate grade")
        ng = g[nb][np.isfinite(g[nb])] if len(nb) else np.array([])
        if ng.size and np.isfinite(g).sum() > 2:
            r = float((ng.mean() > np.nanmedian(g)))
            parts.append((0.3 + 0.7 * r, 0.7))
            if r:
                why.append("good grades in neighbouring pits")
        nd = dep[nb][np.isfinite(dep[nb])] if len(nb) else np.array([])
        if np.isfinite(dep[i]) and nd.size:
            room = float(np.clip((nd.max() - dep[i]) / max(nd.max(), 1e-9), 0, 1))
            parts.append((0.4 + 0.6 * room, 0.8))
            if room > 0.25:
                why.append(f"shallower ({dep[i]:.0f} m) than neighbours that reached {nd.max():.0f} m: room at depth")
            elif room == 0:
                why.append("already the deepest pit here")
        s = sum(a * b for a, b in parts) / sum(b for _, b in parts)
        if pits.barren.iloc[i]:
            s = min(s, 0.2)
            why = ["recorded as barren / worked out"]
        elif pits.hindered.iloc[i]:
            why.append("access problem recorded (abandoned / flooded / collapsed), not proof the gold is finished")
        scores.append(s)
        labels.append("High" if s >= 0.62 else "Moderate" if s >= 0.42 else "Low")
        reasons.append("; ".join(why))
    out = pits.copy()
    out["score"], out["potential"], out["why"] = np.round(scores, 2), labels, reasons
    return out


def _nearby_grade(xy, g, pts, reach):
    """Inverse-distance mean grade of the pits around each target (0-1 scaled); 0.5 when there are no grades."""
    ok = np.isfinite(g)
    if ok.sum() < 2:
        return np.full(len(pts), 0.5)
    gs = pd.Series(g[ok]).rank(pct=True).to_numpy()
    d = np.linalg.norm(pts[:, None, :] - xy[ok][None, :, :], axis=2)
    w = 1 / (d + 0.25 * reach) ** 2
    return (w * gs[None, :]).sum(1) / w.sum(1)


def propose(pits, tr, epsg, max_targets=12, step_outs=2):
    """New pit positions: fill the gaps along the trend, then step out beyond both ends."""
    sp = spacing(pits, tr)
    on = tr["weights"] > 0
    order = np.argsort(tr["along"])
    order = order[on[order]]
    al, ac = tr["along"][order], tr["across"][order]
    g, dep = pits.grade.to_numpy(float), pits.depth.to_numpy(float)
    rows = []
    for a0, a1, c0, c1 in zip(al[:-1], al[1:], ac[:-1], ac[1:]):
        gap = a1 - a0
        if gap > 1.6 * sp:
            m = int(round(gap / sp)) - 1
            for j in range(1, m + 1):
                f = j / (m + 1)
                rows.append(("Infill", a0 + f * gap, c0 + f * (c1 - c0), 1.0,
                             f"untested {gap:.0f} m gap between two mineralised pits"))
    tail = min(4, len(al))
    for sign, ends in ((1, order[-tail:]), (-1, order[:tail])):
        a_end = tr["along"][ends[-1] if sign > 0 else ends[0]]
        c_end = float(np.mean(tr["across"][ends]))
        for j in range(1, step_outs + 1):
            rows.append(("Step-out", a_end + sign * j * sp, c_end, 0.85 ** j,
                         f"{j * sp:.0f} m beyond the last pit along the trend"))
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows, columns=["kind", "along", "across", "decay", "why"])
    pts = tr["centre"] + np.outer(t.along, tr["u"]) + np.outer(t.across, tr["v"])
    xy = pits[["x", "y"]].to_numpy(float)
    d, near = cKDTree(xy).query(pts)
    keep = d > 0.5 * sp
    t, pts, near = t[keep].reset_index(drop=True), pts[keep], near[keep]
    if not len(t):
        return t
    t["score"] = np.round(t.decay * (0.4 + 0.6 * _nearby_grade(xy, np.where(on, g, np.nan), pts, sp)), 2)
    t["x"], t["y"] = pts[:, 0], pts[:, 1]
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(pts[:, 0], pts[:, 1])
    t["lat"], t["lon"] = lat, lon
    t["nearest_pit"] = pits.pit.to_numpy()[near]
    if tr["plunge"]:
        a, dd = tr["along"][on & np.isfinite(dep)], dep[on & np.isfinite(dep)]
        k = np.polyfit(a, dd, 1)
        t["expected_depth"] = np.round(np.clip(np.polyval(k, t.along), 1, None), 0)
    elif np.isfinite(dep).any():
        t["expected_depth"] = np.round(np.where(np.isfinite(dep[near]), dep[near], np.nanmedian(dep)), 0)
    else:
        t["expected_depth"] = np.nan
    t = t.sort_values("score", ascending=False).head(max_targets).reset_index(drop=True)
    t.insert(0, "target", [f"P{i + 1}" for i in range(len(t))])
    t["priority"] = np.where(t.score >= 0.7, "1 (first)", np.where(t.score >= 0.5, "2", "3"))
    return t.drop(columns=["decay"])


def summary(pits, tr, targets):
    """Plain-language interpretation."""
    lines = [f"The mineralised pits line up along a **{strike_label(tr['azimuth'])}** trend (strike "
             f"**{tr['azimuth']:.0f}°**), about **{tr['length']:.0f} m** long and **{2 * tr['half_width']:.0f} m** "
             f"wide, found from {tr['basis']}."]
    if tr["linearity"] < 0.5:
        lines.append("The pits are spread out rather than in a line, so this trend is weak: treat the proposed "
                     "pits with caution, or add more pits / grades.")
    if tr["measured_strike"] is not None:
        diff = abs((tr["measured_strike"] - tr["azimuth"] + 90) % 180 - 90)
        lines.append(f"Strikes measured in the pits average **{tr['measured_strike']:.0f}°**, "
                     + ("which agrees with the trend of the pits." if diff <= 20 else
                        f"{diff:.0f}° away from the trend of the pits: the veins may be oblique to the mineralised "
                        "zone (en-echelon veins), so check in the field."))
    if tr["plunge"]:
        lines.append(f"Pits get deeper towards **{tr['plunge']['toward']:.0f}°**: the ore shoot appears to plunge "
                     f"about **{tr['plunge']['angle']:.0f}°** that way, so pits on that side must go deeper.")
    if "potential" in pits:
        c = pits.potential.value_counts()
        lines.append(f"Remaining potential: **{c.get('High', 0)} high**, {c.get('Moderate', 0)} moderate, "
                     f"{c.get('Low', 0)} low, out of {len(pits)} pits.")
    if len(targets):
        k = targets.kind.value_counts()
        lines.append(f"**{len(targets)} new pits proposed**: {k.get('Infill', 0)} in untested gaps along the trend "
                     f"and {k.get('Step-out', 0)} beyond its two ends. Best first: **{targets.target.iloc[0]}** "
                     f"({targets.why.iloc[0]}).")
    return lines


# ---------- pit register (kept on disk for future use) ----------
def register_rows(project, pits, targets, epsg):
    today = date.today().isoformat()
    a = pd.DataFrame({"project": project, "pit": pits.pit, "type": "existing", "easting": pits.x.round(2),
                      "northing": pits.y.round(2), "epsg": epsg, "latitude": pits.lat.round(7),
                      "longitude": pits.lon.round(7), "depth_m": pits.depth, "grade": pits.grade,
                      "status": pits.status, "potential": pits.get("potential", ""), "priority": "",
                      "note": pits.get("why", ""), "saved": today})
    if not len(targets):
        return a[REGISTER_COLUMNS]
    b = pd.DataFrame({"project": project, "pit": targets.target, "type": "proposed", "easting": targets.x.round(2),
                      "northing": targets.y.round(2), "epsg": epsg, "latitude": targets.lat.round(7),
                      "longitude": targets.lon.round(7), "depth_m": targets.expected_depth, "grade": np.nan,
                      "status": targets.kind, "potential": "", "priority": targets.priority, "note": targets.why,
                      "saved": today})
    return pd.concat([a, b], ignore_index=True)[REGISTER_COLUMNS]


def load_register(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path, dtype={"project": str, "pit": str})
        except Exception:
            pass
    return pd.DataFrame(columns=REGISTER_COLUMNS)


def save_register(path, rows):
    """Add rows to the register; a pit saved again under the same project replaces its old row."""
    old = load_register(path)
    project = str(rows.project.iloc[0])
    # proposed pits of this project are replaced as a set; existing pits are replaced by name
    old = old[~((old.project == project) & ((old.type == "proposed") | old.pit.isin(rows.pit.astype(str))))]
    new = pd.concat([old, rows], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new.to_csv(path, index=False)
    return new


def to_kml(project, pits, targets):
    def mark(name, lon, lat, desc, style):
        return (f"<Placemark><name>{name}</name><description><![CDATA[{desc}]]></description>"
                f"<styleUrl>#{style}</styleUrl><Point><coordinates>{lon:.7f},{lat:.7f},0</coordinates></Point></Placemark>")
    icon = "http://maps.google.com/mapfiles/kml/paddle/{}.png"
    styles = "".join(f'<Style id="{k}"><IconStyle><Icon><href>{icon.format(v)}</href></Icon></IconStyle></Style>'
                     for k, v in {"High": "grn-circle", "Moderate": "ylw-circle", "Low": "red-circle",
                                  "pit": "wht-circle", "new": "purple-stars"}.items())
    body = [mark(r.pit, r.lon, r.lat, f"{getattr(r, 'potential', '')} potential. {getattr(r, 'why', '')}",
                 getattr(r, "potential", "pit") or "pit") for r in pits.itertuples()]
    body += [mark(r.target, r.lon, r.lat, f"{r.kind}, priority {r.priority}. {r.why}", "new")
             for r in targets.itertuples()] if len(targets) else []
    return (f'<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            f"<name>{project} pits</name>{styles}{''.join(body)}</Document></kml>")


# ---------- figures ----------
def map_figure(pits, tr, targets, title="", background=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    if background is not None:
        img, (w, e, s, n) = background
        ax.imshow(np.asarray(img), extent=[w, e, s, n], alpha=0.75, zorder=0)
    c, u, v = tr["centre"], tr["u"], tr["v"]
    a0, a1 = tr["along"].min(), tr["along"].max()
    if len(targets):
        a0, a1 = min(a0, targets.along.min()), max(a1, targets.along.max())
    hw = tr["half_width"]
    box = np.array([c + a0 * u + hw * v, c + a1 * u + hw * v, c + a1 * u - hw * v, c + a0 * u - hw * v])
    ax.fill(box[:, 0], box[:, 1], color="#f2c14e", alpha=0.25, zorder=1, label="Mineralised corridor")
    line = np.array([c + a0 * u, c + a1 * u])
    ax.plot(line[:, 0], line[:, 1], "--", color="#b8860b", lw=1.5, zorder=2,
            label=f"Trend {tr['azimuth']:.0f}° ({strike_label(tr['azimuth'])})")
    g = pits.grade.to_numpy(float)
    size = 60 + 160 * (pd.Series(g).rank(pct=True).fillna(0.3).to_numpy()) if np.isfinite(g).any() else 90
    cols = {"High": "#2e9e4f", "Moderate": "#f0a202", "Low": "#c0392b"}
    for lab, col in cols.items():
        m = (pits.potential == lab).to_numpy()
        if m.any():
            ax.scatter(pits.x[m], pits.y[m], s=np.asarray(size)[m] if np.ndim(size) else size, c=col,
                       edgecolors="k", linewidths=0.6, zorder=4, label=f"{lab} potential ({m.sum()})")
    for r in pits.itertuples():
        ax.annotate(r.pit, (r.x, r.y), xytext=(5, 5), textcoords="offset points", fontsize=7, zorder=5)
    if len(targets):
        ax.scatter(targets.x, targets.y, marker="*", s=260, c="#7d3c98", edgecolors="k", linewidths=0.6, zorder=6,
                   label=f"Proposed pits ({len(targets)})")
        for r in targets.itertuples():
            ax.annotate(r.target, (r.x, r.y), xytext=(6, -11), textcoords="offset points", fontsize=8,
                        weight="bold", color="#5b2c6f", zorder=7)
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.margins(0.12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def section_figure(pits, tr, targets, title=""):
    """Long section: distance along the trend against pit depth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.6))
    dep = pits.depth.to_numpy(float)
    ok = np.isfinite(dep)
    g = pits.grade.to_numpy(float)
    for a, d in zip(tr["along"][ok], dep[ok]):
        ax.plot([a, a], [0, d], color="#888", lw=3, solid_capstyle="butt", zorder=1)
    if np.isfinite(g[ok]).any():
        sc = ax.scatter(tr["along"][ok], dep[ok], c=g[ok], cmap="YlOrRd", s=70, edgecolors="k", linewidths=0.5, zorder=3)
        plt.colorbar(sc, ax=ax, label="Gold grade", shrink=0.9)
    else:
        ax.scatter(tr["along"][ok], dep[ok], c="#b8860b", s=70, edgecolors="k", linewidths=0.5, zorder=3)
    for r, a in zip(pits[ok].itertuples(), tr["along"][ok]):
        ax.annotate(r.pit, (a, 0), xytext=(0, 4), textcoords="offset points", fontsize=7, ha="center", rotation=60)
    if len(targets):
        t = targets[np.isfinite(targets.expected_depth)]
        ax.scatter(t.along, t.expected_depth, marker="*", s=200, c="#7d3c98", edgecolors="k", linewidths=0.5, zorder=4,
                   label="Proposed pits (expected depth)")
        for r in t.itertuples():
            ax.annotate(r.target, (r.along, r.expected_depth), xytext=(5, -10), textcoords="offset points", fontsize=8,
                        color="#5b2c6f", weight="bold")
        ax.legend(fontsize=8, loc="lower left")
    ax.axhline(0, color="k", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Distance along the trend (m), towards {tr['azimuth']:.0f}° →")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title, fontsize=10, pad=34)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
