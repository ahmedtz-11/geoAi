"""Soil gold geochemistry: one assay column, read and interpreted for targeting.

What the page needs that plain statistics do not give:
  * whether the samples hold one population or two (background, and a mineralised population on top of it),
    which is the question a soil survey is run to answer;
  * where the anomalous samples are, whether they join into zones with a trend, and whether a zone runs off
    the edge of the survey;
  * which single highs stand alone, since one isolated high sample is as often a nugget or contamination
    as it is a discovery.

No Streamlit inside.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm

# ppb of gold in residual soil: rules of thumb used to put a survey in context, not thresholds for it
GUIDE = [(10, "typical background"), (30, "weakly anomalous"), (100, "anomalous"), (1000, "strongly anomalous")]
UNITS = {"ppm (= g/t)": 1000.0, "ppb (= µg/t)": 1.0, "g/t (= ppm)": 1000.0}
CLASSES = ["Background", "Weak anomaly", "Anomalous", "Strongly anomalous"]
CLASS_COLORS = ["#c9c9c9", "#f2c14e", "#e4572e", "#7b0d1e"]
# the first asks the data where background stops; the rest are the usual fixed rules
METHODS = ["Two populations (where background stops)", "Mean + \u03c3 on log values",
           "Percentiles (75 / 90 / 98)", "Median + MAD on log values"]


def guess_unit(values):
    """ppm or ppb, from the size of the numbers. Soil gold is a few ppb to a few thousand ppb."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v) & (v > 0)]
    if not v.size:
        return list(UNITS)[0]
    return "ppb (= µg/t)" if np.median(v) > 5 else "ppm (= g/t)"


def layout(x, y):
    """How the samples are laid out: spacing, area, duplicates."""
    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    n = len(pts)
    out = {"n": n, "width": float(np.ptp(pts[:, 0])), "height": float(np.ptp(pts[:, 1])), "duplicates": 0,
           "spacing": np.nan, "area_ha": np.nan}
    if n < 2:
        return out
    d, _ = cKDTree(pts).query(pts, k=2)
    nn = d[:, 1]
    out["duplicates"] = int((nn < 1e-6).sum())
    live = nn[nn > 1e-6]
    out["spacing"] = float(np.median(live)) if live.size else np.nan
    if n >= 3:
        try:
            from scipy.spatial import ConvexHull
            out["area_ha"] = float(ConvexHull(pts).volume / 10000.0)
        except Exception:
            out["area_ha"] = out["width"] * out["height"] / 10000.0
    return out


def detection_floor(values):
    """The lab's detection limit, if the low end of the data piles up on one value."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 10:
        return None
    lo = v.min()
    at_floor = int((v <= lo * 1.001).sum())
    return {"value": float(lo), "n": at_floor} if at_floor >= max(3, 0.03 * v.size) else None


def summary(values, unit_ppb):
    """Statistics, in the data's own units and in ppb."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not v.size:
        return {}
    pos = v[v > 0]
    lv = np.log10(pos) if pos.size else np.array([0.0])
    return {"n": int(v.size), "min": float(v.min()), "p25": float(np.percentile(v, 25)),
            "median": float(np.median(v)), "mean": float(v.mean()),
            "geomean": float(10 ** lv.mean()) if pos.size else np.nan,
            "p75": float(np.percentile(v, 75)), "p90": float(np.percentile(v, 90)),
            "p95": float(np.percentile(v, 95)), "p98": float(np.percentile(v, 98)), "max": float(v.max()),
            "cv": float(v.std() / v.mean()) if v.mean() else np.nan,
            "log_sd": float(lv.std()), "max_ppb": float(v.max() * unit_ppb),
            "median_ppb": float(np.median(v) * unit_ppb),
            "contrast": float(v.max() / np.median(v)) if np.median(v) > 0 else np.nan}


def populations(values, seed=0):
    """Is there a mineralised population on top of background?

    Fits one and two log-normal components and compares them by BIC. Two components fit better only when
    the data really do hold a separate higher-grade population; a survey entirely in background, or
    entirely inside one ore zone, stays at one.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v) & (v > 0)]
    out = {"ok": False, "n_pop": 1, "crossover": np.nan}
    if v.size < 30:
        return out
    lv = np.log10(v).reshape(-1, 1)
    try:
        from sklearn.mixture import GaussianMixture
    except Exception:
        return out
    fits = {}
    for k in (1, 2):
        g = GaussianMixture(k, covariance_type="full", random_state=seed, n_init=5).fit(lv)
        fits[k] = (g, g.bic(lv))
    out["ok"] = True
    out["bic"] = {k: float(b) for k, (_, b) in fits.items()}
    if fits[2][1] >= fits[1][1] - 6:                   # 6 = clearly better, not just a better fit by chance
        g = fits[1][0]
        out.update(n_pop=1, mean=[float(10 ** g.means_[0, 0])], sd=[float(np.sqrt(g.covariances_[0, 0, 0]))],
                   weight=[1.0])
        return out
    g = fits[2][0]
    order = np.argsort(g.means_.ravel())
    lo, hi = order
    out.update(n_pop=2, mean=[float(10 ** g.means_[i, 0]) for i in (lo, hi)],
               sd=[float(np.sqrt(g.covariances_[i, 0, 0])) for i in (lo, hi)],
               weight=[float(g.weights_[i]) for i in (lo, hi)])
    # how sure we can be, at each assay value, that a sample belongs to the upper population
    grid = np.linspace(lv.min(), lv.max() + 1, 4000).reshape(-1, 1)
    upper = g.predict_proba(grid)[:, hi]
    above = np.flatnonzero(upper >= 0.5)
    if not above.size:
        return out
    cross = float(10 ** grid[above[0], 0])
    out["crossover"] = cross
    # Weak = where the upper population takes over. The two populations overlap, so grading the rest by
    # further probability levels would bunch every class into a few hundredths; split the upper population
    # at its own median and 90th percentile instead.
    top = v[v >= cross]
    out["n_upper"] = int(top.size)
    out["limits"] = [cross, float(np.median(top)), float(np.percentile(top, 90))] if top.size >= 4 else None
    return out


def thresholds(values, method, pops=None):
    """Three class limits. The two-population method asks the data where background stops; the others
    are the usual fixed rules, which need the analyst to know what background looks like here."""
    from . import geochem
    if method == METHODS[0]:
        lim = (pops or {}).get("limits")
        if lim and np.isfinite(lim).all():
            return [float(v) for v in lim]
        return geochem.thresholds(values, geochem.METHODS[0])     # no second population: fall back
    return geochem.thresholds(values, method)


def probability_points(values):
    """Log-probability plot: sorted values against the normal score of their cumulative frequency.
    A single log-normal population is a straight line; a break means a second population."""
    v = np.asarray(values, float)
    v = np.sort(v[np.isfinite(v) & (v > 0)])
    if v.size < 5:
        return np.array([]), np.array([])
    p = (np.arange(1, v.size + 1) - 0.5) / v.size
    return v, norm.ppf(p)


def zones(x, y, values, threshold, spacing, min_samples=1):
    """Group the samples at or above `threshold` into zones: anomalies that touch each other are one find.

    Samples within 1.8 sample spacings are joined (single linkage), so a zone is a run of neighbouring
    anomalous samples rather than a circle drawn around each one.
    """
    x, y, v = (np.asarray(a, float) for a in (x, y, values))
    hot = np.flatnonzero(np.isfinite(v) & (v >= threshold))
    if not hot.size:
        return pd.DataFrame()
    pts = np.column_stack([x[hot], y[hot]])
    reach = 1.8 * (spacing if np.isfinite(spacing) and spacing > 0 else 1.0)
    parent = np.arange(len(pts))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(pts).query_pairs(reach):
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
    groups = {}
    for i in range(len(pts)):
        groups.setdefault(find(i), []).append(i)
    rows = []
    for members in groups.values():
        if len(members) < min_samples:
            continue
        m = np.array(members)
        px, py, pv = pts[m, 0], pts[m, 1], v[hot[m]]
        k = int(np.argmax(pv))
        row = {"samples": len(m), "peak": float(pv.max()), "mean": float(pv.mean()),
               "peak_x": float(px[k]), "peak_y": float(py[k]), "peak_i": int(hot[m[k]]),
               "centre_x": float(px.mean()), "centre_y": float(py.mean()),
               "x0": float(px.min()), "x1": float(px.max()), "y0": float(py.min()), "y1": float(py.max()),
               "indices": hot[m]}
        if len(m) >= 3:                               # long axis of the zone
            d = np.column_stack([px - px.mean(), py - py.mean()])
            val, vec = np.linalg.eigh(d.T @ d / len(m))
            u = vec[:, 1]
            row["azimuth"] = float(np.degrees(np.arctan2(u[0], u[1])) % 180)
            along = d @ u
            row["length"] = float(np.ptp(along))
            row["width"] = float(np.ptp(d @ np.array([u[1], -u[0]])))
            row["elongation"] = float(np.sqrt(max(val[1], 1e-12) / max(val[0], 1e-12)))
        else:
            row.update(azimuth=np.nan, length=max(float(np.ptp(px)), float(np.ptp(py))), width=0.0,
                       elongation=np.nan)
        rows.append(row)
    z = pd.DataFrame(rows).sort_values(["peak", "samples"], ascending=False).reset_index(drop=True)
    z.insert(0, "zone", [f"G{i + 1}" for i in range(len(z))])
    return z


def open_edges(zone, x, y, spacing):
    """Which way a zone runs off the edge of the sampled ground, so it is bigger than the survey shows."""
    if not np.isfinite(spacing) or spacing <= 0:
        return []
    reach = 1.5 * spacing
    out = []
    for side, test in (("north", lambda: zone["y1"] >= np.max(y) - reach),
                       ("south", lambda: zone["y0"] <= np.min(y) + reach),
                       ("east", lambda: zone["x1"] >= np.max(x) - reach),
                       ("west", lambda: zone["x0"] <= np.min(x) + reach)):
        if test():
            out.append(side)
    return out


def sampling_direction(x, y):
    """Is the sampling itself directional? Returns (compass azimuth, 0-1 strength).

    Samples walked along widely spaced lines make any anomaly look elongated along those lines, so a zone
    trend parallel to the sampling has to be read with care.
    """
    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    if len(pts) < 10:
        return np.nan, 0.0
    _, idx = cKDTree(pts).query(pts, k=2)
    d = pts[idx[:, 1]] - pts
    a2 = 2 * np.arctan2(d[:, 0], d[:, 1])              # compass: from north, clockwise
    c, sn = float(np.cos(a2).mean()), float(np.sin(a2).mean())
    return float(np.degrees(np.arctan2(sn, c)) / 2 % 180), float(np.hypot(c, sn))


def angle_between(a, b):
    """Smallest angle between two strike directions (0-90)."""
    return float(abs((a - b + 90) % 180 - 90))


def strike_label(az):
    names = ["N-S", "NNE-SSW", "NE-SW", "ENE-WSW", "E-W", "ESE-WNW", "SE-NW", "SSE-NNW"]
    return names[int(((az % 180) + 11.25) // 22.5) % 8]


def guide_label(ppb):
    for limit, name in GUIDE:
        if ppb < limit:
            return name
    return "strongly anomalous"


def narrative(stats, lay, pops, limits, z, x, y, unit_ppb, unit_name, floor=None):
    """The interpretation, as plain sentences. Each one has to be defensible from the numbers above."""
    u = unit_name.split(" ")[0]
    L = []
    L.append(f"**The survey.** {lay['n']:,} soil samples over {lay['width']:,.0f} by {lay['height']:,.0f} m "
             + (f"({lay['area_ha']:,.0f} ha)" if np.isfinite(lay["area_ha"]) else "")
             + (f", about {lay['spacing']:,.0f} m apart." if np.isfinite(lay["spacing"]) else ".")
             + (f" {lay['duplicates']} sample(s) share a position with another, which usually means a field "
                "duplicate or a typing error in the coordinates." if lay["duplicates"] else ""))

    med_ppb = stats["median_ppb"]
    L.append(f"**Gold levels.** Median {stats['median']:,.4g} {u} ({med_ppb:,.0f} ppb), highest "
             f"{stats['max']:,.4g} {u} ({stats['max_ppb']:,.0f} ppb), a contrast of {stats['contrast']:,.0f}× "
             f"between the peak and the median. On the usual rule of thumb for residual soil, the median of "
             f"this survey is already **{guide_label(med_ppb)}**"
             + (", so the whole grid sits inside a gold system rather than over barren ground; the "
                "thresholds below separate the better parts of it, not anomaly from background."
                if med_ppb >= 30 else "."))
    if floor:
        L.append(f"**Detection limit.** {floor['n']} samples sit exactly on {floor['value']:,.4g} {u}, so that "
                 "is the lab's lower limit. Those samples are 'at or below' that value, not equal to it, and "
                 "statistics on them are weaker than they look.")

    if pops.get("ok") and pops["n_pop"] == 2:
        lo_p, hi_p = pops["mean"]
        share = 100 * pops["weight"][1]
        L.append(f"**Two populations.** The assays are better described by two log-normal populations than by "
                 f"one: a lower one centred on {lo_p:,.4g} {u} holding about {100 - share:.0f}% of the samples, "
                 f"and a higher one centred on {hi_p:,.4g} {u} holding about {share:.0f}%. They cross at "
                 f"**{pops['crossover']:,.4g} {u}**, which is the statistically cleanest line between ordinary "
                 f"ground and mineralised ground here; {pops.get('n_upper', 0)} samples are above it. A break "
                 "like this is what a soil survey is run to find.")
    elif pops.get("ok"):
        L.append("**One population.** The assays fit a single log-normal population, with no separate "
                 "higher-grade group. That happens when the grid is entirely in background, entirely inside "
                 "one mineralised zone, or when the sampling is too sparse to resolve the two. Judge the "
                 "thresholds below as percentiles of this one population.")
    else:
        L.append("**Populations.** Too few samples (under 30) to test for a separate mineralised population.")

    if len(z):
        n_multi = int((z.samples >= 3).sum())
        n_pair = int((z.samples == 2).sum())
        n_single = int((z.samples == 1).sum())
        parts = [f"{n} with {w}" for n, w in ((n_multi, "three or more samples"), (n_pair, "two"),
                                              (n_single, "one")) if n]
        L.append(f"**Anomalous ground.** {int(z.samples.sum())} samples reach the anomalous threshold "
                 f"({limits[1]:,.4g} {u}), forming **{len(z)} zone(s)**: " + ", ".join(parts) + ". "
                 + ("Zones of three or more samples are the ones worth spending on. " if n_multi else
                    "None of them has three or more samples, which is too little to call a target yet. ")
                 + ("A single high sample with ordinary neighbours is as often a nugget in the sieve or "
                    "contamination as it is a discovery, so re-sample those before anything else."
                    if n_single else ""))
        best = z.iloc[0]
        az = (f", elongated {strike_label(best.azimuth)} ({best.azimuth:.0f}°) over {best.length:,.0f} m"
              if np.isfinite(best.get("azimuth", np.nan)) and best.get("elongation", 0) > 1.5 else "")
        edges = open_edges(best, x, y, lay["spacing"])
        L.append(f"**Best zone, {best.zone}.** {int(best.samples)} sample(s) centred on "
                 f"{best.centre_x:,.0f} E, {best.centre_y:,.0f} N{az}. Peak {best.peak:,.4g} {u} "
                 f"({best.peak * unit_ppb:,.0f} ppb) at {best.peak_x:,.0f} E, {best.peak_y:,.0f} N."
                 + (f" It runs to the {' and '.join(edges)} edge of the grid, so it is open that way and the "
                    "survey should be extended before drilling." if edges else
                    " It closes inside the grid, so its size is real and not a limit of the survey."))
        trends = z[(z.samples >= 3) & np.isfinite(z.azimuth) & (z.elongation > 1.5)]
        if len(trends) >= 1:
            a2 = 2 * np.radians(trends.azimuth.to_numpy())
            spread = float(np.hypot(np.cos(a2).mean(), np.sin(a2).mean()))
            mean_az = float(np.degrees(np.arctan2(np.sin(a2).mean(), np.cos(a2).mean())) / 2 % 180)
            samp_az, samp_strength = sampling_direction(x, y)
            if len(trends) == 1:
                line = (f"**Trend.** The one zone with a shape to measure runs {strike_label(mean_az)} "
                        f"({mean_az:.0f}°). One zone is not a trend yet, but it is the direction to test.")
            else:
                line = (f"**Trend.** {len(trends)} zones are elongated, averaging {strike_label(mean_az)} "
                        f"({mean_az:.0f}°)"
                        + (", and they agree closely, so the gold follows one structural direction. Orient "
                           "trenches and drill lines across it." if spread > 0.8 else
                           ", but they point in different directions, so more than one structure may be "
                           "carrying the gold, or the sample spacing is too coarse to resolve the trend."))
            if samp_strength > 0.5 and angle_between(mean_az, samp_az) < 25:
                line += (f" Treat this with care: the samples themselves lie along {strike_label(samp_az)} "
                         "lines, so an anomaly will tend to look stretched that way whatever its real shape.")
            L.append(line)
    else:
        L.append(f"**Anomalous ground.** No sample reaches the anomalous threshold ({limits[1]:,.4g} {u}). "
                 "Either this ground is barren, or the gold is covered by transported soil that the sampling "
                 "cannot see through.")

    sp = lay["spacing"]
    if np.isfinite(sp) and len(z):
        wide = z[z.samples >= 3]
        steps = []
        if len(wide):
            steps.append(f"infill the {len(wide)} zone(s) of three or more samples at about {sp / 2:,.0f} m, "
                         "half the present spacing, to find their edges")
        if int((z.samples <= 2).sum()):
            steps.append(f"re-sample beside each of the {int((z.samples <= 2).sum())} one- and two-sample highs "
                         "to see whether they are real")
        steps.append("where a zone holds after infill, trench across its long axis before drilling")
        L.append("**What to do next.** " + "; ".join(steps).capitalize() +
                 ". Soil gold creeps downslope, so check the topography before placing a trench: the source "
                 "is upslope of the anomaly, by more on a steep hillside than on a flat.")
    return L


def report_table(ids, x, y, values, cls, lon=None, lat=None, unit_ppb=1.0, zone_of=None):
    """Every sample, with its class and its zone: the table the page offers for download."""
    out = pd.DataFrame({"Sample": np.asarray(ids, str), "Easting": np.round(x, 1), "Northing": np.round(y, 1)})
    if lon is not None:
        out["Latitude"] = np.round(lat, 6)
        out["Longitude"] = np.round(lon, 6)
    out["Au"] = values
    out["Au_ppb"] = np.round(np.asarray(values, float) * unit_ppb, 1)
    out["Class"] = [CLASSES[c] if c >= 0 else "" for c in cls]
    if zone_of is not None:
        out["Zone"] = zone_of
    return out


# ---------- figures ----------
def _fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def probability_figure(values, pops, limits, unit="ppm", title="Log-probability plot"):
    """The classic test for a second population: a straight line is one population, a break is two."""
    plt = _fig()
    v, z = probability_points(values)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    if not v.size:
        ax.text(0.5, 0.5, "Not enough values", ha="center")
        return fig
    ax.plot(v, z, "o", ms=4, color="#333", label=f"{v.size} samples")
    if pops.get("ok") and pops["n_pop"] == 2:
        grid = np.logspace(np.log10(v.min()), np.log10(v.max()), 200)
        for mean, sd, w, c, name in zip(pops["mean"], pops["sd"], pops["weight"],
                                        ("#2471a3", "#c0392b"), ("background", "mineralised")):
            ax.plot(grid, (np.log10(grid) - np.log10(mean)) / sd, "-", lw=1.8, color=c, alpha=0.85,
                    label=f"{name}: centre {mean:,.4g} {unit}, {100 * w:.0f}% of samples")
        if np.isfinite(pops.get("crossover", np.nan)):
            ax.axvline(pops["crossover"], color="#7d3c98", lw=2, ls="--",
                       label=f"populations cross at {pops['crossover']:,.4g} {unit}")
    for lim, name, c in zip(limits, CLASSES[1:], CLASS_COLORS[1:]):
        if np.isfinite(lim):
            ax.axvline(lim, color=c, lw=1.2, ls=":")
            ax.text(lim, z.max(), f" {name}", rotation=90, va="top", fontsize=7, color=c)
    ax.set_xscale("log")
    ax.set_ylim(z.min() - 0.5, z.max() + 1.2)      # the fitted lines run far past the data; show the data
    ax.set_xlabel(f"Gold ({unit}), log scale")
    ax.set_ylabel("Normal score of cumulative frequency")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    return fig


def zone_map_figure(x, y, values, cls, z, unit="ppm", title="Gold in soil: classes and anomaly zones"):
    """Samples coloured by class, with the anomaly zones ringed and labelled."""
    plt = _fig()
    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    v = np.asarray(values, float)
    size = 24 + 170 * (pd.Series(v).rank(pct=True).fillna(0.2).to_numpy() ** 2)
    for i, (name, col) in enumerate(zip(CLASSES, CLASS_COLORS)):
        m = cls == i
        if m.any():
            ax.scatter(np.asarray(x)[m], np.asarray(y)[m], s=size[m], c=col, edgecolors="k", linewidths=0.4,
                       zorder=3 + i, label=f"{name} ({m.sum()})")
    for r in z.itertuples():
        pad = 0.18 * max(r.x1 - r.x0, r.y1 - r.y0, 30)
        ax.add_patch(Ellipse(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2), r.x1 - r.x0 + 2 * pad,
                             r.y1 - r.y0 + 2 * pad, facecolor="none", edgecolor="#7d3c98", lw=1.8, ls="--",
                             zorder=8))
        ax.annotate(f"{r.zone} · {r.peak:,.3g}", (r.peak_x, r.peak_y), xytext=(8, 8),
                    textcoords="offset points", fontsize=8, weight="bold", color="#5b2c6f", zorder=9)
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"{title}  ({unit})", fontsize=10)
    ax.grid(alpha=0.25)
    ax.margins(0.1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8, title="Class",
              title_fontsize=8)
    fig.tight_layout()
    return fig
