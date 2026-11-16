"""Structural geology: field measurements of planes (shears, faults, veins, bedding) read for gold targeting.

Three things a list of strikes and dips is worth reading for:
  * which way the structures really face. A strike on its own leaves two possible dip directions, so the
    recorded dip direction (often rounded to the nearest 45 deg in the field) is used to settle it, and
    measurements whose dip direction cannot belong to their strike are flagged rather than used.
  * how many sets there are. Planes group into sets, and the set a shear belongs to usually says more about
    whether it carries gold than its individual reading does.
  * where two sets cross. In orogenic gold the ore shoot commonly follows the line where two structures
    intersect, and that line's plunge is what a drill hole has to be aimed down.

No Streamlit inside.
"""
import numpy as np
import pandas as pd

STRUCT_COLS = ["structure", "struct", "type", "feature", "description", "kind"]
STOP_COLS = ["stop", "station", "site", "point", "id", "sample"]
DIPDIR_COLS = ["azimuth", "dipdir", "dip_direction", "dip dir", "dipdirection", "facing"]
ELEV_COLS = ["elevation", "elev", "rl", "height", "z"]
# structure names that carry gold in orogenic systems, in the order they are worth drilling
FAVOURABLE = ["shear", "quartz", "vein", "lode", "reef", "breccia", "mylonite", "fault", "contact"]


def guess(df, names):
    low = {c: str(c).strip().lower() for c in df.columns}
    for n in names:
        for c, l in low.items():
            if l == n:
                return c
    for n in names:
        for c, l in low.items():
            if len(n) > 3 and n in l:
                return c
    return None


def resolve(strike, dip, dipdir=None, tol=45.0):
    """Settle which way each plane dips.

    A strike allows two dip directions, 180 deg apart. The recorded dip direction picks one; because it is
    often rounded to the nearest 45 deg, anything within `tol` is accepted and the exact value is taken from
    the strike. Returns a frame with the right-hand-rule strike, the dip direction, and `ok` False where the
    recorded dip direction fits neither choice, which means a reading to go back and check.
    """
    s = np.asarray(strike, float) % 360
    d = np.asarray(dip, float)
    n = len(s)
    out = pd.DataFrame({"strike_in": s, "dip": d, "strike": s % 180, "dipdir": (s + 90) % 360,
                        "ok": np.isfinite(s) & np.isfinite(d) & (d >= 0) & (d <= 90),
                        "offset": np.full(n, np.nan), "assumed": np.ones(n, bool)})
    if dipdir is None:
        return out
    a = np.asarray(dipdir, float) % 360
    have = np.isfinite(a)
    for i in np.flatnonzero(have & out.ok.to_numpy()):
        cand = np.array([(s[i] + 90) % 360, (s[i] - 90) % 360])
        err = np.abs((cand - a[i] + 180) % 360 - 180)
        k = int(np.argmin(err))
        out.loc[i, "dipdir"] = cand[k]
        out.loc[i, "strike"] = (cand[k] - 90) % 360        # right-hand rule: dip is 90 deg clockwise of strike
        out.loc[i, "offset"] = err[k]
        out.loc[i, "assumed"] = False
        if err[k] > tol:
            out.loc[i, "ok"] = False
    return out


def pole_vectors(strike, dip):
    """Downward unit normals to the planes (right-hand-rule strike), as (n, 3) east/north/down."""
    s, d = np.radians(np.asarray(strike, float)), np.radians(np.asarray(dip, float))
    trend = s - np.pi / 2                                   # the pole points away from the dip direction
    plunge = np.pi / 2 - d
    return np.column_stack([np.cos(plunge) * np.sin(trend), np.cos(plunge) * np.cos(trend), np.sin(plunge)])


def plane_of(vec):
    """Inverse of pole_vectors: a downward normal back to (right-hand-rule strike, dip)."""
    v = np.asarray(vec, float)
    v = v / np.linalg.norm(v)
    if v[2] < 0:
        v = -v
    plunge = np.arcsin(np.clip(v[2], -1, 1))
    trend = np.arctan2(v[0], v[1]) % (2 * np.pi)
    dip = np.degrees(np.pi / 2 - plunge)
    strike = (np.degrees(trend) + 90) % 360
    return float(strike), float(dip)


def line_of(vec):
    """A vector as (trend, plunge) of the downward-pointing direction."""
    v = np.asarray(vec, float)
    v = v / np.linalg.norm(v)
    if v[2] < 0:
        v = -v
    return float(np.degrees(np.arctan2(v[0], v[1])) % 360), float(np.degrees(np.arcsin(np.clip(v[2], -1, 1))))


def fisher(vectors):
    """Mean orientation of a group of axes, with Fisher's concentration and 95% cone.

    Axes have no up or down, so each vector is flipped onto the same side of the mean before averaging.
    kappa is how tightly the group clusters (above about 20 is tight); a95 is the half-angle of the cone
    the true mean lies in, 19 times out of 20.
    """
    v = np.asarray(vectors, float)
    n = len(v)
    if n == 0:
        return None
    m = v.mean(0)
    for _ in range(12):                                     # axial: flip each vector towards the mean
        flip = np.where(v @ m < 0, -1.0, 1.0)[:, None]
        m_new = (v * flip).mean(0)
        if np.linalg.norm(m_new - m) < 1e-10:
            m = m_new
            break
        m = m_new
    R = float(np.linalg.norm(m))
    mean = m / (R or 1)
    out = {"n": int(n), "mean": mean, "R": R, "kappa": np.nan, "a95": np.nan,
           "scatter": float(np.degrees(np.arccos(np.clip(R, 0, 1))))}
    if n > 1 and R < 1:
        out["kappa"] = float((n - 1) / (n * (1 - R)))
        if n * R > 0:
            c = 1 - ((n - n * R) / (n * R)) * ((1 / 0.05) ** (1 / (n - 1)) - 1)
            out["a95"] = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return out


def find_sets(strike, dip, max_sets=3, min_members=3, seed=0):
    """Split the planes into structural sets by clustering their poles.

    Spherical k-means on the poles, treating them as axes. The number of sets is the one that separates the
    data best (silhouette on angles between poles), and a split that would leave a set with fewer than
    `min_members` readings is not taken: three readings is already a weak set.
    """
    P = pole_vectors(strike, dip)
    ok = np.isfinite(P).all(1)
    P = P[ok]
    n = len(P)
    if n < min_members:
        return [], np.full(len(strike), -1)
    ang = np.degrees(np.arccos(np.clip(np.abs(P @ P.T), 0, 1)))     # axial angle between every pair
    best = (0.0, np.zeros(n, int), 1)
    for k in range(2, min(max_sets, n // min_members) + 1):
        lab = _spherical_kmeans(P, k, seed)
        counts = np.bincount(lab, minlength=k)
        if (counts < min_members).any():
            continue
        if not _separated(P, lab, k):
            continue
        score = _silhouette(ang, lab)
        if score > best[0] + 0.02:               # among the splits that pass, take the cleanest
            best = (score, lab, k)
    _, lab, k = best
    sets = []
    for j in range(k):
        f = fisher(P[lab == j])
        s, d = plane_of(f["mean"])
        sets.append({"set": chr(ord("A") + j), "n": f["n"], "strike": s % 180, "dip": d, "dipdir": (s + 90) % 360,
                     "kappa": f["kappa"], "a95": f["a95"], "scatter": f["scatter"], "pole": f["mean"]})
    order = np.argsort([-s["n"] for s in sets])
    sets = [dict(sets[i], set=chr(ord("A") + r)) for r, i in enumerate(order)]
    remap = {int(old): r for r, old in enumerate(order)}
    full = np.full(len(strike), -1)
    full[np.flatnonzero(ok)] = [remap[int(v)] for v in lab]
    return sets, full


MIN_APART = 25.0        # degrees between mean poles before two groups count as different sets


def _separated(P, lab, k, min_apart=MIN_APART, ratio=2.2):
    """Are these really different sets, or one cloud cut in half?

    k-means always finds a split, so it has to be judged afterwards. Two sets count as different when
    their mean poles stand at least `min_apart` apart on the sphere AND that separation is larger than
    the scatter of the readings inside them by `ratio`. Both tests are needed: a tight cloud can be cut
    into halves whose means are well clear of the scatter but only a few degrees apart, while a wide
    cloud can be cut into halves tens of degrees apart that are still obviously one population.
    """
    means, scatter = [], []
    for j in range(k):
        f = fisher(P[lab == j])
        means.append(f["mean"])
        scatter.append(f["scatter"])
    for i in range(k):
        for j in range(i + 1, k):
            apart = np.degrees(np.arccos(np.clip(abs(means[i] @ means[j]), 0, 1)))
            if apart < min_apart or apart < ratio * np.mean([scatter[i], scatter[j]]):
                return False
    return True


def _spherical_kmeans(P, k, seed=0, iters=60):
    rng = np.random.default_rng(seed)
    C = P[rng.choice(len(P), k, replace=False)]
    lab = np.zeros(len(P), int)
    for _ in range(iters):
        sim = np.abs(P @ C.T)                               # axial: direction of the pole does not matter
        new = sim.argmax(1)
        if (new == lab).all() and _:
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                f = fisher(P[m])
                C[j] = f["mean"]
    return lab


def _silhouette(ang, lab):
    s = []
    for i in range(len(lab)):
        same = (lab == lab[i])
        same[i] = False
        if not same.any():
            continue
        a = ang[i, same].mean()
        b = min(ang[i, lab == j].mean() for j in set(lab) if j != lab[i])
        s.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(s)) if s else 0.0


def intersections(sets):
    """Where each pair of sets crosses: the line two planes share, as trend and plunge.

    In orogenic gold the ore shoot often follows this line, and its plunge is the angle a drill hole has
    to follow to stay in the shoot.
    """
    out = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            v = np.cross(sets[i]["pole"], sets[j]["pole"])
            norm = np.linalg.norm(v)
            if norm < 1e-9:
                continue
            trend, plunge = line_of(v / norm)
            angle = float(np.degrees(np.arccos(np.clip(abs(sets[i]["pole"] @ sets[j]["pole"]), 0, 1))))
            out.append({"sets": f"{sets[i]['set']} × {sets[j]['set']}", "trend": trend, "plunge": plunge,
                        "angle": 180 - angle if angle > 90 else angle})
    return out


def stereonet_xy(vectors):
    """Equal-area (Schmidt) lower-hemisphere projection of downward vectors."""
    v = np.asarray(vectors, float)
    v = np.where((v[:, 2] < 0)[:, None], -v, v)
    plunge = np.arcsin(np.clip(v[:, 2], -1, 1))
    trend = np.arctan2(v[:, 0], v[:, 1])
    r = np.sqrt(2) * np.sin((np.pi / 2 - plunge) / 2)
    return r * np.sin(trend), r * np.cos(trend)


def strike_label(az):
    names = ["N-S", "NNE-SSW", "NE-SW", "ENE-WSW", "E-W", "ESE-WNW", "SE-NW", "SSE-NNW"]
    return names[int(((az % 180) + 11.25) // 22.5) % 8]


def dip_word(d):
    return "shallow" if d < 35 else "moderate" if d < 60 else "steep" if d < 80 else "near-vertical"


def plunge_word(p):
    return "shallow" if p < 25 else "moderate" if p < 55 else "steep"


def spread(x, y, ids=None):
    """Where the stations are and how far apart, plus any that sit far from the rest."""
    from scipy.spatial import cKDTree
    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    uniq = np.unique(np.round(pts, 2), axis=0)
    out = {"stations": len(uniq), "width": float(np.ptp(pts[:, 0])), "height": float(np.ptp(pts[:, 1])),
           "spacing": np.nan, "outliers": []}
    if len(uniq) > 2:
        d, _ = cKDTree(uniq).query(uniq, k=2)
        nn = d[:, 1]
        out["spacing"] = float(np.median(nn))
        far = nn > 4 * np.median(nn)
        if far.any() and ids is not None:
            for p in uniq[far]:
                m = np.argmin(np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1]))
                out["outliers"].append((str(np.asarray(ids)[m]), float(p[0]), float(p[1])))
    return out


def classify_types(names):
    """Group the free-text structure names into the kinds that matter for gold."""
    out = []
    for v in np.asarray(names, str):
        low = str(v).strip().lower()
        hit = next((f for f in FAVOURABLE if f in low), None)
        out.append(hit.capitalize() if hit else (str(v).strip().capitalize() or "Unrecorded"))
    return np.array(out, object)


def bisector(sets):
    """Acute bisector of two planes: the shortening direction if the two formed as a conjugate pair."""
    if len(sets) < 2:
        return None
    a, b = sets[0]["pole"], sets[1]["pole"]
    b = b if a @ b >= 0 else -b
    v = a + b                                               # bisects the poles = bisects the planes obtusely
    if np.linalg.norm(v) < 1e-9:
        return None
    trend, plunge = line_of(v / np.linalg.norm(v))
    return {"trend": trend, "plunge": plunge}


def narrative(res, types, sets, labels, inters, sp, structures=None):
    """The interpretation, as plain sentences, each one backed by the numbers above."""
    L = []
    n_ok = int(res.ok.sum())
    bad = int((~res.ok).sum())
    assumed = int(res.assumed[res.ok].sum())
    L.append(f"**The measurements.** {len(res)} readings at {sp['stations']} stations over "
             f"{sp['width']:,.0f} by {sp['height']:,.0f} m"
             + (f", about {sp['spacing']:,.0f} m apart" if np.isfinite(sp["spacing"]) else "") + ". "
             + (f"{bad} reading(s) were set aside because the recorded dip direction cannot belong to the "
                "recorded strike, so one of the two was written down wrong; they are listed below and are "
                "worth re-checking in your notebook. " if bad else "")
             + (f"{assumed} reading(s) have no dip direction, so the right-hand rule was assumed. "
                if assumed else ""))
    if sp["outliers"]:
        names = ", ".join(f"{s} ({x:,.0f} E, {y:,.0f} N)" for s, x, y in sp["outliers"][:3])
        L.append(f"**Away from the rest.** {names} sits far from the other stations. One reading on its own "
                 "says little about the structure there; treat it as a pointer to ground that has not been "
                 "mapped rather than as a measurement of a set.")
    if types is not None and len(types):
        c = pd.Series(types).value_counts()
        parts = ", ".join(f"{v} {k.lower()}" for k, v in c.items())
        shears = int(sum(v for k, v in c.items() if k.lower() in ("shear", "mylonite", "breccia")))
        L.append(f"**What was mapped.** {parts}. "
                 + ("Shears are the structures that carry gold in an orogenic system: they are where rock has "
                    "broken and fluid has passed. Faults matter mostly for where they cut, offset or dilate "
                    "the shears." if shears else
                    "No shear zones were recorded, so these structures bear on where gold could sit rather "
                    "than hosting it directly."))
    if not sets:
        L.append("**Sets.** Too few usable readings to separate structural sets; at least three per set are "
                 "needed. Add measurements before reading a pattern into these.")
        return L
    main = sets[0]
    tight = ("tightly, so this is one structure or one consistent fabric" if main["kappa"] >= 20 else
             "loosely, so it is a general direction rather than a single plane" if main["kappa"] >= 8 else
             "very loosely, so treat it as a rough trend only")
    L.append(f"**The main set ({main['set']}).** {main['n']} of {n_ok} readings strike "
             f"**{main['strike']:.0f}° ({strike_label(main['strike'])})** and dip **{main['dip']:.0f}° "
             f"{dip_word(main['dip'])} towards {main['dipdir']:.0f}°**. They group {tight} "
             f"(κ = {main['kappa']:.0f}, mean within ±{main['a95']:.0f}°). This is the direction to cross with "
             "trenches and drill lines: a hole along the strike of a shear can stay in it and still miss it.")
    for s in sets[1:]:
        L.append(f"**Second set ({s['set']}).** {s['n']} readings strike **{s['strike']:.0f}° "
                 f"({strike_label(s['strike'])})**, dipping {s['dip']:.0f}° {dip_word(s['dip'])} towards "
                 f"{s['dipdir']:.0f}° (κ = {s['kappa']:.0f}). A second direction is what makes ground "
                 "prospective: gold collects where two structures meet and the rock is pulled open.")
    if inters:
        i = inters[0]
        L.append(f"**Where the sets cross.** Sets {i['sets']} meet along a line plunging "
                 f"**{i['plunge']:.0f}° ({plunge_word(i['plunge'])}) towards {i['trend']:.0f}°**, with "
                 f"{i['angle']:.0f}° between the two planes. In orogenic gold the ore shoot commonly follows "
                 "this line, so that plunge is the direction to follow down-dip once a shoot is found, and "
                 "the places where mapped structures of both sets come together are the first ground to "
                 "trench.")
        b = bisector(sets)
        if b and 25 <= i["angle"] <= 75:
            L.append(f"**Shortening direction.** The {i['angle']:.0f}° between the two sets is the angle a "
                     "conjugate pair makes, so if they formed together the rock was being shortened towards "
                     f"about **{b['trend']:.0f}°**. Worth testing against the regional structure: if it "
                     "agrees, the two sets are one event and the intersection above is a real ore-shoot "
                     "direction; if not, they are separate events and the younger one may simply cut the "
                     "older.")
    steep = int((res.dip[res.ok] >= 60).sum())
    if steep >= 0.6 * max(n_ok, 1):
        L.append(f"**Dips.** {steep} of {n_ok} readings dip 60° or more. Steep structures stay in roughly the "
                 "same place with depth, so a surface anomaly sits near its source and vertical or steeply "
                 "angled holes will cross them at a poor angle; incline holes against the dip.")
    elif steep <= 0.3 * max(n_ok, 1):
        L.append(f"**Dips.** Only {steep} of {n_ok} readings dip 60° or more, so these structures are shallow "
                 "to moderate. A shallow structure moves sideways with depth, so its source can lie well to "
                 "one side of the surface anomaly, up-dip of it.")
    return L


# ---------- figures ----------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def great_circle(strike, dip, n=181):
    """Points of a plane's great circle on an equal-area lower-hemisphere net."""
    s, d = np.radians(strike), np.radians(dip)
    t = np.linspace(0, np.pi, n)
    # a plane as the set of lines in it: rotate a horizontal line about the pole
    along = np.array([np.sin(s), np.cos(s), 0.0])
    down = np.array([np.sin(s + np.pi / 2) * np.cos(np.pi / 2 - d), np.cos(s + np.pi / 2) * np.cos(np.pi / 2 - d),
                     np.sin(np.pi / 2 - d)])
    v = np.outer(np.cos(t), along) + np.outer(np.sin(t), down)
    return stereonet_xy(v)


SET_COLORS = ["#c0392b", "#2471a3", "#1e8449", "#7d3c98"]


def stereonet_figure(res, labels, sets, inters, title="Poles to planes (equal area, lower hemisphere)"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    t = np.linspace(0, 2 * np.pi, 361)
    ax.plot(np.cos(t), np.sin(t), "k-", lw=1.2)
    for a in (0, 90):
        ax.plot([-np.sin(np.radians(a)), np.sin(np.radians(a))], [-np.cos(np.radians(a)), np.cos(np.radians(a))],
                color="#ddd", lw=0.7)
    ok = res.ok.to_numpy()
    P = pole_vectors(res.strike[ok], res.dip[ok])
    px, py = stereonet_xy(P)
    for j, s in enumerate(sets):
        m = labels == j
        c = SET_COLORS[j % len(SET_COLORS)]
        if m.any():
            ax.scatter(px[m], py[m], s=42, c=c, edgecolors="k", linewidths=0.5, zorder=4,
                       label=f"Set {s['set']}: {s['strike']:.0f}°/{s['dip']:.0f}° ({s['n']})")
        gx, gy = great_circle(s["strike"], s["dip"])
        ax.plot(gx, gy, "-", color=c, lw=1.8, alpha=0.8, zorder=3)
        mx, my = stereonet_xy(np.array([s["pole"]]))
        ax.scatter(mx, my, marker="s", s=90, facecolor=c, edgecolors="k", linewidths=1.1, zorder=5)
    other = labels < 0
    if other.any():
        ax.scatter(px[other], py[other], s=28, c="#999", edgecolors="k", linewidths=0.4, zorder=2,
                   label="unassigned")
    for i in inters[:1]:
        v = np.array([np.cos(np.radians(i["plunge"])) * np.sin(np.radians(i["trend"])),
                      np.cos(np.radians(i["plunge"])) * np.cos(np.radians(i["trend"])),
                      np.sin(np.radians(i["plunge"]))])
        ix, iy = stereonet_xy(np.array([v]))
        ax.scatter(ix, iy, marker="*", s=320, c="#f1c40f", edgecolors="k", linewidths=0.9, zorder=6,
                   label=f"intersection {i['trend']:.0f}°/{i['plunge']:.0f}°")
    ax.text(0, 1.07, "N", ha="center", fontsize=11, weight="bold")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(0.98, 1.0), frameon=False, fontsize=7.5)
    fig.tight_layout()
    return fig


def map_figure(x, y, res, labels, types=None, sets=None, background=None, ids=None,
               title="Structural measurements"):
    """Stations with a tick along each plane's strike, coloured by set."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    if background is not None:
        img, (x0, x1, y0, y1) = background
        ax.imshow(np.asarray(img), extent=[x0, x1, y0, y1], alpha=0.8, zorder=0)
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = res.ok.to_numpy()
    span = max(np.ptp(x), np.ptp(y), 1.0)
    half = 0.022 * span
    seen = set()
    pos = {int(v): k for k, v in enumerate(np.flatnonzero(ok))}   # row -> its place in `labels`
    for i in np.flatnonzero(ok):
        j = int(labels[pos[int(i)]])
        c = SET_COLORS[j % len(SET_COLORS)] if j >= 0 else "#888"
        a = np.radians(res.strike.iloc[i])
        dx, dy = half * np.sin(a), half * np.cos(a)
        lab = None
        if sets and j >= 0 and j not in seen:
            seen.add(j)
            lab = f"Set {sets[j]['set']}: {sets[j]['strike']:.0f}°/{sets[j]['dip']:.0f}°"
        ax.plot([x[i] - dx, x[i] + dx], [y[i] - dy, y[i] + dy], "-", color=c, lw=2.4, zorder=4, label=lab)
        d = np.radians(res.dipdir.iloc[i])                   # short tick towards the dip
        ax.plot([x[i], x[i] + 0.4 * half * np.sin(d)], [y[i], y[i] + 0.4 * half * np.cos(d)], "-", color=c,
                lw=1.6, zorder=4)
    bad = ~ok
    if bad.any():
        ax.scatter(x[bad], y[bad], marker="x", s=60, c="k", linewidths=1.4, zorder=5, label="set aside")
    ax.scatter(x[ok], y[ok], s=12, c="k", zorder=5)
    if ids is not None:
        for xi, yi, lb in zip(x, y, np.asarray(ids, str)):
            ax.annotate(lb, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=6.5, zorder=6)
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.margins(0.08)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
