"""Step 1: four optional upload slots, each file configured in its own panel."""
import io

import numpy as np
import pandas as pd
import streamlit as st

from geoai import coords, geochem, geology, geophysics, imagegrid, readers
from geoai.coords import parse_column

from .common import ss

SLOT_INFO = {
    "geophysics": ("Magnetic / IP (induced polarisation)",
                   "Magnetic survey points (X, Y, nT), IP chargeability or resistivity as plan points or as sections "
                   "(line, station, n-level or depth), gridded GeoTIFFs, or a picture of a map."),
    "geology": ("Geological data",
                "Mapping / outcrop points with lithology and optional strike and dip, or a geological map "
                "(picture or GeoTIFF) to use as a background."),
    "remote": ("Remote sensing data",
               "GeoTIFFs exported from Section 1 (alteration indices, lineament density, prospectivity…), "
               "CSV points with values, or a picture of a map."),
    "geochem": ("Geochemical data",
                "Soil, rock-chip or stream samples: X, Y and element columns (Au, As, Sb, Bi, Cu…). "
                "Values like '<5' or negative codes are treated as below detection."),
}
NONE = "(none)"


@st.cache_data(show_spinner=False, max_entries=40)
def _read(name, raw, sheet):
    buf = io.BytesIO(raw)
    buf.name = name
    return readers.read_upload(buf, sheet)


@st.cache_data(show_spinner=False, max_entries=40)
def _sheets(name, raw):
    return readers.excel_sheets(io.BytesIO(raw))


def _crs_select(label, values_x, values_y, key, help_=None):
    opts = list(coords.CRS_OPTIONS) + ["Other EPSG code…"]
    geo = coords.looks_geographic(pd.Series(values_x).dropna(), pd.Series(values_y).dropna()) if len(values_x) else False
    default_i = 0 if geo else opts.index("WGS84 / UTM zone 36S (EPSG:32736)")
    choice = st.selectbox(label, opts, default_i, key=key, help=help_ or "Auto-detected; change if needed.")
    if choice in coords.CRS_OPTIONS:
        return coords.CRS_OPTIONS[choice]
    return int(st.number_input("EPSG code", 1024, 999999, 32736, key=key + "_epsg"))


def _pick(label, cols, guess, key, optional=False, help_=None):
    options = ([NONE] if optional else []) + list(cols)
    idx = options.index(guess) if guess in options else 0
    v = st.selectbox(label, options, idx, key=key, help=help_)
    return None if v == NONE else v


def _guess_col(cols, words):
    """First column matching a word. Short words (x, y, z, n, ip) must be the whole column name."""
    low = {c: str(c).strip().lower() for c in cols}
    for w in words:
        for c, l in low.items():
            if (l == w) if len(w) <= 2 else (w in l):
                return c
    return None


def _guess_survey(cols):
    if _guess_col(cols, ["charg", "mv/v", "msec", "ip"]):
        return 1
    if _guess_col(cols, ["resist", "ohm", "rho"]):
        return 2
    return 0


def _xy_cols(df, key, geom_hint=""):
    cols = list(df.columns)
    gx, gy = coords.guess_columns(df)
    c1, c2 = st.columns(2)
    xcol = c1.selectbox(f"X / Easting / Longitude{geom_hint}", cols, cols.index(gx) if gx in cols else 0, key=key + "_x")
    ycol = c2.selectbox("Y / Northing / Latitude", cols, cols.index(gy) if gy in cols else min(1, len(cols) - 1),
                        key=key + "_y")
    return xcol, ycol


def _image_extent(key, name):
    aoi = ss.get("rs_aoi")
    d = aoi["bounds"] if aoi else (0.0, 0.0, 0.0, 0.0)
    st.caption("A picture has no coordinates: give the map's edges" +
               (" (pre-filled from the Section 1 area)." if aoi else "."))
    c1, c2, c3, c4 = st.columns(4)
    xmin = c1.number_input("West (min X)", value=float(d[0]), key=key + "_x0", format="%.2f")
    xmax = c2.number_input("East (max X)", value=float(d[1]), key=key + "_x1", format="%.2f")
    ymin = c3.number_input("South (min Y)", value=float(d[2]), key=key + "_y0", format="%.2f")
    ymax = c4.number_input("North (max Y)", value=float(d[3]), key=key + "_y1", format="%.2f")
    epsg = _crs_select("Coordinate system of the edges", [xmin, xmax], [ymin, ymax], key + "_crs")
    if xmax <= xmin or ymax <= ymin:
        st.warning("Enter the map edges (east must be greater than west, north greater than south).")
        return None, epsg
    return (xmin, xmax, ymin, ymax), epsg


def configure(slot, f, idx, survey=None):
    """Show the settings for one uploaded file; return a spec dict, or None if it can't be used yet."""
    raw = f.getvalue()
    key = f"cfg_{slot}_{idx}_{f.name}"
    sheet = None
    if f.name.lower().endswith((".xls", ".xlsx")):
        try:
            sheets = _sheets(f.name, raw)
        except Exception as e:
            st.error(f"Could not open the workbook: {e}")
            return None
        if len(sheets) > 1:
            sheet = st.selectbox("Sheet", sheets, key=key + "_sheet")
    try:
        res = _read(f.name, raw, sheet)
    except Exception as e:
        st.error(f"Could not read {f.name}: {e}")
        return None
    spec = {"slot": slot, "name": f.name, "kind": res["kind"], "data": res}

    if res["kind"] == "image":
        st.image(res["image"], width=240)
        if slot in ("geophysics", "remote"):
            scale = st.selectbox("Colour scale of the picture", imagegrid.SCALES, key=key + "_scale",
                                 help="Lets GeoAI turn colours back into relative values (0 = low, 1 = high).")
            spec["scale"] = scale
            if slot == "geophysics" and not scale.startswith("Reference"):
                spec["survey"] = survey or st.selectbox("What the map shows", geophysics.SURVEYS, key=key + "_survey")
            if slot == "remote":
                spec["layer_name"] = st.text_input("Layer name", f.name.rsplit(".", 1)[0], key=key + "_lname")
        else:
            st.caption("Used as a reference background map.")
        ext, epsg = _image_extent(key, f.name)
        if ext is None:
            return None
        spec.update(extent=ext, epsg=epsg)
        return spec

    if res["kind"] == "raster":
        layers = list(res["layers"])
        a0 = res["layers"][layers[0]]
        st.caption(f"GeoTIFF · {len(layers)} band(s) · {a0.shape[1]}×{a0.shape[0]} cells · EPSG:{res.get('epsg')}")
        if slot == "geophysics":
            spec["survey"] = survey or st.selectbox("What the grid shows", geophysics.SURVEYS, key=key + "_survey")
            spec["band"] = st.selectbox("Band", layers, key=key + "_band")
        elif slot == "remote":
            spec["bands"] = st.multiselect("Bands to use as evidence", layers, layers, key=key + "_bands")
            if not spec["bands"]:
                st.warning("Choose at least one band.")
                return None
        elif slot == "geology":
            st.caption("Used as a reference background map.")
        else:
            st.warning("Geochemical data must be a table of samples (CSV / Excel), not a raster.")
            return None
        if not res.get("epsg"):
            st.warning("This GeoTIFF's coordinate system has no EPSG code; it may not line up with other data.")
        return spec

    df = res["df"]
    st.caption(f"{len(df):,} rows · {len(df.columns)} columns")
    st.dataframe(df.head(4).astype(str), height=150, width="stretch")
    cols = list(df.columns)

    if slot == "geophysics":
        spec["survey"] = survey or st.selectbox("Survey type", geophysics.SURVEYS, _guess_survey(cols),
                                                key=key + "_survey")
        vguess = _guess_col(cols, ["mag", "tmi", "nt", "charg", "mv/v", "resist", "ohm", "value", "rho"])
        if survey and survey.startswith("Magnetic"):
            layout = "Map points"
        else:
            layout = st.radio("Data layout", ["Map points (X, Y, value)", "Section / profile (station, depth or n-level)"],
                              index=1 if (_guess_col(cols, ["n_level", "nlevel", "depth"]) or "n" in [c.lower() for c in cols])
                              and spec["survey"].startswith("IP") else 0, key=key + "_layout", horizontal=True)
        num = readers.numeric_columns(df)
        spec["vcol"] = _pick("Magnetic value column (nT)" if survey else "Value column", num or cols, vguess, key + "_v")
        if layout.startswith("Map"):
            spec["geometry"] = "points"
            spec["xcol"], spec["ycol"] = _xy_cols(df, key)
        else:
            spec["geometry"] = "section"
            spec["linecol"] = _pick("Line column", cols, _guess_col(cols, ["line"]), key + "_line", optional=True)
            spec["xcol"] = _pick("Station / easting", num or cols, _guess_col(cols, ["station", "east", "x"]), key + "_x")
            spec["ycol"] = _pick("Northing of the line (optional)", num or cols,
                                 _guess_col(cols, ["north", "y"]), key + "_y", optional=True,
                                 help_="Needed to place the section on the map and use it in the prediction.")
            mode = st.radio("Depth given as", ["n-level", "Depth column"], key=key + "_dmode", horizontal=True,
                            index=0 if next((c for c in cols if str(c).lower() in ("n", "n_level", "nlevel", "level")), None) else 1)
            spec["depth_mode"] = mode
            if mode == "n-level":
                c1, c2, c3 = st.columns(3)
                spec["ncol"] = c1.selectbox("n-level column", num or cols,
                                            (num or cols).index(next((c for c in (num or cols) if str(c).lower() in
                                                                      ("n", "n_level", "nlevel", "level")), (num or cols)[0])),
                                            key=key + "_n")
                spec["array"] = c2.selectbox("Array", ["Dipole-dipole", "Pole-dipole", "Wenner / Schlumberger"],
                                             key=key + "_arr")
                spec["a_spacing"] = c3.number_input("Dipole spacing a (m)", 1.0, 1000.0, 50.0, key=key + "_a")
            else:
                spec["depthcol"] = _pick("Depth column (m)", num or cols, _guess_col(cols, ["depth", "z"]), key + "_d")
        if spec.get("ycol") or spec["geometry"] == "points":
            spec["epsg"] = _crs_select("Coordinate system", parse_column(df[spec["xcol"]]).head(200),
                                       parse_column(df[spec["ycol"]]).head(200), key + "_crs")
        else:
            spec["epsg"] = None
        return spec

    spec["xcol"], spec["ycol"] = _xy_cols(df, key)
    spec["epsg"] = _crs_select("Coordinate system", parse_column(df[spec["xcol"]]).head(200),
                               parse_column(df[spec["ycol"]]).head(200), key + "_crs")

    if slot == "geology":
        c1, c2, c3 = st.columns(3)
        with c1:
            spec["lithcol"] = _pick("Lithology column", [c for c in cols if c not in (spec["xcol"], spec["ycol"])],
                                    geology.guess(df, geology.LITH_COLS), key + "_lith", optional=True)
        with c2:
            spec["strikecol"] = _pick("Strike column", cols, geology.guess(df, geology.STRIKE_COLS), key + "_strike",
                                      optional=True)
        with c3:
            spec["dipcol"] = _pick("Dip column", cols, geology.guess(df, geology.DIP_COLS), key + "_dip", optional=True)
        if spec["strikecol"]:
            spec["strike_is_dipdir"] = st.checkbox("Strike column holds dip direction (not strike)", key=key + "_dd")
        if not spec["lithcol"] and not spec["strikecol"]:
            st.warning("Choose a lithology column and/or a strike column.")
            return None
        return spec

    if slot == "remote":
        num = [c for c in readers.numeric_columns(df) if c not in (spec["xcol"], spec["ycol"])]
        spec["vcols"] = st.multiselect("Value columns to use as evidence", num, num[:3], key=key + "_vcols")
        if not spec["vcols"]:
            st.warning("Choose at least one value column.")
            return None
        return spec

    if slot == "geochem":
        els = geochem.detect_elements(df)
        elcols = list(els) or [c for c in readers.numeric_columns(df) if c not in (spec["xcol"], spec["ycol"])]
        if not elcols:
            st.warning("No element columns found (expected names like Au_ppb, As_ppm).")
            return None
        au_guess = next((c for c, (e, _) in els.items() if e == "Au"), None)
        c1, c2 = st.columns([1, 2])
        spec["au"] = _pick("Gold column", elcols, au_guess, key + "_au", optional=True)
        others = [c for c in elcols if c != spec["au"]]
        pf_default = [c for c in others if els.get(c, (None,))[0] in geochem.PATHFINDERS] or others[:4]
        spec["pathfinders"] = st.multiselect("Pathfinder / other elements", others, pf_default, key=key + "_pf")
        spec["idcol"] = _pick("Sample ID column", cols, _guess_col(cols, ["sample", "id"]), key + "_id", optional=True)
        if els:
            st.caption("Detected: " + ", ".join(f"{c} → {e}{' (' + u + ')' if u else ''}" for c, (e, u) in els.items()))
        if not spec["au"] and not spec["pathfinders"]:
            return None
        return spec
    return spec


def render_slot(slot):
    title, desc = SLOT_INFO[slot]
    specs = []
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.caption(desc)
        files = st.file_uploader(f"Upload {title}", type=readers.ACCEPT, accept_multiple_files=True,
                                 key=f"up_{slot}", label_visibility="collapsed")
        for i, f in enumerate(files or []):
            with st.expander(f"⚙️ {f.name}", expanded=slot not in ss.get("s2_runs", {})):  # collapse once results exist
                sp = configure(slot, f, i)
            if sp:
                specs.append(sp)
            else:
                st.caption(f"⚠️ {f.name} needs attention before it can be used.")
    return specs


def signature(specs, settings):
    """Changes to files or choices since the last run."""
    rows = []
    for s in specs:
        d = {k: v for k, v in s.items() if k != "data"}
        rows.append(repr(sorted((k, repr(v)) for k, v in d.items())))
    return hash((tuple(rows), repr(sorted(settings.items()))))
