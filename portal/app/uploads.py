"""Reading an uploaded file and working out what still needs to be chosen about it.

Three separable things:

    inspect(path, sheet)              what is in the file: kind, columns, a preview, the guesses
    fields(slot, info, ...)           what the person still has to choose, as plain descriptors
    build_spec(slot, path, choices)   the spec the engine wants, assembled from those choices

A field descriptor is a dict, and one template macro renders any of them, so adding a control to a
section is a line of Python rather than a lump of HTML.
"""
import os

import pandas as pd

from geoai import coords, geochem, geology, geophysics, imagegrid, readers
from geoai.coords import parse_column

NONE = "(none)"
ACCEPT = readers.ACCEPT
ACCEPT_ATTR = ",".join("." + e for e in readers.ACCEPT)

SLOT_INFO = {
    "geophysics": ("Magnetic / IP (induced polarisation)",
                   "Magnetic survey points (X, Y, nT), IP chargeability or resistivity as plan points or as "
                   "sections (line, station, n-level or depth), gridded GeoTIFFs, or a picture of a map."),
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


# ------------------------------------------------------------------ field descriptors
def field(name, label, kind="select", options=None, value=None, help=None, optional=False,
          width="half", **extra):
    f = {"name": name, "label": label, "kind": kind, "options": list(options or []), "value": value,
         "help": help, "optional": optional, "width": width}
    f.update(extra)
    return f


def _options(values, optional=False):
    opts = [NONE] if optional else []
    return opts + [str(v) for v in values]


def pick(name, label, cols, guess, optional=False, help=None, width="half", **extra):
    opts = _options(cols, optional)
    value = str(guess) if guess is not None and str(guess) in opts else opts[0] if opts else None
    return field(name, label, "select", opts, value, help, optional, width, **extra)


# ------------------------------------------------------------------ guessing
def guess_col(cols, words):
    """First column matching a word. Short words (x, y, z, n, ip) must be the whole column name."""
    low = {c: str(c).strip().lower() for c in cols}
    for w in words:
        for c, l in low.items():
            if (l == w) if len(w) <= 2 else (w in l):
                return c
    return None


def guess_survey(cols):
    if guess_col(cols, ["charg", "mv/v", "msec", "ip"]):
        return 1 if len(geophysics.SURVEYS) > 1 else 0
    if guess_col(cols, ["resist", "ohm", "rho"]):
        return min(2, len(geophysics.SURVEYS) - 1)
    return 0


def default_crs_label(values_x=(), values_y=()):
    """Latitude/longitude when the numbers look like degrees, otherwise the common local UTM zone."""
    try:
        if len(values_x) and coords.looks_geographic(pd.Series(values_x).dropna(),
                                                     pd.Series(values_y).dropna()):
            return list(coords.CRS_OPTIONS)[0]
    except Exception:
        pass
    return "WGS84 / UTM zone 36S (EPSG:32736)"


def default_crs(values_x=(), values_y=()):
    return coords.CRS_OPTIONS[default_crs_label(values_x, values_y)]


def crs_field(name, label, values_x=(), values_y=(), help=None, width="half"):
    opts = list(coords.CRS_OPTIONS) + ["Other EPSG code…"]
    return field(name, label, "crs", opts, default_crs_label(values_x, values_y),
                 help or "Detected from the numbers; change it if that is wrong.", width=width)


def resolve_crs(form, name, fallback=32736):
    choice = form.get(name) or ""
    if choice in coords.CRS_OPTIONS:
        return coords.CRS_OPTIONS[choice]
    try:
        return int(form.get(name + "_epsg") or fallback)
    except (TypeError, ValueError):
        return fallback


# ------------------------------------------------------------------ reading
_cache = {}


def inspect(path, sheet=None):
    """What is in an uploaded file. Cached on the file's path, size and time, because a page may ask
    for it several times while the person is still choosing columns."""
    path = str(path)
    try:
        st = os.stat(path)
        key = (path, st.st_size, int(st.st_mtime), sheet)
    except OSError as e:
        raise ValueError(f"The uploaded file is no longer there: {e}") from e
    if key in _cache:
        return _cache[key]

    name = os.path.basename(path)
    sheets = []
    if name.lower().endswith((".xls", ".xlsx")):
        try:
            with open(path, "rb") as fh:
                sheets = list(readers.excel_sheets(fh))
        except Exception as e:
            raise ValueError(f"Could not open the workbook: {e}") from e
    with open(path, "rb") as fh:
        buf = _named(fh.read(), name)
        try:
            res = readers.read_upload(buf, sheet)
        except Exception as e:
            raise ValueError(f"Could not read {name}: {e}") from e

    info = {"name": name, "path": path, "kind": res["kind"], "sheets": sheets, "sheet": sheet,
            "result": res}
    if res["kind"] == "table":
        df = res["df"]
        info["columns"] = [str(c) for c in df.columns]
        info["numeric"] = [str(c) for c in readers.numeric_columns(df)]
        info["rows"] = len(df)
        info["preview"] = df.head(5).astype(str)
    elif res["kind"] == "raster":
        info["layers"] = [str(k) for k in res["layers"]]
        first = res["layers"][list(res["layers"])[0]]
        info["shape"] = (int(first.shape[0]), int(first.shape[1]))
        info["epsg"] = res.get("epsg")
    if len(_cache) > 24:
        for old in list(_cache)[:8]:
            _cache.pop(old, None)
    _cache[key] = info
    return info


def _named(raw, name):
    """readers.read_upload picks its parser from the file name, which BytesIO does not have."""
    import io
    buf = io.BytesIO(raw)
    buf.name = name
    return buf


def forget(path):
    for k in [k for k in _cache if k[0] == str(path)]:
        _cache.pop(k, None)


# ------------------------------------------------------------------ what still needs choosing
def fields(slot, info, survey=None, aoi_bounds=None):
    """The controls for one uploaded file, in the order they should appear."""
    out = []
    if len(info.get("sheets") or []) > 1:
        out.append(field("sheet", "Sheet", "select", info["sheets"], info.get("sheet") or info["sheets"][0],
                         "Which sheet of the workbook holds the data.", width="half", reload=True))

    if info["kind"] == "image":
        out += _image_fields(slot, info, survey, aoi_bounds)
    elif info["kind"] == "raster":
        out += _raster_fields(slot, info, survey)
    else:
        out += _table_fields(slot, info, survey)
    return out


def _image_fields(slot, info, survey, aoi_bounds):
    out = []
    if slot in ("geophysics", "remote"):
        out.append(field("scale", "Colour scale of the picture", "select", imagegrid.SCALES,
                         imagegrid.SCALES[0],
                         "Lets GeoAI turn the colours back into relative values (0 = low, 1 = high).",
                         width="full"))
        if slot == "geophysics":
            out.append(field("survey", "What the map shows", "select", geophysics.SURVEYS,
                             survey or geophysics.SURVEYS[0],
                             "Ignored when the colour scale is a reference background."))
        if slot == "remote":
            out.append(field("layer_name", "Layer name", "text", value=info["name"].rsplit(".", 1)[0]))
    b = aoi_bounds or (0.0, 0.0, 0.0, 0.0)
    hint = " (pre-filled from your Section 1 area)" if aoi_bounds else ""
    for n, (nm, label, v) in enumerate([("x0", "West (min X)", b[0]), ("x1", "East (max X)", b[1]),
                                        ("y0", "South (min Y)", b[2]), ("y1", "North (max Y)", b[3])]):
        out.append(field(nm, label, "number", value=float(v), width="quarter", step="any",
                         help=f"A picture carries no coordinates: give the map's edges{hint}." if n == 0 else None))
    out.append(crs_field("crs", "Coordinate system of the edges", width="full"))
    return out


def _raster_fields(slot, info, survey):
    layers = info["layers"]
    out = []
    if slot == "geophysics":
        out.append(field("survey", "What the grid shows", "select", geophysics.SURVEYS,
                         survey or geophysics.SURVEYS[0]))
        out.append(field("band", "Band", "select", layers, layers[0]))
    elif slot == "remote":
        out.append(field("bands", "Bands to use as evidence", "multi", layers, layers, width="full"))
    return out


def _table_fields(slot, info, survey):
    cols, num = info["columns"], info["numeric"] or info["columns"]
    df = info["result"]["df"]
    gx, gy = coords.guess_columns(df)
    out = []

    if slot == "geophysics":
        is_mag = bool(survey and str(survey).startswith("Magnetic"))
        if not survey:
            out.append(field("survey", "Survey type", "select", geophysics.SURVEYS,
                             geophysics.SURVEYS[guess_survey(cols)]))
        vguess = guess_col(cols, ["mag", "tmi", "nt", "charg", "mv/v", "resist", "ohm", "value", "rho"])
        if not is_mag:
            section_like = bool(guess_col(cols, ["n_level", "nlevel", "depth"])
                                or "n" in [str(c).lower() for c in cols])
            out.append(field("layout", "Data layout", "radio",
                             ["Map points (X, Y, value)", "Section / profile (station, depth or n-level)"],
                             "Section / profile (station, depth or n-level)" if section_like
                             else "Map points (X, Y, value)", width="full"))
        out.append(pick("vcol", "Magnetic value column (nT)" if is_mag else "Value column", num, vguess))
        out.append(pick("xcol", "X / Easting / Longitude", cols, gx))
        out.append(pick("ycol", "Y / Northing / Latitude", cols, gy))
        if is_mag:
            # a magnetic survey is always map points, so none of the section controls below apply
            out.append(crs_field("crs", "Coordinate system",
                                 parse_column(df[gx]).head(200) if gx in df else (),
                                 parse_column(df[gy]).head(200) if gy in df else (), width="full"))
            return out
        # section-only controls; the template hides them unless the layout says section
        out.append(pick("linecol", "Line column", cols, guess_col(cols, ["line"]), optional=True,
                        help="Which readings belong to the same survey line.", width="half", only="section"))
        out.append(field("depth_mode", "Depth given as", "radio", ["n-level", "Depth column"],
                         "n-level" if next((c for c in cols if str(c).lower() in
                                            ("n", "n_level", "nlevel", "level")), None) else "Depth column",
                         width="half", only="section"))
        nguess = next((c for c in num if str(c).lower() in ("n", "n_level", "nlevel", "level")),
                      num[0] if num else None)
        out.append(pick("ncol", "n-level column", num, nguess, width="third", only="section"))
        out.append(field("array", "Array", "select",
                         ["Dipole-dipole", "Pole-dipole", "Wenner / Schlumberger"], "Dipole-dipole",
                         width="third", only="section"))
        out.append(field("a_spacing", "Dipole spacing a (m)", "number", value=50.0, width="third",
                         min=1.0, max=1000.0, step=1.0, only="section"))
        out.append(pick("depthcol", "Depth column (m)", num, guess_col(cols, ["depth", "z"]),
                        optional=True, width="third", only="section"))
        out.append(crs_field("crs", "Coordinate system",
                             parse_column(df[gx]).head(200) if gx in df else (),
                             parse_column(df[gy]).head(200) if gy in df else (), width="full"))
        return out

    out.append(pick("xcol", "X / Easting / Longitude", cols, gx))
    out.append(pick("ycol", "Y / Northing / Latitude", cols, gy))
    out.append(crs_field("crs", "Coordinate system",
                         parse_column(df[gx]).head(200) if gx in df else (),
                         parse_column(df[gy]).head(200) if gy in df else (), width="full"))

    if slot == "geology":
        others = [c for c in cols if c not in (gx, gy)]
        out.append(pick("lithcol", "Lithology column", others, geology.guess(df, geology.LITH_COLS),
                        optional=True, width="third"))
        out.append(pick("strikecol", "Strike column", cols, geology.guess(df, geology.STRIKE_COLS),
                        optional=True, width="third"))
        out.append(pick("dipcol", "Dip column", cols, geology.guess(df, geology.DIP_COLS),
                        optional=True, width="third"))
        out.append(field("strike_is_dipdir", "The strike column holds dip direction, not strike",
                         "check", value=False, width="full"))
    elif slot == "remote":
        vals = [c for c in num if c not in (gx, gy)]
        out.append(field("vcols", "Value columns to use as evidence", "multi", vals, vals[:3], width="full"))
    elif slot == "geochem":
        els = geochem.detect_elements(df)
        elcols = [str(c) for c in els] or [c for c in num if c not in (gx, gy)]
        au_guess = next((str(c) for c, (e, _) in els.items() if e == "Au"), None)
        out.append(pick("au", "Gold column", elcols, au_guess, optional=True))
        others = [c for c in elcols if c != (au_guess or "")]
        pf_default = [c for c in others if els.get(c, (None,))[0] in geochem.PATHFINDERS] or others[:4]
        out.append(field("pathfinders", "Pathfinder / other elements", "multi", others, pf_default,
                         width="full"))
        out.append(pick("idcol", "Sample ID column", cols, guess_col(cols, ["sample", "id"]),
                        optional=True))
        if els:
            out.append(field("_detected", "", "note",
                             value="Detected: " + ", ".join(
                                 f"{c} → {e}{' (' + u + ')' if u else ''}" for c, (e, u) in els.items()),
                             width="full"))
    return out


# ------------------------------------------------------------------ building the engine's spec
def build_spec(slot, path, form, survey=None, prefix=""):
    """Turn the submitted form into the spec dict geoai.datasets.build expects.

    Raises ValueError with a sentence the person can act on when something essential is missing.
    """
    def g(name, default=None):
        return form.get(prefix + name, default)

    def glist(name):
        getter = getattr(form, "getlist", None)
        return [v for v in (getter(prefix + name) if getter else (form.get(prefix + name) or "").split(","))
                if v]

    def gnum(name, default=0.0):
        try:
            return float(g(name, default))
        except (TypeError, ValueError):
            return float(default)

    def gcol(name):
        v = g(name)
        return None if v in (None, "", NONE) else v

    sheet = g("sheet") or None
    info = inspect(path, sheet)
    res = info["result"]
    spec = {"slot": slot, "name": info["name"], "kind": res["kind"], "data": res}

    if res["kind"] == "image":
        ext = (gnum("x0"), gnum("x1"), gnum("y0"), gnum("y1"))
        if ext[1] <= ext[0] or ext[3] <= ext[2]:
            raise ValueError(f"{info['name']}: give the map's edges — east must be greater than west, "
                             "and north greater than south.")
        spec.update(extent=ext, epsg=resolve_crs(form, prefix + "crs"))
        if slot in ("geophysics", "remote"):
            spec["scale"] = g("scale", imagegrid.SCALES[0])
            if slot == "geophysics" and not str(spec["scale"]).startswith("Reference"):
                spec["survey"] = survey or g("survey", geophysics.SURVEYS[0])
            if slot == "remote":
                spec["layer_name"] = g("layer_name") or info["name"].rsplit(".", 1)[0]
        return spec

    if res["kind"] == "raster":
        if slot == "geophysics":
            spec["survey"] = survey or g("survey", geophysics.SURVEYS[0])
            spec["band"] = g("band", info["layers"][0])
        elif slot == "remote":
            spec["bands"] = glist("bands") or info["layers"]
        elif slot == "geochem":
            raise ValueError(f"{info['name']}: geochemical data must be a table of samples, not a raster.")
        return spec

    df = res["df"]
    if slot == "geophysics":
        spec["survey"] = survey or g("survey", geophysics.SURVEYS[0])
        is_mag = str(spec["survey"]).startswith("Magnetic")
        layout = "Map points" if is_mag else g("layout", "Map points (X, Y, value)")
        spec["vcol"] = gcol("vcol")
        if not spec["vcol"]:
            raise ValueError(f"{info['name']}: choose the column holding the readings.")
        if str(layout).startswith("Map"):
            spec["geometry"] = "points"
            spec["xcol"], spec["ycol"] = gcol("xcol"), gcol("ycol")
        else:
            spec["geometry"] = "section"
            spec["linecol"] = gcol("linecol")
            spec["xcol"] = gcol("xcol")
            spec["ycol"] = gcol("ycol")
            spec["depth_mode"] = g("depth_mode", "n-level")
            if spec["depth_mode"] == "n-level":
                spec["ncol"] = gcol("ncol")
                spec["array"] = g("array", "Dipole-dipole")
                spec["a_spacing"] = gnum("a_spacing", 50.0)
            else:
                spec["depthcol"] = gcol("depthcol")
        if not spec.get("xcol"):
            raise ValueError(f"{info['name']}: choose the X / station column.")
        spec["epsg"] = (resolve_crs(form, prefix + "crs")
                        if (spec.get("ycol") or spec["geometry"] == "points") else None)
        return spec

    spec["xcol"], spec["ycol"] = gcol("xcol"), gcol("ycol")
    if not spec["xcol"] or not spec["ycol"]:
        raise ValueError(f"{info['name']}: choose the X and Y columns.")
    spec["epsg"] = resolve_crs(form, prefix + "crs")

    if slot == "geology":
        spec["lithcol"] = gcol("lithcol")
        spec["strikecol"] = gcol("strikecol")
        spec["dipcol"] = gcol("dipcol")
        spec["strike_is_dipdir"] = bool(g("strike_is_dipdir"))
        if not spec["lithcol"] and not spec["strikecol"]:
            raise ValueError(f"{info['name']}: choose a lithology column, a strike column, or both.")
    elif slot == "remote":
        spec["vcols"] = glist("vcols")
        if not spec["vcols"]:
            raise ValueError(f"{info['name']}: choose at least one value column.")
    elif slot == "geochem":
        spec["au"] = gcol("au")
        spec["pathfinders"] = glist("pathfinders")
        spec["idcol"] = gcol("idcol")
        if not spec["au"] and not spec["pathfinders"]:
            raise ValueError(f"{info['name']}: choose the gold column, some pathfinder elements, or both.")
    return spec
