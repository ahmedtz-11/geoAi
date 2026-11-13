"""Read any uploaded field-data file into one of three kinds.

table   pandas DataFrame (CSV, TXT, XYZ, Geosoft-style .dat, Excel .xls/.xlsx)
raster  georeferenced grid(s) (GeoTIFF with a coordinate system)
image   picture (PNG/JPG, or a TIFF without coordinates), needs an extent to be used as data
"""
import io
import re

import numpy as np
import pandas as pd
from PIL import Image

TABLE_EXT = (".csv", ".txt", ".xyz", ".dat", ".xls", ".xlsx")
RASTER_EXT = (".tif", ".tiff")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")
ACCEPT = ["csv", "txt", "xyz", "dat", "xls", "xlsx", "tif", "tiff", "png", "jpg", "jpeg"]
DUMMIES = {"*", "-9999", "-99999", "-9999.0", "1e+32", "-1e+32", "nan", "NaN", "NA", "N/A", "", "------", "INFINI",
           "INF", "-INF", "inf", "-inf", "OVER", "over"}
MAX_IMAGE_PX = 600


def _bytes(f):
    if hasattr(f, "getvalue"):
        return f.getvalue()
    if hasattr(f, "read"):
        return f.read()
    with open(f, "rb") as fh:
        return fh.read()


def _is_num(tok):
    try:
        float(tok.replace(",", "."))
        return True
    except ValueError:
        return False


def parse_geosoft_xyz(text):
    """Geosoft XYZ / generic .dat: '/' comment lines (last one with names = header),
    'Line 100' / 'Tie 10' separators, whitespace or comma separated values, '*' = dummy."""
    header, rows, line_id, has_line = None, [], None, False
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("/"):
            toks = re.split(r"[\s,;]+", s.lstrip("/").strip())
            toks = [t for t in toks if t]
            if len(toks) >= 2 and not any(_is_num(t) for t in toks):
                header = toks
            continue
        low = s.lower()
        parts = s.split()
        if low.startswith(("line", "tie", "trend")) and len(parts) <= 3:
            line_id = parts[1] if len(parts) > 1 else parts[0]
            has_line = True
            continue
        toks = [t for t in re.split(r"[\s,;]+", s) if t]
        if not toks:
            continue
        rows.append(([line_id] if has_line else []) + toks)
    if not rows:
        raise ValueError("No data rows found.")
    width = max(len(r) for r in rows)
    rows = [r + [None] * (width - len(r)) for r in rows]
    df = pd.DataFrame(rows)
    names = ([] if not has_line else ["line"])
    data_w = width - len(names)
    if header and len(header) == data_w:
        names += header
    else:
        names += [f"col_{i}" for i in range(data_w)]
    df.columns = names
    return df.replace({d: np.nan for d in DUMMIES})


def _split(line):
    for sep in (",", ";", "\t"):
        if sep in line:
            return [t.strip() for t in line.split(sep)]
    return line.split()


def _numeric_share(tokens):
    toks = [t for t in tokens if t]
    if not toks:
        return 0.0, 0
    return sum(_is_num(t) for t in toks) / len(toks), len(toks)


def header_row(lines, max_scan=40):
    """Index of the header row when instrument files start with information lines (IRIS Prosys/ELREC,
    ABEM, Syscal ...): a mostly-text row followed by rows that are mostly numbers. 0 if not found."""
    rows = [_split(l) for l in lines[:max_scan + 4]]
    for i in range(min(max_scan, len(rows) - 1)):
        share, n = _numeric_share(rows[i])
        if n < 3 or share > 0.4:
            continue
        nxt = [_numeric_share(r) for r in rows[i + 1:i + 4]]
        if nxt and all(sh >= 0.6 and k >= 0.6 * n for sh, k in nxt):
            return i
    return 0


def parse_text_table(text):
    """CSV / TXT / XYZ / DAT. Uses the Geosoft parser when '/' comments or Line markers are present.
    Information lines above the real header (instrument exports) are skipped automatically."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("The file is empty.")
    geosoft = any(l.lstrip().startswith("/") for l in lines[:50]) or \
        any(re.match(r"^\s*(line|tie)\s+\S+\s*$", l, re.I) for l in lines[:200])
    if geosoft:
        return parse_geosoft_xyz(text)
    skip = header_row(lines)
    if skip:
        text = "\n".join(lines[skip:])
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    df = df.loc[:, [not (str(c).startswith("Unnamed") and df[c].isna().all()) for c in df.columns]]
    if all(_is_num(str(c)) for c in df.columns):  # no header row
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", header=None)
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    df.columns = [str(c).strip() for c in df.columns]
    return df.replace({d: np.nan for d in DUMMIES if d})


def excel_sheets(f):
    return pd.ExcelFile(io.BytesIO(_bytes(f))).sheet_names


def _image_result(img):
    img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_PX, MAX_IMAGE_PX))
    return {"kind": "image", "image": img}


def read_upload(f, sheet=None):
    """Return dict(kind=table|raster|image, ...) for an uploaded file."""
    name = getattr(f, "name", str(f)).lower()
    raw = _bytes(f)
    if name.endswith(IMAGE_EXT):
        return _image_result(Image.open(io.BytesIO(raw)))
    if name.endswith(RASTER_EXT):
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(raw) as mem, mem.open() as src:
            if src.crs is None or src.transform == rasterio.Affine.identity():
                arr = src.read()
                if arr.shape[0] >= 3:
                    rgb = np.dstack(arr[:3]).astype(float)
                else:
                    g = arr[0].astype(float)
                    rgb = np.dstack([g, g, g])
                rgb = (255 * (rgb - np.nanmin(rgb)) / ((np.nanmax(rgb) - np.nanmin(rgb)) or 1)).astype(np.uint8)
                return _image_result(Image.fromarray(rgb))
            layers = {}
            for i in range(1, src.count + 1):
                a = src.read(i).astype(np.float32)
                if src.nodata is not None:
                    a[a == src.nodata] = np.nan
                desc = src.descriptions[i - 1] if src.descriptions else None
                layers[desc or f"band_{i}"] = a
            return {"kind": "raster", "layers": layers, "transform": src.transform, "crs": src.crs.to_wkt(),
                    "epsg": src.crs.to_epsg()}
    if name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet or 0)
        df.columns = [str(c).strip() for c in df.columns]
        return {"kind": "table", "df": df.replace({d: np.nan for d in DUMMIES if d})}
    text = raw.decode("utf-8", errors="ignore")
    return {"kind": "table", "df": parse_text_table(text)}


def numeric_columns(df, min_frac=0.8):
    from .coords import parse_column
    return [c for c in df.columns if parse_column(df[c]).notna().mean() >= min_frac]
