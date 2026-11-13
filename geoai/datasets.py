"""Turn uploaded files + the user's column choices ("specs") into analysis-ready datasets.

All coordinates are converted to one working projected system (metres): the common EPSG of the inputs when
they share a projected one, otherwise the UTM zone of the data centre.
"""
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from . import geochem, geophysics, imagegrid
from .coords import parse_column, utm_epsg_for

SLOTS = {
    "geophysics": "Magnetic / IP",
    "geology": "Geological data",
    "remote": "Remote sensing data",
    "geochem": "Geochemical data",
}


def _src_epsg(spec):
    if spec["kind"] == "raster":
        return spec["data"].get("epsg")
    return spec.get("epsg")


def working_epsg(specs):
    epsgs = {_src_epsg(s) for s in specs if _src_epsg(s)}
    if len(epsgs) == 1:
        e = next(iter(epsgs))
        try:
            if CRS.from_epsg(e).is_projected:
                return e
        except Exception:
            pass
    for s in specs:  # UTM zone of the first dataset with coordinates
        try:
            lon, lat = _centre_lonlat(s)
            return utm_epsg_for(lon, lat)
        except Exception:
            continue
    return 32736  # nothing georeferenced (e.g. IP sections in local distances): sections are still drawn


def _centre_lonlat(spec):
    if spec["kind"] == "table" and spec.get("line_origin") and spec.get("epsg"):
        cx, cy, e = spec["line_origin"][0], spec["line_origin"][1], spec["epsg"]
        return Transformer.from_crs(e, 4326, always_xy=True).transform(cx, cy) if e != 4326 else (cx, cy)
    if spec["kind"] == "table":
        if not spec.get("ycol") or not spec.get("epsg"):
            raise ValueError("no map coordinates")
        x = parse_column(spec["data"]["df"][spec["xcol"]]).to_numpy(float)
        y = parse_column(spec["data"]["df"][spec["ycol"]]).to_numpy(float)
        cx, cy, e = np.nanmean(x), np.nanmean(y), spec["epsg"]
    elif spec["kind"] == "raster":
        t = spec["data"]["transform"]
        a = next(iter(spec["data"]["layers"].values()))
        cx, cy, e = t.c + t.a * a.shape[1] / 2, t.f + t.e * a.shape[0] / 2, spec["data"]["epsg"]
    else:
        x0, x1, y0, y1 = spec["extent"]
        cx, cy, e = (x0 + x1) / 2, (y0 + y1) / 2, spec["epsg"]
    if e == 4326:
        return cx, cy
    return Transformer.from_crs(e, 4326, always_xy=True).transform(cx, cy)


def _xy(spec, work, xcol=None, ycol=None):
    df = spec["data"]["df"]
    x = parse_column(df[xcol or spec["xcol"]]).to_numpy(float)
    y = parse_column(df[ycol or spec["ycol"]]).to_numpy(float)
    if spec["epsg"] != work:
        x, y = Transformer.from_crs(spec["epsg"], work, always_xy=True).transform(x, y)
        x, y = np.asarray(x, float), np.asarray(y, float)
    return x, y


def _reproject_layers(layers, transform, src_epsg, work):
    if src_epsg == work:
        return layers, transform
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject
    a0 = next(iter(layers.values()))
    h, w = a0.shape
    left, top = transform.c, transform.f
    right, bottom = left + w * transform.a, top + h * transform.e
    dst_t, dw, dh = calculate_default_transform(f"EPSG:{src_epsg}", f"EPSG:{work}", w, h, left, bottom, right, top)
    out = {}
    for k, a in layers.items():
        d = np.full((dh, dw), np.nan, np.float32)
        reproject(a.astype(np.float32), d, src_transform=transform, src_crs=f"EPSG:{src_epsg}", dst_transform=dst_t,
                  dst_crs=f"EPSG:{work}", src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
        out[k] = d
    return out, dst_t


def _image_extent(spec, work):
    x0, x1, y0, y1 = spec["extent"]
    if spec["epsg"] == work:
        return x0, x1, y0, y1
    t = Transformer.from_crs(spec["epsg"], work, always_xy=True)
    xs, ys = t.transform([x0, x1, x1, x0], [y0, y0, y1, y1])
    return min(xs), max(xs), min(ys), max(ys)


def build(spec, work):
    """One spec -> one dataset dict (see geoai.interpret for the formats)."""
    slot, kind, name = spec["slot"], spec["kind"], spec["name"]
    base = {"slot": slot, "name": name}

    if kind == "image":
        ext = _image_extent(spec, work)
        img = spec["data"]["image"]
        scale = spec.get("scale", imagegrid.SCALES[-1])
        if slot in ("geophysics", "remote") and not scale.startswith("Reference"):
            vals, _ = imagegrid.image_values(img, scale)
            tr = imagegrid.extent_transform(img.width, img.height, *ext)
            if slot == "geophysics":
                return {**base, "survey": spec["survey"], "kind": "grid", "array": vals, "transform": tr,
                        "from_image": True}
            return {**base, "kind": "grid", "layers": {spec.get("layer_name") or name: vals}, "transform": tr,
                    "from_image": True}
        return {**base, "kind": "image_ref", "image": img, "extent": ext}

    if kind == "raster":
        layers, tr = _reproject_layers(spec["data"]["layers"], spec["data"]["transform"], spec["data"]["epsg"], work)
        if slot == "geophysics":
            return {**base, "survey": spec["survey"], "kind": "grid", "array": layers[spec["band"]], "transform": tr}
        if slot == "remote":
            return {**base, "kind": "grid", "layers": {b: layers[b] for b in spec["bands"]}, "transform": tr}
        from PIL import Image
        arrs = list(layers.values())
        rgb = np.dstack(arrs[:3]) if len(arrs) >= 3 else np.dstack([arrs[0]] * 3)
        rgb = (rgb - np.nanmin(rgb)) / ((np.nanmax(rgb) - np.nanmin(rgb)) or 1)
        img = Image.fromarray((np.nan_to_num(rgb, nan=1.0) * 255).astype(np.uint8))
        h, w = arrs[0].shape
        return {**base, "kind": "image_ref", "image": img, "extent": (tr.c, tr.c + w * tr.a, tr.f + h * tr.e, tr.f)}

    df = spec["data"]["df"]
    if slot == "geophysics":
        v = parse_column(df[spec["vcol"]]).to_numpy(float)
        if spec.get("geometry", "points") == "section":
            if spec["depth_mode"] == "n-level":
                depth = geophysics.n_level_to_depth(parse_column(df[spec["ncol"]]).to_numpy(float),
                                                    spec["a_spacing"], spec["array"])
            else:
                depth = np.abs(parse_column(df[spec["depthcol"]]).to_numpy(float))
            line = df[spec["linecol"]].astype(str).to_numpy() if spec.get("linecol") else \
                np.array([spec.get("line_name") or "Line 1"] * len(df))
            if spec.get("ycol"):
                xm, ym = _xy(spec, work)
            elif spec.get("line_origin") is not None:  # stations are distances along the line
                e0, n0, brg = spec["line_origin"]
                dist = parse_column(df[spec["xcol"]]).to_numpy(float)
                xm = e0 + dist * np.sin(np.radians(brg))
                ym = n0 + dist * np.cos(np.radians(brg))
                if spec.get("epsg") and spec["epsg"] != work:
                    xm, ym = (np.asarray(a, float) for a in
                              Transformer.from_crs(spec["epsg"], work, always_xy=True).transform(xm, ym))
                return {**base, "survey": spec["survey"], "kind": "section", "line": line, "x": dist, "x_map": xm,
                        "y": ym, "depth": depth, "v": v}
            elif spec.get("line_northing") is not None:  # one file per line: northing typed in by the user
                xm = parse_column(df[spec["xcol"]]).to_numpy(float)
                ym = np.full(len(df), float(spec["line_northing"]))
                if spec.get("epsg") and spec["epsg"] != work:
                    xm, ym = (np.asarray(a, float) for a in
                              Transformer.from_crs(spec["epsg"], work, always_xy=True).transform(xm, ym))
            else:
                xm, ym = parse_column(df[spec["xcol"]]).to_numpy(float), None
            return {**base, "survey": spec["survey"], "kind": "section", "line": line, "x": xm, "x_map": xm,
                    "y": ym, "depth": depth, "v": v}
        x, y = _xy(spec, work)
        return {**base, "survey": spec["survey"], "kind": "points", "x": x, "y": y, "v": v}

    x, y = _xy(spec, work)
    if slot == "geology":
        strike = dip = None
        if spec.get("strikecol"):
            from .geology import strike_from
            strike = strike_from(parse_column(df[spec["strikecol"]]).to_numpy(float), spec.get("strike_is_dipdir"))
        if spec.get("dipcol"):
            dip = parse_column(df[spec["dipcol"]]).to_numpy(float)
        return {**base, "kind": "points", "x": x, "y": y,
                "lith": df[spec["lithcol"]].astype(str).to_numpy() if spec.get("lithcol") else np.array([""] * len(df)),
                "strike": strike, "dip": dip}
    if slot == "remote":
        return {**base, "kind": "points", "x": x, "y": y,
                "layers": {c: parse_column(df[c]).to_numpy(float) for c in spec["vcols"]}}
    if slot == "geochem":
        cols = [spec["au"]] + [p for p in spec["pathfinders"] if p != spec["au"]] if spec.get("au") else spec["pathfinders"]
        table, bdl = {}, {}
        for c in cols:
            table[c], bdl[c] = geochem.clean_values(df[c])
        ids = df[spec["idcol"]].astype(str).to_numpy() if spec.get("idcol") else np.arange(1, len(df) + 1).astype(str)
        return {**base, "kind": "points", "x": x, "y": y, "table": pd.DataFrame(table), "bdl": bdl,
                "au": spec.get("au"), "pathfinders": [p for p in spec["pathfinders"] if p != spec.get("au")],
                "ids": ids}
    raise ValueError(f"Unsupported data for {SLOTS.get(slot, slot)}: {name}")
