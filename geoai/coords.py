"""Corner coordinates -> Area of Interest (AOI).

Accepts decimal degrees, degrees-minutes-seconds strings (e.g. 2°52'10"S) or projected
coordinates (UTM, Arc 1960 ...). Everything is converted to WGS84 longitude/latitude.
"""
import json
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyproj import CRS, Geod, Transformer
from shapely.geometry import Polygon, mapping

# Common systems for Tanzania / East Africa first; any EPSG code can be typed in the UI.
CRS_OPTIONS = {
    "WGS84 latitude / longitude (EPSG:4326)": 4326,
    "WGS84 / UTM zone 35S (EPSG:32735)": 32735,
    "WGS84 / UTM zone 36S (EPSG:32736)": 32736,
    "WGS84 / UTM zone 37S (EPSG:32737)": 32737,
    "Arc 1960 / UTM zone 35S (EPSG:21035)": 21035,
    "Arc 1960 / UTM zone 36S (EPSG:21036)": 21036,
    "Arc 1960 / UTM zone 37S (EPSG:21037)": 21037,
    "WGS84 / UTM zone 36N (EPSG:32636)": 32636,
    "WGS84 / UTM zone 37N (EPSG:32637)": 32637,
}

X_NAMES = ["longitude", "long", "lon", "lng", "easting", "east", "x", "e"]
Y_NAMES = ["latitude", "lat", "northing", "north", "y", "n"]

_GEOD = Geod(ellps="WGS84")


# ---------- parsing ----------
def parse_coord(value):
    """Parse one coordinate: 32.25, '32,25', '2°52\\'10"S', 'S 2 52 10', '-2 52.5'. NaN if unreadable."""
    if value is None:
        return math.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    s = str(value).strip()
    if not s:
        return math.nan
    try:
        return float(s.replace(",", "."))
    except ValueError:
        pass
    nums = re.findall(r"\d+(?:[.,]\d+)?", s)
    if not nums or len(nums) > 3:
        return math.nan
    parts = [float(n.replace(",", ".")) for n in nums]
    deg = parts[0] + (parts[1] / 60 if len(parts) > 1 else 0) + (parts[2] / 3600 if len(parts) > 2 else 0)
    if re.search(r"[SWsw]", s) or s.startswith("-"):
        deg = -deg
    return deg


def parse_column(series):
    return series.map(parse_coord).astype(float)


def _match(cols, names):
    low = {c: str(c).strip().lower() for c in cols}
    for n in names:  # exact name first
        for c, l in low.items():
            if l == n:
                return c
    for n in names:  # then "contains" for longer names only (avoid matching 'x' inside 'index')
        if len(n) < 3:
            continue
        for c, l in low.items():
            if n in l:
                return c
    return None


def guess_columns(df):
    """Best guess for (x/longitude/easting, y/latitude/northing) column names."""
    cols = list(df.columns)
    x, y = _match(cols, X_NAMES), _match(cols, Y_NAMES)
    if x is None or y is None or x == y:
        numeric = [c for c in cols if parse_column(df[c]).notna().mean() > 0.8]
        if len(numeric) >= 2:
            x, y = numeric[0], numeric[1]
    return x, y


def looks_geographic(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return bool(np.all(np.abs(xs) <= 180) and np.all(np.abs(ys) <= 180))


def wide_to_long(df):
    """A single row like x1,y1,x2,y2,... -> one row per corner."""
    vals = [parse_coord(v) for v in df.iloc[0].tolist()]
    vals = [v for v in vals if not math.isnan(v)]
    if len(vals) < 6 or len(vals) % 2:
        return df
    return pd.DataFrame({"x": vals[0::2], "y": vals[1::2]})


def utm_epsg_for(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


# ---------- AOI ----------
@dataclass
class AOI:
    lon: np.ndarray  # ordered corners, WGS84
    lat: np.ndarray
    source_epsg: int = 4326
    notes: tuple = ()

    @property
    def polygon(self):
        return Polygon(zip(self.lon, self.lat))

    @property
    def bbox(self):
        return float(self.lon.min()), float(self.lat.min()), float(self.lon.max()), float(self.lat.max())

    @property
    def centroid(self):
        c = self.polygon.centroid
        return float(c.x), float(c.y)

    @property
    def area_m2(self):
        a, _ = _GEOD.geometry_area_perimeter(self.polygon)
        return abs(a)

    @property
    def perimeter_m(self):
        _, p = _GEOD.geometry_area_perimeter(self.polygon)
        return abs(p)

    @property
    def utm_epsg(self):
        return utm_epsg_for(*self.centroid)

    @property
    def key(self):
        """Hashable identity, used for caching."""
        return tuple(np.round(self.lon, 7)), tuple(np.round(self.lat, 7))

    def to_utm(self, epsg=None):
        t = Transformer.from_crs(4326, epsg or self.utm_epsg, always_xy=True)
        return t.transform(self.lon, self.lat)

    def corners_table(self):
        e, n = self.to_utm()
        return pd.DataFrame({
            "Corner": [f"C{i + 1}" for i in range(len(self.lon))],
            "Latitude": np.round(self.lat, 6),
            "Longitude": np.round(self.lon, 6),
            "Easting": np.round(e, 1),
            "Northing": np.round(n, 1),
        })

    @property
    def utm_name(self):
        return CRS.from_epsg(self.utm_epsg).name

    def geometry_json(self):
        return json.dumps(mapping(self.polygon))

    def geojson(self):
        feat = {"type": "Feature", "properties": {"name": "GeoAI AOI",
                                                  "area_km2": round(self.area_m2 / 1e6, 4)},
                "geometry": mapping(self.polygon)}
        return json.dumps({"type": "FeatureCollection", "features": [feat]}, indent=2)


def build_aoi(xs, ys, epsg=4326):
    """Validate corners, convert to WGS84 and order them around the centre.

    Raises ValueError with a message that can be shown to the user.
    """
    xs = np.asarray([parse_coord(v) for v in xs], float)
    ys = np.asarray([parse_coord(v) for v in ys], float)
    ok = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[ok], ys[ok]
    notes = []
    if (~ok).any():
        notes.append(f"{int((~ok).sum())} row(s) skipped: coordinates could not be read.")
    if len(xs) < 3:
        raise ValueError("At least 3 valid corner points are needed (4 expected).")
    if len(xs) != 4:
        notes.append(f"{len(xs)} points found (4 expected); using them all as a polygon.")

    if epsg == 4326:
        lon, lat = xs, ys
        if np.any(np.abs(lat) > 90) and np.all(np.abs(lon) <= 90):
            lon, lat = lat, lon
            notes.append("Latitude and longitude looked swapped, so they were switched.")
        if np.any(np.abs(lat) > 90) or np.any(np.abs(lon) > 180):
            raise ValueError("Values are outside latitude/longitude range. "
                             "Are these projected (UTM) coordinates? Choose the right coordinate system.")
    else:
        if looks_geographic(xs, ys):
            raise ValueError("These look like latitude/longitude degrees, but a projected system "
                             f"(EPSG:{epsg}) is selected. Choose WGS84 latitude / longitude.")
        lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(xs, ys)
        lon, lat = np.asarray(lon), np.asarray(lat)

    # drop exact duplicates, then order by angle around the centre (avoids a crossed "bow-tie")
    pts = np.unique(np.column_stack([lon, lat]).round(9), axis=0)
    if len(pts) < 3:
        raise ValueError("Corner points are duplicates of each other.")
    cx, cy = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))[::-1]  # clockwise
    lon, lat = pts[order, 0], pts[order, 1]

    aoi = AOI(lon=lon, lat=lat, source_epsg=epsg, notes=tuple(notes))
    if not aoi.polygon.is_valid or aoi.area_m2 < 1:
        raise ValueError("The corners do not form a valid area (points on a line?).")
    return aoi


def aoi_from_table(df, xcol, ycol, epsg):
    if len(df) == 1:
        df = wide_to_long(df)
        if {"x", "y"} <= set(df.columns):
            xcol, ycol = "x", "y"
    return build_aoi(df[xcol].tolist(), df[ycol].tolist(), epsg)
