"""GeoAI core package.

Modules
-------
io        Reading tabular field data (CSV, TXT, DAT, XYZ, Excel).
coords    Parsing corner coordinates and building an Area of Interest (AOI).
grid      Gridding helpers used by the interpretation tools (Section 2).
indices   Sentinel-2 band maths: composites, alteration indices, target zones.
export    PNG / GeoTIFF / KML / GeoJSON exports.
imagery   Satellite image sources (Esri basemap, Sentinel-2 via Planetary Computer).
"""

__version__ = "0.2.0"
