"""Offline tests for the GeoAI core (no internet needed).  Run:  python -m pytest tests"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geoai import coords, export, indices  # noqa: E402
from geoai.imagery import basemap, sentinel  # noqa: E402

LAT = [-2.85, -2.85, -2.89, -2.89]
LON = [32.15, 32.19, 32.19, 32.15]


def test_parse_coord_formats():
    assert coords.parse_coord("32,25") == pytest.approx(32.25)
    assert coords.parse_coord("2°51'00\"S") == pytest.approx(-2.85)
    assert coords.parse_coord("S 2 51 0") == pytest.approx(-2.85)
    assert coords.parse_coord("32 09 00 E") == pytest.approx(32.15)
    assert np.isnan(coords.parse_coord("abc"))


def test_aoi_latlon_area_and_order():
    aoi = coords.build_aoi(LON[::-1], LAT[::-1], 4326)  # any input order
    # ~4.44 km x 4.44 km at the equator
    assert aoi.area_m2 / 1e6 == pytest.approx(19.7, rel=0.02)
    assert aoi.polygon.is_valid
    assert aoi.utm_epsg == 32736


def test_crossed_order_fixed():
    aoi = coords.build_aoi([32.15, 32.19, 32.15, 32.19], [-2.85, -2.89, -2.89, -2.85], 4326)
    assert aoi.polygon.is_valid and aoi.area_m2 / 1e6 == pytest.approx(19.7, rel=0.02)


def test_aoi_from_utm_matches_latlon():
    ref = coords.build_aoi(LON, LAT, 4326)
    e, n = ref.to_utm(32736)
    aoi = coords.build_aoi(e, n, 32736)
    assert aoi.area_m2 == pytest.approx(ref.area_m2, rel=1e-6)


def test_swapped_latlon_and_wrong_crs():
    aoi = coords.build_aoi([-2.85, -2.85, -2.89, -2.89], [120.1, 120.2, 120.2, 120.1], 4326)  # lat in X column
    assert any("swapped" in n for n in aoi.notes)
    with pytest.raises(ValueError):
        coords.build_aoi(LON, LAT, 32736)
    with pytest.raises(ValueError):
        coords.build_aoi([1, 2], [1, 2], 4326)


def test_guess_columns_and_wide_row():
    df = pd.DataFrame({"Point": ["A", "B", "C", "D"], "Northing": [1, 2, 3, 4], "Easting": [5, 6, 7, 8]})
    assert coords.guess_columns(df) == ("Easting", "Northing")
    wide = pd.DataFrame([[32.15, -2.85, 32.19, -2.85, 32.19, -2.89, 32.15, -2.89]])
    aoi = coords.aoi_from_table(wide, 0, 1, 4326)
    assert len(aoi.lon) == 4


def test_basemap_stitch_with_fake_tiles():
    aoi = coords.build_aoi(LON, LAT, 4326)
    bbox = basemap.padded_bbox(aoi.bbox)
    fake = lambda z, x, y: Image.new("RGB", (256, 256), ((x * 40) % 255, (y * 40) % 255, 120))  # noqa: E731
    img, z, failed = basemap.satellite_image(bbox, max_tiles=36, fetcher=fake)
    assert failed == 0 and 10 <= z <= 18 and img.width > 100
    fig = basemap.annotated_figure(img, bbox, aoi)
    assert len(export.fig_to_png(fig)) > 1000


def ndimage_smooth(a):
    a = gaussian_filter(a, 6)
    return a / (a.std() or 1)


def _fake_scene(aoi, res=20):
    epsg, tr, w, h = sentinel.target_grid(aoi, res)
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-((xx - w * 0.3) ** 2 + (yy - h * 0.6) ** 2) / (2 * (w * 0.06) ** 2))
    shade = 1 + 0.3 * ndimage_smooth(rng.standard_normal((h, w)))  # shared brightness / topography (multiplies)
    b = {k: (0.1 * shade + 0.01 * rng.standard_normal((h, w))).astype(np.float32) for k in sentinel.BANDS}
    b["B11"] = b["B11"] + 0.15 * blob  # clay anomaly: bright in SWIR-1 ...
    b["B12"] = b["B12"] - 0.03 * blob  # ... absorbing in SWIR-2
    b["B04"] = b["B04"] + 0.08 * blob  # iron anomaly
    b["B08"] = b["B08"] + 0.05
    b["SCL"] = np.full((h, w), 5, np.float32)
    inside = sentinel.polygon_mask(aoi, tr, w, h)
    return b, tr, epsg, inside


def test_indices_targets_and_exports():
    aoi = coords.build_aoi(LON, LAT, 4326)
    bands, tr, epsg, inside = _fake_scene(aoi)
    assert inside.sum() > 0.5 * inside.size
    for k in indices.INDICES:
        assert np.isfinite(indices.compute_index(bands, k)).any()
    for c in indices.COMPOSITES:
        rgb = indices.composite(bands, c, inside)
        assert rgb.shape[2] == 3 and rgb.max() <= 1
    res = indices.alteration_targets(bands, inside, {"clay_hydroxyl": 1, "iron_oxide": 1}, 0.9, 3, 4, tr)
    assert res["zones"], "the planted anomaly should be found"
    # strongest zone should be near the planted blob (30% across, 60% down)
    h, w = inside.shape
    z = res["zones"][0]
    assert abs(z["col"] / w - 0.3) < 0.1 and abs(z["row"] / h - 0.6) < 0.1
    tif = export.geotiff_bytes({"score": res["score"]}, tr, epsg)
    assert tif[:2] in (b"II", b"MM")
    assert export.rgb_geotiff_bytes(indices.composite(bands, "True colour (B4-B3-B2)"), tr, epsg)[:2] in (b"II", b"MM")
    assert "<Polygon>" in export.aoi_kml(aoi)


def test_load_scene_with_local_cogs(tmp_path):
    """load_scene reads/warps rasters onto the AOI grid (local files stand in for Sentinel-2 COGs)."""
    import rasterio
    from rasterio.transform import from_origin

    aoi = coords.build_aoi(LON, LAT, 4326)
    e, n = aoi.to_utm()
    tr = from_origin(min(e) - 2000, max(n) + 2000, 10, 10)
    w, h = int((max(e) - min(e) + 4000) / 10), int((max(n) - min(n) + 4000) / 10)
    assets = {}
    for b in sentinel.BANDS:
        p = tmp_path / f"{b}.tif"
        data = np.full((h, w), 5 if b == "SCL" else 2000, np.uint16)
        with rasterio.open(p, "w", driver="GTiff", height=h, width=w, count=1, dtype="uint16",
                           crs="EPSG:32736", transform=tr, nodata=0) as dst:
            dst.write(data, 1)
        assets[b] = {"href": str(p)}
    item = {"id": "fake", "assets": assets,
            "properties": {"datetime": "2026-10-01T08:00:00Z", "s2:processing_baseline": "05.10"}}
    sc = sentinel.load_scene([item], aoi, 20, opener=rasterio.open)
    v = sc["bands"]["B04"][sc["inside"]]
    assert np.allclose(v, (2000 - 1000) / 10000)  # offset applied
    assert sc["clear"].sum() == sc["inside"].sum()


# ---------- extended Step-3 outputs ----------
def _fake_dem(shape, res, az_deg=45):
    """Gentle slope with a straight valley (fault) striking az_deg."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    e, n = xx * res, -yy * res
    t = np.radians(az_deg)
    across = (e - e.mean()) * np.cos(t) - (n - n.mean()) * np.sin(t)  # distance across the strike
    return (1200 + 0.02 * e + 40 * (1 - np.exp(-(across / 60) ** 2))).astype(np.float32)


def test_terrain():
    from geoai import terrain
    dem = _fake_dem((120, 120), 20)
    hs = terrain.hillshade(dem, 20)
    assert hs.shape == dem.shape and 0 <= np.nanmin(hs) and np.nanmax(hs) <= 1
    flat = terrain.slope_deg(np.full((10, 10), 100.0), 10)
    assert np.allclose(flat, 0)
    ramp = np.tile(np.arange(10, dtype=float) * 10, (10, 1))  # rises to the east 1 m/m
    assert np.allclose(terrain.slope_deg(ramp, 10)[5, 5], 45, atol=0.5)
    assert abs(terrain.aspect_deg(ramp, 10)[5, 5] - 270) < 1  # faces west


def test_lineaments_find_planted_fault():
    from geoai import lineaments, terrain
    res = 20
    dem = _fake_dem((200, 200), res, az_deg=45)
    valid = np.ones(dem.shape, bool)
    imgs = [terrain.hillshade(dem, res, a, 30) for a in (0, 45, 90, 135)]
    lines, _ = lineaments.detect(imgs, valid, res, sigma=2, high_pct=90, min_length_m=600, max_gap_m=60)
    assert lines
    starts, lens = lineaments.rose_stats(lines)
    main = starts[np.argmax(lens)] + 5
    assert abs(main - 45) <= 15, main
    d = lineaments.density(lines, dem.shape, res)
    assert d.max() > 0
    geo = lineaments.to_geojson(lines, __import__("rasterio").transform.from_origin(400000, 9700000, res, res), 32736)
    assert "LineString" in geo


def test_pca_crosta_and_lithology():
    from geoai import lithology, pca
    aoi = coords.build_aoi(LON, LAT, 4326)
    bands, tr, epsg, inside = _fake_scene(aoi)
    res = pca.pca(bands, pca.PCA_BANDS, inside)
    assert np.isclose(res["explained"].sum(), 1) and res["pcs"].shape[0] == 6
    cr = pca.crosta(bands, inside, "hydroxyl")
    h, w = inside.shape
    blob = cr["image"][int(h * 0.6) - 3:int(h * 0.6) + 3, int(w * 0.3) - 3:int(w * 0.3) + 3]
    assert np.nanmean(blob) > np.nanpercentile(cr["image"], 90)  # planted clay is bright
    lr = lithology.classify(bands, inside, k=4)
    assert set(np.unique(lr["labels"][inside])) <= {0, 1, 2, 3} and len(lr["means"]) == 4


def test_prospectivity_combine():
    from geoai import prospectivity
    valid = np.ones((50, 50), bool)
    a = np.zeros((50, 50), np.float32)
    a[10:15, 10:15] = 1
    b = np.zeros((50, 50), np.float32)
    b[10:15, 10:15] = 1
    b[30:35, 30:35] = 1
    la = prospectivity.fuzzy_linear(a, valid, 0, 100)
    lb = prospectivity.fuzzy_linear(b, valid, 0, 100)
    for m in ["Weighted sum", "Fuzzy gamma"]:
        s = prospectivity.combine({"a": la, "b": lb}, {"a": 1, "b": 1}, valid, m)
        assert s[12, 12] > s[32, 32] > s[0, 0]
    z = prospectivity.targets(s, valid, top_pct=1, min_pixels=4)
    assert z["zones"]
    dist = np.where(np.arange(10) < 5, 0, 500)[None, :].repeat(3, 0).astype(np.float32)
    f = prospectivity.fuzzy_distance(dist, np.ones_like(dist, bool), 250)
    assert f[0, 0] == 1 and f[0, 9] < 0.25


def test_load_dem_with_local_tile(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from geoai.imagery import dem as dem_mod
    aoi = coords.build_aoi(LON, LAT, 4326)
    p = tmp_path / "dem.tif"
    with rasterio.open(p, "w", driver="GTiff", height=100, width=100, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(32.1, -2.8, 0.002, 0.002), nodata=-32767) as d:
        d.write(np.full((100, 100), 1234.5, np.float32), 1)
    epsg, tr, w, h = sentinel.target_grid(aoi, 30)
    out = dem_mod.load_dem(aoi, epsg, tr, w, h, items=[{"assets": {"data": {"href": str(p)}}}], opener=rasterio.open)
    assert np.allclose(out[np.isfinite(out)], 1234.5) and np.isfinite(out).mean() > 0.9
