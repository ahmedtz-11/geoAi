"""Offline tests for Section 2 (interpretation & prediction).  Run:  python -m pytest tests"""
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SAMPLES = os.path.join(ROOT, "sample_data")

from geoai import geochem, geophysics, interpret, readers  # noqa: E402


def test_read_geosoft_xyz():
    txt = "/ survey\n/ X Y MAG\nLine 10\n100 200 33000\n110 200 *\nLine 20\n100 300 33010\n"
    df = readers.parse_text_table(txt)
    assert list(df.columns) == ["line", "X", "Y", "MAG"]
    assert df.line.tolist() == ["10", "10", "20"] and pd.isna(df.MAG.iloc[1])


def test_read_plain_table_without_header():
    df = readers.parse_text_table("1 2 3\n4 5 6\n7 8 9\n")
    assert df.shape == (3, 3) and df.columns[0] == "col_0"


def test_geochem_cleaning_and_thresholds():
    v, bdl = geochem.clean_values(pd.Series(["<5", "12", -2, "BDL", ">1000", 3.5]))
    assert np.allclose(v[[0, 1, 2, 4, 5]], [2.5, 12, 1, 1000, 3.5]) and np.isnan(v[3])
    assert bdl.tolist() == [True, False, True, False, False, False]
    assert geochem.element_of("Au_ppb") == ("Au", "ppb") and geochem.element_of("As ppm")[0] == "As"
    assert geochem.element_of("Station")[0] is None and geochem.element_of("Easting")[0] is None
    rng = np.random.default_rng(0)
    vals = np.exp(rng.normal(1, 0.5, 2000))
    lim = geochem.thresholds(vals)
    assert lim[0] < lim[1] < lim[2]
    assert 0.01 < (vals >= lim[1]).mean() < 0.05  # ~2.3% above mean+2σ for lognormal data


def _sphere_grid(depth=120.0, cell=20.0, n=128):
    xs = np.arange(n) * cell
    X, Y = np.meshgrid(xs, xs[::-1])
    cx = cy = xs[n // 2]
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    T = 1e9 * (2 * depth ** 2 - r2) / (r2 + depth ** 2) ** 2.5  # vertical dipole, vertical field (SI = 3)
    from rasterio.transform import from_origin
    return geophysics.grid_from_raster(T, from_origin(-cell / 2, xs[-1] + cell / 2, cell, cell)), (cx, cy)


def test_euler_recovers_sphere_depth():
    grid, (cx, cy) = _sphere_grid(depth=120)
    _, derivs, zs = geophysics.magnetic_products(grid)
    sols = geophysics.euler(grid, zs, derivs, si=3.0, window=10)
    near = sols[np.hypot(sols.x - cx, sols.y - cy) < 150]
    assert len(near) >= 3
    assert abs(near.depth.median() - 120) < 25, near.depth.median()


def test_derivative_signs():
    from rasterio.transform import from_origin
    a = np.tile(np.arange(50, dtype=float), (50, 1))  # increases to the east, 1 unit per cell
    g = geophysics.grid_from_raster(a, from_origin(0, 1000, 10, 10))
    dx, dy, dz = geophysics.derivatives(g["zf"], 10)
    assert np.allclose(dx[10:40, 10:40], 0.1, atol=0.02) and np.allclose(dy[10:40, 10:40], 0, atol=0.02)
    b = np.tile(np.arange(50, dtype=float)[::-1, None], (1, 50))  # increases to the north
    dx, dy, _ = geophysics.derivatives(b, 10)
    assert np.allclose(dy[10:40, 10:40], 0.1, atol=0.02)


def test_section_anomalies():
    x = np.repeat(np.arange(0, 500, 50.0), 4)
    d = np.tile([10, 20, 30, 40.0], 10)
    v = np.where((x >= 200) & (x <= 250) & (d >= 20), 20, 3.0)
    an = geophysics.section_anomalies(x, d, v, 10)
    assert len(an) == 1 and an.iloc[0].from_x == 200 and an.iloc[0].depth_top == 20


@pytest.mark.skipif(not os.path.exists(os.path.join(SAMPLES, "soil_geochem.xlsx")), reason="samples not generated")
def test_full_run_on_samples_finds_planted_target():
    def up(name):
        with open(os.path.join(SAMPLES, name), "rb") as fh:
            b = io.BytesIO(fh.read())
        b.name = name
        return readers.read_upload(b)

    specs = [
        {"slot": "geophysics", "name": "mag", "kind": "table", "data": up("magnetics_tmi.dat"), "survey": geophysics.SURVEYS[0],
         "geometry": "points", "xcol": "X", "ycol": "Y", "vcol": "MAG_nT", "epsg": 32736},
        {"slot": "geophysics", "name": "ip", "kind": "table", "data": up("ip_dipole_dipole.csv"),
         "survey": geophysics.SURVEYS[1], "geometry": "section", "vcol": "Chargeability_mVV", "linecol": "Line",
         "xcol": "Station_E", "ycol": "Northing", "depth_mode": "n-level", "ncol": "n", "array": "Dipole-dipole",
         "a_spacing": 50.0, "epsg": 32736},
        {"slot": "geology", "name": "geo", "kind": "table", "data": up("geology_mapping.csv"), "xcol": "Easting",
         "ycol": "Northing", "epsg": 32736, "lithcol": "Lithology", "strikecol": "Strike", "dipcol": "Dip"},
        {"slot": "remote", "name": "rs", "kind": "raster", "data": up("rs_alteration_indices.tif"),
         "bands": ["clay_hydroxyl_B11_B12", "iron_oxide_B4_B2"]},
        {"slot": "geochem", "name": "soil", "kind": "table", "data": up("soil_geochem.xlsx"), "xcol": "Easting",
         "ycol": "Northing", "epsg": 32736, "au": "Au_ppb", "pathfinders": ["As_ppm", "Sb_ppm", "Bi_ppm"],
         "idcol": "Sample_ID"},
    ]
    run = interpret.run(specs)
    assert not run["errors"] and run["epsg"] == 32736
    ev = run["evidence"]
    mem = {k: interpret.membership(e["raw"], e["favourable"]) for k, e in ev.items()}
    w = {k: e["default_weight"] for k, e in ev.items()}
    score, _ = interpret.knowledge_driven({k: m for k, m in mem.items() if w[k] > 0}, {k: v for k, v in w.items() if v})
    tr, shape, cell = run["grid"]
    z, t = interpret.describe_targets(score, np.isfinite(score), tr, 5, 20, mem, run["results"], run["datasets"])
    assert len(t)
    top = t.iloc[0]
    # planted gold zone: 045° shear through (407300, 9682300), hot spots there and at (408500, 9683500)
    d1 = np.hypot(top.Easting - 407300, top.Northing - 9682300)
    d2 = np.hypot(top.Easting - 408500, top.Northing - 9683500)
    assert min(d1, d2) < 600, (top.Easting, top.Northing)
    assert top["Est. depth (m)"] is not None and "NE" in str(top.Trend)


# ---------- magnetics page ----------
def _dipole_grid(inc, dec, depth=150.0, cell=20.0, n=160):
    from rasterio.transform import from_origin
    from geoai import magnetics
    xs = np.arange(n) * cell
    X, Y = np.meshgrid(xs, xs[::-1])
    src = (xs[n // 2], xs[n // 2])
    T = magnetics.dipole_tmi(X, Y, src, depth, inc, dec)
    return geophysics.grid_from_raster(T, from_origin(-cell / 2, xs[-1] + cell / 2, cell, cell)), X, Y, src


@pytest.mark.parametrize("inc,dec", [(-60, 10), (-28, 2), (35, -5)])
def test_rtp_matches_vertical_field(inc, dec):
    from geoai import magnetics
    g, X, Y, src = _dipole_grid(inc, dec)
    truth = magnetics.dipole_tmi(X, Y, src, 150, 90, 0)
    rtp = magnetics.reduce(g["zf"], g["cell"], inc, dec, "RTP")
    assert np.corrcoef((rtp - rtp.mean()).ravel(), truth.ravel())[0, 1] > 0.95
    iy, ix = np.unravel_index(np.argmax(rtp), rtp.shape)
    assert abs(ix - 80) <= 2 and abs(iy - 80) <= 2  # peak over the source


def test_rte_matches_equator_field():
    from geoai import magnetics
    g, X, Y, src = _dipole_grid(-10, 2)
    eq = magnetics.dipole_tmi(X, Y, src, 150, 0, 2)
    rte = magnetics.reduce(g["zf"], g["cell"], -10, 2, "RTE")
    assert np.corrcoef(rte.ravel(), eq.ravel())[0, 1] > 0.95


def test_spi_depth_of_contact():
    from rasterio.transform import from_origin
    from geoai import magnetics
    cell, n, h = 20.0, 160, 80.0
    xs = np.arange(n) * cell
    X, _ = np.meshgrid(xs, xs)
    T = 100 * (np.pi / 2 + np.arctan((X - xs[n // 2]) / h))  # vertical contact, vertical field
    g = geophysics.grid_from_raster(T, from_origin(0, n * cell, cell, cell))
    spi, _ = magnetics.spi_depth(g["zf"], cell, smooth_cells=0)
    assert abs(np.nanmedian(spi[20:-20, n // 2 - 1:n // 2 + 2]) - h) < 10


def test_field_direction_low_latitude():
    from geoai import magnetics
    inc, dec, f, src = magnetics.field_direction(32.17, -2.87)
    assert -40 < inc < -15 and abs(dec) < 10
    assert magnetics.auto_method(inc) in ("RTP", "RTE") and magnetics.auto_method(-10) == "RTE"


def test_single_line_is_not_gridded():
    assert geophysics.is_collinear(np.arange(10.0), np.full(10, 5.0))
    with pytest.raises(ValueError):
        geophysics.grid_points(np.arange(10.0), np.full(10, 5.0), np.arange(10.0))
    an = geophysics.section_anomalies(np.array([0, 50, 100.0]), np.array([20, 20, 20.0]), np.array([1, 9, 1.0]), 5)
    assert an.empty  # a single noisy station is not an anomaly


# ---------- instrument IP exports (IRIS Prosys / ELREC style) ----------
IRIS_TEXT = """Version,PPC:,1.27.2.0,Version,Rx:,0.2.5.17,Rx,SN:,1506,,,,
Project:,TEST,,,,,,,,,,,
Windows:,20,Setting:,Arithmetic,Delay,(ms):,240,,,,,,
,Mem,Array,n,Tx1,Tx2,Rx1,Rx2,Contact,Rho,M,ErrM
,1,P-DP,1,9999999,0,20,40,749.05,13.27,66.0,19.7
,1,P-DP,2,9999999,0,40,60,1.24,23.43,1.5,1.8
,1,P-DP,3,9999999,0,60,80,1.34,41.49,3.2,3.6
,2,P-DP,1,9999999,20,40,60,INFINI,20.1,-4.9,0.5
,2,P-DP,2,9999999,20,60,80,------,25.0,2.0,0.4
,2,P-DP,3,9999999,20,80,100,0.9,30.0,2.5,0.3
"""


def test_header_row_after_information_lines():
    df = readers.parse_text_table(IRIS_TEXT)
    assert list(df.columns)[:4] == ["Mem", "Array", "n", "Tx1"] and len(df) == 6
    assert pd.isna(df.Contact.iloc[3]) and pd.isna(df.Contact.iloc[4])  # INFINI / ------ are missing values
    plain = readers.parse_text_table("Station,n,M\n0,1,2.5\n20,1,3.1\n40,2,2.2\n")
    assert list(plain.columns) == ["Station", "n", "M"] and len(plain) == 3


def test_iris_geometry_and_quality():
    from geoai import ipdata
    df = readers.parse_text_table(IRIS_TEXT)
    assert ipdata.is_iris(df) and ipdata.array_name(df) == "Pole-dipole"
    g, arr, a = ipdata.geometry(df)
    assert a == 20 and g.station_x.iloc[0] == 15  # midway between C (0) and the dipole centre (30)
    assert np.allclose(g.pseudo_depth.iloc[:3], 20 * np.array([0.52, 0.91, 1.30]))
    keep, why = ipdata.quality_mask(df, "M", max_err=10)
    assert keep.tolist() == [False, True, True, False, False, True]
    assert why["error above 10"] == 1 and why["negative or zero"] == 1


# ---------- gridding of line-survey data ----------
def _line_survey(station=20.0, line_sp=100.0, ext=1000.0):
    X, Y = np.meshgrid(np.arange(0, ext + 1, station), np.arange(0, ext + 1, line_sp))
    return X.ravel(), Y.ravel()


def _smooth_field(X, Y):
    return (100 * np.exp(-(((X - 500) ** 2 + (Y - 500) ** 2) / (2 * 180.0 ** 2)))
            + 60 * np.sin(X / 300.0) * np.cos(Y / 250.0) + 0.01 * X)


def test_survey_geometry_detects_lines():
    g = geophysics.survey_geometry(*_line_survey())
    assert g["is_line_survey"] and g["n_lines"] == 11
    assert g["station"] == pytest.approx(20) and g["line_spacing"] == pytest.approx(100)
    rng = np.random.default_rng(0)
    assert not geophysics.survey_geometry(*rng.uniform(0, 1000, (2, 500)))["is_line_survey"]
    sq = np.meshgrid(np.arange(0, 1001, 100.0), np.arange(0, 1001, 100.0))
    assert not geophysics.survey_geometry(sq[0].ravel(), sq[1].ravel())["is_line_survey"]


def test_minimum_curvature_reconstructs_field_between_lines():
    x, y = _line_survey()
    z, known = geophysics.minimum_curvature(x, y, _smooth_field(x, y), 0, 1000.0, 25.0, 41, 41)
    assert np.isfinite(z).all()                       # no gaps anywhere
    X, Y = np.meshgrid(np.arange(41) * 25.0, 1000.0 - np.arange(41) * 25.0)
    err = (z - _smooth_field(X, Y))[3:-3, 3:-3]       # interior (edges use a reflecting boundary)
    assert np.sqrt((err ** 2).mean()) < 0.02 * 130    # < 2% of the field range
    at_data = np.abs(z - np.where(known, z, np.nan))[known]
    assert np.nanmax(at_data) == 0                    # readings are honoured exactly


def test_grid_points_fills_between_lines():
    x, y = _line_survey()
    v = _smooth_field(x, y)
    g = geophysics.grid_points(x, y, v)
    assert g["cell"] == pytest.approx(25, rel=0.2)    # about a quarter of the 100 m line spacing
    assert np.isfinite(g["z"][g["valid"]]).all()      # no holes inside the survey
    assert g["coverage"].sum() < g["valid"].sum()     # 'as measured' shows less than the filled view
    assert g["valid"].mean() > 0.9                    # the survey rectangle is covered
    sparse = geophysics.grid_points(x, y, v, method="Linear interpolation", cell=10.0, mask_factor=2.5)
    assert (sparse["valid"] & ~np.isfinite(sparse["z"])).sum() == 0
    assert sparse["valid"].mean() < g["valid"].mean()  # the old masking leaves gaps between the lines


# ---------- pit mapping ----------
def _pits():
    from geoai import coords, pits
    df = pd.read_excel(os.path.join(os.path.dirname(__file__), "..", "sample_data", "pits_sample.xlsx"))
    x, y = coords.guess_columns(df)
    p, epsg = pits.build(df, x, y, 32736, pits.guess_fields(df, used=(x, y)))
    return pits, p, epsg


def test_pit_trend_and_targets():
    pits, p, epsg = _pits()
    tr = pits.trend(p)
    assert abs(tr["azimuth"] - 45) < 5 and tr["linearity"] > 0.9
    assert tr["plunge"] and abs(tr["plunge"]["toward"] - 45) < 5      # pits deepen to the NE
    p = pits.potential(p, tr)
    assert set(p[p.barren].potential) == {"Low"}
    t = pits.propose(p, tr, epsg, 12, 2)
    infill = t[t.kind == "Infill"]
    assert len(infill) == 2 and (infill.along.between(tr["along"].min(), tr["along"].max())).all()
    assert (t.kind == "Step-out").sum() == 4
    d = np.hypot(t.x.to_numpy()[:, None] - p.x.to_numpy()[None], t.y.to_numpy()[:, None] - p.y.to_numpy()[None])
    assert d.min() > 10                                               # never on top of an existing pit


def test_pit_register_roundtrip(tmp_path):
    pits, p, epsg = _pits()
    tr = pits.trend(p)
    p = pits.potential(p, tr)
    t = pits.propose(p, tr, epsg)
    path = str(tmp_path / "data" / "reg.csv")
    rows = pits.register_rows("Site A", p, t, epsg)
    pits.save_register(path, rows)
    again = pits.save_register(path, rows)                            # saving twice must not duplicate
    assert len(again) == len(rows) == len(p) + len(t)
    pits.save_register(path, pits.register_rows("Site B", p, t.head(0), epsg))
    assert set(pits.load_register(path).project) == {"Site A", "Site B"}
    assert "<Placemark>" in pits.to_kml("Site A", p, t)


# ---------- IP inversion ----------
def test_ip_sensitivity_kernel_integrates_to_halfspace_total():
    from geoai import ipinv
    ze = np.r_[0, np.cumsum(5 * 1.1 ** np.arange(60))]
    for s in (10.0, -30.0):
        T = ipinv._template(s, 10.0, 600, ze)
        assert T.sum() * 2 * np.pi * abs(s) == pytest.approx(1.0, abs=0.01)


def test_ip_inversion_recovers_chargeable_body():
    from geoai import ipdata, ipinv
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "sample_data", "ip_electrodes_L1.csv"))
    g, arr, a = ipdata.geometry(df)
    assert arr == "Dipole-dipole" and a == 20
    inv = ipinv.invert(df.Tx1, df.Tx2, df.Rx1, df.Rx2, df.M, g.station_x, g.pseudo_depth)
    an = ipinv.anomalies(inv, np.percentile(inv["model"][inv["mask"]], 95))
    assert 270 <= an["peak_x"] <= 350 and 20 <= an["peak_depth"] <= 65      # body planted at 280-340 m, 25-60 m deep
    assert an["peak"] > 3 * np.median(inv["model"][inv["mask"]])
    assert inv["abs_error"] < 0.1 * np.median(df.M) + 0.5
    import matplotlib.pyplot as plt
    plt.close(ipinv.figure(inv, "L1"))


def test_resistivity_inversion_recovers_conductor_and_resistor():
    from geoai import ipdata, ipinv
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "sample_data", "ip_electrodes_L1.csv"))
    g, _, _ = ipdata.geometry(df)
    inv = ipinv.invert(df.Tx1, df.Tx2, df.Rx1, df.Rx2, df.Rho, g.station_x, g.pseudo_depth, log=True)
    (lo, lx, lz), (hi, hx, hz) = ipinv.extremes(inv)
    assert 270 <= lx <= 350 and 20 <= lz <= 65 and lo < 120            # conductor planted at 280-340 m, 25-60 m, 40 ohm.m
    assert 90 <= hx <= 190 and hz < 20 and hi > 700                    # resistive cap at 100-180 m, top 15 m, 1500 ohm.m
    assert inv["abs_error"] < 6                                        # per cent; 3 % noise was added
    import matplotlib.pyplot as plt
    plt.close(ipinv.figure(inv, "L1"))


# ---------- Res2DInv .dat ----------
def test_res2dinv_general_array_file_matches_electrode_csv():
    import io
    from geoai import ipdata, readers
    here = os.path.join(os.path.dirname(__file__), "..", "sample_data")
    f = io.BytesIO(open(os.path.join(here, "ip_res2dinv_L1.dat"), "rb").read())
    f.name = "ip_res2dinv_L1.dat"
    out = readers.read_upload(f)
    df, ref = out["df"], pd.read_csv(os.path.join(here, "ip_electrodes_L1.csv"))
    assert out["meta"]["format"] == "Res2DInv" and out["meta"]["ip_unit"] == "mV/V" and len(df) == len(ref)
    for a, b in (("Tx1", "Tx1"), ("Rx2", "Rx2"), ("Rho", "Rho"), ("M", "M")):
        assert np.allclose(df[a], ref[b], atol=1e-3)
    g, arr, a = ipdata.geometry(df)                       # n-level worked out from the electrodes
    assert arr == "Dipole-dipole" and a == 20 and np.allclose(g.n_level, ref.n)


def test_res2dinv_pole_dipole_and_index_format():
    from geoai import ipdata, res2d
    general = "\r\n".join(["Line 1", "20.0", "11", "6", "Type of measurement (0=app. resistivity,1=resistance)", "0",
                           "2", "0", "1", "Chargeability", "msec.", "0.0,1.0",
                           "3 -20.0 0.0 20.0 0.0 0.0 0.0 63.8 , 4.129", "3 -20.0 0.0 40.0 0.0 20.0 0.0 138.98 , 3.928",
                           "0", "0"])
    assert res2d.looks_like(general)
    df, meta = res2d.parse(general)
    assert meta["array"] == "Pole-dipole" and meta["ip_unit"] == "msec." and list(df.M) == [4.129, 3.928]
    g, arr, a = ipdata.geometry(df)
    assert arr == "Pole-dipole" and list(g.n_level) == [1, 2] and list(g.station_x) == [-5.0, 5.0]
    index = "\n".join(["Dipole line", "10.0", "3", "2", "1", "0", "25.0 10.0 3 120.5", "40.0 10.0 1 80.0", "0"])
    df, meta = res2d.parse(index)                         # x = midpoint, a = 10, n = 3 -> electrodes 0,10 | 40,50
    assert "M" not in df and sorted(df.iloc[0][["Tx1", "Tx2", "Rx1", "Rx2"]]) == [0, 10, 40, 50]
    assert not res2d.looks_like("X,Y,TMI\n1,2,3\n4,5,6\n7,8,9")
