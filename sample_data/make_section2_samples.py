"""Generate a consistent synthetic test dataset for Section 2 (all in WGS84 / UTM 36S, EPSG:32736).

A NE-striking (045°) gold-bearing shear zone crosses the area. It shows up as:
  magnetics   induced field at inclination -28°, declination 0.6° (as at the site): a narrow magnetic low
              (magnetite destroyed) next to a magnetic dyke, plus a NW dyke and a deep body
  IP          high chargeability 60-130 m deep where the shear crosses each survey line
  geology     sheared schist, quartz veins and greenstone along the shear; granite to the SE; NE strikes
  geochem     Au (ppb, with '<5' values) and As/Sb/Bi pathfinders along the shear, hot spot at the target
  remote      clay/hydroxyl and iron oxide indices (GeoTIFF) higher over the altered zone

Run:  python sample_data/make_section2_samples.py
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(42)
E0, E1, N0, N1 = 405600.0, 410000.0, 9680600.0, 9685000.0
TARGET = (407300.0, 9682300.0)
SHEAR_AZ = 45.0


def across(e, n, az, p0):
    t = np.radians(az)
    return (e - p0[0]) * np.cos(t) - (n - p0[1]) * np.sin(t)  # perpendicular distance to a line of strike az


def along(e, n, az, p0):
    t = np.radians(az)
    return (e - p0[0]) * np.sin(t) + (n - p0[1]) * np.cos(t)


INC, DEC = -28.0, 0.6  # Earth's field at the site (IGRF, Geita area): low magnetic latitude


def dipole(e, n, src, depth, moment):
    """TMI of an induced point dipole (field and magnetisation along INC / DEC)."""
    I, D = np.radians(INC), np.radians(DEC)
    f = np.array([np.cos(I) * np.sin(D), np.cos(I) * np.cos(D), np.sin(I)])
    rx, ry, rz = e - src[0], n - src[1], -depth
    r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    md = f[0] * rx + f[1] * ry + f[2] * rz
    return moment * ((3 * md * rx / r ** 2 - f[0]) * f[0] + (3 * md * ry / r ** 2 - f[1]) * f[1] +
                     (3 * md * rz / r ** 2 - f[2]) * f[2]) / r ** 3


def dyke(e, n, az, p0, depth, amp, length=7000.0, step=40.0):
    """Thin dyke / sheet as a row of induced dipoles along strike (top at `depth`); peak ~2 x amp nT."""
    t = np.radians(az)
    out = np.zeros(np.shape(e))
    for s in np.arange(-length / 2, length / 2 + step, step):
        out += dipole(e, n, (p0[0] + s * np.sin(t), p0[1] + s * np.cos(t)), depth, amp * step * depth ** 2)
    return out


def sphere(e, n, p0, depth, amp):
    return dipole(e, n, p0, depth, amp * depth ** 3)


def tmi(e, n):
    f = 32886 + 0.004 * (e - E0) - 0.002 * (n - N0)
    f = f + dyke(e, n, SHEAR_AZ, (TARGET[0] + 250, TARGET[1] - 250), 80, 90)      # magnetic dyke / BIF beside shear
    f = f - dyke(e, n, SHEAR_AZ, TARGET, 30, 30)                                  # shear: magnetite destroyed
    f = f + dyke(e, n, 135.0, (408200, 9682800), 60, 60)                          # NW-trending dolerite dyke
    f = f + sphere(e, n, (409200, 9684100), 150, 200)                              # deep mafic plug
    return f + rng.normal(0, 1.0, np.shape(e))


def gold_signal(e, n):
    x = across(e, n, SHEAR_AZ, TARGET)
    a = along(e, n, SHEAR_AZ, TARGET)
    shear = np.exp(-(x / 80) ** 2) * np.exp(-(a / 1800) ** 2)
    hot = np.exp(-((e - TARGET[0]) ** 2 + (n - TARGET[1]) ** 2) / (2 * 180 ** 2))
    hot2 = 0.6 * np.exp(-((e - 408500) ** 2 + (n - 9683500) ** 2) / (2 * 150 ** 2))
    return np.clip(shear, 0, 1) * 0.6 + hot + hot2


def magnetics():
    rows = []
    for k, n in enumerate(np.arange(N0, N1 + 1, 100.0)):
        e = np.arange(E0, E1 + 1, 20.0)
        rows.append(f"Line {1000 + k * 10}")
        for ee, f in zip(e, tmi(e, np.full_like(e, n))):
            rows.append(f"{ee:10.1f} {n:11.1f} {f:10.2f}")
    with open(os.path.join(OUT, "magnetics_tmi.dat"), "w") as fh:
        fh.write("/ GeoAI synthetic ground magnetic survey, UTM 36S (EPSG:32736)\n/ X Y MAG_nT\n")
        fh.write("\n".join(rows) + "\n")


def ip():
    rows = []
    a = 50.0
    for line_n in (9681600.0, 9682300.0, 9683000.0):
        for st in np.arange(E0 + 200, E1 - 200, 50.0):
            for nl in range(1, 7):
                depth = a * (0.416 + 0.2 * (nl - 1))
                x = across(st, line_n, SHEAR_AZ, TARGET)
                body = np.exp(-(x / 90) ** 2) * np.exp(-((depth - 95) / 40) ** 2)
                charge = 4 + 22 * body + rng.normal(0, 0.8)
                res = 900 - 500 * body + 300 * (st > TARGET[0] + 800) + rng.normal(0, 40)
                rows.append({"Line": f"L{int(line_n) % 10000}", "Station_E": st, "Northing": line_n, "n": nl,
                             "Chargeability_mVV": round(charge, 2), "Resistivity_ohmm": round(res, 1)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "ip_dipole_dipole.csv"), index=False)
    for line, g in df.groupby("Line"):  # one file per line (for the IP page); northing typed in by the user
        g[["Station_E", "n", "Chargeability_mVV", "Resistivity_ohmm"]].to_csv(
            os.path.join(OUT, f"ip_line_{line}_N{int(g.Northing.iloc[0])}.csv"), index=False)


def geology():
    pts = []
    for _ in range(160):
        e, n = rng.uniform(E0, E1), rng.uniform(N0, N1)
        x = across(e, n, SHEAR_AZ, TARGET)
        if abs(x) < 70:
            lith = rng.choice(["Sheared schist", "Quartz vein"], p=[0.7, 0.3])
        elif x > 700:
            lith = "Granite"
        elif x < -900:
            lith = rng.choice(["Laterite cover", "Granite"], p=[0.6, 0.4])
        elif abs(x - 350) < 120:
            lith = "BIF"
        else:
            lith = "Mafic metavolcanic (greenstone)"
        strike = (SHEAR_AZ + rng.normal(0, 8)) % 360 if lith != "Laterite cover" and rng.random() < 0.6 else np.nan
        dip = np.clip(rng.normal(70, 8), 30, 89) if np.isfinite(strike) else np.nan
        pts.append({"Station": f"GS{len(pts) + 1:03d}", "Easting": round(e, 1), "Northing": round(n, 1),
                    "Lithology": lith, "Strike": round(strike) if np.isfinite(strike) else None,
                    "Dip": round(dip) if np.isfinite(dip) else None})
    pd.DataFrame(pts).to_csv(os.path.join(OUT, "geology_mapping.csv"), index=False)


def geochem():
    e, n = np.meshgrid(np.arange(E0 + 50, E1, 100.0), np.arange(N0 + 50, N1, 100.0))
    e, n = e.ravel(), n.ravel()
    g = gold_signal(e, n)
    au = np.exp(np.log(3) + 0.7 * rng.standard_normal(e.size) + 3.6 * g)
    rows = []
    for i in range(e.size):
        def lognorm(base, k, s=0.4):
            return float(np.exp(np.log(base) + s * rng.standard_normal() + k * g[i]))
        a = au[i]
        rows.append({"Sample_ID": f"SS{i + 1:05d}", "Easting": e[i], "Northing": n[i],
                     "Au_ppb": f"<5" if a < 5 else round(a, 1),
                     "As_ppm": round(lognorm(8, 2.2), 1), "Sb_ppm": round(lognorm(0.4, 1.6), 2),
                     "Bi_ppm": round(lognorm(0.15, 1.2), 2), "Cu_ppm": round(lognorm(30, 0.5), 1),
                     "Pb_ppm": round(lognorm(12, 0.0), 1), "Zn_ppm": round(lognorm(40, 0.0), 1)})
    pd.DataFrame(rows).to_excel(os.path.join(OUT, "soil_geochem.xlsx"), index=False)


def remote():
    import rasterio
    from rasterio.transform import from_origin
    from scipy.ndimage import gaussian_filter

    res = 20.0
    w, h = int((E1 - E0) / res), int((N1 - N0) / res)
    tr = from_origin(E0, N1, res, res)
    e, n = np.meshgrid(E0 + res * (np.arange(w) + 0.5), N1 - res * (np.arange(h) + 0.5))
    g = gold_signal(e, n)
    tex = gaussian_filter(rng.standard_normal((h, w)), 6) * 4
    clay = 1.30 + 0.12 * g + 0.02 * tex + 0.01 * rng.standard_normal((h, w))
    iron = 1.60 + 0.25 * g + 0.05 * tex + 0.02 * rng.standard_normal((h, w))
    with rasterio.open(os.path.join(OUT, "rs_alteration_indices.tif"), "w", driver="GTiff", height=h, width=w,
                       count=2, dtype="float32", crs="EPSG:32736", transform=tr, nodata=np.nan) as dst:
        dst.write(clay.astype("float32"), 1)
        dst.write(iron.astype("float32"), 2)
        dst.set_band_description(1, "clay_hydroxyl_B11_B12")
        dst.set_band_description(2, "iron_oxide_B4_B2")


def map_image():
    """A coloured TMI map picture (to test image upload). Extent = the survey area above."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e, n = np.meshgrid(np.linspace(E0, E1, 300), np.linspace(N1, N0, 300))
    f = tmi(e, n)
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(f, cmap="jet", vmin=np.percentile(f, 2), vmax=np.percentile(f, 98))
    ax.axis("off")
    fig.savefig(os.path.join(OUT, "magnetic_map_image.png"))
    plt.close(fig)


if __name__ == "__main__":
    magnetics()
    ip()
    geology()
    geochem()
    remote()
    map_image()
    print("Sample files written to", OUT)
