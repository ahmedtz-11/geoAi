"""Export helpers: PNG figures, GeoTIFF rasters, KML / GeoJSON / CSV for the AOI."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fig_to_png(fig, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def image_to_png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def geotiff_bytes(layers, transform, epsg, nodata=np.nan):
    """Multi-band float32 GeoTIFF from {name: 2-D array}; band descriptions = names."""
    import rasterio
    from rasterio.io import MemoryFile

    names = list(layers)
    h, w = layers[names[0]].shape
    profile = dict(driver="GTiff", height=h, width=w, count=len(names), dtype="float32",
                   crs=f"EPSG:{epsg}", transform=transform, nodata=nodata, compress="deflate")
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            for i, n in enumerate(names, 1):
                dst.write(np.asarray(layers[n], np.float32), i)
                dst.set_band_description(i, n)
        return mem.read()


def rgb_geotiff_bytes(rgb, transform, epsg):
    """8-bit RGB GeoTIFF from an (H, W, 3) 0-1 array (opens directly in QGIS / ArcGIS)."""
    from rasterio.io import MemoryFile

    data = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    h, w, _ = data.shape
    profile = dict(driver="GTiff", height=h, width=w, count=3, dtype="uint8",
                   crs=f"EPSG:{epsg}", transform=transform, photometric="RGB", compress="deflate")
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            for i in range(3):
                dst.write(data[:, :, i], i + 1)
        return mem.read()


def aoi_kml(aoi, name="GeoAI AOI"):
    ring = list(zip(aoi.lon, aoi.lat)) + [(aoi.lon[0], aoi.lat[0])]
    coords = " ".join(f"{x:.7f},{y:.7f},0" for x, y in ring)
    marks = "".join(
        f"<Placemark><name>C{i + 1}</name><Point><coordinates>{x:.7f},{y:.7f},0</coordinates></Point></Placemark>"
        for i, (x, y) in enumerate(zip(aoi.lon, aoi.lat)))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>{name}</name>
<Style id="aoi"><LineStyle><color>ff00d4ff</color><width>3</width></LineStyle>
<PolyStyle><color>3300d4ff</color></PolyStyle></Style>
<Placemark><name>{name}</name><description>Area {aoi.area_m2 / 1e6:.4f} km2</description><styleUrl>#aoi</styleUrl>
<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon>
</Placemark>{marks}</Document></kml>"""


def raster_figure(arr, transform, aoi_utm=None, title="", cmap="viridis", valid=None,
                  overlay=None, cbar_label="", lo=2, hi=98):
    """Map of a single-band raster in UTM metres with AOI outline and optional target overlay."""
    h, w = arr.shape
    left, top = transform.c, transform.f
    ext = [left, left + w * transform.a, top + h * transform.e, top]
    show = np.where(valid, arr, np.nan) if valid is not None else arr
    finite = show[np.isfinite(show)]
    vmin, vmax = (np.percentile(finite, [lo, hi]) if finite.size else (0, 1))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(show, extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    cb = plt.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(cbar_label)
    if overlay is not None and overlay.any():
        ov = np.ma.masked_where(~overlay, overlay)
        ax.imshow(ov, extent=ext, cmap="cool", alpha=0.9, interpolation="nearest", vmin=0, vmax=1)
    if aoi_utm is not None:
        e, n = aoi_utm
        ax.plot(list(e) + [e[0]], list(n) + [n[0]], color="black", lw=1.6)
    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(useOffset=False, style="plain")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def rgb_figure(rgb, transform, aoi_utm=None, title=""):
    h, w, _ = rgb.shape
    left, top = transform.c, transform.f
    ext = [left, left + w * transform.a, top + h * transform.e, top]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(rgb, extent=ext, interpolation="nearest")
    if aoi_utm is not None:
        e, n = aoi_utm
        ax.plot(list(e) + [e[0]], list(n) + [n[0]], color="#ffd400", lw=1.8)
    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(useOffset=False, style="plain")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def class_figure(cls, transform, labels, colors, aoi_utm=None, title=""):
    """Map of a classified raster (values 0..n-1, -1 = no data) with a legend."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    h, w = cls.shape
    left, top = transform.c, transform.f
    ext = [left, left + w * transform.a, top + h * transform.e, top]
    show = np.ma.masked_where(cls < 0, cls)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(show, extent=ext, cmap=ListedColormap(colors), vmin=-0.5, vmax=len(colors) - 0.5,
              interpolation="nearest")
    if aoi_utm is not None:
        e, n = aoi_utm
        ax.plot(list(e) + [e[0]], list(n) + [n[0]], color="black", lw=1.6)
    handles = [Patch(color=c, label=l) for c, l in zip(colors, labels)]
    if max(len(l) for l in labels) > 24:  # long names: legend under the map so the map keeps its size
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False,
                  ncol=2, fontsize=8)
    else:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(useOffset=False, style="plain")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


# ---------- figures for the extended remote-sensing outputs ----------
def _extent(transform, shape):
    h, w = shape[:2]
    return [transform.c, transform.c + w * transform.a, transform.f + h * transform.e, transform.f]


def _outline(ax, aoi_utm, color="black", lw=1.4):
    if aoi_utm is not None:
        e, n = aoi_utm
        ax.plot(list(e) + [e[0]], list(n) + [n[0]], color=color, lw=lw)


def _axes_style(ax, title="", labels=True):
    ax.set_title(title, fontsize=10)
    ax.ticklabel_format(useOffset=False, style="plain")
    if labels:
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def panels_figure(arrays, titles, cmaps, transform, valid=None, aoi_utm=None, ncols=3, suptitle="",
                  lo=2, hi=98):
    """Small multiples of single-band rasters (each with its own percentile stretch)."""
    n = len(arrays)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), squeeze=False)
    for ax, a, t, cm in zip(axes.ravel(), arrays, titles, cmaps):
        show = np.where(valid, a, np.nan) if valid is not None else a
        f = show[np.isfinite(show)]
        vmin, vmax = np.percentile(f, [lo, hi]) if f.size else (0, 1)
        im = ax.imshow(show, extent=_extent(transform, a.shape), cmap=cm, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, shrink=0.75)
        _outline(ax, aoi_utm)
        _axes_style(ax, t, labels=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def relief_figure(dem, hs, transform, aoi_utm=None, title="", contour_step=None):
    """Elevation colours blended with hillshade, plus contours."""
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(8, 7))
    ext = _extent(transform, dem.shape)
    f = dem[np.isfinite(dem)]
    norm = Normalize(*np.percentile(f, [1, 99])) if f.size else Normalize(0, 1)
    rgb = plt.get_cmap("terrain")(norm(np.nan_to_num(dem, nan=float(np.nanmean(dem)))))[..., :3]
    shade = np.nan_to_num(hs, nan=0.5)[..., None]
    blend = np.clip(rgb * (0.35 + 0.75 * shade), 0, 1)
    blend[np.isnan(dem)] = 1
    ax.imshow(blend, extent=ext, interpolation="bilinear")
    sm = plt.cm.ScalarMappable(norm=norm, cmap="terrain")
    plt.colorbar(sm, ax=ax, shrink=0.8, label="Elevation (m)")
    if contour_step:
        h, w = dem.shape
        xs = np.linspace(ext[0], ext[1], w)
        ys = np.linspace(ext[3], ext[2], h)
        levels = np.arange(np.floor(np.nanmin(dem) / contour_step) * contour_step, np.nanmax(dem) + contour_step,
                           contour_step)
        cs = ax.contour(xs, ys, dem, levels=levels, colors="k", linewidths=0.4, alpha=0.6)
        ax.clabel(cs, levels[::2], fontsize=6, fmt="%d")
    _outline(ax, aoi_utm, "#d00000", 1.8)
    _axes_style(ax, title)
    fig.tight_layout()
    return fig


def lines_figure(background, lines, transform, aoi_utm=None, title="", cmap="gray"):
    """Lineaments over a greyscale background, coloured by strike direction."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ext = _extent(transform, background.shape)
    f = background[np.isfinite(background)]
    vmin, vmax = np.percentile(f, [2, 98]) if f.size else (0, 1)
    ax.imshow(background, extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
    cm = plt.get_cmap("hsv")
    for l in lines:
        x = [transform.c + transform.a * (l["x0"] + 0.5), transform.c + transform.a * (l["x1"] + 0.5)]
        y = [transform.f + transform.e * (l["y0"] + 0.5), transform.f + transform.e * (l["y1"] + 0.5)]
        ax.plot(x, y, color=cm(l["azimuth"] / 180), lw=1.6)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 180), cmap="hsv")
    cb = plt.colorbar(sm, ax=ax, shrink=0.8)
    cb.set_label("Strike (° from north)")
    _outline(ax, aoi_utm, "#ffd400", 1.8)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    _axes_style(ax, title)
    fig.tight_layout()
    return fig


def rose_figure(bin_starts, lengths, bin_deg=10, title="Lineament strike rose"):
    """Length-weighted rose diagram (bidirectional, 0-360°)."""
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    th = np.radians(np.concatenate([bin_starts, bin_starts + 180]) + bin_deg / 2)
    r = np.concatenate([lengths, lengths])
    ax.bar(th, r, width=np.radians(bin_deg) * 0.95, color="#c0392b", edgecolor="white", alpha=0.85)
    ax.set_title(title, pad=16)
    ax.set_yticklabels([])
    fig.tight_layout()
    return fig


def loadings_figure(loadings, names, explained, title="PCA loadings"):
    k = loadings.shape[0]
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(names), 0.6 + 0.45 * k))
    im = ax.imshow(loadings, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(k), [f"PC{i + 1} ({100 * e:.1f}%)" for i, e in enumerate(explained)])
    for i in range(k):
        for j in range(len(names)):
            ax.text(j, i, f"{loadings[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(loadings[i, j]) > 0.55 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig
