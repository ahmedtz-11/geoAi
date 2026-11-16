"""Interactive 3-D view of the DEM, with a map draped over the terrain.

Two Plotly primitives, each used where it fits:
  * a measured quantity (elevation, slope, a mineral index) -> Surface, which carries a colour bar and
    reads values back on hover;
  * a picture (shaded relief, satellite image, false colour) -> Mesh3d with a colour per vertex, which
    keeps the exact pixel colours instead of pushing them through a colour scale.

No Streamlit inside.
"""
import warnings

import numpy as np

MAX_SIDE = 400          # grid points along the longer side; 400 x 400 stays smooth to rotate in a browser


def block_mean(a, step):
    """Average `step` x `step` blocks, ignoring NaN. Works on (H, W) and (H, W, C)."""
    if step <= 1:
        return a
    h, w = a.shape[:2]
    h, w = (h // step) * step, (w // step) * step
    if h == 0 or w == 0:
        return a
    b = a[:h, :w]
    shape = (h // step, step, w // step, step) + a.shape[2:]
    with warnings.catch_warnings():                 # a block of nothing but holes is itself a hole
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(b.reshape(shape), axis=(1, 3))


def coarsen(res_m, max_side, *arrays):
    """Reduce every array by the same factor so the longer side is at most `max_side` points."""
    h, w = arrays[0].shape[:2]
    step = max(1, int(np.ceil(max(h, w) / max(max_side, 16))))
    return [block_mean(np.asarray(a, float), step) for a in arrays], res_m * step


def _xy(transform, shape, step):
    """Cell-centre eastings and northings of the coarsened grid."""
    h, w = shape
    x = transform.c + (np.arange(w) + 0.5) * transform.a * step
    y = transform.f + (np.arange(h) + 0.5) * transform.e * step
    return x, y


def _mesh(X, Y, Z, rgb, name, hover_unit=""):
    """Triangulated surface coloured pixel by pixel. Triangles touching a hole are dropped."""
    import plotly.graph_objects as go

    h, w = Z.shape
    idx = np.arange(h * w).reshape(h, w)
    tl, tr, bl, br = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    i = np.concatenate([tl.ravel(), tr.ravel()])
    j = np.concatenate([bl.ravel(), br.ravel()])
    k = np.concatenate([tr.ravel(), bl.ravel()])
    z = Z.ravel()
    ok = np.isfinite(z)
    keep = ok[i] & ok[j] & ok[k]
    colour = (np.clip(np.nan_to_num(rgb, nan=0.6), 0, 1) * 255).astype(np.uint8).reshape(-1, 3)
    return go.Mesh3d(x=np.repeat(X[None, :], h, 0).ravel(), y=np.repeat(Y[:, None], w, 1).ravel(),
                     z=np.nan_to_num(z, nan=float(np.nanmin(z)) if ok.any() else 0.0),
                     i=i[keep], j=j[keep], k=k[keep], vertexcolor=colour, flatshading=False, name=name,
                     lighting=dict(ambient=0.95, diffuse=0.25, specular=0.05, roughness=0.9),
                     hovertemplate="E %{x:,.0f}<br>N %{y:,.0f}<br>elevation %{z:,.0f} m" + hover_unit +
                                   "<extra></extra>")


def _surface(X, Y, Z, colour, colourscale, label, reverse=False):
    import plotly.graph_objects as go

    return go.Surface(x=X, y=Y, z=Z, surfacecolor=colour, colorscale=colourscale, reversescale=reverse,
                      colorbar=dict(title=label, len=0.6, thickness=14),
                      lighting=dict(ambient=0.8, diffuse=0.5, specular=0.1, roughness=0.9),
                      hovertemplate="E %{x:,.0f}<br>N %{y:,.0f}<br>elevation %{z:,.0f} m<br>" +
                                    label + " %{surfacecolor:,.2f}<extra></extra>")


def outline_trace(aoi_utm, dem, transform, step, lift):
    """The area-of-interest boundary, lifted onto the terrain so it is visible over the surface.

    aoi_utm is (eastings, northings) of the corners, as geoai.coords.AOI.to_utm returns them.
    """
    import plotly.graph_objects as go

    e, n = (np.asarray(a, float) for a in aoi_utm)
    if e.size < 3:
        return None
    ring = np.column_stack([np.r_[e, e[0]], np.r_[n, n[0]]])        # close the polygon
    # resample the ring so it follows the ground rather than cutting through hills
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(ring, axis=0).T))]
    t = np.linspace(0, seg[-1], 400)
    xs, ys = np.interp(t, seg, ring[:, 0]), np.interp(t, seg, ring[:, 1])
    h, w = dem.shape
    col = np.clip(((xs - transform.c) / (transform.a * step)).astype(int), 0, w - 1)
    row = np.clip(((ys - transform.f) / (transform.e * step)).astype(int), 0, h - 1)
    z = dem[row, col]
    if not np.isfinite(z).any():
        return None
    return go.Scatter3d(x=xs, y=ys, z=np.nan_to_num(z, nan=np.nanmedian(z)) + lift, mode="lines",
                        line=dict(color="#d00000", width=5), name="Area of interest", hoverinfo="skip")


def figure(dem, res_m, transform, drape, exaggeration=2.0, max_side=MAX_SIDE, aoi_utm=None, title="",
           hillshade=None, scalar=None, scalar_label="", colourscale="Viridis", reverse=False):
    """3-D terrain.

    drape: "rgb" with an (H, W, 3) picture in `hillshade`-blended form passed as `scalar`=None, or
           "scalar" with a measured grid in `scalar`.
    Everything is coarsened together, so the picture always lines up with the terrain.
    """
    import plotly.graph_objects as go

    layers = [dem] + ([hillshade] if hillshade is not None else []) + ([scalar] if scalar is not None else [])
    if drape is not None and np.ndim(drape) == 3:
        layers.append(drape)
    h0, w0 = dem.shape
    step = max(1, int(np.ceil(max(h0, w0) / max(max_side, 16))))
    done = [block_mean(np.asarray(a, float), step) for a in layers]
    Z = done[0]
    rest = done[1:]
    rgb = rest.pop() if (drape is not None and np.ndim(drape) == 3) else None
    sc = rest.pop() if scalar is not None else None
    X, Y = _xy(transform, Z.shape, step)
    relief = float(np.nanmax(Z) - np.nanmin(Z)) if np.isfinite(Z).any() else 1.0

    if rgb is not None:
        trace = _mesh(X, Y, Z, rgb, title)
    else:
        trace = _surface(X, Y, Z, sc if sc is not None else Z, colourscale, scalar_label or "Elevation (m)", reverse)
    data = [trace]
    if aoi_utm is not None:
        t = outline_trace(aoi_utm, Z, transform, step, 0.01 * max(relief, 1.0) + 2)
        if t is not None:
            data.append(t)
    width_m = float(np.ptp(X)) or 1.0
    height_m = float(np.ptp(Y)) or 1.0
    long_side = max(width_m, height_m)
    z_aspect = float(np.clip(exaggeration * relief / long_side, 0.04, 1.6))
    # Plotly frames the whole bounding box, and the box of a landscape is a thin slab, so the default
    # camera leaves most of the picture empty. Move the eye in as the slab gets thinner.
    eye = 1.05 + 0.55 * min(z_aspect, 1.0)
    direction = np.array([0.62, -0.66, 0.42])
    direction = direction / np.linalg.norm(direction) * eye
    fig = go.Figure(data)
    fig.update_layout(
        height=640, margin=dict(l=0, r=0, t=36, b=0), title=title, showlegend=False,
        scene=dict(xaxis_title="Easting (m)", yaxis_title="Northing (m)", zaxis_title="Elevation (m)",
                   aspectmode="manual",
                   aspectratio=dict(x=width_m / long_side, y=height_m / long_side,
                                    z=z_aspect),
                   camera=dict(eye=dict(x=direction[0], y=direction[1], z=direction[2])),
                   xaxis=dict(showbackground=False), yaxis=dict(showbackground=False),
                   zaxis=dict(showbackground=True, backgroundcolor="rgba(240,240,240,0.5)")))
    return fig, step, res_m * step


def shaded_rgb(dem, hs, cmap="terrain", lo=1, hi=99):
    """Elevation colours blended with hillshade: the shaded-relief picture, as an (H, W, 3) array."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    f = dem[np.isfinite(dem)]
    norm = Normalize(*np.percentile(f, [lo, hi])) if f.size else Normalize(0, 1)
    base = plt.get_cmap(cmap)(norm(np.nan_to_num(dem, nan=float(np.nanmean(dem)) if f.size else 0.0)))[..., :3]
    shade = np.nan_to_num(hs, nan=0.5)[..., None]
    return np.clip(base * (0.35 + 0.75 * shade), 0, 1)


def shade_rgb(rgb, hs, strength=0.75):
    """Light an existing picture (satellite, false colour) with the hillshade so relief shows through."""
    shade = np.nan_to_num(hs, nan=0.5)[..., None]
    return np.clip(np.asarray(rgb, float) * ((1 - strength) + (strength + 0.35) * shade), 0, 1)
