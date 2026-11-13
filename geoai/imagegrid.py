"""Turn a map picture (e.g. a coloured magnetic or chargeability map) into an approximate value grid.

Only relative values are recovered (0 = lowest colour, 1 = highest), so results are indicative.
"""
import numpy as np

SCALES = ["Rainbow (blue = low, red = high)", "Greyscale (dark = low)", "Greyscale (dark = high)",
          "Reference only (do not use as data)"]


def image_values(img, scale=SCALES[0]):
    """Returns (values 0-1 with NaN where no data, valid mask)."""
    rgb = np.asarray(img.convert("RGB"), float) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    if scale.startswith("Rainbow"):
        import matplotlib.colors as mcolors
        hsv = mcolors.rgb_to_hsv(rgb)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h = np.where(h > 0.85, 0.0, h)  # magenta/pink tops of some scales -> red end
        vals = 1 - np.clip(h, 0, 0.7) / 0.7
        valid = (s > 0.25) & (v > 0.2)  # white background, black text and grey lines are not data
    else:
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        vals = lum if "dark = low" in scale else 1 - lum
        valid = (mx - mn < 0.15) & (lum < 0.98)
    return np.where(valid, vals, np.nan).astype(np.float32), valid


def extent_transform(width, height, xmin, xmax, ymin, ymax):
    from rasterio.transform import from_bounds
    return from_bounds(xmin, ymin, xmax, ymax, width, height)
