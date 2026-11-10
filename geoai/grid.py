"""Gridding helpers for point survey data (used by Section 2 tools)."""
import numpy as np
from scipy.interpolate import griddata


def bounds_of(d):
    return d.x.min(), d.x.max(), d.y.min(), d.y.max()


def make_grid(d, n, method, bounds):
    x0, x1, y0, y1 = bounds
    xi = np.linspace(x0, x1, n)
    yi = np.linspace(y0, y1, n)
    XI, YI = np.meshgrid(xi, yi)
    Z = griddata((d.x.values, d.y.values), d.v.values, (XI, YI), method=method)
    return XI, YI, Z


def fill(Z):
    return np.where(np.isnan(Z), np.nanmean(Z), Z)
