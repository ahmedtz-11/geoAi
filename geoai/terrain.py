"""Terrain derivatives from a DEM grid (metres): hillshade, slope, aspect, contours."""
import numpy as np


def _fill(dem):
    if np.isnan(dem).any():
        return np.where(np.isnan(dem), np.nanmean(dem), dem)
    return dem


def gradients(dem, res):
    z = _fill(dem)
    dzdy, dzdx = np.gradient(z, res)  # rows go south, so dz/dnorth = -dzdy
    return dzdx, -dzdy


def slope_deg(dem, res):
    dx, dy = gradients(dem, res)
    s = np.degrees(np.arctan(np.hypot(dx, dy)))
    return np.where(np.isnan(dem), np.nan, s)


def aspect_deg(dem, res):
    """Direction the slope faces, degrees clockwise from north."""
    dx, dy = gradients(dem, res)
    a = (np.degrees(np.arctan2(-dx, -dy)) + 360) % 360
    return np.where(np.isnan(dem), np.nan, a)


def hillshade(dem, res, azimuth=315.0, altitude=45.0, z_factor=1.0):
    """Illumination 0-1 (Horn method). Azimuth clockwise from north, altitude above horizon."""
    dx, dy = gradients(dem, res)
    dx, dy = dx * z_factor, dy * z_factor
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, -dy)  # direction of steepest descent, clockwise from north
    az, alt = np.radians(azimuth), np.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.where(np.isnan(dem), np.nan, np.clip(hs, 0, 1))


def multi_hillshade(dem, res, azimuths=(0, 45, 90, 135, 180, 225, 270, 315), altitude=45.0, z_factor=1.0):
    """Average of several illumination directions: shows features of every orientation."""
    return np.nanmean([hillshade(dem, res, a, altitude, z_factor) for a in azimuths], axis=0)


def contour_interval(dem):
    rng = np.nanmax(dem) - np.nanmin(dem)
    for step in [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500]:
        if rng / step <= 25:
            return step
    return 1000
