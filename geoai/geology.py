"""Geological field data: lithology points and structural measurements (strike / dip)."""
import numpy as np
import pandas as pd

LITH_HINTS = ["greenstone", "bif", "banded iron", "ironstone", "quartz", "vein", "shear", "schist", "mafic",
              "basalt", "andesite", "metavolcanic", "felsic volcanic", "chert", "mylonite", "altered", "sulphide",
              "sulfide", "gossan", "dolerite", "tonalite", "porphyry"]
LITH_COLS = ["lithology", "litho", "lith", "rock", "rock_type", "rocktype", "unit", "geology", "formation"]
STRIKE_COLS = ["strike", "azimuth", "trend", "dipdir", "dip_direction", "dip_dir"]
DIP_COLS = ["dip", "plunge"]


def guess(df, names):
    low = {c: str(c).strip().lower() for c in df.columns}
    for n in names:
        for c, l in low.items():
            if l == n:
                return c
    for n in names:
        for c, l in low.items():
            if len(n) > 3 and n in l:
                return c
    return None


def favourable_default(liths):
    """Lithologies whose names suggest a gold-favourable host (greenstone, BIF, quartz veins, shear...)."""
    return [l for l in liths if any(h in str(l).lower() for h in LITH_HINTS)]


def strike_from(values, is_dip_direction=False):
    v = np.asarray(values, float) % 360
    if is_dip_direction:
        v = (v - 90) % 360  # right-hand rule: strike = dip direction - 90
    return v % 180


def rose_from_strikes(strikes, bin_deg=10):
    s = np.asarray(strikes, float)
    s = s[np.isfinite(s)] % 180
    bins = np.arange(0, 180 + bin_deg, bin_deg)
    h, _ = np.histogram(s, bins=bins)
    return bins[:-1], h.astype(float)


def poles(strike, dip):
    """Equal-area (Schmidt, lower hemisphere) x, y of poles to planes (right-hand-rule strike)."""
    s, d = np.radians(np.asarray(strike, float)), np.radians(np.asarray(dip, float))
    trend = s - np.pi / 2  # pole trend = strike - 90 (opposite dip direction)
    plunge = np.pi / 2 - d
    r = np.sqrt(2) * np.sin((np.pi / 2 - plunge) / 2)
    return r * np.sin(trend), r * np.cos(trend)


def lith_summary(liths):
    s = pd.Series(liths).fillna("(blank)").astype(str)
    return s.value_counts().rename_axis("Lithology").reset_index(name="Points")
