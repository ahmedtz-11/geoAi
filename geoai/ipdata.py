"""IP / resistivity field files with electrode positions (IRIS Instruments Prosys / ELREC Pro exports and
similar): station position and pseudo-depth from the electrodes, and data-quality filtering.

IRIS columns used: Array (P-DP, DP-DP, W, ...), n, Tx1, Tx2 (current electrodes C1, C2), Rx1, Rx2
(potential electrodes P1, P2), Rho (apparent resistivity, ohm.m), M (chargeability, mV/V), ErrM, Contact.
A remote ("infinite") electrode is recorded with a huge coordinate such as 9999999.
"""
import numpy as np
import pandas as pd

from .coords import parse_column

REMOTE = 1e6  # electrode coordinates beyond this are remote electrodes
ARRAYS = {"P-DP": "Pole-dipole", "PDP": "Pole-dipole", "DP-DP": "Dipole-dipole", "DD": "Dipole-dipole",
          "D-D": "Dipole-dipole", "DPDP": "Dipole-dipole", "W": "Wenner", "WEN": "Wenner", "WS": "Schlumberger",
          "SCH": "Schlumberger", "P-P": "Pole-pole", "PP": "Pole-pole", "GRAD": "Gradient"}


def _col(df, *names):
    low = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def electrode_columns(df):
    """{'c1','c2','p1','p2','n','array','rho','m','errm','contact'} -> column names (None if absent)."""
    return {"c1": _col(df, "Tx1", "C1", "A", "Ax"), "c2": _col(df, "Tx2", "C2", "B", "Bx"),
            "p1": _col(df, "Rx1", "P1", "M_pos", "Mx"), "p2": _col(df, "Rx2", "P2", "N_pos", "Nx"),
            "n": _col(df, "n", "n_level", "nlevel", "level"), "array": _col(df, "Array"),
            "rho": _col(df, "Rho", "Res", "Resistivity", "App_Res"), "m": _col(df, "M", "Charg", "Chargeability", "IP"),
            "errm": _col(df, "ErrM", "Dev_M", "M_err"), "contact": _col(df, "Contact", "Rs", "Rcontact")}


def has_electrodes(df):
    c = electrode_columns(df)
    return all(c[k] is not None for k in ("c1", "c2", "p1", "p2"))


def is_iris(df):
    c = electrode_columns(df)
    return has_electrodes(df) and c["m"] is not None and c["rho"] is not None


def array_name(df):
    c = electrode_columns(df)
    if c["array"] is None:
        return None
    v = df[c["array"]].dropna().astype(str).str.strip().str.upper()
    if v.empty:
        return None
    code = v.mode().iloc[0]
    return ARRAYS.get(code, ARRAYS.get(code.replace("-", ""), code))


def depth_factor(n, array):
    """Median depth of investigation / a (Edwards, 1977), n = dipole separation factor."""
    n = np.asarray(n, float)
    if array == "Pole-dipole":
        return 0.52 + 0.39 * (n - 1)
    if array == "Dipole-dipole":
        return 0.416 + 0.26 * (n - 1)
    if array == "Pole-pole":
        return 0.867 * n
    if array == "Wenner":
        return 0.519 * n
    return 0.5 * n  # Schlumberger / others (rough)


def geometry(df, array=None):
    """Station x (plot point), dipole size a, n-level and pseudo-depth from electrode positions.

    Plot point = midway between the centre of the active current electrodes and the centre of the
    potential dipole (standard pseudosection convention); remote electrodes are ignored."""
    c = electrode_columns(df)
    e = {k: parse_column(df[c[k]]).to_numpy(float) for k in ("c1", "c2", "p1", "p2")}
    for k in e:
        e[k] = np.where(np.abs(e[k]) >= REMOTE, np.nan, e[k])
    cc = np.nanmean(np.vstack([e["c1"], e["c2"]]), axis=0)
    pc = np.nanmean(np.vstack([e["p1"], e["p2"]]), axis=0)
    x = np.where(np.isnan(cc), pc, (cc + pc) / 2)
    a = np.abs(e["p2"] - e["p1"])
    a_med = float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan
    if c["n"] is not None:
        n = parse_column(df[c["n"]]).to_numpy(float)
    else:  # separation between the nearest current electrode and P1, in dipole lengths
        near = np.where(np.isnan(e["c2"]), e["c1"], e["c2"])
        n = np.round(np.abs(e["p1"] - near) / np.where(a > 0, a, np.nan))
    arr = array or array_name(df) or ("Pole-dipole" if np.isnan(e["c1"]).all() or np.isnan(e["c2"]).all()
                                      else "Dipole-dipole")
    depth = depth_factor(n, arr) * np.where(np.isfinite(a), a, a_med)
    return pd.DataFrame({"station_x": x, "a": a, "n_level": n, "pseudo_depth": depth}), arr, a_med


def quality_mask(df, value_col, max_err=None, drop_negative=True, drop_bad_contact=True, err_col=None):
    """True for readings to keep. Returns (mask, reasons dict with counts)."""
    c = electrode_columns(df)
    v = parse_column(df[value_col]).to_numpy(float)
    keep = np.isfinite(v)
    reasons = {"missing value": int((~keep).sum())}
    if drop_negative:
        bad = keep & (v <= 0)
        reasons["negative or zero"] = int(bad.sum())
        keep &= ~bad
    err_col = err_col or c["errm"]
    if max_err is not None and err_col is not None and value_col == c["m"]:
        err = parse_column(df[err_col]).to_numpy(float)
        bad = keep & np.isfinite(err) & (err > max_err)
        reasons[f"error above {max_err:g}"] = int(bad.sum())
        keep &= ~bad
    if drop_bad_contact and c["contact"] is not None:
        raw = df[c["contact"]].astype(str).str.strip().str.upper()
        bad = keep & (raw.isin(["INFINI", "------", "INF", "NAN", "", "OVER"]) |
                      parse_column(df[c["contact"]]).isna()).to_numpy()
        reasons["bad electrode contact"] = int(bad.sum())
        keep &= ~bad
    return keep, reasons
