"""Res2DInv data files (.dat): apparent resistivity, with or without IP (chargeability).

Read into the same columns as an IRIS export (Array, n, Tx1, Tx2, Rx1, Rx2, Rho, M), so the rest of the IP
page treats both alike. Supported: the general array format (array type 11, each reading lists its own
electrode positions) and the index formats for Wenner (1), pole-pole (2), dipole-dipole (3), pole-dipole (6)
and Wenner-Schlumberger (7). Topography and other sections after the data are ignored.
"""
import numpy as np
import pandas as pd

REMOTE = 9999999.0
CODES = {1: "W", 2: "P-P", 3: "DP-DP", 6: "P-DP", 7: "WS"}
NAMES = {1: "Wenner", 2: "Pole-pole", 3: "Dipole-dipole", 6: "Pole-dipole", 7: "Wenner-Schlumberger",
         11: "General array"}


def _nums(line):
    out = []
    for t in line.replace(",", " ").split():
        try:
            out.append(float(t))
        except ValueError:
            return None
    return out


def looks_like(text):
    """True when the first lines have the Res2DInv header: title, unit spacing, array type number."""
    lines = [l.strip() for l in text.splitlines()[:4]]
    if len(lines) < 4:
        return False
    a, b = _nums(lines[1]), _nums(lines[2])
    return bool(a and len(a) == 1 and a[0] > 0 and b and len(b) == 1 and b[0] == int(b[0]) and 1 <= b[0] <= 15
                and _nums(lines[0]) is None)


def parse(text):
    """Returns (DataFrame, meta). Raises ValueError with a plain reason when the file cannot be used."""
    L = [l.strip() for l in text.splitlines()]
    L = [l for l in L if l != ""]
    title, unit, array = L[0], float(_nums(L[1])[0]), int(_nums(L[2])[0])
    i, sub, resistance = 3, None, False
    if array == 11:
        sub = int(_nums(L[i])[0])
        i += 1
        if _nums(L[i]) is None:                      # "Type of measurement (0=app. resistivity,1=resistance)"
            resistance = int(_nums(L[i + 1])[0]) == 1
            i += 2
    elif array not in CODES:
        raise ValueError(f"Res2DInv array type {array} is not supported yet (supported: Wenner 1, pole-pole 2, "
                         "dipole-dipole 3, pole-dipole 6, Wenner-Schlumberger 7, general array 11).")
    n, xloc, ip = int(_nums(L[i])[0]), int(_nums(L[i + 1])[0]), int(_nums(L[i + 2])[0])
    i += 3
    ip_name, ip_unit = None, None
    if ip == 1:
        ip_name, ip_unit = L[i], L[i + 1]
        i += 3                                       # name, unit, "delay, integration time"
    rows = []
    for line in L[i:i + n]:
        v = _nums(line)
        if not v:
            continue
        if array == 11:
            k = int(v[0])
            xs, rest = v[1:1 + 2 * k:2], v[1 + 2 * k:]
            if k == 4:
                c1, c2, p1, p2 = xs
            elif k == 3:
                c1, c2, p1, p2 = REMOTE, xs[0], xs[1], xs[2]
            elif k == 2:
                c1, c2, p1, p2 = REMOTE, xs[0], xs[1], REMOTE
            else:
                continue
            nlev = np.nan
        else:
            x, a = v[0], v[1]
            nlev = v[2] if array in (3, 6, 7) else 1.0
            rest = v[3:] if array in (3, 6, 7) else v[2:]
            length = {1: 3 * a, 2: a, 3: (nlev + 2) * a, 6: (nlev + 1) * a, 7: (2 * nlev + 1) * a}[array]
            x0 = x - length / 2 if xloc == 1 else x
            if array == 1:
                c1, p1, p2, c2 = x0, x0 + a, x0 + 2 * a, x0 + 3 * a
            elif array == 2:
                c1, c2, p1, p2 = REMOTE, x0, x0 + a, REMOTE
            elif array == 3:
                c2, c1, p1, p2 = x0, x0 + a, x0 + (nlev + 1) * a, x0 + (nlev + 2) * a
            elif array == 6:
                c1, c2, p1, p2 = REMOTE, x0, x0 + nlev * a, x0 + (nlev + 1) * a
            else:
                c1, p1, p2, c2 = x0, x0 + nlev * a, x0 + (nlev + 1) * a, x0 + (2 * nlev + 1) * a
        if not rest:
            continue
        rho = rest[0]
        if resistance:                               # resistance (ohm) -> apparent resistivity
            def inv(a_, b_):
                return 0.0 if REMOTE in (a_, b_) else 1.0 / abs(a_ - b_)
            g = inv(c1, p1) - inv(c2, p1) - inv(c1, p2) + inv(c2, p2)
            rho = 2 * np.pi * rho / g if g else np.nan
        rows.append((nlev, c1, c2, p1, p2, abs(rho) if np.isfinite(rho) else np.nan,
                     rest[1] if ip == 1 and len(rest) > 1 else np.nan))
    if not rows:
        raise ValueError("No readings found in the Res2DInv file.")
    df = pd.DataFrame(rows, columns=["n", "Tx1", "Tx2", "Rx1", "Rx2", "Rho", "M"])
    code = CODES.get(sub if array == 11 else array)
    if code:
        df.insert(0, "Array", code)
    if df.n.isna().all():
        df = df.drop(columns="n")
    if ip != 1:
        df = df.drop(columns="M")
    meta = {"format": "Res2DInv", "title": title, "unit_spacing": unit, "array": NAMES.get(sub if array == 11 else array,
            NAMES.get(array, str(array))), "ip_name": ip_name, "ip_unit": ip_unit, "declared": n, "read": len(df)}
    return df, meta
