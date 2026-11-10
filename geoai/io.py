"""Reading tabular field data uploaded through Streamlit (or any file-like object)."""
import io

import pandas as pd

TABLE_TYPES = ["csv", "xyz", "txt", "dat", "xlsx", "xls"]


def _name(f):
    return getattr(f, "name", str(f)).lower()


def _text(f):
    if hasattr(f, "getvalue"):
        raw = f.getvalue()
    elif hasattr(f, "read"):
        raw = f.read()
    else:
        with open(f, "rb") as fh:
            raw = fh.read()
    return raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw


def read_raw_table(f):
    """Read a table keeping every column as it is (strings allowed, e.g. DMS coordinates)."""
    name = _name(f)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(f)
    text = _text(f)
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    # No header row (first line is numeric)? Re-read without header.
    try:
        [float(str(c).replace(",", ".")) for c in df.columns]
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", header=None)
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    except ValueError:
        pass
    return df


def read_table(f):
    """Read a numeric survey table (X, Y, value ...). Same as read_raw_table today."""
    return read_raw_table(f)
