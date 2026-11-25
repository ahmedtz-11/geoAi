"""Turning what the engine returns into things a browser can show.

A matplotlib figure becomes a PNG on disk that a route serves, a plotly figure becomes a <div> plus
its JSON, and a dataframe becomes an HTML table with a CSV beside it. Every one of them is addressed
by a short name, so a template can ask for "tmi" and the download route finds the same file.
"""
import base64
import io
import json
import threading

import matplotlib
matplotlib.use("Agg")                      # no display on a server; must be set before pyplot is loaded
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402
from markupsafe import Markup              # noqa: E402

DPI = 130
MAX_TABLE_ROWS = 400

# matplotlib's pyplot keeps global state and is not thread-safe. The browser asks for several
# pictures at once, and two threads building figures together corrupt that state — most visibly as
# a mathtext parser error on a perfectly ordinary axis label. Every figure is built under this lock.
draw_lock = threading.Lock()


def safe_name(s):
    """A name that is safe in a file path, a URL and an HTML id."""
    return "".join(c if c.isalnum() else "-" for c in str(s)).strip("-").lower()[:60] or "item"


# ------------------------------------------------------------------ matplotlib
def save_fig(fig, path, dpi=DPI, close=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)
    return path


def fig_to_png_bytes(fig, dpi=DPI, close=True):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)
    return buf.getvalue()


def fig_to_data_uri(fig, dpi=90, close=True):
    """For a small picture that is cheaper to inline than to fetch — a legend swatch, a sparkline."""
    return "data:image/png;base64," + base64.b64encode(fig_to_png_bytes(fig, dpi, close)).decode("ascii")


# ------------------------------------------------------------------ plotly
def plotly_div(fig, div_id, height=620):
    """A plotly figure as a div plus the JSON to draw it. plotly.js is loaded once by the layout."""
    import plotly.io as pio
    spec = json.loads(pio.to_json(fig))
    spec.setdefault("layout", {})["autosize"] = True
    return Markup(
        f'<div class="plotly" id="{div_id}" style="height:{height}px"></div>'
        f'<script type="application/json" class="plotly-spec" data-target="{div_id}">'
        f"{json.dumps(spec)}</script>")


# ------------------------------------------------------------------ tables
def table_html(df, max_rows=MAX_TABLE_ROWS, index=False, classes="data", float_fmt="{:,.4g}"):
    """A dataframe as a plain HTML table. Long tables are cut, with a line saying so."""
    if df is None or len(df) == 0:
        return Markup('<p class="muted small">Nothing to show.</p>')
    shown = df.head(max_rows)
    out = shown.to_html(index=index, classes=classes, border=0, escape=True, na_rep="",
                        float_format=lambda v: float_fmt.format(v) if np.isfinite(v) else "")
    if len(df) > max_rows:
        out += (f'<p class="muted small">Showing the first {max_rows:,} of {len(df):,} rows. '
                f"The download has them all.</p>")
    return Markup(out)


def df_to_csv_bytes(df, index=False):
    buf = io.StringIO()
    df.to_csv(buf, index=index)
    return buf.getvalue().encode("utf-8")


def stats_row(arr, unit=""):
    """The min / mean / max line that sits beside each map."""
    v = np.asarray(arr, float)
    v = v[np.isfinite(v)]
    if not v.size:
        return []
    u = f" {unit}" if unit else ""
    return [("Minimum", f"{v.min():,.4g}{u}"), ("Mean", f"{v.mean():,.4g}{u}"),
            ("Maximum", f"{v.max():,.4g}{u}"), ("Readings", f"{v.size:,}")]


def fmt(value, nd=4):
    """A number for display: no exponent soup, no trailing noise."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "—"
    if f == int(f) and abs(f) < 1e15:
        return f"{int(f):,}"
    return f"{f:,.{nd}g}"


def describe(df, columns=None):
    """A small summary table for a set of numeric columns."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if not len(v):
            continue
        rows.append({"Column": c, "Count": len(v), "Minimum": v.min(), "Median": v.median(),
                     "Mean": v.mean(), "Maximum": v.max(), "Std dev": v.std()})
    return pd.DataFrame(rows)
