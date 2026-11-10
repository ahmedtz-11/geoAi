import base64
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="GeoAI Platform", layout="wide")
MAX_MB = 50  # per file; data lives in memory only and is dropped when the session ends


# ---------- helpers ----------
def read_table(f):
    name = f.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(f)
    text = f.getvalue().decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    # No header row (first line is numeric)? Re-read without header.
    try:
        [float(c) for c in df.columns]
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", header=None)
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    except ValueError:
        pass
    return df


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


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    return buf.getvalue()


# ---------- UI: upload ----------
st.title("GeoAI Platform")
st.caption("Upload data, pick a tool, get results. Nothing is stored: refresh to reset.")

files = st.sidebar.file_uploader(
    "Upload data (CSV, XYZ, TXT, DAT, Excel)",
    type=["csv", "xyz", "txt", "dat", "xlsx", "xls"],
    accept_multiple_files=True,
)

datasets = {}
for f in files or []:
    if f.size > MAX_MB * 1e6:
        st.sidebar.warning(f"{f.name}: larger than {MAX_MB} MB, skipped.")
        continue
    try:
        df = read_table(f)
    except Exception as e:
        st.sidebar.error(f"{f.name}: could not read ({e})")
        continue
    num = df.select_dtypes("number").columns.tolist()
    if len(num) < 3:
        st.sidebar.warning(f"{f.name}: needs at least 3 numeric columns (X, Y, value).")
        continue
    with st.sidebar.expander(f"Columns: {f.name}", expanded=len(files) == 1):
        cx = st.selectbox("X / Easting", num, 0, key=f"x{f.name}")
        cy = st.selectbox("Y / Northing", num, 1, key=f"y{f.name}")
        cv = st.selectbox("Value", num, 2, key=f"v{f.name}")
    d = df[[cx, cy, cv]].dropna()
    d.columns = ["x", "y", "v"]
    d = d.drop_duplicates(subset=["x", "y"])
    if len(d) < 10:
        st.sidebar.warning(f"{f.name}: too few valid points.")
        continue
    datasets[f.name] = d

if not datasets:
    st.info("Upload one or more files in the sidebar to begin. Each needs X, Y and a measured value column.")
    st.stop()

tools = ["Gridded map", "Regional-residual separation", "Horizontal gradient (edges)", "AI anomaly detection"]
if len(datasets) > 1:
    tools.append("Multi-dataset integration (target map)")
tool = st.sidebar.radio("Tool", tools)
n = st.sidebar.slider("Grid resolution (cells per side)", 50, 400, 150, 10)
method = st.sidebar.selectbox("Interpolation", ["linear", "cubic", "nearest"])

# ---------- processing ----------
table, notes, params = None, [], {"grid": n, "interpolation": method}
cmap = "viridis"

if tool.startswith("Multi-dataset"):
    allb = np.array([bounds_of(d) for d in datasets.values()])
    bounds = (allb[:, 0].min(), allb[:, 1].max(), allb[:, 2].min(), allb[:, 3].max())
    st.sidebar.markdown("**Weights** (negative = low values are favourable)")
    top = st.sidebar.slider("Top % of cells as targets", 1, 30, 10)
    zs, ws = [], []
    for name, d in datasets.items():
        XI, YI, Z = make_grid(d, n, method, bounds)
        zs.append((Z - np.nanmean(Z)) / (np.nanstd(Z) or 1))
        ws.append(st.sidebar.slider(name, -1.0, 1.0, 1.0, 0.1, key=f"w{name}"))
    zs, ws = np.array(zs), np.array(ws)[:, None, None]
    valid = ~np.isnan(zs)
    denom = np.where(valid, np.abs(ws), 0).sum(0)
    score = np.where(denom > 0, np.nansum(zs * ws, 0) / np.where(denom == 0, 1, denom), np.nan)
    field, mask, title, cmap = score, np.isnan(score), "Integrated target score", "magma"
    thr = np.nanpercentile(score, 100 - top)
    hit = score >= thr
    table = pd.DataFrame({"x": XI[hit], "y": YI[hit], "score": score[hit]}).sort_values("score", ascending=False)
    notes.append(f"{len(datasets)} datasets normalised and combined; top {top}% of cells flagged as targets.")
    params.update({"datasets": list(datasets), "weights": ws.ravel().tolist(), "top_percent": top})
    pts = table.rename(columns={"score": "class"}).assign(**{"class": 1})
else:
    name = st.sidebar.selectbox("Dataset", list(datasets))
    d = datasets[name]
    XI, YI, Z = make_grid(d, n, method, bounds_of(d))
    mask = np.isnan(Z)
    Zf = fill(Z)
    pts = None
    if tool == "Gridded map":
        field, title = Z, f"Gridded {name}"
    elif tool == "Horizontal gradient (edges)":
        gy, gx = np.gradient(Zf)
        field, title, cmap = np.hypot(gx, gy), "Horizontal gradient magnitude", "inferno"
        notes.append("High gradient values often mark contacts, faults or source edges.")
    else:
        sigma = st.sidebar.slider("Regional smoothing (sigma, cells)", 2, 60, 15)
        residual = Zf - gaussian_filter(Zf, sigma)
        params["sigma"] = sigma
        if tool.startswith("Regional"):
            field, title, cmap = residual, "Residual field", "RdBu_r"
            notes.append("Residual = data minus a Gaussian-smoothed regional trend.")
        else:
            cont = st.sidebar.slider("Expected anomaly fraction", 0.01, 0.20, 0.05, 0.01)
            k = st.sidebar.slider("Number of anomaly classes", 2, 6, 3)
            gy, gx = np.gradient(Zf)
            feats = np.column_stack([Zf.ravel(), residual.ravel(), np.hypot(gx, gy).ravel()])
            feats = StandardScaler().fit_transform(feats)
            iso = IsolationForest(contamination=cont, random_state=42).fit(feats)
            is_anom = (iso.predict(feats) == -1) & ~mask.ravel()
            field, title, cmap = residual, "Residual field with AI-detected anomalies", "RdBu_r"
            if is_anom.sum() >= k:
                lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(feats[is_anom]) + 1
                pts = pd.DataFrame({"x": XI.ravel()[is_anom], "y": YI.ravel()[is_anom],
                                    "value": Zf.ravel()[is_anom],
                                    "residual": residual.ravel()[is_anom], "class": lab})
                summ = pts.groupby("class").agg(cells=("x", "size"), mean_residual=("residual", "mean"))
                notes.append(f"IsolationForest flagged {int(is_anom.sum())} cells; KMeans grouped them into {k} classes.")
                notes += [f"Class {c}: {r.cells} cells, mean residual {r.mean_residual:.2f} "
                          f"({'positive/high' if r.mean_residual > 0 else 'negative/low'} anomaly)"
                          for c, r in summ.iterrows()]
                table = pts
            else:
                notes.append("Too few anomalies found; raise the expected anomaly fraction.")
            params.update({"contamination": cont, "classes": k})

field = np.where(mask, np.nan, field)
ext = [XI.min(), XI.max(), YI.min(), YI.max()]

# ---------- map ----------
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(field, origin="lower", extent=ext, cmap=cmap, aspect="equal")
plt.colorbar(im, ax=ax, shrink=0.8)
if pts is not None and len(pts):
    sc = ax.scatter(pts.x, pts.y, c=pts["class"], cmap="tab10", s=6, edgecolors="none")
ax.set_title(title)
ax.set_xlabel("X")
ax.set_ylabel("Y")
png = fig_to_png(fig)

c1, c2 = st.columns([3, 2])
with c1:
    st.subheader("Map")
    st.pyplot(fig)
with c2:
    st.subheader("Summary")
    summary = "\n".join([f"Tool: {tool}", f"Valid cells: {int((~mask).sum())}",
                         f"Min / Max: {np.nanmin(field):.3f} / {np.nanmax(field):.3f}",
                         f"Mean / Std: {np.nanmean(field):.3f} / {np.nanstd(field):.3f}"] + notes)
    st.code(summary, language=None)  # hover to copy
    if table is not None:
        st.dataframe(table.head(200), height=250)

st.subheader("3D surface")
f3 = go.Figure(go.Surface(x=XI[0], y=YI[:, 0], z=field, colorscale="Viridis"))
f3.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0))
st.plotly_chart(f3, use_container_width=True)

# ---------- report + downloads ----------
grid_df = pd.DataFrame({"x": XI.ravel(), "y": YI.ravel(), "value": field.ravel()}).dropna()
rows = table.head(50).to_html(index=False) if table is not None else ""
report = f"""<html><body style="font-family:Arial;max-width:900px;margin:auto">
<h1>GeoAI Report</h1><p>Generated {datetime.now():%Y-%m-%d %H:%M}</p>
<h2>{title}</h2><pre>{summary}</pre>
<h3>Parameters</h3><pre>{params}</pre>
<img src="data:image/png;base64,{base64.b64encode(png).decode()}" style="width:100%">
{('<h3>Detected cells (first 50)</h3>' + rows) if rows else ''}
</body></html>"""

st.subheader("Downloads")
d1, d2, d3, d4 = st.columns(4)
d1.download_button("Map (PNG)", png, "map.png", "image/png")
d2.download_button("Processed grid (CSV)", grid_df.to_csv(index=False), "grid.csv", "text/csv")
d3.download_button("Report (HTML)", report, "report.html", "text/html")
if table is not None:
    d4.download_button("Results table (CSV)", table.to_csv(index=False), "results.csv", "text/csv")