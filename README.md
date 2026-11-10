# GeoAI Platform

Web app that helps geologists and geophysicists find underground minerals, focused on gold.

## Run

```bash
venv\Scripts\activate          # Windows (from the geoAi folder)
pip install -r requirements.txt
streamlit run app.py
```

An internet connection is needed for satellite imagery (Section 1).

## Sections

| Section | Input | Output |
|---|---|---|
| **1 · Remote sensing** | CSV / Excel with the 4 corner coordinates (lat/long, DMS or UTM) | AOI map and area; Step 3 maps: high-resolution image, iron oxide / ferric iron, clay / hydroxyl alteration, band ratio images, PCA images (incl. Crósta), lithological classification, geological lineaments (rose diagram, density), DEM / hillshade, integrated prospectivity with ranked targets. Downloads: PNG / GeoTIFF / KML / GeoJSON / CSV |
| **2 · Interpretation & prediction** | Magnetic, IP, geological, geochemical data (.xls/.xlsx, .dat, .csv, images) | Interpreted maps, trend/strike, depth, gold prediction *(in development; general gridding/anomaly tools available)* |

## Project layout

```
app.py                     entry point + navigation
views/
  home.py                  landing page
  remote_sensing.py        Section 1 page (steps 1-2, then the Step-3 tabs)
  interpretation.py        Section 2 page
ui/rs/                     Section 1 Step-3 interface, one module per tab
  data_panel.py            loads Sentinel-2 scene + Copernicus DEM (shared by all tabs)
  tab_image.py  tab_index.py (iron / clay)  tab_ratios.py  tab_pca.py
  tab_lithology.py  tab_lineaments.py  tab_dem.py  tab_prospectivity.py
  common.py                shared state, caching, display helpers
geoai/                     reusable logic (no Streamlit inside)
  io.py                    read CSV / TXT / DAT / Excel
  coords.py                parse corners (decimal, DMS, UTM, Arc 1960) -> AOI, area, KML/GeoJSON
  grid.py                  gridding for point surveys
  indices.py               Sentinel-2 indices, band ratios, sigma classes, target zones
  pca.py                   standard PCA + Crósta feature-oriented PCs
  lithology.py             K-means spectral (lithological) classification
  lineaments.py            Canny + Hough lineament extraction, rose stats, density, GeoJSON
  terrain.py               hillshade, slope, aspect
  prospectivity.py         fuzzy evidence layers, weighted sum / fuzzy gamma, targets
  export.py                PNG, GeoTIFF, KML exports and map figures
  imagery/
    basemap.py             Esri World Imagery tiles -> stitched image
    sentinel.py            Sentinel-2 L2A search + loading (Microsoft Planetary Computer)
    dem.py                 Copernicus DEM GLO-30 loading (Microsoft Planetary Computer)
sample_data/               example corner files and a test magnetic grid
tests/test_core.py         offline tests (python -m pytest tests)
```

## Corner file format

One row per corner. Column names are detected automatically
(`latitude/longitude`, `lat/lon`, `Y/X`, `northing/easting`):

```
corner,latitude,longitude
C1,-2.8500,32.1500
C2,-2.8500,32.1900
C3,-2.8900,32.1900
C4,-2.8900,32.1500
```

Supported coordinate systems: WGS84 lat/long, WGS84 UTM 35S/36S/37S, Arc 1960 UTM 35S/36S/37S,
or any EPSG code. Values like `2°51'00"S` also work. Corners can be in any order.

## Data sources

- High-resolution image: Esri World Imagery (Esri, Maxar, Earthstar Geographics). Mosaic, date varies.
- Recent multispectral: Copernicus Sentinel-2 L2A via Microsoft Planetary Computer (free, no key).
- Elevation: Copernicus DEM GLO-30 via Microsoft Planetary Computer (free, no key).

## Gold alteration indices (Sentinel-2)

| Index | Formula | Highlights |
|---|---|---|
| Clay / hydroxyl | B11 / B12 | sericite, kaolinite, illite (phyllic / argillic alteration) |
| Iron oxide | B4 / B2 | hematite, goethite, jarosite (gossans over sulphides) |
| Ferric oxides | B11 / B8A | ferric oxide minerals |
| Ferrous iron | B12/B8 + B3/B4 | chlorite, mafic rocks (propylitic alteration) |
| Gossan | B11 / B4 | iron-rich weathered caps |
| NDVI | (B8-B4)/(B8+B4) | vegetation (used as a mask) |

Results are surface screening only and must be checked with geology, geochemistry, geophysics and fieldwork.
