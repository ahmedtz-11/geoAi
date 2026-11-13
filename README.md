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
| **2 · Magnetic** | One magnetic survey file: .dat (incl. Geosoft XYZ), .csv, .txt, .xls/.xlsx, GeoTIFF or a map picture | Spike removal, minimum-curvature grid (cell = ¼ of the survey line spacing) clipped to the survey outline, two image types to switch between: **Normal** (original gridding and colours) and **Filled** (Oasis montaj style: Geosoft rainbow, equal-area stretch); 12 maps: TMI, RTP/RTE (IGRF field direction, low-latitude stabilised), 1VD, 2VD, analytic signal, tilt derivative, THDR, lineaments (+ strike rose), upward continuation, SPI depth, Euler 3D, magnetic target map; ZIP of all maps |
| **3 · IP** | One .csv / .txt file per survey line (station, n-level or depth, chargeability / resistivity) | One pseudosection image per line with the anomaly outlined, its position and depth range |
| **4 · Geology** | Mapping points (lithology, strike, dip) or a geological map picture / GeoTIFF | Lithology map, gold-favourable host rocks, strike rose, stereonet |
| **5 · Geochemistry** | Soil / rock / stream samples (CSV, Excel) with Au and pathfinders | Below-detection handling, log thresholds, anomaly classes, correlations, multi-element gold index, anomalous samples |
| **6 · Integrated gold prediction** | Whatever was submitted in sections 2–5, plus Section 1 results and/or uploaded remote sensing GeoTIFFs | Weighted evidence (knowledge-driven) and/or Random Forest on gold samples; prospectivity map; ranked targets with location, estimated depth and trend; HTML report |

## Pit mapping (separate from sections 1-6)

Upload one row per pit (Excel, CSV or TXT; lat/long, DMS or UTM). Depth, gold grade, status and vein strike
are used when the file has them. An optional picture (sketch, photo, or a map drawn behind the pits) and
field notes can be added.

- **Mineralisation trend**: the best-fit line through the mineralised pits, weighted by gold grade; strike,
  length, corridor width, plunge (when pits deepen one way), and a check against strikes measured in the pits.
- **What is left in each pit**: High / Moderate / Low remaining potential with the reasons (on or off the
  trend, grade here and next door, room below the pit compared with deeper neighbours, recorded status).
  This is a screening comparison between pits, not a resource estimate.
- **Proposed pits**: infill pits in untested gaps along the trend and step-out pits beyond both ends, ranked,
  with coordinates, expected depth and the reason. CSV and Google Earth (KML) downloads.
- **Pit register**: existing and proposed pits are saved to `data/pit_register.csv` under a project name and
  stay there between sessions. Saving a project again updates it.

Sample: `sample_data/pits_sample.xlsx` and `pits_notes.txt` (trend 045°, an untested gap, two barren pits).

## IP: inverse model chargeability and resistivity sections

For files that carry electrode positions (Tx1, Tx2, Rx1, Rx2, as exported by IRIS Prosys), each
line is inverted and drawn in the Res2DInv layout: "Inverse Model Chargeability Section", the 17-colour scale
with its block legend, depth axis at the model layers, iteration number and absolute error, and unit electrode
spacing. A switch above the images goes back to the measured pseudosection.

The ground under the line is divided into blocks one electrode spacing wide, in layers that thicken with
depth. Apparent chargeability is the sensitivity-weighted average of the block chargeabilities (Seigel, 1959),
so the block values are found by smoothness-constrained least squares (Loke & Barker, 1995), kept positive,
with readings that will not fit down-weighted. The sensitivities are those of a homogeneous half-space; they
are checked in the tests against direct integration and against the analytic half-space total.

The **Inverse Model Resistivity Section** uses the same solver on logarithms of apparent resistivity (the
Rho column), has the same layout, reports its error in per cent, and uses logarithmic colour intervals. Each
quantity applies its own quality filters, so a reading with a bad chargeability can still serve resistivity.
The ZIP download holds both sections for every line.

Res2DInv goes one step further: it inverts resistivity too and recomputes the sensitivities for that
resistivity model. Expect the same anomalies in the same places, with differences in shape and amplitude
where the ground resistivity changes sharply. Files without electrode positions keep the pseudosection.

Sample: `sample_data/ip_electrodes_L1.csv` (dipole-dipole, a = 20 m, chargeable body planted at 280-340 m,
25-60 m deep, which is also a 40 ohm.m conductor; a 1500 ohm.m resistive cap at 100-180 m in the top 15 m).

## Project layout

```
app.py                     entry point + navigation
views/
  home.py                  landing page
  remote_sensing.py        Section 1 page (steps 1-2, then the Step-3 tabs)
  magnetic.py              Section 2 page (one file -> 12 maps)
  ip.py                    Section 3 page (one file per line -> one image per line)
  geology.py               Section 4 page
  geochemistry.py          Section 5 page
  interpretation.py        Section 6 page (integrated gold prediction)
  pit_mapping.py           Pit mapping page (trend, remaining potential, proposed pits, register)
ui/state.py                keeps widget choices when moving between tabs and pages
ui/interp/                 Sections 2-5 interface
  page.py                  shared upload -> submit -> results layout of the data pages
  upload.py                upload slots and per-file column settings
  tab_magnetic.py          the 13 magnetic result tabs
  tab_summary.py  tab_geology.py  tab_remote.py  tab_geochem.py  tab_prediction.py
  predict.py               prediction settings + cached computation
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
  readers.py               any upload -> table / raster / image (Geosoft XYZ .dat, CSV, Excel, GeoTIFF, PNG)
  datasets.py              uploads + column choices -> datasets in one working UTM system
  geophysics.py            survey geometry, minimum-curvature gridding, FFT filters, Euler deconvolution,
                           spectral depth, IP sections
  magnetics.py             IGRF field direction, RTP / RTE, VD, AS, tilt, THDR, SPI, magnetic analysis
  ipinv.py                 IP inversion: half-space sensitivities, smoothness-constrained solve, Res2DInv-style figure
  geochem.py               lab value cleaning, log thresholds, classes, correlation, gold index
  geology.py               lithology / structural measurements, rose, stereonet poles
  imagegrid.py             map picture colours -> relative values
  pits.py                  pit trend, remaining potential, proposed pits, pit register, KML
  interpret.py             Section 2 pipeline: analysis, evidence layers, prediction, targets
  plots.py                 Section 2 figures
  imagery/
    basemap.py             Esri World Imagery tiles -> stitched image
    sentinel.py            Sentinel-2 L2A search + loading (Microsoft Planetary Computer)
    dem.py                 Copernicus DEM GLO-30 loading (Microsoft Planetary Computer)
sample_data/               example files (see below)
tests/                     offline tests (python -m pytest tests)
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

## Sample data

| File | Use in |
|---|---|
| `aoi_corners_latlong.csv`, `aoi_corners_dms.csv` | Section 1 corners |
| `magnetics_tmi.dat` | 2 · Magnetic (Geosoft XYZ, UTM 36S; induced field I = -28°, D = 0.6°) |
| `ip_line_L1600_N9681600.csv`, `ip_line_L2300_N9682300.csv`, `ip_line_L3000_N9683000.csv` | 3 · IP, one file per line (type the northing from the file name; n-level, dipole-dipole, a = 50 m) |
| `ip_dipole_dipole.csv` | all three lines in one file (Line and Northing columns) |
| `magnetic_map_image.png` | 2 · Magnetic as a picture (extent W 405600, E 410000, S 9680600, N 9685000, UTM 36S) |
| `geology_mapping.csv` | 3 · Geology (lithology, strike, dip) |
| `rs_alteration_indices.tif` | 5 · Integrated gold prediction (add remote sensing layers) (clay and iron oxide indices) |
| `soil_geochem.xlsx` | 4 · Geochemistry (Au_ppb with '<5', As, Sb, Bi, Cu, Pb, Zn) |

All Section 2-5 samples describe one synthetic gold system (a NE-striking shear zone through 407300 E,
9682300 N) so the prediction can be checked. Regenerate with `python sample_data/make_section2_samples.py`.

## Magnetic maps: how they are made (and how to get the Oasis montaj look)

**Spikes first.** A magnetometer swung, dropped, or carried past a fence records jumps of thousands of nT
between readings a metre apart, which no rock produces. Each reading is compared with the median of its
ten nearest neighbours and dropped if it is more than 6 robust standard deviations away. This matters more
than it sounds: minimum curvature has to honour every reading it is given, so one spike pulls a bullseye
into the grid and into every derivative taken from it, and it stretches the colour range until the real
field looks flat. Tolerance and on/off are in Advanced settings; the count removed is reported above the maps.

**Gridding.** Ground and airborne surveys are sampled densely along lines and sparsely across them, so GeoAI
measures both spacings and sets the grid cell to a quarter of the **line** spacing (the 1/3-1/5 rule used by
the USGS and Geosoft). The line direction comes from a local PCA over each reading's nearest neighbours,
which still works on a survey walked by hand where the GPS scatter is as large as the step between readings;
the lines themselves are found as peaks in the density of the across-line coordinate, so they need not be
evenly spaced. The grid is then filled by **minimum curvature** (Briggs, 1974), the smoothest surface through
every reading, and clipped to the survey outline (a concave hull) so the fill stops where the survey stopped
instead of running away into false highs along the edges.

**Two image types, chosen after submitting.** Both are processed on Submit, so switching is instant.

- **Normal** (shown first): the original processing. Every reading is kept, gridded by linear interpolation
  with the cell taken from the station spacing, and drawn in each map's original colours with a linear
  stretch. Ground between survey lines that no reading reaches stays blank.
- **Filled**: everything described above (spikes removed, minimum curvature, outline clipping), drawn in the
  Geosoft rainbow with an **equal-area** stretch, Oasis montaj's "histogram equalization", where every colour
  gets the same number of cells. This is the continuous image Oasis montaj produces.

Every map, depth estimate, lineament and target on the page follows the selected type, as do the downloads.
The Integrated gold prediction page uses the Filled processing.

**Microlevelling** (Advanced settings, off by default) removes stripes running along the survey lines, left
by diurnal drift and instrument offsets between lines (Minty, 1991). It helps a survey walked over several
days; on a short clean survey it can take out real structure that happens to trend along the lines, so
compare both before using it.
