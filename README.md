# GeoAI Platform

Web app that helps geologists and geophysicists find underground minerals, focused on gold.

## Run

```bash
venv\Scripts\activate          # Windows (from the geoAi folder)
pip install -r requirements.txt
python run_local.py
```

Then open **http://localhost:8000**. The first run makes an administrator and prints how to sign in;
[`LOCAL.md`](LOCAL.md) walks through the whole flow. An internet connection is needed for satellite
imagery (Section 1); everything else works offline.

## Sections

| Section | Input | Output |
|---|---|---|
| **1 · Remote sensing** | CSV / Excel with the 4 corner coordinates (lat/long, DMS or UTM) | AOI map and area; Step 3 maps: high-resolution image, iron oxide / ferric iron, clay / hydroxyl alteration, band ratio images, PCA images (incl. Crósta), lithological classification, geological lineaments (rose diagram, density), DEM / hillshade, 3-D terrain, integrated prospectivity with ranked targets. Downloads: PNG / GeoTIFF / KML / GeoJSON / CSV |
| **2 · Magnetic** | One magnetic survey file: .dat (incl. Geosoft XYZ), .csv, .txt, .xls/.xlsx, GeoTIFF or a map picture | Spike removal, minimum-curvature grid (cell = ¼ of the survey line spacing) clipped to the survey outline, two image types to switch between: **Normal** (original gridding and colours) and **Filled** (Oasis montaj style: Geosoft rainbow, equal-area stretch); 12 maps: TMI, RTP/RTE (IGRF field direction, low-latitude stabilised), 1VD, 2VD, analytic signal, tilt derivative, THDR, lineaments (+ strike rose), upward continuation, SPI depth, Euler 3D, magnetic target map; ZIP of all maps |
| **3 · IP** | One .csv / .txt file per survey line (station, n-level or depth, chargeability / resistivity) | One pseudosection image per line with the anomaly outlined, its position and depth range |
| **4 · Geology** | Structural measurements (CSV / Excel) and / or a QDS geological map sheet (PNG, JPG, GeoTIFF); either alone is enough | Dip directions resolved and bad readings flagged, structural sets with mean orientation and Fisher statistics, the line where sets cross (ore-shoot plunge), shortening direction, stereonet, strike rose, structure map over the sheet, written interpretation |
| **5 · Geochemistry** | Soil samples with a gold assay (CSV or Excel): position, sample ID, Au in ppm or % | Where background stops (two-population fit), anomaly classes, ranked anomaly zones with trend, extent and whether they are open, log-probability plot, interpolated map (natural neighbour, IDW, kriging with its variogram and error map, minimum curvature or linear, at a grid interval you choose), written interpretation and follow-up advice |
| **6 · Integrated gold prediction** | Whatever was submitted in sections 2–5, plus Section 1 results and/or uploaded remote sensing GeoTIFFs | Weighted evidence (knowledge-driven) and/or Random Forest on gold samples; prospectivity map; ranked targets with location, estimated depth and trend; HTML report |

## 3-D terrain (Section 1, Step 3 tab)

Its own tab, after DEM / hillshade: an interactive 3-D view of the landform. Drag to rotate, scroll to zoom. What is draped
over the terrain is chosen from a list, and the choice decides how it is drawn:

- a **picture** (shaded relief, satellite true colour, geology false colour, the clay / iron / ferric ratio
  composite) keeps its own pixel colours, lit by the hillshade so the relief still reads;
- a **measured grid** (elevation, slope, clay / hydroxyl, iron oxide) gets a colour bar and shows its value
  when you hover over the ground.

Seeing the alteration maps on the landform separates anomalies that sit on ridges of resistant rock from those
following valleys eroded along faults, and from those that are only slope or shadow. Elevations are the real
Copernicus DEM GLO-30 values; vertical exaggeration stretches the drawing only, and the axis and hover still
read true metres. The area of interest is drawn on the ground as a red line. The whole view downloads as an
interactive HTML file that opens in any browser.

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
spacing. For every line the resistivity section is shown above the chargeability section, on the same
axes, for direct comparison. The measured pseudosection is shown only for files without electrode positions.

The ground under the line is divided into blocks one electrode spacing wide, in layers that thicken with
depth. Apparent chargeability is the sensitivity-weighted average of the block chargeabilities (Seigel, 1959),
so the block values are found by smoothness-constrained least squares (Loke & Barker, 1995), kept positive,
with readings that will not fit down-weighted. The sensitivities are those of a homogeneous half-space; they
are checked in the tests against direct integration and against the analytic half-space total.

The **Inverse Model Resistivity Section** uses the same solver on logarithms of apparent resistivity (the
Rho column), has the same layout, reports its error in per cent, and uses logarithmic colour intervals. Each
quantity applies its own quality filters, so a reading with a bad chargeability can still serve resistivity.
The ZIP download holds both sections for every line.

**Res2DInv .dat files** are read directly (general array format 11, and the index formats for Wenner,
pole-pole, dipole-dipole, pole-dipole and Wenner-Schlumberger), with or without IP. The title, array, unit
spacing and IP unit (for example msec) come from the file header and are used on the sections.

Res2DInv goes one step further: it inverts resistivity too and recomputes the sensitivities for that
resistivity model. Expect the same anomalies in the same places, with differences in shape and amplitude
where the ground resistivity changes sharply. Files without electrode positions keep the pseudosection.

Sample: `sample_data/ip_electrodes_L1.csv` (dipole-dipole, a = 20 m, chargeable body planted at 280-340 m,
25-60 m deep, which is also a 40 ohm.m conductor; a 1500 ohm.m resistive cap at 100-180 m in the top 15 m).

## Structural geology (Section 4)

Two optional inputs, either alone or together: a table of field measurements (CSV or Excel) and a QDS
geological map sheet (PNG, JPG or GeoTIFF). A GeoTIFF brings its own coordinates; a picture needs its edges
typed in, and the measurement stations are drawn on it so the fit can be checked.

**Which way each plane dips** is settled first. A strike allows two dip directions 180° apart, so the
recorded dip direction picks one; because it is usually rounded in the field, anything within 45° is
accepted and the exact value comes from the strike. A reading whose dip direction fits neither choice was
mis-recorded, and is set aside and listed rather than used.

**Structural sets** are found by clustering the poles to the planes (spherical k-means on axes). A split is
only accepted when the set mean poles stand at least 25° apart AND that separation is more than 2.2 times
the scatter inside them: k-means will always split a single cloud if asked, and both tests are needed to
stop it. Each set is reported with its mean strike and dip, Fisher concentration κ, the 95% cone on the
mean, and its spread.

**Where the sets cross** is the line two planes share. In orogenic gold the ore shoot commonly follows it,
so its plunge is the direction a hole has to follow to stay in the shoot. When the two sets meet at a
conjugate angle (25-75°), the acute bisector is reported as the likely shortening direction, with the
caveat that it only holds if the two sets formed together.

Everything is stored for **6 · Integrated gold prediction**: a structure-type column (shear, vein, fault)
is read there as the host-rock column, so shears become the gold-favourable hosts behind the "proximity to
favourable lithology" evidence layer, and the measurement positions behind "proximity to measured
structures". The QDS sheet is kept as the background map.

## Soil gold geochemistry (Section 5)

One CSV or Excel file of soil samples: position, sample ID and a gold assay. The unit (ppm / g/t or ppb) is
guessed from the size of the numbers and can be changed; everything is reported in both.

**Where background stops** is the question the page answers first. It fits one and two log-normal populations
to the assays and compares them by BIC. Two populations win only when the data really hold a mineralised
group on top of background, and the line between them becomes the anomaly threshold. The usual fixed rules
(mean + σ on logs, percentiles, median + MAD) are still selectable. A log-probability plot shows the same
thing by eye: one population is a straight line of points, two make a step.

**Assay units** are ppm (the same as g/t) or per cent, and you pick which. The choice decides every
threshold on the page, so it is worth checking against the lab certificate; results are reported in your
unit and in ppb.

**Interpolating between samples** offers five methods, because they disagree in ways that matter:

| Method | What it does | Watch for |
|---|---|---|
| Natural neighbour (default) | Each sample gets the share of ground a new sample at that spot would steal from it (Sibson, 1981) | Stops at the edge of the sampled ground |
| Inverse distance (IDW) | Weighted average of the nearest samples, 1/distance^p | A bull's-eye at every sample; choose the power |
| Ordinary kriging | Fits a variogram, then weights samples to minimise the estimation error | The only one that also maps its own uncertainty; smooths towards the local mean wherever there is a nugget |
| Minimum curvature | Smoothest surface through every reading | Can overshoot beyond the data range in gaps |
| Linear triangulation | Flat triangles between samples | Visible facets |

Kriging reports the fitted variogram (model, range, nugget-to-sill ratio and fit) with its plot, and a
standard-error map. The **range** is roughly the size of the anomalies and a good guide to infill spacing;
the **nugget ratio** is the share of variation already present between samples a metre apart, and above
about 50 % the page says so, because the map is then mostly noise. The variogram model can be fixed
(spherical, exponential, Gaussian) or left on Auto, which fits all three and keeps the best.

The **grid interval** is yours to set. Left automatic it is about a third of the sample spacing; a finer
interval than that only smooths the picture, it adds no real detail. Assays are interpolated in log space
by default, because gold is log-normal and one high sample would otherwise dominate the whole map. The
result downloads as a PNG or a GeoTIFF.

**Anomaly zones.** Anomalous samples within 1.8 sample spacings are joined into one zone, so a zone is a run
of neighbouring anomalous samples rather than a ring drawn around each one. Each zone is reported with its
peak and mean assay, peak coordinates, length, trend, and whether it runs off the edge of the survey and is
therefore open. Zones of one or two samples are flagged for re-sampling rather than follow-up: a single high
sample beside ordinary ones is as often one grain in the sieve as it is a discovery. Where the sampling
itself is strongly directional, a zone trend parallel to it is called out as possibly an artefact.

The written interpretation covers the survey layout, the gold levels against the usual rules of thumb for
residual soil, the detection limit if the low end piles up on one value, the populations, the zones, the
trend and what to do next. Everything is stored for **6 · Integrated gold prediction**, which uses the log
z-scores of the assays as its gold geochemistry evidence layer.

## Accounts, subscriptions and hosting

One Flask application serves all of it: the public site, the sign-in, the admin dashboard and the
workbench. A section of the workbench is an ordinary route behind `@login_required`, so a person
whose plan does not include a section simply has no route to it.

```
            ┌──────── nginx (443, TLS) ────────┐
browser ───▶│  everything → gunicorn :8000     │
            └──────────────────────────────────┘
```

**How access is decided.** `portal/app/__init__.py` wraps every section view in a decorator that
checks the signed-in person's plan. There is nothing signed, forwarded or trusted between two
servers, because there is only one server. Results are addressed from the owner's own user id, so
another person's run id does not resolve.

**Heavy work runs in the background.** A satellite fetch or an IP inversion takes longer than a web
request should, so it goes to a worker thread and the browser watches a small progress endpoint. The
result is written to disk the moment it is finished, which is also what lets section 6 combine work
done on different days.

**A plan is data, not code** (`portal/models.py`). Which sections it opens, whether downloads are
included, the monthly upload cap and the seat count are columns in a table, so the client adds a tier or
changes what one includes from the admin dashboard without anyone touching Python.

**The admin dashboard** at `/admin` lists accounts, who is paying, revenue this month, and subscriptions
ending within seven days. An administrator activates a subscription by recording what was paid (amount,
method, receipt reference); re-activating the same plan adds days rather than stacking. Suspending,
cancelling and granting admin rights are there too, and every one of those actions is written to an
audit log with who did it.

**Payments** are manual to begin with: someone pays however the client prefers and an administrator
records it. `portal/billing.py` has the two functions a gateway needs and the webhook route already
wired, with signature checking failing closed until a real provider is connected. For Tanzania the
candidates are Flutterwave, Pesapal, DPO Pay, Selcom or M-Pesa direct; Stripe does not onboard Tanzanian
merchants. Worth knowing before choosing: mobile money generally cannot keep a card on file and charge
it silently each month, so renewal is normally a request the subscriber approves, which is why the model
here is expiry-and-renew rather than a silent auto-charge.

Deployment is in [`deploy/README.md`](deploy/README.md): a plain Ubuntu VPS with nginx and one
systemd unit, or **cPanel**, which works now that there is a single WSGI application.

## Project layout

```
run_local.py               starts everything on your own machine
wsgi.py                    what gunicorn runs; `flask --app wsgi admin-user` makes the first admin
passenger_wsgi.py          the same application, for cPanel
portal/                    the whole web application (Flask)
  models.py                users, plans, subscriptions, payments, audit log
  config.py  auth.py  public.py  billing.py  admin.py
  app/                     the workbench
    __init__.py            section registry, access control, job and figure routes
    store.py               per-person uploads, runs and cached results on disk
    jobs.py                background work with progress the browser can watch
    render.py              figures, tables and downloads
    uploads.py             reading a file and working out what still needs choosing
    sections/
      base.py              the upload -> submit -> results cycle every section shares
      remote.py            1 · Remote sensing (AOI, scene, ten image tabs)
      magnetic.py          2 · Magnetic (one file -> 12 maps)
      ip.py                3 · IP (one file per line -> inverse model sections)
      geology.py           4 · Geology (structure and/or a QDS sheet)
      geochemistry.py      5 · Geochemistry (soil gold)
      prediction.py        6 · Integrated gold prediction
      pits.py              Pit mapping
  templates/               the pages; app/ holds the workbench, one folder per section
  static/                  style.css, app.css, app.js
deploy/                    nginx config, systemd unit, env example, VPS and cPanel guide
geoai/                     the science: all of it independent of the web layer
  io.py                    read CSV / TXT / DAT / Excel
  coords.py                parse corners (decimal, DMS, UTM, Arc 1960) -> AOI, area, KML/GeoJSON
  grid.py                  gridding for point surveys
  indices.py               Sentinel-2 indices, band ratios, sigma classes, target zones
  pca.py                   standard PCA + Crósta feature-oriented PCs
  lithology.py             K-means spectral (lithological) classification
  lineaments.py            Canny + Hough lineament extraction, rose stats, density, GeoJSON
  terrain.py               hillshade, slope, aspect
  terrain3d.py             interactive 3-D terrain (Plotly) with a picture or a measured grid draped over it
  prospectivity.py         fuzzy evidence layers, weighted sum / fuzzy gamma, targets
  export.py                PNG, GeoTIFF, KML exports and map figures
  readers.py               any upload -> table / raster / image (Geosoft XYZ .dat, CSV, Excel, GeoTIFF, PNG)
  datasets.py              uploads + column choices -> datasets in one working UTM system
  geophysics.py            survey geometry, minimum-curvature gridding, FFT filters, Euler deconvolution,
                           spectral depth, IP sections
  magnetics.py             IGRF field direction, RTP / RTE, VD, AS, tilt, THDR, SPI, magnetic analysis
  res2d.py                 Res2DInv .dat reader (general array and index formats)
  ipinv.py                 IP inversion: half-space sensitivities, smoothness-constrained solve, Res2DInv-style figure
  geochem.py               lab value cleaning, log thresholds, classes, correlation, gold index
  soilgold.py              soil gold: survey layout, two-population fit, anomaly zones, written interpretation
  interpolate.py           natural neighbour, IDW, ordinary kriging (+ variogram), on a chosen grid interval
  geology.py               lithology / structural measurements, rose, stereonet poles
  structure.py             structural sets, Fisher statistics, set intersections, structural interpretation
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
| `structure_measurements.csv` | 4 · Geology (21 shear and fault readings, strike / dip / dip direction) |
| `qds_sheet.png` | 4 · Geology (a made-up map sheet for testing the background; edges 548050-548450 E, 9048450-9049250 N, UTM 36S) |
| `geology_mapping.csv` | older lithology sample (lithology, strike, dip) |
| `rs_alteration_indices.tif` | 5 · Integrated gold prediction (add remote sensing layers) (clay and iron oxide indices) |
| `soil_gold.xlsx` | 5 · Geochemistry (609 soil samples on a 25 m grid, Au in ppm, a 020° shear open to the north) |
| `soil_geochem.xlsx` | older multi-element sample (Au_ppb with '<5', As, Sb, Bi, Cu, Pb, Zn) |

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
