# Running GeoAI on your own machine

One command, in the geoAi folder:

```
venv\Scripts\python run_local.py
```

It starts everything on **http://localhost:8000** — the public site, the sign-in, the admin dashboard
and the workbench — makes an administrator the first time it runs, and prints the sign-in details:

| | |
|---|---|
| Site | http://localhost:8000 |
| Workbench | http://localhost:8000/app/ |
| Admin dashboard | http://localhost:8000/admin |
| Administrator | `admin@local` / `local-admin-password` |

Ctrl+C stops it. The first run makes `data\portal.db`; delete that file to start over with no
accounts. Uploads and results live in `data\work\<your user id>\`.

If the packages are not installed yet: `venv\Scripts\pip install -r requirements.txt`.

## Walk through it as a customer would

1. **Sign up.** Open the site, press *Create account*. You land on the Free plan, so the workbench
   shows Remote sensing and nothing else.
2. **Open the workbench.** The locked sections are greyed out with a 🔒.
3. **Become the administrator.** Open a **private / incognito window** so you stay signed in as the
   customer in the first one, go to http://localhost:8000/login and sign in as `admin@local`.
4. **Take a payment.** Admin → Users → *Manage* on the customer → pick a plan, type what they paid
   and the receipt number, press *Activate*. The plan, the payment and who recorded it are all stored.
5. **Back in the customer's window.** Reload the workbench: Magnetic, IP and Pit mapping are now
   open. Upgrade them to Pro and Geology, Geochemistry and the integrated prediction appear too.
6. **Try to get in without paying.** Sign out and open http://localhost:8000/app/magnetic — it sends
   you to the sign-in page. Signed in on a Free plan, it sends you to the pricing page.

Other things worth trying from the admin dashboard: *Plans* to change what each plan opens (add
Geochemistry to Basic and the customer has it on their next reload — plans are data, not code),
*Payments* for the money taken, *Audit* for who did what, and *Suspend* on a user.

## Trying the sections

There are sample files in `sample_data\`:

| Section | File |
|---|---|
| 1 · Remote sensing | type the corners in, or `aoi_corners_latlong.csv` |
| 2 · Magnetic | `magnetics_tmi.dat` |
| 3 · IP | `ip_res2dinv_L1.dat`, or the three `ip_line_*.csv` together |
| 4 · Geology | `structure_measurements.csv` and/or `qds_sheet.png` |
| 5 · Geochemistry | `soil_gold.xlsx` |
| Pit mapping | `pits_sample.xlsx` plus `pits_notes.txt` |
| 6 · Prediction | tick whatever the other sections have produced |

Remote sensing downloads Sentinel-2 imagery and a Copernicus DEM, so that one needs an internet
connection; everything else works offline.

## How this differs from the server

Only two things: `run_local.py` uses a fixed development secret, and it serves over plain HTTP with
Flask's own server instead of gunicorn behind nginx. There is no separate app process, no token and
no gate to configure — the workbench is part of the same application as the sign-in, so what you see
locally is what runs on the server. `deploy/README.md` covers the real thing, including cPanel.

## If something does not start

| What you see | What it means |
|---|---|
| `Address already in use` | Something already holds port 8000. Close it, or set `GEOAI_PORT`. |
| `No module named flask` | `venv\Scripts\pip install -r requirements.txt` |
| A section says "not part of your plan" | You are signed in as a customer, not the administrator. |
| A run never finishes | Look at the terminal window; the error is printed there. |
