# Putting GeoAI on a server

GeoAI is one Flask application: the public site, the sign-in, the admin dashboard and the workbench
all come from the same process, and a section of the workbench is an ordinary route behind
`@login_required`. That makes hosting simple, and it means two things are now true that were not
before: **cPanel works**, and there is no gate between two servers that could be walked around.

Two routes are written up here. A **VPS** is the better home for this app and is covered first;
**cPanel** works and is covered after it, with its limits stated plainly.

---

# A. A plain Ubuntu VPS

```
            ┌──────── nginx (443, TLS) ────────┐
browser ───▶│  everything → gunicorn :8000     │
            └──────────────────────────────────┘
```

Gunicorn listens on 127.0.0.1, so nginx is the only way in.

## What the VPS needs

2 vCPU and **4 GB of RAM** is a sensible start: the satellite pages hold several bands of imagery in
memory at once, and 2 GB leaves no headroom once two people are working. 20 GB of disk, plus room for
what people upload. Ubuntu 22.04 or 24.04.

## 1. System packages

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential nginx git \
                    libgdal-dev gdal-bin libgeos-dev libproj-dev proj-data
```

`rasterio` and `shapely` normally install from wheels that carry their own GDAL and GEOS, and then the
`-dev` packages are not needed. Install them anyway: when a wheel is missing for your Python version,
pip falls back to building from source and will stop without them.

## 2. A user and the code

```bash
sudo adduser --system --group --home /srv/geoai geoai
sudo -u geoai -H bash
cd /srv/geoai
git clone <your repository> .          # or upload the folder
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
mkdir -p data
exit
```

## 3. Settings

```bash
sudo -u geoai cp /srv/geoai/deploy/env.example /srv/geoai/.env
python3 -c "import secrets; print(secrets.token_urlsafe(48))"   # paste as GEOAI_SECRET_KEY
sudo -u geoai nano /srv/geoai/.env
sudo chmod 600 /srv/geoai/.env
```

`GEOAI_SECRET_KEY` signs the session cookie. Change it and everyone is signed out; leak it and anyone
can forge a session, so keep that file at mode 600.

## 4. The first administrator

```bash
cd /srv/geoai
sudo -u geoai -H env $(grep -v '^#' .env | xargs) venv/bin/flask --app wsgi admin-user \
    you@example.com 'a long password you will remember'
sudo -u geoai -H env $(grep -v '^#' .env | xargs) venv/bin/flask --app wsgi show-config
```

`show-config` prints the settings and warns about anything unsafe. Fix what it reports before step 6.

## 5. The service

```bash
sudo cp /srv/geoai/deploy/geoai.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now geoai
systemctl status geoai
```

## 6. nginx and TLS

```bash
sudo cp /srv/geoai/deploy/nginx.conf /etc/nginx/sites-available/geoai
sudo nano /etc/nginx/sites-available/geoai          # change geoai.example.com
sudo ln -s /etc/nginx/sites-available/geoai /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d geoai.example.com           # writes the TLS block and the redirect
```

Then close everything except SSH and the web:

```bash
sudo ufw allow OpenSSH && sudo ufw allow 'Nginx Full' && sudo ufw enable
```

## 7. Check that access control holds

```bash
curl -I https://geoai.example.com/app/            # expect 302 to /login
curl -I https://geoai.example.com/app/magnetic    # expect 302 to /login
curl -I http://127.0.0.1:8000/                    # answers on the VPS itself, never from outside
```

From another machine, `http://your.ip:8000` must not answer at all. Then sign in with a Free account
and confirm that `/app/magnetic` sends you to the pricing page rather than opening.

---

# B. cPanel

cPanel runs a Python application through Passenger, which is exactly what GeoAI is. Use **Setup
Python App** in cPanel:

| Field | Value |
|---|---|
| Python version | 3.11 or newer |
| Application root | `geoai` (where you uploaded the folder) |
| Application URL | your domain or subdomain |
| Application startup file | `passenger_wsgi.py` |
| Application Entry point | `application` |

Then, in the same screen:

1. Add every variable from `deploy/env.example` under **Environment variables** — above all a long
   random `GEOAI_SECRET_KEY`, and `GEOAI_DATA` pointing at a folder **outside** `public_html`.
2. Press **Run Pip Install** against `requirements.txt`.
3. Open the app's terminal (or SSH) and make the first administrator:
   `flask --app wsgi admin-user you@example.com 'a long password'`

`passenger_wsgi.py` is in the repository root and needs no editing.

**What to watch on shared cPanel hosting**

- **Memory.** Shared plans often cap a process at 512 MB–1 GB. The satellite pages need more than
  that for a large area. Sections 2–6 and Pit mapping are comfortable; Remote sensing over a big area
  may be killed. Keep areas small, or put that section's users on a VPS.
- **Binary wheels.** `rasterio`, `shapely` and `pyproj` must install as wheels, because shared hosts
  have no compiler. They normally do on CPython 3.11/3.12 on x86-64. If pip starts building from
  source, ask support to install GDAL, or move to a VPS.
- **Long requests.** Passenger usually kills a request after 60–120 seconds. GeoAI already runs heavy
  work in a background thread and polls for it, so that is handled — but Passenger also stops an idle
  application, and a stopped application loses a job that was still running. Set
  `PassengerMinInstances 1` if your host allows it.
- **One process.** Do not raise the instance count (see the next section).

---

## More than a handful of people

The service runs **one worker with threads**, on purpose. A background job lives in the process that
started it, and its progress is read from there, so a second process would not know about a job the
first one is running. The results themselves are written to disk the moment a job finishes, so
nothing of value is lost when the process restarts — only a run that was still going.

That is fine for a handful of geologists working at once. Beyond that, the right change is not more
workers: it is to move the job runner out of the web process into a queue (RQ or Celery with Redis),
after which any number of workers can serve pages while the queue does the computing.

## Updating

```bash
cd /srv/geoai && sudo -u geoai git pull
sudo -u geoai venv/bin/pip install -r requirements.txt
sudo systemctl restart geoai
```

## Backups

Two things matter: the database and the work folder.

```bash
sudo -u geoai sqlite3 /srv/geoai/data/portal.db ".backup '/srv/geoai/data/portal-$(date +%F).db'"
sudo tar czf /srv/backups/geoai-work-$(date +%F).tar.gz -C /srv/geoai/data work
```

Put those in cron and copy the results off the machine. Moving to PostgreSQL later is a change to
`GEOAI_DATABASE_URL` and nothing else.

Each person's uploads and results live in `data/work/<user id>/`. `GEOAI_KEEP_RUNS` (5 by default)
decides how many finished runs of each section are kept per person before the oldest is dropped.

## What is not done yet

- **A payment gateway.** `GEOAI_PAYMENT_PROVIDER=manual` is complete and working: a subscriber pays,
  an administrator records it on their user page, access starts. `portal/billing.py` has the two
  functions a gateway needs (`start_checkout` and `handle_webhook`) and the webhook route, with
  signature checking left deliberately failing closed until a real provider is wired in.
- **Email.** Nothing is sent: no confirmation, no password reset, no renewal reminder. The admin
  dashboard lists subscriptions ending within seven days so they can be chased by hand meanwhile.
- **A job queue.** See "More than a handful of people" above.
