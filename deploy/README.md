# Putting GeoAI on a VPS

A plain Ubuntu VPS, no cPanel. The portal (Flask) answers on `/`, the Streamlit app on `/app`, and nginx
only lets a signed-in subscriber reach the app.

```
            ┌─────────── nginx (443, TLS) ───────────┐
browser ───▶│  /       → portal   (gunicorn :8000)   │
            │  /app/   → ask the portal first,       │
            │            then Streamlit (:8501)      │
            └────────────────────────────────────────┘
```

Both services listen on 127.0.0.1, so nginx is the only way in and the gate cannot be walked around.

## What the VPS needs

2 vCPU and **4 GB of RAM** is a sensible starting point: the satellite pages hold several bands of
imagery in memory at once, and 2 GB leaves no headroom once two people are working. 20 GB of disk.
Ubuntu 22.04 or 24.04.

## 1. System packages

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential nginx git \
                    libgdal-dev gdal-bin libgeos-dev libproj-dev proj-data
```

`rasterio` and `shapely` normally install from wheels that carry their own GDAL and GEOS, and then the
`-dev` packages above are not needed. Install them anyway: when a wheel is missing for your Python
version, pip falls back to building from source and will stop without them.

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

`GEOAI_SECRET_KEY` does two jobs: it signs the login cookie and it signs the gate token. **Setting it is
what turns the gate on.** With it unset, `streamlit run app.py` stays open for local work; with it set,
a request that arrives without a valid token is refused. Use the same value in both services, which the
shared `.env` takes care of.

## 4. The first administrator

```bash
cd /srv/geoai
sudo -u geoai -H env $(grep -v '^#' .env | xargs) venv/bin/flask --app wsgi admin-user \
    you@example.com 'a long password you will remember'
sudo -u geoai -H env $(grep -v '^#' .env | xargs) venv/bin/flask --app wsgi show-config
```

`show-config` prints the settings and warns about anything unsafe. Fix what it reports before step 6.

## 5. The two services

```bash
sudo cp /srv/geoai/deploy/geoai-portal.service /etc/systemd/system/
sudo cp /srv/geoai/deploy/geoai-app.service    /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now geoai-portal geoai-app
systemctl status geoai-portal geoai-app
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

## 7. Check the gate actually holds

```bash
curl -I https://geoai.example.com/app/          # expect 302 to /login
curl -I http://127.0.0.1:8501/                  # only answers on the VPS itself, never from outside
```

From another machine, `http://your.ip:8501` must not answer at all. If it does, Streamlit is listening
on the wrong address: check `--server.address 127.0.0.1` in the service file.

## Updating

```bash
cd /srv/geoai && sudo -u geoai git pull
sudo -u geoai venv/bin/pip install -r requirements.txt
sudo systemctl restart geoai-portal geoai-app
```

## Backups

Everything that matters is one SQLite file:

```bash
sudo -u geoai sqlite3 /srv/geoai/data/portal.db ".backup '/srv/geoai/data/portal-$(date +%F).db'"
```

Put that in cron and copy the result off the machine. Moving to PostgreSQL later is a change to
`GEOAI_DATABASE_URL` and nothing else.

## What is not done yet

- **A payment gateway.** `GEOAI_PAYMENT_PROVIDER=manual` is complete and working: a subscriber pays, an
  administrator records it on their user page, access starts. `portal/billing.py` has the two functions a
  gateway needs (`start_checkout` and `handle_webhook`) and the webhook route, with signature checking
  left deliberately failing closed until a real provider is wired in.
- **Email.** Nothing is sent: no confirmation, no password reset, no renewal reminder. The admin
  dashboard lists subscriptions ending within seven days so they can be chased by hand meanwhile.
- **Several app servers.** One Streamlit process serves everyone. That is fine for a handful of
  concurrent users; beyond that, give heavy work its own queue rather than adding processes, because each
  one would hold its own copy of the imagery.
