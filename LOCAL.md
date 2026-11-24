# Seeing the whole thing on your own machine

Two halves run side by side: the **portal** (accounts, plans, admin) and the **app** (the eight GeoAI
sections). One command starts both and wires the gate between them.

```
C:\Users\ADMIN\Documents\geoAi> python run_local.py
```

It prints two addresses and an administrator to sign in with:

| | |
|---|---|
| Portal | http://localhost:8000 |
| Admin dashboard | http://localhost:8000/admin |
| The app itself | http://localhost:8501 |
| Administrator | `admin@local` / `local-admin-password` |

Ctrl+C in that window stops both. The first run makes `data\portal.db`; delete that file to start over
with no accounts.

## Walk through it as a customer would

1. **Sign up.** Open http://localhost:8000, press *Create account*, fill it in. You land on the pricing
   page with a Free plan.
2. **Open the app.** Press *Open the app* in the top bar. The sidebar has Home and Remote sensing, the
   home page shows the other six with 🔒 *Comes with a higher plan*, and the bottom of the sidebar shows
   your name and plan.
3. **Become the administrator.** Open a **private / incognito window** so you stay signed in as the
   customer in the first one, go to http://localhost:8000/login and sign in as `admin@local`.
4. **Take a payment.** Admin → Users → *Manage* on the customer → pick a plan, type what they paid and
   the receipt number, press *Activate*. The plan, the payment and who recorded it are all stored.
5. **Back to the customer's window.** Reload, press *Open the app* again: Magnetic, IP and Pit mapping
   are now in the sidebar. Upgrade them to Pro and Geology, Geochemistry and the integrated prediction
   appear as well.
6. **Try to get in without paying.** Open http://localhost:8501 directly: "Please sign in". That is the
   gate doing its job.

Other things worth trying from the admin dashboard: *Plans* to change what each plan opens (edit the
Basic plan to include Geochemistry and the customer has it on their next reload — plans are data, not
code), *Payments* for the money taken, *Audit* for who did what, and *Suspend* on a user.

## How this differs from the real server

On the VPS, nginx sits in front, asks the portal "is this person allowed?" on every request, and puts
the answer in a header the app reads. On your laptop there is no nginx, so `run_local.py` sets two
things that exist only for this:

- `GEOAI_APP_URL=http://localhost:8501` — tells the portal the app is at its own address rather than
  behind `/app`.
- `GEOAI_ALLOW_TOKEN_IN_URL=1` — lets the gate token travel in the link, since there is no header.

Neither is set in `deploy/env.example`, so the server never accepts a token from a URL. The development
secret in `run_local.py` is also fixed and public; the server gets a random one (see `deploy/README.md`,
step 3). This is why you will see `?t=…` in the address bar locally and never will in production.

## If something does not start

| What you see | What it means |
|---|---|
| `Address already in use` | Something already holds 8000 or 8501. Close it, or change the ports at the bottom of `run_local.py`. |
| `No module named flask` | `pip install -r requirements.txt` |
| The app says "Please sign in" after you pressed *Open the app* | The link was opened more than 12 hours after it was made. Go back to the portal and press it again. |
| You refreshed and got signed out | Keep the `?t=…` on the end of the address; it is what identifies you locally. |

## Running the app on its own, with no portal

Still works, exactly as before:

```
streamlit run app.py
```

With no `GEOAI_SECRET_KEY` set, the gate stays open, every section is there, and no sign-in badge
appears. That is the quickest way to work on a section without thinking about plans.
