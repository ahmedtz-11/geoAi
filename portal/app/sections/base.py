"""What every section of the workbench does the same way.

A section is always the same four steps: choose files, say how to read them, submit, look at the
results. The differences live in each section's own module; everything here is shared so those modules
stay about geology rather than about HTTP.
"""
import time

from flask import (abort, current_app, flash, redirect, render_template, request, send_file,
                   url_for)
from flask_login import current_user
from werkzeug.utils import secure_filename

from .. import jobs, store, uploads
from ..render import df_to_csv_bytes


# ------------------------------------------------------------------ uploads
def take_upload(section_key, field="file", multiple=False):
    """Save whatever the browser just sent into the draft area. Returns how many files arrived."""
    files = request.files.getlist(field) if multiple else (
        [request.files[field]] if field in request.files else [])
    kept = 0
    for f in files:
        if not f or not f.filename:
            continue
        name = secure_filename(f.filename) or "upload"
        if "." not in name or name.rsplit(".", 1)[1].lower() not in uploads.ACCEPT:
            flash(f"{f.filename}: GeoAI does not read that kind of file. "
                  f"Accepted: {', '.join(uploads.ACCEPT)}.", "error")
            continue
        store.add_draft(current_user.id, section_key, name, f.read(), replace=not multiple and kept == 0)
        kept += 1
    return kept


def drafts(section_key):
    """The staged files, each with what is in it — or why it cannot be read."""
    out = []
    for path in store.draft_files(current_user.id, section_key):
        sheet = request.values.get(f"f{len(out)}_sheet") or None
        try:
            out.append({"path": path, "name": path.name, "info": uploads.inspect(path, sheet),
                        "error": None, "index": len(out)})
        except ValueError as e:
            out.append({"path": path, "name": path.name, "info": None, "error": str(e),
                        "index": len(out)})
    return out


def draft_fields(section_key, slot, survey=None, aoi_bounds=None):
    """Staged files with the controls each one still needs, ready for the template."""
    items = drafts(section_key)
    for it in items:
        it["prefix"] = f"f{it['index']}_"
        it["fields"] = (uploads.fields(slot, it["info"], survey, aoi_bounds)
                        if it["info"] else [])
    return items


def effective(items):
    """What the choices currently are: each field's default, overridden by anything the person has
    actually submitted. Lets the page act on the guessed columns before the form is ever sent."""
    from werkzeug.datastructures import MultiDict
    md = MultiDict()
    for it in items:
        for f in it.get("fields", []):
            value = f.get("value")
            if isinstance(value, (list, tuple)):
                for v in value:
                    md.add(it["prefix"] + f["name"], v)
            elif value is not None:
                md.add(it["prefix"] + f["name"], value)
    for key in request.values:
        md.setlist(key, request.values.getlist(key))
    return md


def build_specs(section_key, slot, survey=None):
    """Every staged file turned into an engine spec. Raises ValueError naming the file that is wrong."""
    specs = []
    for it in drafts(section_key):
        if it["error"]:
            raise ValueError(it["error"])
        specs.append(uploads.build_spec(slot, it["path"], request.form, survey=survey,
                                        prefix=f"f{it['index']}_"))
    if not specs:
        raise ValueError("Choose a file first.")
    return specs


# ------------------------------------------------------------------ submitting
def submit(section_key, work, label="", settings=None, keep_files=True):
    """Make a run, hand the work to a background thread, and send the browser to the waiting page.

    `work(run, job)` does the computing and saves whatever it wants into the run.
    """
    from .. import note_usage, upload_allowance
    busy = jobs.running_for(current_user.id)
    if len(busy) >= jobs.MAX_RUNNING:
        flash("You already have work running. Let it finish, then submit this.", "error")
        return redirect(url_for(_endpoint(section_key)))

    used, limit, left = upload_allowance()
    if limit is not None and left <= 0:
        flash(f"Your plan allows {limit} runs a month and you have used {used}. "
              f"The count resets on the first of next month.", "error")
        return redirect(url_for("billing.pricing"))

    run = store.Run.create(current_user.id, section_key)
    if keep_files:
        for path in store.draft_files(current_user.id, section_key):
            run.keep_upload(path.name, path.read_bytes())
    run.write_meta({"state": "running", "label": label, "settings": settings or {},
                    "files": [p.name for p in store.draft_files(current_user.id, section_key)]})

    app = current_app._get_current_object()
    user_id = current_user.id

    def task(job):
        started = time.time()
        try:
            work(run, job)
        except Exception:
            run.write_meta({"state": "failed", "seconds": round(time.time() - started, 1)})
            raise
        run.write_meta({"state": "done", "seconds": round(time.time() - started, 1)})
        store.prune(user_id, section_key, app.config.get("KEEP_RUNS", store.KEEP_RUNS))

    note_usage("upload", section_key, label)
    job = jobs.start(app, user_id, section_key, run.id, task, label)
    return redirect(url_for("app.wait", section_key=section_key, job_id=job.id))


def _endpoint(section_key):
    from .. import BY_KEY
    return BY_KEY[section_key]["endpoint"]


# ------------------------------------------------------------------ showing a section
def page(section_key, template, **ctx):
    """Render a section page with everything its shell needs."""
    from .. import BY_KEY
    s = BY_KEY[section_key]
    run = store.Run.latest(current_user.id, section_key)
    if run is not None and not run.ok:
        run = next((r for r in store.Run.list(current_user.id, section_key) if r.ok), None)
    running = jobs.running_for(current_user.id, section_key)
    ctx.setdefault("run", run)
    ctx.setdefault("running", running[0] if running else None)
    ctx.setdefault("history", [r for r in store.Run.list(current_user.id, section_key) if r.ok])
    return render_template(template, section_key=section_key, section=s,
                           section_title=s["title"], **ctx)


def current_run(section_key, run_id=None):
    """The run a page is showing: the one asked for, or the newest finished one."""
    if run_id:
        run = store.Run.load(current_user.id, section_key, run_id)
        if run is None or not run.ok:
            abort(404)
        return run
    for run in store.Run.list(current_user.id, section_key):
        if run.ok:
            return run
    return None


def need_run(section_key, run_id=None):
    run = current_run(section_key, run_id)
    if run is None:
        abort(404)
    return run


# ------------------------------------------------------------------ downloads
def send_bytes(data, filename, mimetype="application/octet-stream"):
    import io
    return send_file(io.BytesIO(data), mimetype=mimetype, as_attachment=True, download_name=filename)


def send_csv(df, filename, index=False):
    return send_bytes(df_to_csv_bytes(df, index=index), filename, "text/csv")


def send_png(path, filename=None):
    if not path or not path.is_file():
        abort(404)
    return send_file(path, mimetype="image/png", as_attachment=True,
                     download_name=filename or path.name)
