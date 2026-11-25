"""Where a person's work lives between requests.

Each person gets a folder, and each run in it keeps its settings, its result object and its rendered
pictures. Nothing depends on the browser session, so a reload keeps the results, and section 6 can
combine work done on different days:

    data/work/<user id>/<section>/<run id>/
        run.pkl        the engine's result, as returned by geoai.interpret.run
        meta.json      what was uploaded, which settings, when, how long it took
        fig/*.png      pictures rendered from the run
        files/*        the uploaded files, kept so a run can be repeated

The newest run of a section is the "current" one; older runs stay until they are pruned, so section 6
can combine work done last week.

A run is pickled. That is safe here because nothing but this application ever writes these files, they
never leave the machine, and the user's own uploads reach them only as numpy arrays and dataframes. They
are read back only for the user who owns the folder.
"""
import json
import os
import pickle
import shutil
import time
import uuid
from pathlib import Path

from flask import current_app

SECTIONS = ["remote-sensing", "magnetic", "ip", "geology", "geochemistry", "prediction", "pit-mapping"]
KEEP_RUNS = 5                      # per section, per person
PICKLE_PROTOCOL = 4


def root():
    base = Path(current_app.config["DATA_DIR"]) / "work"
    base.mkdir(parents=True, exist_ok=True)
    return base


def home(user_id):
    p = root() / str(user_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe(section):
    if section not in SECTIONS:
        raise ValueError(f"unknown section {section!r}")
    return section


class Run:
    """One submitted job and everything that came out of it."""

    def __init__(self, user_id, section, run_id):
        self.user_id = str(user_id)
        self.section = _safe(section)
        self.id = run_id
        self.dir = home(user_id) / section / run_id

    # ---------------------------------------------------------------- making one
    @classmethod
    def create(cls, user_id, section):
        run = cls(user_id, section, time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
        (run.dir / "fig").mkdir(parents=True, exist_ok=True)
        (run.dir / "files").mkdir(parents=True, exist_ok=True)
        run.write_meta({"id": run.id, "section": section, "created": time.time(), "state": "new"})
        return run

    @classmethod
    def load(cls, user_id, section, run_id):
        run = cls(user_id, section, run_id)
        return run if run.dir.is_dir() else None

    @classmethod
    def latest(cls, user_id, section):
        runs = cls.list(user_id, section)
        return runs[0] if runs else None

    @classmethod
    def list(cls, user_id, section):
        d = home(user_id) / _safe(section)
        if not d.is_dir():
            return []
        runs = [cls(user_id, section, p.name) for p in d.iterdir()
                if p.is_dir() and not p.name.startswith("_")]     # "_draft" is not a run
        runs.sort(key=lambda r: r.id, reverse=True)          # the id starts with the timestamp
        return runs

    # ---------------------------------------------------------------- what is in it
    def meta(self):
        try:
            return json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def write_meta(self, data):
        current = self.meta()
        current.update(data)
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(current, default=str), encoding="utf-8")
        tmp.replace(self.dir / "meta.json")
        return current

    @property
    def ok(self):
        return self.meta().get("state") == "done"

    def save_result(self, obj):
        tmp = self.dir / "run.pkl.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(obj, fh, protocol=PICKLE_PROTOCOL)
        tmp.replace(self.dir / "run.pkl")

    def result(self):
        """The engine's run object. Cached per request, because it can be tens of megabytes."""
        from flask import g
        key = f"_run_{self.user_id}_{self.section}_{self.id}"
        if not hasattr(g, key):
            path = self.dir / "run.pkl"
            with open(path, "rb") as fh:
                setattr(g, key, pickle.load(fh))
        return getattr(g, key)

    def cached(self, key, build):
        """A derived result kept beside the run — lineaments at one selectivity, a target score for one
        set of weights. Computed once per set of settings and then read back, which is what
        makes flipping between tabs quick."""
        import hashlib
        name = hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:20]
        path = self.dir / "cache" / f"{name}.pkl"
        if path.is_file():
            try:
                with open(path, "rb") as fh:
                    return pickle.load(fh)
            except Exception:
                path.unlink(missing_ok=True)       # a half-written file from a restart
        value = build()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "wb") as fh:
                pickle.dump(value, fh, protocol=PICKLE_PROTOCOL)
            tmp.replace(path)
        except Exception:
            tmp.unlink(missing_ok=True)            # not cacheable; the value is still good
        return value

    # ---------------------------------------------------------------- files and pictures
    def keep_upload(self, filename, raw):
        name = os.path.basename(filename).replace("\\", "_") or "upload"
        path = self.dir / "files" / name
        path.write_bytes(raw)
        return path

    def fig_path(self, name):
        return self.dir / "fig" / f"{name}.png"

    def has_fig(self, name):
        p = self.fig_path(name)
        return p.is_file() and p.stat().st_size > 0

    def delete(self):
        shutil.rmtree(self.dir, ignore_errors=True)


# ---------------------------------------------------------------- files waiting to be submitted
def draft_dir(user_id, section):
    """Where a file sits between being chosen and being submitted, so the column choices on the page
    survive a reload and the person can change one setting without uploading again."""
    d = home(user_id) / _safe(section) / "_draft"
    d.mkdir(parents=True, exist_ok=True)
    return d


def draft_files(user_id, section):
    return sorted((p for p in draft_dir(user_id, section).iterdir() if p.is_file()),
                  key=lambda p: p.stat().st_mtime)


def add_draft(user_id, section, filename, raw, replace=True):
    name = os.path.basename(filename).replace("\\", "_").replace("/", "_") or "upload"
    d = draft_dir(user_id, section)
    if replace:
        clear_draft(user_id, section)
    path = d / name
    path.write_bytes(raw)
    return path


def clear_draft(user_id, section):
    from . import uploads
    for p in draft_dir(user_id, section).iterdir():
        if p.is_file():
            uploads.forget(p)
            p.unlink(missing_ok=True)


def drop_draft(user_id, section, name):
    from . import uploads
    p = draft_dir(user_id, section) / os.path.basename(name)
    if p.is_file():
        uploads.forget(p)
        p.unlink(missing_ok=True)


def prune(user_id, section, keep=KEEP_RUNS):
    for old in Run.list(user_id, section)[keep:]:
        old.delete()


def sections_with_results(user_id):
    """Which sections this person has a finished run for — what section 6 can draw on."""
    out = {}
    for section in SECTIONS:
        run = Run.latest(user_id, section)
        if run and run.ok:
            out[section] = run
    return out


def usage_bytes(user_id):
    total = 0
    for dirpath, _dirs, files in os.walk(home(user_id)):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total
