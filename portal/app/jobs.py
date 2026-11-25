"""Long work, run off the request thread, with progress the browser can watch.

A satellite fetch or an IP inversion takes longer than a web request should: a request that blocks for
three minutes hits a proxy timeout. So the work goes to a worker thread, the browser polls a small
JSON endpoint, and the page swaps itself for the results when the job reports it is done.

Jobs live in memory. That is the right trade here: one process serves the app (see deploy/README.md), a
job is worthless once the process restarts, and the *result* is on disk the moment the job finishes, so
nothing of value is lost — a restart mid-run costs the run, not the data.
"""
import threading
import time
import traceback
import uuid

MAX_RUNNING = 2            # heavy work; more at once would exhaust memory before it finished faster
KEEP_FINISHED = 60 * 30    # forget a finished job after half an hour

_lock = threading.Lock()
_jobs = {}
_slots = threading.Semaphore(MAX_RUNNING)


class Job:
    def __init__(self, user_id, section, run_id, label=""):
        self.id = uuid.uuid4().hex
        self.user_id = str(user_id)
        self.section = section
        self.run_id = run_id
        self.label = label
        self.state = "queued"          # queued | running | done | failed | cancelled
        self.progress = 0.0
        self.message = "Waiting for a free slot…"
        self.error = ""
        self.started = time.time()
        self.finished = None
        self.cancel = threading.Event()

    def say(self, fraction, text=""):
        """The progress callback handed to the engine. Raises to stop a cancelled run."""
        if self.cancel.is_set():
            raise Cancelled()
        self.progress = max(0.0, min(float(fraction), 1.0))
        if text:
            self.message = str(text)

    def as_dict(self):
        return {"id": self.id, "state": self.state, "progress": round(self.progress, 3),
                "message": self.message, "error": self.error, "section": self.section,
                "run_id": self.run_id, "label": self.label,
                "seconds": round((self.finished or time.time()) - self.started, 1)}


class Cancelled(Exception):
    """Raised inside a worker when the person pressed Cancel."""


def _run(app, job, work):
    with _slots:
        if job.cancel.is_set():
            job.state, job.message = "cancelled", "Cancelled before it started."
            job.finished = time.time()
            return
        job.state, job.message = "running", "Starting…"
        with app.app_context():
            try:
                work(job)
                job.state, job.progress, job.message = "done", 1.0, "Finished."
            except Cancelled:
                job.state, job.message = "cancelled", "Cancelled."
            except Exception as e:                     # noqa: BLE001 - the browser must see any failure
                job.state = "failed"
                job.error = str(e) or e.__class__.__name__
                job.message = "It did not finish."
                app.logger.error("job %s (%s) failed: %s\n%s", job.id, job.section, e,
                                 traceback.format_exc())
            finally:
                job.finished = time.time()
                _forget_old()


def start(app, user_id, section, run_id, work, label=""):
    """Queue `work(job)` and return the Job. `work` gets the job so it can report progress."""
    job = Job(user_id, section, run_id, label)
    with _lock:
        _jobs[job.id] = job
    t = threading.Thread(target=_run, args=(app, job, work), name=f"geoai-{section}", daemon=True)
    t.start()
    return job


def get(job_id, user_id=None):
    job = _jobs.get(job_id)
    if job is None:
        return None
    if user_id is not None and job.user_id != str(user_id):
        return None                                    # never report on someone else's work
    return job


def running_for(user_id, section=None):
    return [j for j in list(_jobs.values())
            if j.user_id == str(user_id) and j.state in ("queued", "running")
            and (section is None or j.section == section)]


def _forget_old():
    cutoff = time.time() - KEEP_FINISHED
    with _lock:
        for jid, j in list(_jobs.items()):
            if j.finished and j.finished < cutoff:
                _jobs.pop(jid, None)
