"""One module per section of the workbench.

Each module gets `attach(bp)` called once, adds its own routes, and may offer:

    redraw(run, name, args) -> Path | None     draw a picture that was not pre-rendered
    deliver(run, kind, name, args) -> Response hand over a download (PNG, CSV, GeoTIFF, KML, ZIP)

Both are dispatched from the generic /fig and /download routes so every section gets the same
access checks without repeating them.
"""
from flask import abort

MODULES = {}


def attach(bp):
    from . import geochemistry, geology, ip, magnetic, pits, prediction, remote
    for mod in (remote, magnetic, ip, geology, geochemistry, prediction, pits):
        mod.attach(bp)
        MODULES[mod.KEY] = mod


def redraw(run, name, args):
    """Draw a picture that is not on disk yet, one at a time (see render.draw_lock)."""
    from ..render import draw_lock
    mod = MODULES.get(run.section)
    fn = getattr(mod, "redraw", None) if mod else None
    if fn is None:
        return None
    with draw_lock:
        if run.has_fig(name):            # another thread drew it while this one waited
            return run.fig_path(name)
        return fn(run, name, args)


def deliver(run, kind, name, args):
    from ..render import draw_lock
    mod = MODULES.get(run.section)
    fn = getattr(mod, "deliver", None) if mod else None
    if fn is None:
        abort(404)
    with draw_lock:
        return fn(run, kind, name, args)
