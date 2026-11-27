"""Icons, as one inlined SVG sprite.

Lucide, built into portal/static/icons.svg by a script kept with the design notes. The sprite is
inlined once per page rather than linked, so an icon always inherits the colour of the text around
it — the same markup works on the black rail and on the white page — and there is no second request
to lose on a slow connection.
"""
import functools
import pathlib

from markupsafe import Markup

SPRITE = pathlib.Path(__file__).parent / "static" / "icons.svg"


@functools.lru_cache(maxsize=1)
def sprite():
    try:
        return Markup(SPRITE.read_text(encoding="utf-8"))
    except OSError:
        return Markup("")


@functools.lru_cache(maxsize=1)
def names():
    """Every icon in the sprite, so a typo can be caught by a test rather than by a blank square."""
    import re
    return frozenset(re.findall(r'id="i-([a-z0-9-]+)"', str(sprite())))


def icon(name, cls="", label=None):
    """One icon. Decorative by default; pass `label` when it is the only thing naming a control."""
    if name not in names():
        raise KeyError(f"no icon named {name!r} in {SPRITE.name}")
    aria = (f'role="img" aria-label="{label}"' if label else 'aria-hidden="true" focusable="false"')
    classes = f"ico {cls}".strip()
    return Markup(f'<svg class="{classes}" {aria}><use href="#i-{name}"></use></svg>')


def init_app(app):
    app.jinja_env.globals["icon"] = icon
    app.jinja_env.globals["icon_sprite"] = sprite
