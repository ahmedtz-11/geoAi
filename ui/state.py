"""Keep widget choices when moving between tabs and pages.

Streamlit deletes the state of widgets that are not on screen, and a fragment rerun deletes every widget
state outside that fragment. So each persistent widget (keys starting with PREFIXES) is copied to a plain
"mirror" key after every run, and restored from it when the widget comes back.
"""
import functools

import streamlit as st

ss = st.session_state
MIRROR = "_keep_"
PREFIXES = ("rs_", "s2w_")
NOT_WIDGETS = {"rs_memo_store", "rs_aoi"}


def _widget_keys():
    return [k for k in list(ss.keys()) if isinstance(k, str) and k.startswith(PREFIXES) and k not in NOT_WIDGETS]


def remember():
    for k in _widget_keys():
        ss[MIRROR + k] = ss[k]


def restore():
    """Call at the start of every full run (app.py)."""
    for k in [k for k in list(ss.keys()) if isinstance(k, str) and k.startswith(MIRROR)]:
        key = k[len(MIRROR):]
        if key not in ss:
            ss[key] = ss[k]
    for k in _widget_keys():
        ss[k] = ss[k]


def initial(key, value):
    """Default for a widget key: the remembered value if there is one."""
    if key not in ss:
        ss[key] = ss.get(MIRROR + key, value)
    return key


def forget(prefix):
    for k in [k for k in list(ss.keys()) if isinstance(k, str) and (k.startswith(prefix) or
                                                                    k.startswith(MIRROR + prefix))]:
        del ss[k]


def fragment(fn):
    """st.fragment that remembers widget choices after each of its own reruns."""
    @functools.wraps(fn)
    def run(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            remember()
    return st.fragment(run)
