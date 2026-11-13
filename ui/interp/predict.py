"""Prediction settings (shared by the Summary and Gold prediction tabs) and the cached computation."""
import numpy as np
import streamlit as st

from geoai import interpret

from .common import PREFIX, memo, ss

METHODS = ["Knowledge-driven (expert weights)", "Data-driven (Random Forest on gold samples)", "Average of both"]


def wkey(name, field):
    return f"{PREFIX}ev_{field}_{abs(hash(name)) % 10 ** 8}"


def evidence(run):
    """Evidence layers, with the lithology layer rebuilt from the favourable lithologies chosen."""
    ev = dict(run["evidence"])
    fav = tuple(ss.get(f"{PREFIX}fav", run["fav_default"]))
    if run["datasets"]["geology"] and run["grid"]:
        tr, shape, _ = run["grid"]
        lay = memo(run, "lith_layer", lambda: interpret.lithology_layer(run["datasets"], fav, tr, shape), fav)
        ev.pop("Proximity to favourable lithology", None)
        if lay is not None:
            ev["Proximity to favourable lithology"] = lay
    return ev


def settings(run, ev):
    w = {k: float(ss.get(wkey(k, "w"), e["default_weight"])) for k, e in ev.items()}
    fav = {k: ss.get(wkey(k, "dir"), e["favourable"]) for k, e in ev.items()}
    return {"weights": w, "favourable": fav, "method": ss.get(f"{PREFIX}method", METHODS[0]),
            "top": float(ss.get(f"{PREFIX}top", 5.0)), "min_ha": float(ss.get(f"{PREFIX}minha", 1.0))}


def gold_labels(run):
    for ds, r in zip(run["datasets"]["geochem"], run["results"]["geochem"]):
        if "au_class" in r:
            return np.column_stack([ds["x"], ds["y"]]), (r["au_class"] >= 2).astype(int), ds["name"]
    return None, None, None


def compute(run, ev, s):
    tr, shape, cell = run["grid"]
    members = {k: memo(run, "member", lambda e=e, d=s["favourable"][k]: interpret.membership(e["raw"], d), k,
                       s["favourable"][k]) for k, e in ev.items()}
    out = {"members": members, "rf": None, "knowledge": None}
    used = {k: v for k, v in s["weights"].items() if v > 0}
    if s["method"] != METHODS[1]:
        out["knowledge"], out["cover"] = interpret.knowledge_driven({k: members[k] for k in used}, used)
    if s["method"] != METHODS[0]:
        xy, lab, src = gold_labels(run)
        if xy is None:
            out["rf"] = {"error": "Needs geochemical data with a gold column."}
        else:
            feats = {k: members[k] for k in members if ev[k]["source"] != "Geochemistry" and s["weights"].get(k, 0) > 0}
            out["rf"] = memo(run, "rf", lambda: interpret.data_driven(feats, xy, lab, tr, shape), tuple(sorted(feats)),
                             tuple(sorted(s["favourable"].items())))
            out["rf_source"] = src
    rf_prob = out["rf"].get("prob") if out["rf"] and "prob" in out["rf"] else None
    if s["method"] == METHODS[0]:
        score = out["knowledge"]
    elif s["method"] == METHODS[1]:
        score = rf_prob
    else:
        score = out["knowledge"] if rf_prob is None else (
            rf_prob if out["knowledge"] is None else np.nanmean([out["knowledge"], rf_prob], axis=0))
    out["score"] = score
    if score is None:
        return out
    valid = np.isfinite(score)
    min_px = max(1, int(round(s["min_ha"] * 1e4 / cell ** 2)))
    used_members = {k: members[k] for k in used}
    out["zones"], out["targets"] = interpret.describe_targets(score, valid, tr, s["top"], min_px, used_members,
                                                             run["results"], run["datasets"])
    return out


def current(run):
    """Prediction with the current settings (cached)."""
    if not run["grid"] or not run["evidence"] and not run["datasets"]["geology"]:
        return None, None, None
    ev = evidence(run)
    if not ev:
        return ev, None, None
    s = settings(run, ev)
    key = (tuple(sorted(s["weights"].items())), tuple(sorted(s["favourable"].items())), s["method"], s["top"],
           s["min_ha"], tuple(ss.get(f"{PREFIX}fav", run["fav_default"])))
    return ev, s, memo(run, "prediction", lambda: compute(run, ev, s), key)
