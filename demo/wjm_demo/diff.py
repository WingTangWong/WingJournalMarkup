"""Differential update between two captures of the same page (a light M8).

Text-level: which parsed elements appeared / disappeared / changed state, and
which metadata fields moved.
"""

from __future__ import annotations


def _element_key(el: dict) -> str:
    if el["kind"] == "bullet":
        return f"bullet::{el['data'].get('item', el['text']).lower()}"
    return f"{el['kind']}::{el['text'].strip().lower()}"


def diff_captures(prev: dict, curr: dict) -> dict:
    prev_els = {_element_key(e): e for e in prev.get("detected_elements", [])}
    curr_els = {_element_key(e): e for e in curr.get("detected_elements", [])}

    added = [curr_els[k] for k in curr_els if k not in prev_els]
    removed = [prev_els[k] for k in prev_els if k not in curr_els]

    changed = []
    for k in prev_els.keys() & curr_els.keys():
        a, b = prev_els[k], curr_els[k]
        if a["kind"] == "bullet" and a["data"].get("state") != b["data"].get("state"):
            changed.append({
                "item": b["data"].get("item", b["text"]),
                "from": a["data"].get("state"),
                "to": b["data"].get("state"),
            })

    meta_changes = {}
    pm, cm = prev.get("page_metadata") or {}, curr.get("page_metadata") or {}
    for field in ("document_id", "page_id", "left", "above", "below", "right"):
        if pm.get(field) != cm.get(field):
            meta_changes[field] = {"from": pm.get(field), "to": cm.get(field)}
    if set(pm.get("topic_tags") or []) != set(cm.get("topic_tags") or []):
        meta_changes["topic_tags"] = {
            "from": pm.get("topic_tags") or [], "to": cm.get("topic_tags") or []
        }

    return {
        "added": [{"kind": e["kind"], "text": e["text"]} for e in added],
        "removed": [{"kind": e["kind"], "text": e["text"]} for e in removed],
        "changed": changed,
        "metadata_changes": meta_changes,
        "boundary_method": _pair(prev.get("page_boundary_method"),
                                 curr.get("page_boundary_method")),
        "orientation": _pair(prev.get("orientation_degrees"),
                             curr.get("orientation_degrees")),
    }


def _pair(a, b):
    return {"from": a, "to": b} if a != b else None


def is_empty(d: dict) -> bool:
    return not any(
        d[k] for k in ("added", "removed", "changed", "metadata_changes")
    ) and not d["boundary_method"] and not d["orientation"]
