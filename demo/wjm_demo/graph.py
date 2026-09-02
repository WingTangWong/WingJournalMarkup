"""Build the page-graph view: nodes (pages), edges (spatial + document
membership), orphans, and a grid layout from the L/A/B/R relationships.
"""

from __future__ import annotations

from wjm_demo.store import DemoStore

_OFFSET = {"LEFT": (-1, 0), "RIGHT": (1, 0), "ABOVE": (0, -1), "BELOW": (0, 1)}


def build_graph(store: DemoStore) -> dict:
    pages = store.pages()
    by_uuid = {p.uuid: p for p in pages}

    rels = []
    for p in pages:
        for r in store.wjm.relationships_for_page(p.uuid):
            if r.source_page == p.uuid:  # emit once
                rels.append(r)

    degree: dict[str, int] = {p.uuid: 0 for p in pages}
    edges = []
    for r in rels:
        if r.target_page not in by_uuid:
            continue
        degree[r.source_page] += 1
        degree[r.target_page] += 1
        edges.append({
            "source": r.source_page, "target": r.target_page,
            "relation": r.relation,
            "kind": "explicit" if r.explicitly_declared else "inferred",
        })

    # document membership edges (dashed grouping)
    docs: dict[str, list[str]] = {}
    for p in pages:
        doc = p.document_id_resolved or p.document_id_explicit
        if doc:
            docs.setdefault(doc, []).append(p.uuid)

    nodes = []
    for p in pages:
        doc = p.document_id_resolved or p.document_id_explicit
        nodes.append({
            "id": p.uuid,
            "label": p.page_id_explicit or p.uuid[:8],
            "page_id": p.page_id_explicit,
            "document": doc,
            "topics": p.topic_tags,
            "captures": len(p.capture_uuids),
            "orphan": degree[p.uuid] == 0,
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "documents": {k: v for k, v in docs.items()},
        "layout": _grid_layout(pages, rels, by_uuid),
        "orphans": [n["id"] for n in nodes if n["orphan"]],
    }


def _grid_layout(pages, rels, by_uuid) -> dict[str, list[int]]:
    """Place pages on an integer grid by walking the L/A/B/R relationships."""

    adj: dict[str, list[tuple[str, str]]] = {p.uuid: [] for p in pages}
    for r in rels:
        if r.target_page in by_uuid and r.relation in _OFFSET:
            adj[r.source_page].append((r.target_page, r.relation))
            inv = {"LEFT": "RIGHT", "RIGHT": "LEFT", "ABOVE": "BELOW", "BELOW": "ABOVE"}
            adj[r.target_page].append((r.source_page, inv[r.relation]))

    pos: dict[str, tuple[int, int]] = {}
    free_col = 0
    for start in sorted(adj, key=lambda u: -len(adj[u])):
        if start in pos:
            continue
        pos[start] = (free_col, 0)
        stack = [start]
        while stack:
            u = stack.pop()
            ux, uy = pos[u]
            for v, rel in adj[u]:
                if v in pos:
                    continue
                dx, dy = _OFFSET[rel]
                pos[v] = (ux + dx, uy + dy)
                stack.append(v)
        max_x = max((x for x, _ in (pos[n] for n in pos)), default=free_col)
        free_col = max_x + 3

    return {u: [x, y] for u, (x, y) in pos.items()}
