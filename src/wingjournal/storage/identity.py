"""Page-identity resolution (spec §39) and conflict surfacing (spec §46).

Evidence, strongest first:

    permanent machine page id
    > handwritten metadata page id
    > resolved spatial-graph identity
    > visual match against previous captures
    > new unknown page

Conflicting *explicit* evidence is reported, never silently chosen.
"""

from __future__ import annotations

from dataclasses import dataclass

from wingjournal.models import Conflict, Page
from wingjournal.storage.store import Store


@dataclass
class IdentityResult:
    page: Page
    source: str  # machine_id | handwritten_id | spatial | visual | new
    created: bool
    conflicts: list[Conflict]


def resolve_identity(
    store: Store,
    *,
    page_id_machine: str | None = None,
    page_id_explicit: str | None = None,
    topic_tags: list[str] | None = None,
    document_id_explicit: str | None = None,
    capture_uuid: str | None = None,
) -> IdentityResult:
    """Find or create the :class:`Page` this capture belongs to."""

    conflicts: list[Conflict] = []
    match: Page | None = None
    source = "new"

    if page_id_machine:
        match = store.find_page(page_id_machine=page_id_machine)
        source = "machine_id"
    if match is None and page_id_explicit:
        match = store.find_page(page_id_explicit=page_id_explicit)
        source = "handwritten_id"

    if match is not None:
        # existing page - check the new evidence does not contradict it
        if (
            page_id_machine and match.page_id_machine
            and page_id_machine != match.page_id_machine
        ):
            conflicts.append(Conflict(
                kind="page_id",
                detail=f"machine id {page_id_machine!r} vs stored {match.page_id_machine!r}",
                page_uuid=match.uuid, capture_uuid=capture_uuid,
            ))
        if (
            page_id_explicit and match.page_id_explicit
            and page_id_explicit != match.page_id_explicit
        ):
            conflicts.append(Conflict(
                kind="page_id",
                detail=f"handwritten id {page_id_explicit!r} vs stored "
                       f"{match.page_id_explicit!r}",
                page_uuid=match.uuid, capture_uuid=capture_uuid,
            ))
        if (
            document_id_explicit and match.document_id_explicit
            and document_id_explicit != match.document_id_explicit
        ):
            conflicts.append(Conflict(
                kind="document_id",
                detail=f"document {document_id_explicit!r} vs stored "
                       f"{match.document_id_explicit!r}",
                page_uuid=match.uuid, capture_uuid=capture_uuid,
            ))
        # fill in anything not yet known (does not overwrite)
        match.page_id_machine = match.page_id_machine or page_id_machine
        match.page_id_explicit = match.page_id_explicit or page_id_explicit
        match.document_id_explicit = match.document_id_explicit or document_id_explicit
        if topic_tags:
            match.topic_tags = sorted(set(match.topic_tags) | set(topic_tags))
        store.upsert_page(match)
        page = match
        created = False
    else:
        page = Page(
            page_id_machine=page_id_machine,
            page_id_explicit=page_id_explicit,
            document_id_explicit=document_id_explicit,
            topic_tags=sorted(set(topic_tags or [])),
        )
        store.upsert_page(page)
        source = "new"
        created = True

    for c in conflicts:
        store.add_conflict(c)
    return IdentityResult(page=page, source=source, created=created, conflicts=conflicts)
