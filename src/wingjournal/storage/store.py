"""The WJM store: a directory holding a SQLite DB and a content-addressed blob
tree (spec §41, §43). One :class:`Store` == one ``--store`` directory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
from pathlib import Path

from wingjournal.models import (
    Capture,
    Conflict,
    Document,
    Page,
    PageRelationship,
)
from wingjournal.storage.schema import DDL, SCHEMA_VERSION

_DB_NAME = "wjm.sqlite"
_BLOB_DIR = "blobs"


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / _BLOB_DIR).mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.path / _DB_NAME)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(DDL)
        self.db.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ blobs
    def put_blob(self, data: bytes | str | Path) -> str:
        if isinstance(data, (str, Path)):
            data = Path(data).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        dest = self.path / _BLOB_DIR / digest[:2] / digest
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        return digest

    def blob_path(self, digest: str) -> Path:
        return self.path / _BLOB_DIR / digest[:2] / digest

    def get_blob(self, digest: str) -> bytes:
        return self.blob_path(digest).read_bytes()

    # --------------------------------------------------------------- documents
    def upsert_document(self, doc: Document) -> Document:
        self.db.execute(
            "INSERT INTO documents(uuid, name, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(uuid) DO UPDATE SET name = excluded.name",
            (doc.uuid, doc.name, doc.created_at),
        )
        self.db.commit()
        return doc

    # ------------------------------------------------------------------- pages
    def upsert_page(self, page: Page) -> Page:
        self.db.execute(
            """INSERT INTO pages(
                   uuid, created_at, document_id_explicit, document_id_resolved,
                   document_id_resolution_source, page_id_explicit, page_id_machine,
                   topic_tags, left_ref, above_ref, below_ref, right_ref)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(uuid) DO UPDATE SET
                   document_id_explicit          = excluded.document_id_explicit,
                   document_id_resolved          = excluded.document_id_resolved,
                   document_id_resolution_source = excluded.document_id_resolution_source,
                   page_id_explicit              = excluded.page_id_explicit,
                   page_id_machine               = excluded.page_id_machine,
                   topic_tags                    = excluded.topic_tags,
                   left_ref = excluded.left_ref, above_ref = excluded.above_ref,
                   below_ref = excluded.below_ref, right_ref = excluded.right_ref""",
            (
                page.uuid, page.created_at, page.document_id_explicit,
                page.document_id_resolved, page.document_id_resolution_source,
                page.page_id_explicit, page.page_id_machine,
                json.dumps(page.topic_tags),
                page.left, page.above, page.below, page.right,
            ),
        )
        self.db.commit()
        return page

    def get_page(self, uuid: str) -> Page | None:
        row = self.db.execute("SELECT * FROM pages WHERE uuid = ?", (uuid,)).fetchone()
        if row is None:
            return None
        return self._page_from_row(row)

    def find_page(
        self, page_id_explicit: str | None = None, page_id_machine: str | None = None
    ) -> Page | None:
        if page_id_machine:
            row = self.db.execute(
                "SELECT * FROM pages WHERE page_id_machine = ?", (page_id_machine,)
            ).fetchone()
            if row:
                return self._page_from_row(row)
        if page_id_explicit:
            row = self.db.execute(
                "SELECT * FROM pages WHERE page_id_explicit = ?", (page_id_explicit,)
            ).fetchone()
            if row:
                return self._page_from_row(row)
        return None

    def _page_from_row(self, row: sqlite3.Row) -> Page:
        return Page(
            uuid=row["uuid"],
            created_at=row["created_at"],
            document_id_explicit=row["document_id_explicit"],
            document_id_resolved=row["document_id_resolved"],
            document_id_resolution_source=row["document_id_resolution_source"],
            page_id_explicit=row["page_id_explicit"],
            page_id_machine=row["page_id_machine"],
            topic_tags=json.loads(row["topic_tags"]),
            left=row["left_ref"], above=row["above_ref"],
            below=row["below_ref"], right=row["right_ref"],
            capture_uuids=[
                r["uuid"] for r in self.db.execute(
                    "SELECT uuid FROM captures WHERE page_uuid = ? ORDER BY timestamp",
                    (row["uuid"],),
                )
            ],
        )

    # ---------------------------------------------------------------- captures
    def add_capture(self, capture: Capture) -> Capture:
        self.db.execute(
            """INSERT INTO captures(
                   uuid, page_uuid, timestamp, source_type, raw_blob, normalized_blob,
                   page_boundary_method, page_boundary_conf, orientation_degrees,
                   orientation_method, previous_capture_uuid, data)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                capture.uuid, capture.page_uuid, capture.timestamp, capture.source_type,
                capture.raw_blob, capture.normalized_blob,
                capture.page_boundary_method, capture.page_boundary_confidence,
                capture.orientation_degrees, capture.orientation_method,
                capture.previous_capture_uuid,
                json.dumps(dataclasses.asdict(capture)),
            ),
        )
        self.db.commit()
        return capture

    def get_capture(self, uuid: str) -> dict | None:
        row = self.db.execute("SELECT data FROM captures WHERE uuid = ?", (uuid,)).fetchone()
        return json.loads(row["data"]) if row else None

    def captures_for_page(self, page_uuid: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT data FROM captures WHERE page_uuid = ? ORDER BY timestamp", (page_uuid,)
        )
        return [json.loads(r["data"]) for r in rows]

    # ----------------------------------------------------------- relationships
    def add_relationship(self, rel: PageRelationship) -> PageRelationship:
        self.db.execute(
            """INSERT INTO page_relationships(
                   uuid, source_page, target_page, relation, explicitly_declared,
                   source_capture, confidence)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(source_page, target_page, relation) DO NOTHING""",
            (
                rel.uuid, rel.source_page, rel.target_page, rel.relation,
                int(rel.explicitly_declared), rel.source_capture, rel.confidence,
            ),
        )
        self.db.commit()
        return rel

    def relationships_for_page(self, page_uuid: str) -> list[PageRelationship]:
        rows = self.db.execute(
            "SELECT * FROM page_relationships WHERE source_page = ? OR target_page = ?",
            (page_uuid, page_uuid),
        )
        return [
            PageRelationship(
                uuid=r["uuid"], source_page=r["source_page"], target_page=r["target_page"],
                relation=r["relation"], explicitly_declared=bool(r["explicitly_declared"]),
                source_capture=r["source_capture"], confidence=r["confidence"],
            )
            for r in rows
        ]

    # --------------------------------------------------------------- conflicts
    def add_conflict(self, conflict: Conflict) -> Conflict:
        self.db.execute(
            "INSERT INTO conflicts(uuid, kind, detail, page_uuid, capture_uuid, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (
                conflict.uuid, conflict.kind, conflict.detail,
                conflict.page_uuid, conflict.capture_uuid, conflict.created_at,
            ),
        )
        self.db.commit()
        return conflict

    def conflicts(self, page_uuid: str | None = None) -> list[Conflict]:
        if page_uuid:
            rows = self.db.execute(
                "SELECT * FROM conflicts WHERE page_uuid = ? ORDER BY created_at", (page_uuid,)
            )
        else:
            rows = self.db.execute("SELECT * FROM conflicts ORDER BY created_at")
        return [
            Conflict(
                uuid=r["uuid"], kind=r["kind"], detail=r["detail"],
                page_uuid=r["page_uuid"], capture_uuid=r["capture_uuid"],
                created_at=r["created_at"],
            )
            for r in rows
        ]
