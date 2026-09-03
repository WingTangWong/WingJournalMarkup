"""Demo storage: the wingjournal Store plus a small SQLite for demo-only state
(the uploaded-scan registry and capture diffs) and a scans/ folder.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from wingjournal.storage import Store

_DEMO_DDL = """
CREATE TABLE IF NOT EXISTS scans (
    uuid          TEXT PRIMARY KEY,
    original_name TEXT,
    stored_path   TEXT NOT NULL,
    uploaded_at   TEXT NOT NULL,
    capture_uuid  TEXT,
    page_uuid     TEXT
);
CREATE TABLE IF NOT EXISTS diffs (
    capture_uuid      TEXT PRIMARY KEY,
    prev_capture_uuid TEXT,
    page_uuid         TEXT,
    created_at        TEXT NOT NULL,
    summary           TEXT NOT NULL
);
"""


@dataclass
class ScanRecord:
    uuid: str
    original_name: str
    stored_path: str
    uploaded_at: str
    capture_uuid: str | None
    page_uuid: str | None


class DemoStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.scans_dir = self.root / "scans"
        self.scans_dir.mkdir(parents=True, exist_ok=True)
        # the Flask dev server is multi-threaded; one process, so share the
        # connections across threads and serialise writes with a lock
        self.wjm = Store(self.root / "store", check_same_thread=False)
        self.db = sqlite3.connect(self.root / "demo.sqlite", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_DEMO_DDL)
        self.db.commit()
        self.write_lock = threading.Lock()

    def close(self) -> None:
        self.wjm.close()
        self.db.close()

    # -- scan files --------------------------------------------------------
    def archive_scan(self, scan_uuid: str, original_name: str, data: bytes) -> Path:
        ext = Path(original_name).suffix.lower() or ".png"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = self.scans_dir / f"{scan_uuid}_{stamp}{ext}"
        dest.write_bytes(data)
        self.db.execute(
            "INSERT INTO scans(uuid, original_name, stored_path, uploaded_at) "
            "VALUES (?,?,?,?)",
            (scan_uuid, original_name, str(dest), datetime.now(timezone.utc).isoformat()),
        )
        self.db.commit()
        return dest

    def link_scan(self, scan_uuid: str, capture_uuid: str, page_uuid: str) -> None:
        self.db.execute(
            "UPDATE scans SET capture_uuid = ?, page_uuid = ? WHERE uuid = ?",
            (capture_uuid, page_uuid, scan_uuid),
        )
        self.db.commit()

    def scans(self) -> list[ScanRecord]:
        rows = self.db.execute("SELECT * FROM scans ORDER BY uploaded_at DESC")
        return [ScanRecord(**dict(r)) for r in rows]

    def scan_for_capture(self, capture_uuid: str) -> ScanRecord | None:
        r = self.db.execute(
            "SELECT * FROM scans WHERE capture_uuid = ?", (capture_uuid,)
        ).fetchone()
        return ScanRecord(**dict(r)) if r else None

    # -- diffs ------------------------------------------------------------
    def save_diff(self, capture_uuid: str, prev: str | None, page_uuid: str, summary: dict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO diffs VALUES (?,?,?,?,?)",
            (capture_uuid, prev, page_uuid,
             datetime.now(timezone.utc).isoformat(), json.dumps(summary)),
        )
        self.db.commit()

    def get_diff(self, capture_uuid: str) -> dict | None:
        r = self.db.execute(
            "SELECT summary FROM diffs WHERE capture_uuid = ?", (capture_uuid,)
        ).fetchone()
        return json.loads(r["summary"]) if r else None

    # -- convenience passthroughs --------------------------------------
    def pages(self) -> list:
        rows = self.wjm.db.execute("SELECT uuid FROM pages ORDER BY created_at")
        return [self.wjm.get_page(r["uuid"]) for r in rows]

    def all_captures(self) -> list[dict]:
        rows = self.wjm.db.execute(
            "SELECT data FROM captures ORDER BY timestamp DESC"
        )
        return [json.loads(r["data"]) for r in rows]
