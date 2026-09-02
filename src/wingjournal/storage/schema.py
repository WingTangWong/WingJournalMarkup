"""SQLite schema for the WJM store (spec §41-46).

Rich / nested capture fields (hypotheses, fiducials, notes) live in the
``captures.data`` JSON column; the queryable bits are promoted to real columns.
"""

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    uuid       TEXT PRIMARY KEY,
    name       TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    uuid                          TEXT PRIMARY KEY,
    created_at                    TEXT NOT NULL,
    document_id_explicit          TEXT,
    document_id_resolved          TEXT,
    document_id_resolution_source TEXT,
    page_id_explicit              TEXT,
    page_id_machine               TEXT,
    topic_tags                    TEXT NOT NULL DEFAULT '[]',
    left_ref                      TEXT,
    above_ref                     TEXT,
    below_ref                     TEXT,
    right_ref                     TEXT
);
CREATE INDEX IF NOT EXISTS idx_pages_page_id_explicit ON pages(page_id_explicit);
CREATE INDEX IF NOT EXISTS idx_pages_page_id_machine  ON pages(page_id_machine);

CREATE TABLE IF NOT EXISTS captures (
    uuid                  TEXT PRIMARY KEY,
    page_uuid             TEXT REFERENCES pages(uuid),
    timestamp             TEXT NOT NULL,
    source_type           TEXT,
    raw_blob              TEXT,
    normalized_blob       TEXT,
    page_boundary_method  TEXT,
    page_boundary_conf    REAL,
    orientation_degrees   INTEGER,
    orientation_method    TEXT,
    previous_capture_uuid TEXT,
    data                  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_captures_page ON captures(page_uuid);

CREATE TABLE IF NOT EXISTS page_relationships (
    uuid                TEXT PRIMARY KEY,
    source_page         TEXT NOT NULL REFERENCES pages(uuid),
    target_page         TEXT NOT NULL REFERENCES pages(uuid),
    relation            TEXT NOT NULL,
    explicitly_declared INTEGER NOT NULL DEFAULT 1,
    source_capture      TEXT,
    confidence          REAL NOT NULL DEFAULT 1.0,
    UNIQUE(source_page, target_page, relation)
);

CREATE TABLE IF NOT EXISTS conflicts (
    uuid         TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    detail       TEXT NOT NULL,
    page_uuid    TEXT,
    capture_uuid TEXT,
    created_at   TEXT NOT NULL
);
"""
