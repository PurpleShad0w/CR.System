"""SQLite content catalog for OneNote exporter.

Provides optional local database for incremental export and indexing.
"""
from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS notebooks(
  id TEXT PRIMARY KEY,
  name TEXT,
  slug TEXT,
  created TEXT,
  modified TEXT
);
CREATE TABLE IF NOT EXISTS sections(
  id TEXT PRIMARY KEY,
  notebook_id TEXT,
  name TEXT,
  slug TEXT,
  created TEXT,
  modified TEXT,
  FOREIGN KEY(notebook_id) REFERENCES notebooks(id)
);
CREATE TABLE IF NOT EXISTS pages(
  id TEXT PRIMARY KEY,
  section_id TEXT,
  notebook_id TEXT,
  title TEXT,
  slug TEXT,
  created TEXT,
  modified TEXT,
  md_path TEXT,
  merged_path TEXT,
  jsonl_path TEXT,
  content_hash TEXT,
  word_count INTEGER,
  page_order INTEGER,
  FOREIGN KEY(section_id) REFERENCES sections(id),
  FOREIGN KEY(notebook_id) REFERENCES notebooks(id)
);
CREATE TABLE IF NOT EXISTS assets(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  page_id TEXT,
  rel_path TEXT,
  mime_type TEXT,
  size_bytes INTEGER,
  sha256 TEXT,
  FOREIGN KEY(page_id) REFERENCES pages(id)
);
"""


@dataclass
class PageState:
    id: str
    modified: str | None
    content_hash: str | None


class DBManager:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA foreign_keys=ON;")
        # Execute schema
        for stmt in SCHEMA.strip().split(";\n"):
            if stmt.strip():
                self.conn.execute(stmt)
        self.conn.commit()

    # --- Upserts ------------------------------------------------------------
    def upsert_notebook(self, nb: dict[str, Any]):
        self.conn.execute(
            """
            INSERT INTO notebooks(id, name, slug, created, modified)
            VALUES(:id, :name, :slug, :created, :modified)
            ON CONFLICT(id) DO UPDATE SET
              name=excluded.name, slug=excluded.slug,
              created=excluded.created, modified=excluded.modified
            """,
            nb,
        )

    def upsert_section(self, sec: dict[str, Any]):
        self.conn.execute(
            """
                        INSERT INTO sections(
                            id, notebook_id, name, slug, created, modified
                        )
            VALUES(:id, :notebook_id, :name, :slug, :created, :modified)
            ON CONFLICT(id) DO UPDATE SET
              notebook_id=excluded.notebook_id, name=excluded.name,
              slug=excluded.slug, created=excluded.created,
              modified=excluded.modified
            """,
            sec,
        )

    def upsert_page(self, page: dict[str, Any]):
        self.conn.execute(
            """
                                                INSERT INTO pages(
                                                    id,
                                                    section_id,
                                                    notebook_id,
                                                    title,
                                                    slug,
                                                    created,
                                                    modified,
                                                    md_path,
                                                    merged_path,
                                                    jsonl_path,
                                                    content_hash,
                                                    word_count,
                                                    page_order
                                                ) VALUES (
                                                    :id,
                                                    :section_id,
                                                    :notebook_id,
                                                    :title,
                                                    :slug,
                                                    :created,
                                                    :modified,
                                                    :md_path,
                                                    :merged_path,
                                                    :jsonl_path,
                                                    :content_hash,
                                                    :word_count,
                                                    :page_order
                                                )
            ON CONFLICT(id) DO UPDATE SET
              section_id=excluded.section_id, notebook_id=excluded.notebook_id,
              title=excluded.title, slug=excluded.slug,
              created=excluded.created, modified=excluded.modified,
              md_path=excluded.md_path, merged_path=excluded.merged_path,
              jsonl_path=excluded.jsonl_path,
              content_hash=excluded.content_hash,
              word_count=excluded.word_count,
              page_order=excluded.page_order
            """,
            page,
        )

    def upsert_assets(self, page_id: str, assets: Iterable[dict[str, Any]]):
        # Replace assets for a page
        self.conn.execute("DELETE FROM assets WHERE page_id = ?", (page_id,))
        to_insert = [
            {**a, "page_id": page_id}
            for a in assets
            if a.get("rel_path")
        ]
        if to_insert:
            self.conn.executemany(
                """
                                                                INSERT INTO assets (
                                                                    page_id,
                                                                    rel_path,
                                                                    mime_type,
                                                                    size_bytes,
                                                                    sha256
                                                                )
                VALUES(:page_id, :rel_path, :mime_type, :size_bytes, :sha256)
                """,
                to_insert,
            )

    # --- Queries ------------------------------------------------------------
    def get_page_state(self, page_id: str) -> PageState | None:
        cur = self.conn.execute(
            "SELECT id, modified, content_hash FROM pages WHERE id = ?",
            (page_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return PageState(id=row[0], modified=row[1], content_hash=row[2])

    def commit(self):
        self.conn.commit()

    def close(self):  # pragma: no cover - trivial
        try:
            self.conn.close()
        except sqlite3.Error:
            pass

    # --- Helpers ------------------------------------------------------------
    @staticmethod
    def hash_content(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
