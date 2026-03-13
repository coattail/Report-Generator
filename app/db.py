from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from .config import DB_PATH, ensure_directories


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  base_url TEXT,
  feed_url TEXT,
  page_url TEXT,
  language TEXT,
  source_type TEXT,
  categories_json TEXT NOT NULL DEFAULT '[]',
  enabled INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_articles (
  id TEXT PRIMARY KEY,
  source_id TEXT,
  title TEXT NOT NULL,
  url TEXT NOT NULL UNIQUE,
  published_at TEXT,
  language TEXT,
  category TEXT,
  author TEXT,
  summary TEXT,
  content TEXT,
  content_hash TEXT NOT NULL,
  tags_json TEXT NOT NULL DEFAULT '[]',
  entities_json TEXT NOT NULL DEFAULT '[]',
  trust_level TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS topic_clusters (
  id TEXT PRIMARY KEY,
  week_key TEXT NOT NULL,
  title TEXT NOT NULL,
  slug TEXT NOT NULL,
  summary TEXT,
  category TEXT,
  score REAL NOT NULL DEFAULT 0,
  article_ids_json TEXT NOT NULL DEFAULT '[]',
  entities_json TEXT NOT NULL DEFAULT '[]',
  timeline_json TEXT NOT NULL DEFAULT '[]',
  evidence_pack_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS style_corpus_docs (
  id TEXT PRIMARY KEY,
  file_name TEXT NOT NULL,
  file_type TEXT NOT NULL,
  imported_at TEXT NOT NULL,
  title TEXT,
  content TEXT NOT NULL,
  sections_json TEXT NOT NULL DEFAULT '[]',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  structure_tags_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS style_profiles (
  id TEXT PRIMARY KEY,
  version TEXT NOT NULL,
  voice_notes_json TEXT NOT NULL DEFAULT '[]',
  worldview_notes_json TEXT NOT NULL DEFAULT '[]',
  banned_phrases_json TEXT NOT NULL DEFAULT '[]',
  preferred_patterns_json TEXT NOT NULL DEFAULT '[]',
  stats_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS weekly_issues (
  id TEXT PRIMARY KEY,
  week_key TEXT NOT NULL,
  title TEXT NOT NULL,
  selected_topic_ids_json TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL,
  structure_json TEXT NOT NULL DEFAULT '{}',
  market_scope_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS issue_sections (
  id TEXT PRIMARY KEY,
  issue_id TEXT NOT NULL,
  section_key TEXT NOT NULL,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  citations_json TEXT NOT NULL DEFAULT '[]',
  candidate_rankings_json TEXT NOT NULL DEFAULT '[]',
  critic_notes_json TEXT NOT NULL DEFAULT '[]',
  final_score REAL NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS draft_versions (
  id TEXT PRIMARY KEY,
  issue_id TEXT NOT NULL,
  version_label TEXT NOT NULL,
  markdown_content TEXT NOT NULL,
  html_content TEXT NOT NULL,
  source_model_artifact_id TEXT,
  metrics_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edit_diffs (
  id TEXT PRIMARY KEY,
  issue_id TEXT NOT NULL,
  section_id TEXT,
  original_text TEXT NOT NULL,
  edited_text TEXT NOT NULL,
  diff_json TEXT NOT NULL DEFAULT '{}',
  labels_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS preference_pairs (
  id TEXT PRIMARY KEY,
  issue_id TEXT NOT NULL,
  section_id TEXT,
  chosen_text TEXT NOT NULL,
  rejected_text TEXT NOT NULL,
  reason TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS section_scores (
  id TEXT PRIMARY KEY,
  issue_id TEXT NOT NULL,
  section_id TEXT,
  factuality INTEGER NOT NULL,
  style INTEGER NOT NULL,
  structure INTEGER NOT NULL,
  publishability INTEGER NOT NULL,
  notes TEXT,
  flags_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS publish_decisions (
  id TEXT PRIMARY KEY,
  issue_id TEXT NOT NULL,
  verdict TEXT NOT NULL,
  notes TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_snapshots (
  id TEXT PRIMARY KEY,
  week_key TEXT NOT NULL,
  symbol TEXT NOT NULL,
  display_name TEXT NOT NULL,
  asset_type TEXT NOT NULL,
  window_start TEXT,
  window_end TEXT,
  latest_close REAL,
  weekly_return REAL,
  ytd_return REAL,
  notes TEXT,
  related_article_urls_json TEXT NOT NULL DEFAULT '[]',
  raw_series_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_artifacts (
  id TEXT PRIMARY KEY,
  role TEXT NOT NULL,
  base_model TEXT NOT NULL,
  adapter_path TEXT,
  status TEXT NOT NULL,
  is_production INTEGER NOT NULL DEFAULT 0,
  parent_artifact_id TEXT,
  metrics_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
  id TEXT PRIMARY KEY,
  model_artifact_id TEXT,
  scope TEXT NOT NULL,
  status TEXT NOT NULL,
  summary_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
"""


def init_db() -> None:
    ensure_directories()
    with sqlite3.connect(DB_PATH) as connection:
        connection.executescript(SCHEMA_SQL)
        connection.commit()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    ensure_directories()
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()
