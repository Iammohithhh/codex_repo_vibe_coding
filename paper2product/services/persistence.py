"""SQLite persistence layer for Paper2Product.

Stores projects, workspaces, and experiment runs with JSON serialization
for complex nested structures.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict
from typing import Dict, List, Optional

from ..models.schema import (
    ExperimentRun,
    IngestRequest,
    Project,
    ProjectStatus,
    ReviewComment,
    Workspace,
)

_DB_PATH = "paper2product.db"
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
    return _local.conn


def init_db(db_path: str = "paper2product.db"):
    global _DB_PATH
    _DB_PATH = db_path
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'draft',
            workspace_id TEXT,
            created_at REAL,
            updated_at REAL
        );
        CREATE TABLE IF NOT EXISTS workspaces (
            id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            created_at REAL
        );
        CREATE TABLE IF NOT EXISTS experiment_runs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            data TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at REAL,
            FOREIGN KEY (project_id) REFERENCES projects(id)
        );
        CREATE TABLE IF NOT EXISTS reviews (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            data TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at REAL,
            FOREIGN KEY (project_id) REFERENCES projects(id)
        );
        CREATE INDEX IF NOT EXISTS idx_projects_workspace ON projects(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_experiments_project ON experiment_runs(project_id);
        CREATE INDEX IF NOT EXISTS idx_reviews_project ON reviews(project_id);
    """)
    conn.commit()


def _serialize(obj) -> str:
    if hasattr(obj, "__dataclass_fields__"):
        return json.dumps(asdict(obj), default=str)
    return json.dumps(obj, default=str)


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------
def save_project(project: Project):
    conn = _get_conn()
    project.updated_at = time.time()
    conn.execute(
        "INSERT OR REPLACE INTO projects (id, data, status, workspace_id, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        (project.id, _serialize(project), project.status.value if hasattr(project.status, 'value') else project.status,
         project.workspace_id, project.created_at, project.updated_at),
    )
    conn.commit()


def get_project(project_id: str) -> Optional[Dict]:
    conn = _get_conn()
    row = conn.execute("SELECT data FROM projects WHERE id=?", (project_id,)).fetchone()
    if row:
        return json.loads(row["data"])
    return None


def list_projects(workspace_id: Optional[str] = None) -> List[Dict]:
    conn = _get_conn()
    if workspace_id:
        rows = conn.execute("SELECT data FROM projects WHERE workspace_id=? ORDER BY updated_at DESC", (workspace_id,)).fetchall()
    else:
        rows = conn.execute("SELECT data FROM projects ORDER BY updated_at DESC").fetchall()
    return [json.loads(r["data"]) for r in rows]


def delete_project(project_id: str):
    conn = _get_conn()
    conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
    conn.execute("DELETE FROM experiment_runs WHERE project_id=?", (project_id,))
    conn.execute("DELETE FROM reviews WHERE project_id=?", (project_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# Workspaces
# ---------------------------------------------------------------------------
def save_workspace(workspace: Workspace):
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO workspaces (id, data, created_at) VALUES (?,?,?)",
        (workspace.id, _serialize(workspace), workspace.created_at),
    )
    conn.commit()


def get_workspace(workspace_id: str) -> Optional[Dict]:
    conn = _get_conn()
    row = conn.execute("SELECT data FROM workspaces WHERE id=?", (workspace_id,)).fetchone()
    if row:
        return json.loads(row["data"])
    return None


def list_workspaces() -> List[Dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT data FROM workspaces ORDER BY created_at DESC").fetchall()
    return [json.loads(r["data"]) for r in rows]


# ---------------------------------------------------------------------------
# Experiment Runs
# ---------------------------------------------------------------------------
def save_experiment_run(project_id: str, run: ExperimentRun):
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO experiment_runs (id, project_id, data, status, created_at) VALUES (?,?,?,?,?)",
        (run.id, project_id, _serialize(run), run.status, run.created_at),
    )
    conn.commit()


def list_experiment_runs(project_id: str) -> List[Dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT data FROM experiment_runs WHERE project_id=? ORDER BY created_at DESC", (project_id,)).fetchall()
    return [json.loads(r["data"]) for r in rows]


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------
def save_review(project_id: str, review: ReviewComment):
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO reviews (id, project_id, data, status, created_at) VALUES (?,?,?,?,?)",
        (review.id, project_id, _serialize(review), review.status.value if hasattr(review.status, 'value') else review.status, review.created_at),
    )
    conn.commit()


def list_reviews(project_id: str) -> List[Dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT data FROM reviews WHERE project_id=? ORDER BY created_at DESC", (project_id,)).fetchall()
    return [json.loads(r["data"]) for r in rows]
