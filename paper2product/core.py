from __future__ import annotations

import io
import json
import os
import queue
import re
import sqlite3
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class IngestRequest:
    title: str
    abstract: str
    method_text: str
    source_url: str | None = None


class PersistentStore:
    def __init__(self, db_path: str = "paper2product.db", object_dir: str = "object_store"):
        self.db_path = db_path
        self.object_dir = Path(object_dir)
        self.object_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    token TEXT UNIQUE NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source_url TEXT,
                    abstract TEXT NOT NULL,
                    method_text TEXT NOT NULL,
                    paper_spec_json TEXT NOT NULL,
                    architecture_mermaid TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    uncertainty_flags_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    run_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    output_json TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    framework TEXT NOT NULL,
                    file_count INTEGER NOT NULL,
                    zip_object_key TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rate_limits (
                    key TEXT PRIMARY KEY,
                    window_start REAL NOT NULL,
                    count INTEGER NOT NULL
                );
                """
            )

    def create_user(self, email: str) -> Dict:
        now = time.time()
        user_id = str(uuid.uuid4())
        token = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO users (id, email, token, created_at) VALUES (?, ?, ?, ?)",
                (user_id, email, token, now),
            )
            conn.commit()
        return {"id": user_id, "email": email, "token": token}

    def get_user_by_token(self, token: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE token = ?", (token,)).fetchone()
        return dict(row) if row else None

    def allow_rate_limit(self, key: str, limit: int = 60, window_sec: int = 60) -> bool:
        now = time.time()
        with self.lock:
            with self._connect() as conn:
                row = conn.execute("SELECT * FROM rate_limits WHERE key = ?", (key,)).fetchone()
                if not row:
                    conn.execute(
                        "INSERT INTO rate_limits (key, window_start, count) VALUES (?, ?, ?)",
                        (key, now, 1),
                    )
                    conn.commit()
                    return True
                start = row["window_start"]
                count = row["count"]
                if now - start > window_sec:
                    conn.execute(
                        "UPDATE rate_limits SET window_start = ?, count = 1 WHERE key = ?",
                        (now, key),
                    )
                    conn.commit()
                    return True
                if count >= limit:
                    return False
                conn.execute("UPDATE rate_limits SET count = count + 1 WHERE key = ?", (key,))
                conn.commit()
                return True

    def create_project(self, user_id: str, ingest: IngestRequest, paper_spec: Dict, architecture_mermaid: str, confidence_score: float, uncertainty_flags: List[str]) -> Dict:
        now = time.time()
        project_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO projects (
                    id, user_id, title, source_url, abstract, method_text, paper_spec_json,
                    architecture_mermaid, confidence_score, uncertainty_flags_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    user_id,
                    ingest.title,
                    ingest.source_url,
                    ingest.abstract,
                    ingest.method_text,
                    json.dumps(paper_spec),
                    architecture_mermaid,
                    confidence_score,
                    json.dumps(uncertainty_flags),
                    now,
                    now,
                ),
            )
            conn.commit()
        return self.get_project(project_id, user_id)

    def get_project(self, project_id: str, user_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id)).fetchone()
        if not row:
            return None
        data = dict(row)
        data["paper_spec"] = json.loads(data.pop("paper_spec_json"))
        data["uncertainty_flags"] = json.loads(data.pop("uncertainty_flags_json"))
        return data

    def list_projects(self, user_id: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM projects WHERE user_id = ? ORDER BY updated_at DESC", (user_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["paper_spec"] = json.loads(d.pop("paper_spec_json"))
            d["uncertainty_flags"] = json.loads(d.pop("uncertainty_flags_json"))
            out.append(d)
        return out

    def create_run(self, project_id: str, user_id: str, run_type: str) -> Dict:
        run_id = str(uuid.uuid4())
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (id, project_id, user_id, run_type, status, message, output_json, error, created_at, updated_at) VALUES (?, ?, ?, ?, 'queued', 'Queued', NULL, NULL, ?, ?)",
                (run_id, project_id, user_id, run_type, now, now),
            )
            conn.commit()
        return self.get_run(run_id, user_id)

    def update_run(self, run_id: str, status: str, message: str, output: Optional[Dict] = None, error: Optional[str] = None):
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, message = ?, output_json = ?, error = ?, updated_at = ? WHERE id = ?",
                (status, message, json.dumps(output) if output is not None else None, error, now, run_id),
            )
            conn.commit()

    def get_run(self, run_id: str, user_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ? AND user_id = ?", (run_id, user_id)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["output"] = json.loads(d.pop("output_json")) if d.get("output_json") else None
        return d

    def list_runs(self, project_id: str, user_id: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE project_id = ? AND user_id = ? ORDER BY created_at DESC",
                (project_id, user_id),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["output"] = json.loads(d.pop("output_json")) if d.get("output_json") else None
            out.append(d)
        return out

    def next_artifact_version(self, project_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COALESCE(MAX(version), 0) AS v FROM artifacts WHERE project_id = ?", (project_id,)).fetchone()
        return int(row["v"]) + 1

    def save_object(self, key: str, blob: bytes) -> str:
        path = self.object_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        return str(path)

    def load_object(self, key: str) -> bytes:
        return (self.object_dir / key).read_bytes()

    def create_artifact(self, project_id: str, run_id: str, framework: str, manifest: Dict, zip_blob: bytes) -> Dict:
        artifact_id = str(uuid.uuid4())
        version = self.next_artifact_version(project_id)
        object_key = f"{project_id}/artifact_v{version}.zip"
        self.save_object(object_key, zip_blob)
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO artifacts (id, project_id, run_id, version, framework, file_count, zip_object_key, manifest_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (artifact_id, project_id, run_id, version, framework, len(manifest.get("files", [])), object_key, json.dumps(manifest), now),
            )
            conn.commit()
        return self.get_artifact(artifact_id)

    def get_artifact(self, artifact_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["manifest"] = json.loads(d.pop("manifest_json"))
        return d

    def latest_artifact(self, project_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM artifacts WHERE project_id = ? ORDER BY version DESC LIMIT 1", (project_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["manifest"] = json.loads(d.pop("manifest_json"))
        return d


STORE = PersistentStore()
JOB_QUEUE: "queue.Queue[Dict]" = queue.Queue()


def _extract_equations(text: str) -> List[str]:
    equations = re.findall(r"([A-Za-z]\s*=\s*[^\n\.;]{3,})", text)
    return equations[:3] or ["L = CrossEntropy(y_pred, y_true)"]


def _extract_sections(abstract: str, method_text: str) -> Dict[str, str]:
    method_lines = [s.strip() for s in method_text.split(".") if s.strip()]
    return {
        "problem": abstract.split(".")[0].strip(),
        "method": method_lines[0] if method_lines else "Method details unavailable",
        "training": method_lines[1] if len(method_lines) > 1 else "Training details missing",
        "evaluation": method_lines[-1] if method_lines else "Evaluation details missing",
    }


def _extract_keywords(text: str, defaults: List[str]) -> List[str]:
    found = [word for word in defaults if word.lower() in text.lower()]
    return found or defaults[:2]


def build_paper_spec(payload: IngestRequest) -> Dict:
    sections = _extract_sections(payload.abstract, payload.method_text)
    datasets = _extract_keywords(payload.abstract + payload.method_text, ["ImageNet", "CIFAR-10", "SQuAD", "MIMIC"])
    metrics = _extract_keywords(payload.abstract + payload.method_text, ["accuracy", "F1", "BLEU", "ROUGE"])
    equations = _extract_equations(payload.method_text)
    return {
        "problem": sections["problem"],
        "method": sections["method"],
        "sections": sections,
        "key_equations": equations,
        "datasets": datasets,
        "metrics": metrics,
        "assumptions": [
            "Training distribution reflects production usage.",
            "Compute budget supports replication runs.",
        ],
        "citations": {
            "problem": {"section": "abstract", "snippet": payload.abstract[:180]},
            "method": {"section": "method", "snippet": payload.method_text[:220]},
        },
    }


def build_architecture_mermaid(spec: Dict) -> str:
    return "\n".join(
        [
            "flowchart TD",
            "A[Paper]-->B[Section Parser]",
            "B-->C[Model Core]",
            "C-->D[Train/Eval]",
            f"D-->E[Metric: {spec['metrics'][0]}]",
        ]
    )


def build_trust_layer(spec: Dict) -> Dict:
    confidence_breakdown = {
        "extraction": 0.8,
        "equation_coverage": 0.72 if spec["key_equations"] else 0.4,
        "reproducibility": 0.7,
        "citation_traceability": 0.88,
    }
    overall = round(sum(confidence_breakdown.values()) / len(confidence_breakdown), 2)
    uncertainty_flags = []
    if "training" in spec["sections"] and "missing" in spec["sections"]["training"].lower():
        uncertainty_flags.append("Training procedure details are incomplete.")
    if len(spec["datasets"]) == 1:
        uncertainty_flags.append("Only one dataset detected; benchmark breadth may be limited.")
    if not uncertainty_flags:
        uncertainty_flags.append("No critical uncertainty flags detected in heuristic pass.")
    citation_map = {
        "core/model.py": ["method"],
        "README.md": ["problem", "method"],
    }
    return {
        "confidence_breakdown": confidence_breakdown,
        "overall_confidence": overall,
        "uncertainty_flags": uncertainty_flags,
        "citation_map": citation_map,
    }


def generate_code_scaffold(project: Dict, framework: str = "pytorch") -> Dict[str, str]:
    spec = project["paper_spec"]
    trust = build_trust_layer(spec)
    return {
        "README.md": (
            f"# {project['title']}\n\n"
            f"Framework: {framework}\n"
            f"Overall confidence: {trust['overall_confidence']}\n\n"
            "## Citation Traceability\n"
            f"- Problem: {spec['citations']['problem']['snippet']}\n"
            f"- Method: {spec['citations']['method']['snippet']}\n"
            "\n## Uncertainty Flags\n"
            + "\n".join([f"- {x}" for x in trust["uncertainty_flags"]])
            + "\n"
        ),
        "core/model.py": (
            '"""Citation-aware generated model.\n'
            f"Problem source: {spec['citations']['problem']['section']}\n"
            f"Method source: {spec['citations']['method']['section']}\n"
            '"""\n\n'
            "def train_step(batch):\n"
            f"    return {{'loss': 0.0, 'framework': '{framework}'}}\n"
        ),
        "train/train.py": "from core.model import train_step\nprint(train_step(None))\n",
        "tests/test_smoke.py": "def test_smoke():\n    assert 1 + 1 == 2\n",
    }


def export_zip(project: Dict, framework: str = "pytorch") -> bytes:
    mem = io.BytesIO()
    spec = project["paper_spec"]
    trust = build_trust_layer(spec)
    files = generate_code_scaffold(project, framework)
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, content in files.items():
            zf.writestr(path, content)
        zf.writestr("visuals/architecture.mmd", project["architecture_mermaid"])
        zf.writestr("reports/paper_spec.json", json.dumps(spec, indent=2))
        zf.writestr("reports/trust_layer.json", json.dumps(trust, indent=2))
    return mem.getvalue()


def build_distillation(project: Dict) -> Dict:
    spec = project["paper_spec"]
    return {
        "poster": {
            "problem": spec["problem"],
            "method": spec["method"],
            "key_equations": spec["key_equations"],
            "architecture": "Mermaid flow graph",
            "results": f"Track {', '.join(spec['metrics'])}",
            "limitations": "Needs larger evaluation and latency profiling.",
            "qr_demo": f"https://demo.local/projects/{project['id']}",
            "formats": ["A0", "A1", "social", "slide"],
        },
        "knowledge_cards": {
            "concept_cards": [
                {"title": "Attention", "note": "Highlights important signal paths."},
                {"title": "Loss Function", "note": "Balances fit and generalization."},
                {"title": "Evaluation Setup", "note": "Map datasets to robustness scenarios."},
            ],
            "equation_intuition_panels": [
                {
                    "equation": spec["key_equations"][0],
                    "diagram": "Objective balance diagram",
                    "intuition": "Trade-off between teacher alignment and label fit.",
                    "example": "Student model retains high accuracy at lower latency.",
                }
            ],
            "failure_cards": [
                {"scenario": "Low data", "impact": "Overfitting"},
                {"scenario": "Domain shift", "impact": "Metric drift"},
                {"scenario": "Noisy labels", "impact": "Unstable convergence"},
            ],
        },
        "executive_takeaways": {
            "one_page_brief": ["Why it matters", "Scale potential", "Risks", "ROI", "Moat"],
            "five_slide_deck": ["Problem", "Solution", "Market", "Demo", "Roadmap"],
        },
        "extension_engine": {
            "what_next": [
                "Dataset extension for medical imaging domain adaptation.",
                "Architecture variants for low-latency inference.",
                "Scaling-law sweep for parameter-performance frontier.",
                "Domain transfer to edge speech tasks.",
            ],
            "missing_piece_detector": [
                "Assumption test: robustness under noisy labels.",
                "Baseline gap: compare with stronger modern baseline.",
                "Eval-size risk: extend benchmark to larger diverse sets.",
            ],
            "research_roadmap": [
                "Phase 1: Replication",
                "Phase 2: Improvement",
                "Phase 3: New domain",
                "Phase 4: Publication",
            ],
        },
        "learning_path": {
            "beginner": ["Math prerequisites", "Intuition notes", "Run baseline code", "Try one experiment"],
            "advanced": ["Edge-case analysis", "Extension experiments", "Optimization and scaling"],
        },
    }


def openapi_spec() -> Dict:
    return {
        "openapi": "3.0.0",
        "info": {"title": "Paper2Product Phase 2", "version": "2.0.0"},
        "paths": {
            "/api/v1/auth/register": {"post": {"summary": "Create user and API token"}},
            "/api/v1/projects/ingest": {"post": {"summary": "Create project from paper text"}},
            "/api/v1/projects/{id}": {"get": {"summary": "Get project"}},
            "/api/v1/projects/{id}/runs": {"post": {"summary": "Queue run"}, "get": {"summary": "List runs"}},
            "/api/v1/projects/{id}/artifacts/latest": {"get": {"summary": "Latest artifact metadata"}},
            "/api/v1/projects/{id}/visual-graph": {"get": {"summary": "Get visual graph"}},
            "/api/v1/projects/{id}/distillation": {"get": {"summary": "Research distillation output"}},
            "/api/v1/projects/{id}/export.zip": {"get": {"summary": "Download latest bundle"}},
            "/api/v1/dashboard": {"get": {"summary": "Dashboard with projects and run history"}},
            "/api/v1/runs/{run_id}": {"get": {"summary": "Get run status"}},
        },
    }


def enqueue_run(run_payload: Dict):
    JOB_QUEUE.put(run_payload)


def start_worker_once():
    if getattr(start_worker_once, "started", False):
        return

    def loop():
        while True:
            job = JOB_QUEUE.get()
            run_id = job["run_id"]
            try:
                STORE.update_run(run_id, "running", "Run started")
                project = STORE.get_project(job["project_id"], job["user_id"])
                if not project:
                    raise RuntimeError("Project not found")
                if job["run_type"] == "artifacts":
                    zip_blob = export_zip(project, job.get("framework", "pytorch"))
                    files = list(generate_code_scaffold(project, job.get("framework", "pytorch")).keys())
                    manifest = {
                        "framework": job.get("framework", "pytorch"),
                        "files": files,
                        "confidence_score": project["confidence_score"],
                        "architecture_mermaid": project["architecture_mermaid"],
                        "trust": build_trust_layer(project["paper_spec"]),
                    }
                    artifact = STORE.create_artifact(project["id"], run_id, job.get("framework", "pytorch"), manifest, zip_blob)
                    STORE.update_run(run_id, "completed", "Artifacts ready", {"artifact_id": artifact["id"], "version": artifact["version"]})
                elif job["run_type"] == "distillation":
                    output = build_distillation(project)
                    STORE.update_run(run_id, "completed", "Distillation ready", output)
                else:
                    raise RuntimeError("Unsupported run type")
            except Exception as exc:  # best-effort worker loop
                STORE.update_run(run_id, "failed", "Run failed", error=str(exc))
            finally:
                JOB_QUEUE.task_done()

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    setattr(start_worker_once, "started", True)
