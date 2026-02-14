from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .core import (
    STORE,
    IngestRequest,
    build_architecture_mermaid,
    build_distillation,
    build_paper_spec,
    build_trust_layer,
    enqueue_run,
    openapi_spec,
    start_worker_once,
)

INDEX_HTML = """<!doctype html>
<html><head><meta charset='utf-8'><title>Paper2Product Phase 2</title></head>
<body style='font-family: Arial; max-width: 1000px; margin: 24px auto;'>
<h1>Paper2Product — Phase 2 Backend</h1>
<p>Register, ingest a paper, queue artifact runs, and monitor history.</p>
<ol>
<li>POST /api/v1/auth/register</li>
<li>Use returned token in X-API-Key header</li>
<li>POST /api/v1/projects/ingest</li>
<li>POST /api/v1/projects/{id}/runs with run_type=artifacts|distillation</li>
</ol>
<pre>{"status": "running", "note": "Use API client for authenticated endpoints."}</pre>
</body></html>"""


def _json(handler: BaseHTTPRequestHandler):
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth(self):
        token = self.headers.get("X-API-Key", "").strip()
        if not token:
            return None, self._send_json({"detail": "Missing X-API-Key"}, status=401)
        user = STORE.get_user_by_token(token)
        if not user:
            return None, self._send_json({"detail": "Invalid API key"}, status=401)
        if not STORE.allow_rate_limit(token, limit=120, window_sec=60):
            return None, self._send_json({"detail": "Rate limit exceeded"}, status=429)
        return user, None

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self._send_html(INDEX_HTML)
        if path == "/openapi.json":
            return self._send_json(openapi_spec())

        user, sent = self._auth()
        if sent is not None:
            return

        parts = path.strip("/").split("/")
        if path == "/api/v1/dashboard":
            projects = STORE.list_projects(user["id"])
            payload = []
            for p in projects:
                runs = STORE.list_runs(p["id"], user["id"])
                latest = STORE.latest_artifact(p["id"])
                payload.append(
                    {
                        "project_id": p["id"],
                        "title": p["title"],
                        "confidence_score": p["confidence_score"],
                        "uncertainty_flags": p["uncertainty_flags"],
                        "run_count": len(runs),
                        "latest_artifact_version": latest["version"] if latest else None,
                    }
                )
            return self._send_json({"projects": payload})

        if len(parts) == 4 and parts[:3] == ["api", "v1", "runs"]:
            run = STORE.get_run(parts[3], user["id"])
            if not run:
                return self._send_json({"detail": "Run not found"}, status=404)
            return self._send_json(run)

        if len(parts) >= 4 and parts[0] == "api" and parts[1] == "v1" and parts[2] == "projects":
            project_id = parts[3]
            project = STORE.get_project(project_id, user["id"])
            if not project:
                return self._send_json({"detail": "Project not found"}, status=404)
            if len(parts) == 4:
                return self._send_json(project)
            if len(parts) == 5 and parts[4] == "runs":
                return self._send_json({"runs": STORE.list_runs(project_id, user["id"])})
            if len(parts) == 6 and parts[4] == "artifacts" and parts[5] == "latest":
                latest = STORE.latest_artifact(project_id)
                return self._send_json({"artifact": latest})
            if len(parts) == 5 and parts[4] == "visual-graph":
                return self._send_json(
                    {
                        "nodes": [{"id": "paper"}, {"id": "method"}, {"id": "eval"}],
                        "edges": [{"source": "paper", "target": "method"}, {"source": "method", "target": "eval"}],
                        "mermaid": project["architecture_mermaid"],
                    }
                )
            if len(parts) == 5 and parts[4] == "distillation":
                return self._send_json(build_distillation(project))
            if len(parts) == 5 and parts[4] == "export.zip":
                latest = STORE.latest_artifact(project_id)
                if not latest:
                    return self._send_json({"detail": "No artifact found"}, status=404)
                blob = STORE.load_object(latest["zip_object_key"])
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f"attachment; filename={project_id}_v{latest['version']}.zip")
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)
                return

        self._send_json({"detail": "Not found"}, status=404)

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/v1/auth/register":
            payload = _json(self)
            email = str(payload.get("email", "")).strip().lower()
            if "@" not in email:
                return self._send_json({"detail": "Valid email required"}, status=400)
            try:
                user = STORE.create_user(email)
            except Exception:
                return self._send_json({"detail": "Email already exists"}, status=409)
            return self._send_json(user, status=201)

        user, sent = self._auth()
        if sent is not None:
            return

        payload = _json(self)

        if path == "/api/v1/projects/ingest":
            title = str(payload.get("title", "")).strip()
            abstract = str(payload.get("abstract", "")).strip()
            method_text = str(payload.get("method_text", "")).strip()
            if len(title) < 4 or len(abstract) < 40 or len(method_text) < 80:
                return self._send_json({"detail": "Invalid input lengths"}, status=HTTPStatus.BAD_REQUEST)

            ingest = IngestRequest(title=title, abstract=abstract, method_text=method_text, source_url=payload.get("source_url"))
            paper_spec = build_paper_spec(ingest)
            architecture = build_architecture_mermaid(paper_spec)
            trust = build_trust_layer(paper_spec)
            project = STORE.create_project(
                user_id=user["id"],
                ingest=ingest,
                paper_spec=paper_spec,
                architecture_mermaid=architecture,
                confidence_score=trust["overall_confidence"],
                uncertainty_flags=trust["uncertainty_flags"],
            )
            return self._send_json(project, status=201)

        parts = path.strip("/").split("/")
        if len(parts) == 5 and parts[:3] == ["api", "v1", "projects"] and parts[4] == "runs":
            project_id = parts[3]
            project = STORE.get_project(project_id, user["id"])
            if not project:
                return self._send_json({"detail": "Project not found"}, status=404)
            run_type = str(payload.get("run_type", "artifacts"))
            if run_type not in {"artifacts", "distillation"}:
                return self._send_json({"detail": "run_type must be artifacts|distillation"}, status=400)
            run = STORE.create_run(project_id=project_id, user_id=user["id"], run_type=run_type)
            enqueue_run(
                {
                    "run_id": run["id"],
                    "project_id": project_id,
                    "user_id": user["id"],
                    "run_type": run_type,
                    "framework": payload.get("framework", "pytorch"),
                }
            )
            return self._send_json(run, status=202)

        self._send_json({"detail": "Not found"}, status=404)


def run(host: str = "127.0.0.1", port: int = 8000):
    start_worker_once()
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Paper2Product running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
