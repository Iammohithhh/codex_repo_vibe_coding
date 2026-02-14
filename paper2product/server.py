from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .core import (
    STORE,
    IngestRequest,
    build_distillation,
    create_project,
    export_zip,
    generate_code_scaffold,
    openapi_spec,
    project_to_dict,
)

INDEX_HTML = """<!doctype html>
<html><head><meta charset='utf-8'><title>Paper2Product Phase 1</title></head>
<body style='font-family: Arial; max-width: 900px; margin: 24px auto;'>
<h1>Paper2Product — Phase 1 End-to-End</h1>
<p>Runs ingestion, artifact generation, visual graph, distillation layer, and zip export.</p>
<button onclick="run()">Run End-to-End</button>
<a id="dl"></a>
<pre id="out">Ready.</pre>
<script>
async function run(){
  const payload={
    title:'Efficient Distilled Transformers',
    abstract:'We propose an efficient transformer compression method for low-latency inference while preserving performance for production tasks.',
    method_text:'Our method defines L = alpha * KL(student, teacher) + beta * CE(student, labels). We run pruning and quantization-aware training on CIFAR-10 and report accuracy and F1.'
  };
  const ingest=await fetch('/api/v1/projects/ingest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}).then(r=>r.json());
  const pid=ingest.id;
  const artifacts=await fetch(`/api/v1/projects/${pid}/artifacts`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({framework:'pytorch'})}).then(r=>r.json());
  const visual=await fetch(`/api/v1/projects/${pid}/visual-graph`).then(r=>r.json());
  const distill=await fetch(`/api/v1/projects/${pid}/distillation`).then(r=>r.json());
  document.getElementById('dl').href=`/api/v1/projects/${pid}/export.zip`;
  document.getElementById('dl').innerText='Download ZIP export';
  document.getElementById('out').innerText=JSON.stringify({ingest,artifacts,visual,distill},null,2);
}
</script></body></html>"""


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

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self._send_html(INDEX_HTML)
        if path == "/openapi.json":
            return self._send_json(openapi_spec())

        parts = path.strip("/").split("/")
        if len(parts) >= 4 and parts[0] == "api" and parts[1] == "v1" and parts[2] == "projects":
            project_id = parts[3]
            project = STORE.projects.get(project_id)
            if not project:
                return self._send_json({"detail": "Project not found"}, status=404)
            if len(parts) == 4:
                return self._send_json(project_to_dict(project))
            if len(parts) == 5 and parts[4] == "visual-graph":
                return self._send_json(
                    {
                        "nodes": [{"id": "paper"}, {"id": "method"}, {"id": "eval"}],
                        "edges": [{"source": "paper", "target": "method"}, {"source": "method", "target": "eval"}],
                        "mermaid": project.architecture_mermaid,
                    }
                )
            if len(parts) == 5 and parts[4] == "distillation":
                return self._send_json(build_distillation(project))
            if len(parts) == 5 and parts[4] == "export.zip":
                framework = parse_qs(urlparse(self.path).query).get("framework", ["pytorch"])[0]
                blob = export_zip(project, framework)
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f"attachment; filename={project_id}.zip")
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)
                return

        self._send_json({"detail": "Not found"}, status=404)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")

        if path == "/api/v1/projects/ingest":
            title = payload.get("title", "").strip()
            abstract = payload.get("abstract", "").strip()
            method_text = payload.get("method_text", "").strip()
            if len(title) < 4 or len(abstract) < 40 or len(method_text) < 80:
                return self._send_json({"detail": "Invalid input lengths"}, status=HTTPStatus.BAD_REQUEST)
            project = create_project(IngestRequest(title=title, abstract=abstract, method_text=method_text, source_url=payload.get("source_url")))
            return self._send_json(project_to_dict(project), status=HTTPStatus.OK)

        parts = path.strip("/").split("/")
        if len(parts) == 5 and parts[:3] == ["api", "v1", "projects"] and parts[4] == "artifacts":
            project_id = parts[3]
            project = STORE.projects.get(project_id)
            if not project:
                return self._send_json({"detail": "Project not found"}, status=404)
            framework = payload.get("framework", "pytorch")
            files = list(generate_code_scaffold(project, framework).keys())
            return self._send_json(
                {
                    "project_id": project_id,
                    "framework": framework,
                    "files": files,
                    "confidence_score": project.confidence_score,
                    "architecture_mermaid": project.architecture_mermaid,
                }
            )

        self._send_json({"detail": "Not found"}, status=404)


def run(host: str = "127.0.0.1", port: int = 8000):
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Paper2Product running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
