import json
import threading
import time
import unittest
import urllib.request

from paper2product.server import run


class TestPhase2E2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread = threading.Thread(target=run, kwargs={"host": "127.0.0.1", "port": 8011}, daemon=True)
        cls.thread.start()
        time.sleep(0.4)

    def request_json(self, method: str, path: str, data=None, headers=None):
        body = None
        req_headers = headers or {}
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
        req = urllib.request.Request(f"http://127.0.0.1:8011{path}", data=body, method=method, headers=req_headers)
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if "application/json" in ctype:
                return resp.status, json.loads(content.decode("utf-8")), ctype
            return resp.status, content, ctype

    def wait_run(self, run_id, headers, timeout=5):
        start = time.time()
        while time.time() - start < timeout:
            status, run, _ = self.request_json("GET", f"/api/v1/runs/{run_id}", headers=headers)
            self.assertEqual(status, 200)
            if run["status"] in {"completed", "failed"}:
                return run
            time.sleep(0.1)
        self.fail("Run did not complete")

    def test_phase2_end_to_end(self):
        status, user, _ = self.request_json("POST", "/api/v1/auth/register", {"email": "builder@example.com"})
        self.assertEqual(status, 201)
        headers = {"X-API-Key": user["token"]}

        ingest_payload = {
            "title": "Efficient Distilled Transformers",
            "abstract": "We propose an efficient transformer compression method for low-latency inference while preserving performance for production NLP tasks.",
            "method_text": "Our method defines L = alpha * KL(student, teacher) + beta * CE(student, labels). We run pruning and quantization-aware training. We evaluate on CIFAR-10 and report accuracy and F1.",
        }
        status, project, _ = self.request_json("POST", "/api/v1/projects/ingest", ingest_payload, headers=headers)
        self.assertEqual(status, 201)
        pid = project["id"]

        status, run, _ = self.request_json(
            "POST",
            f"/api/v1/projects/{pid}/runs",
            {"run_type": "artifacts", "framework": "pytorch"},
            headers=headers,
        )
        self.assertEqual(status, 202)
        final_run = self.wait_run(run["id"], headers=headers)
        self.assertEqual(final_run["status"], "completed")

        status, latest, _ = self.request_json("GET", f"/api/v1/projects/{pid}/artifacts/latest", headers=headers)
        self.assertEqual(status, 200)
        self.assertIsNotNone(latest["artifact"])

        status, visual, _ = self.request_json("GET", f"/api/v1/projects/{pid}/visual-graph", headers=headers)
        self.assertEqual(status, 200)
        self.assertIn("mermaid", visual)

        status, distill, _ = self.request_json("GET", f"/api/v1/projects/{pid}/distillation", headers=headers)
        self.assertEqual(status, 200)
        self.assertIn("poster", distill)
        self.assertIn("knowledge_cards", distill)
        self.assertIn("executive_takeaways", distill)
        self.assertIn("extension_engine", distill)
        self.assertIn("learning_path", distill)

        status, dashboard, _ = self.request_json("GET", "/api/v1/dashboard", headers=headers)
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(dashboard["projects"]), 1)

        status, zip_blob, ctype = self.request_json("GET", f"/api/v1/projects/{pid}/export.zip", headers=headers)
        self.assertEqual(status, 200)
        self.assertIn("application/zip", ctype)
        self.assertGreater(len(zip_blob), 100)


if __name__ == "__main__":
    unittest.main()
