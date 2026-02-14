import json
import threading
import time
import unittest
import urllib.request

from paper2product.server import run


class TestPhase1E2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread = threading.Thread(target=run, kwargs={"host": "127.0.0.1", "port": 8011}, daemon=True)
        cls.thread.start()
        time.sleep(0.3)

    def request_json(self, method: str, path: str, data=None):
        body = None
        headers = {}
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(f"http://127.0.0.1:8011{path}", data=body, method=method, headers=headers)
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if "application/json" in ctype:
                return resp.status, json.loads(content.decode("utf-8")), ctype
            return resp.status, content, ctype

    def test_end_to_end(self):
        ingest_payload = {
            "title": "Efficient Distilled Transformers",
            "abstract": "We propose an efficient transformer compression method for low-latency inference while preserving performance for production NLP tasks.",
            "method_text": "Our method defines L = alpha * KL(student, teacher) + beta * CE(student, labels). We run pruning and quantization-aware training. We evaluate on CIFAR-10 and report accuracy and F1.",
        }
        status, project, _ = self.request_json("POST", "/api/v1/projects/ingest", ingest_payload)
        self.assertEqual(status, 200)
        pid = project["id"]

        status, artifacts, _ = self.request_json("POST", f"/api/v1/projects/{pid}/artifacts", {"framework": "pytorch"})
        self.assertEqual(status, 200)
        self.assertIn("core/model.py", artifacts["files"])

        status, visual, _ = self.request_json("GET", f"/api/v1/projects/{pid}/visual-graph")
        self.assertEqual(status, 200)
        self.assertIn("mermaid", visual)

        status, distill, _ = self.request_json("GET", f"/api/v1/projects/{pid}/distillation")
        self.assertEqual(status, 200)
        self.assertIn("poster", distill)
        self.assertIn("knowledge_cards", distill)
        self.assertIn("executive_takeaways", distill)
        self.assertIn("extension_engine", distill)
        self.assertIn("learning_path", distill)

        status, zip_blob, ctype = self.request_json("GET", f"/api/v1/projects/{pid}/export.zip")
        self.assertEqual(status, 200)
        self.assertIn("application/zip", ctype)
        self.assertGreater(len(zip_blob), 100)


if __name__ == "__main__":
    unittest.main()
