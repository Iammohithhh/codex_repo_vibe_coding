"""End-to-end tests for Paper2Product v2 API."""
import json
import threading
import time
import unittest
import urllib.request
import urllib.error


class TestV2E2E(unittest.TestCase):
    """Full v2 API e2e test suite."""

    @classmethod
    def setUpClass(cls):
        from paper2product.services import persistence as db
        db.init_db(":memory:")
        from paper2product.server_v2 import run
        cls.thread = threading.Thread(target=run, kwargs={"host": "127.0.0.1", "port": 8012}, daemon=True)
        cls.thread.start()
        time.sleep(0.5)

    def request(self, method: str, path: str, data=None):
        body = None
        headers = {}
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(f"http://127.0.0.1:8012{path}", data=body, method=method, headers=headers)
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if "application/json" in ctype:
                return resp.status, json.loads(content.decode("utf-8")), ctype
            return resp.status, content, ctype

    def test_full_pipeline(self):
        """Test the complete v2 pipeline: ingest -> artifacts -> visual-graph -> distillation -> export."""
        payload = {
            "title": "Efficient Distilled Transformers",
            "abstract": "We propose an efficient transformer compression method combining knowledge distillation with structured pruning for low-latency inference while preserving performance on production NLP tasks.",
            "method_text": "Our method defines L = alpha * KL(student, teacher) + beta * CE(student, labels). We apply iterative magnitude pruning with quantization-aware training. We train for 50 epochs with batch_size 64 using Adam optimizer with learning_rate 1e-4. We evaluate on CIFAR-10 and SQuAD, reporting accuracy and F1.",
            "framework": "pytorch",
            "persona": "ml_engineer",
        }

        # 1. Ingest
        status, project, _ = self.request("POST", "/api/v2/projects/ingest", payload)
        self.assertEqual(status, 200)
        self.assertIn("id", project)
        pid = project["id"]

        # Verify paper spec
        self.assertIn("paper_spec", project)
        spec = project["paper_spec"]
        self.assertIn("problem", spec)
        self.assertIn("method", spec)
        self.assertIn("key_equations", spec)
        self.assertIn("datasets", spec)
        self.assertIn("metrics", spec)
        self.assertIn("claims", spec)
        self.assertIn("limitations", spec)
        self.assertIn("hyperparameters", spec)

        # Verify reproducibility
        self.assertIn("reproducibility", project)
        repro = project["reproducibility"]
        self.assertIn("overall", repro)
        self.assertIn("blockers", repro)
        self.assertIn("fixes", repro)

        # Verify code scaffold
        self.assertIn("code_scaffold", project)
        scaffold = project["code_scaffold"]
        self.assertIn("files", scaffold)
        self.assertIn("framework", scaffold)
        file_paths = [f.get("path", "") for f in scaffold["files"]]
        self.assertIn("core/model.py", file_paths)
        self.assertIn("train/train.py", file_paths)
        self.assertIn("eval/evaluate.py", file_paths)
        self.assertIn("inference/serve.py", file_paths)
        self.assertIn("configs/default.json", file_paths)
        self.assertIn("tests/test_model.py", file_paths)
        self.assertIn("Dockerfile", file_paths)

        # Verify agent messages
        self.assertIn("agent_messages", project)
        messages = project["agent_messages"]
        self.assertGreater(len(messages), 5)
        roles = {m.get("role") for m in messages}
        self.assertIn("reader", roles)
        self.assertIn("skeptic", roles)
        self.assertIn("implementer", roles)
        self.assertIn("verifier", roles)
        self.assertIn("explainer", roles)

        # Verify confidence
        self.assertIn("confidence_score", project)
        self.assertGreater(project["confidence_score"], 0)

        # 2. List projects
        status, data, _ = self.request("GET", "/api/v2/projects")
        self.assertEqual(status, 200)
        self.assertGreater(data["count"], 0)

        # 3. Get project detail
        status, detail, _ = self.request("GET", f"/api/v2/projects/{pid}")
        self.assertEqual(status, 200)

        # 4. Visual graph
        status, visual, _ = self.request("GET", f"/api/v2/projects/{pid}/visual-graph")
        self.assertEqual(status, 200)
        self.assertIn("architecture_graph", visual)
        self.assertIn("data_flow_timeline", visual)
        self.assertIn("failure_mode_map", visual)
        self.assertIn("mermaid_diagrams", visual)

        # 5. Distillation
        status, distill, _ = self.request("GET", f"/api/v2/projects/{pid}/distillation")
        self.assertEqual(status, 200)
        self.assertIn("poster", distill)
        self.assertIn("knowledge_cards", distill)
        self.assertIn("executive_takeaways", distill)
        self.assertIn("extension_engine", distill)
        self.assertIn("learning_path", distill)
        self.assertIn("summaries", distill)
        self.assertIn("quiz_cards", distill)

        # 6. Reproducibility
        status, repro_data, _ = self.request("GET", f"/api/v2/projects/{pid}/reproducibility")
        self.assertEqual(status, 200)
        self.assertIn("overall", repro_data)

        # 7. Code scaffold detail
        status, scaffold_data, _ = self.request("GET", f"/api/v2/projects/{pid}/code-scaffold")
        self.assertEqual(status, 200)
        self.assertIn("files", scaffold_data)
        self.assertIn("framework", scaffold_data)

        # 8. Agent messages
        status, msgs, _ = self.request("GET", f"/api/v2/projects/{pid}/agent-messages")
        self.assertEqual(status, 200)
        self.assertGreater(msgs["count"], 0)

        # 9. Export ZIP
        status, zip_blob, ctype = self.request("GET", f"/api/v2/projects/{pid}/export.zip")
        self.assertEqual(status, 200)
        self.assertIn("application/zip", ctype)
        self.assertGreater(len(zip_blob), 500)

    def test_productization(self):
        """Test productization: API contract + deployment plan + launch package."""
        payload = {
            "title": "Neural Style Transfer",
            "abstract": "We present a neural style transfer method using deep convolutional neural networks that separates content and style representations to generate artistic images in real-time.",
            "method_text": "Our approach uses a VGG-19 encoder to extract feature representations. We define L_total = alpha * L_content + beta * L_style where L_content = MSE(features_output, features_content) and L_style uses Gram matrices. We train for 20 epochs with batch_size 16 using Adam with learning_rate 1e-3. We evaluate on COCO reporting FID and LPIPS metrics.",
            "framework": "pytorch",
        }

        status, project, _ = self.request("POST", "/api/v2/projects/ingest", payload)
        self.assertEqual(status, 200)
        pid = project["id"]

        status, result, _ = self.request("POST", f"/api/v2/projects/{pid}/productize", {
            "architecture_type": "server",
        })
        self.assertEqual(status, 200)
        self.assertIn("api_contract", result)
        self.assertIn("deployment_plan", result)
        self.assertIn("launch_package", result)

        contract = result["api_contract"]
        self.assertIn("openapi_spec", contract)
        self.assertIn("endpoints", contract)
        self.assertIn("sdk_stubs", contract)
        self.assertIn("python", contract["sdk_stubs"])
        self.assertIn("javascript", contract["sdk_stubs"])

        deploy = result["deployment_plan"]
        self.assertEqual(deploy["architecture_type"], "server")
        self.assertIn("security_checklist", deploy)

        launch = result["launch_package"]
        self.assertIn("web_deploy", launch)
        self.assertIn("mobile_release", launch)
        self.assertIn("release_checklist", launch)

    def test_experiments(self):
        """Test experiment lab."""
        payload = {
            "title": "Test Experiment Project",
            "abstract": "A test project for experiment lab functionality with enough characters to pass validation requirements.",
            "method_text": "We define a simple loss function L = MSE(prediction, target) and train using SGD with learning_rate 0.01 and batch_size 32 for 100 epochs. We evaluate accuracy on the test set using standard metrics.",
        }

        status, project, _ = self.request("POST", "/api/v2/projects/ingest", payload)
        self.assertEqual(status, 200)
        pid = project["id"]

        # Get experiment plans
        status, plans, _ = self.request("GET", f"/api/v2/projects/{pid}/experiments/plan")
        self.assertEqual(status, 200)
        self.assertIn("plans", plans)
        self.assertGreater(len(plans["plans"]), 0)

        # Create experiment
        status, run, _ = self.request("POST", f"/api/v2/projects/{pid}/experiments", {
            "config": {"epochs": 1, "batch_size": 8},
            "expected_metrics": {"accuracy": 0.9},
        })
        self.assertEqual(status, 201)

        # List experiments
        status, exps, _ = self.request("GET", f"/api/v2/projects/{pid}/experiments")
        self.assertEqual(status, 200)
        self.assertGreater(len(exps["experiments"]), 0)

    def test_collaboration(self):
        """Test collaboration: workspace, reviews, approval."""
        status, ws, _ = self.request("POST", "/api/v2/workspaces", {
            "name": "Test Workspace",
            "members": ["alice", "bob"],
        })
        self.assertEqual(status, 201)
        self.assertIn("id", ws)

        status, wsl, _ = self.request("GET", "/api/v2/workspaces")
        self.assertEqual(status, 200)

        payload = {
            "title": "Review Test Project",
            "abstract": "A project to test the review and approval workflow with sufficient content for validation.",
            "method_text": "We implement a standard classification pipeline with L = CrossEntropy(logits, labels) and train using Adam optimizer with learning_rate 1e-3 for 50 epochs on the CIFAR-10 dataset reporting accuracy.",
        }
        status, project, _ = self.request("POST", "/api/v2/projects/ingest", payload)
        pid = project["id"]

        status, review, _ = self.request("POST", f"/api/v2/projects/{pid}/reviews", {
            "author": "alice",
            "content": "Looks good, but needs more baselines.",
            "artifact_path": "core/model.py",
        })
        self.assertEqual(status, 201)

        status, reviews, _ = self.request("GET", f"/api/v2/projects/{pid}/reviews")
        self.assertEqual(status, 200)
        self.assertGreater(len(reviews["reviews"]), 0)

        status, approval, _ = self.request("POST", f"/api/v2/projects/{pid}/approve", {
            "approver": "admin",
        })
        self.assertEqual(status, 200)

    def test_openapi(self):
        """Test OpenAPI spec endpoint."""
        status, spec, _ = self.request("GET", "/openapi.json")
        self.assertEqual(status, 200)
        self.assertEqual(spec["openapi"], "3.0.3")
        self.assertIn("paths", spec)
        self.assertGreater(len(spec["paths"]), 10)

    def test_validation_errors(self):
        """Test input validation."""
        try:
            self.request("POST", "/api/v2/projects/ingest", {
                "title": "Ab",
                "abstract": "x" * 50,
                "method_text": "x" * 100,
            })
            self.fail("Expected error")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_not_found(self):
        """Test 404 for nonexistent project."""
        try:
            self.request("GET", "/api/v2/projects/nonexistent-id")
            self.fail("Expected 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_tensorflow_framework(self):
        """Test TensorFlow code generation."""
        payload = {
            "title": "TensorFlow Generation Test",
            "abstract": "Testing TensorFlow code generation with a simple classification model that demonstrates framework-specific output.",
            "method_text": "We define L = CrossEntropy(logits, labels) and train using Adam optimizer with learning_rate 1e-3 and batch_size 32 for 10 epochs. We evaluate accuracy on CIFAR-10 using standard TensorFlow/Keras APIs.",
            "framework": "tensorflow",
        }
        status, project, _ = self.request("POST", "/api/v2/projects/ingest", payload)
        self.assertEqual(status, 200)
        scaffold = project.get("code_scaffold", {})
        self.assertEqual(scaffold.get("framework"), "tensorflow")

        model_file = None
        for f in scaffold.get("files", []):
            if f.get("path") == "core/model.py":
                model_file = f
                break
        self.assertIsNotNone(model_file)
        self.assertIn("tensorflow", model_file.get("content", "").lower())


if __name__ == "__main__":
    unittest.main()
