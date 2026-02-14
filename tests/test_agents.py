"""Unit tests for the multi-agent pipeline."""
import unittest

from paper2product.models.schema import IngestRequest
from paper2product.agents.reader import read_paper
from paper2product.agents.skeptic import review_paper
from paper2product.agents.implementer import generate_scaffold
from paper2product.agents.verifier import verify_scaffold
from paper2product.agents.explainer import explain_paper
from paper2product.agents.pipeline import run_pipeline


SAMPLE_REQUEST = IngestRequest(
    title="Efficient Distilled Transformers",
    abstract="We propose an efficient transformer compression method combining knowledge distillation with structured pruning for low-latency inference while preserving performance on production NLP tasks.",
    method_text="Our method defines L = alpha * KL(student, teacher) + beta * CE(student, labels). We apply iterative magnitude pruning with quantization-aware training. We train for 50 epochs with batch_size 64 using Adam optimizer with learning_rate 1e-4. We evaluate on CIFAR-10 and SQuAD, reporting accuracy and F1.",
)


class TestReaderAgent(unittest.TestCase):
    def test_read_paper(self):
        spec, messages = read_paper(SAMPLE_REQUEST)
        self.assertIsNotNone(spec.problem)
        self.assertIsNotNone(spec.method)
        self.assertGreater(len(spec.key_equations), 0)
        self.assertGreater(len(spec.datasets), 0)
        self.assertGreater(len(spec.metrics), 0)
        self.assertGreater(len(spec.claims), 0)
        self.assertGreater(len(messages), 0)

    def test_extracts_hyperparameters(self):
        spec, _ = read_paper(SAMPLE_REQUEST)
        self.assertIn("learning_rate", spec.hyperparameters)
        self.assertIn("batch_size", spec.hyperparameters)
        self.assertIn("epochs", spec.hyperparameters)
        self.assertIn("optimizer", spec.hyperparameters)

    def test_extracts_equations(self):
        spec, _ = read_paper(SAMPLE_REQUEST)
        found_kl = any("KL" in eq.raw or "alpha" in eq.raw for eq in spec.key_equations)
        self.assertTrue(found_kl)

    def test_equation_has_intuition(self):
        spec, _ = read_paper(SAMPLE_REQUEST)
        for eq in spec.key_equations:
            self.assertIsNotNone(eq.intuition)
            self.assertGreater(len(eq.intuition), 0)


class TestSkepticAgent(unittest.TestCase):
    def test_review_paper(self):
        spec, _ = read_paper(SAMPLE_REQUEST)
        score, messages = review_paper(spec)
        self.assertIsNotNone(score.overall)
        self.assertGreaterEqual(score.overall, 0)
        self.assertLessEqual(score.overall, 1)
        self.assertGreater(len(messages), 0)

    def test_reproducibility_score_components(self):
        spec, _ = read_paper(SAMPLE_REQUEST)
        score, _ = review_paper(spec)
        self.assertGreaterEqual(score.data_available, 0)
        self.assertGreaterEqual(score.hyperparams_complete, 0)


class TestImplementerAgent(unittest.TestCase):
    def test_generate_pytorch(self):
        spec, _ = read_paper(SAMPLE_REQUEST)
        scaffold, messages = generate_scaffold(spec, "pytorch")
        self.assertEqual(scaffold.framework, "pytorch")
        self.assertGreater(len(scaffold.files), 5)
        paths = [f.path for f in scaffold.files]
        self.assertIn("core/model.py", paths)
        self.assertIn("train/train.py", paths)

    def test_generate_tensorflow(self):
        spec, _ = read_paper(SAMPLE_REQUEST)
        scaffold, messages = generate_scaffold(spec, "tensorflow")
        self.assertEqual(scaffold.framework, "tensorflow")
        model_file = next(f for f in scaffold.files if f.path == "core/model.py")
        self.assertIn("tensorflow", model_file.content.lower())

    def test_has_mermaid(self):
        spec, _ = read_paper(SAMPLE_REQUEST)
        scaffold, _ = generate_scaffold(spec)
        self.assertIn("flowchart", scaffold.architecture_mermaid)


class TestVerifierAgent(unittest.TestCase):
    def test_verify_scaffold(self):
        spec, _ = read_paper(SAMPLE_REQUEST)
        score, _ = review_paper(spec)
        scaffold, _ = generate_scaffold(spec)
        runs, messages = verify_scaffold(spec, scaffold, score)
        self.assertGreater(len(runs), 0)
        self.assertGreater(len(messages), 0)

    def test_experiment_plans(self):
        spec, _ = read_paper(SAMPLE_REQUEST)
        score, _ = review_paper(spec)
        scaffold, _ = generate_scaffold(spec)
        runs, _ = verify_scaffold(spec, scaffold, score)
        statuses = {r.status for r in runs}
        self.assertIn("planned", statuses)


class TestExplainerAgent(unittest.TestCase):
    def test_explain_paper(self):
        spec, _ = read_paper(SAMPLE_REQUEST)
        visual_pack, distillation, messages = explain_paper(spec)
        self.assertGreater(len(visual_pack.nodes), 0)
        self.assertGreater(len(visual_pack.edges), 0)
        self.assertIn("poster", distillation)
        self.assertIn("knowledge_cards", distillation)
        self.assertIn("summaries", distillation)
        self.assertIn("quiz_cards", distillation)

    def test_multi_depth_summaries(self):
        spec, _ = read_paper(SAMPLE_REQUEST)
        _, distillation, _ = explain_paper(spec)
        summaries = distillation.get("summaries", {})
        self.assertIn("eli5", summaries)
        self.assertIn("practitioner", summaries)
        self.assertIn("researcher", summaries)

    def test_mermaid_diagrams(self):
        spec, _ = read_paper(SAMPLE_REQUEST)
        visual_pack, _, _ = explain_paper(spec)
        self.assertGreater(len(visual_pack.mermaid_diagrams), 0)
        self.assertIn("architecture", visual_pack.mermaid_diagrams)


class TestFullPipeline(unittest.TestCase):
    def test_run_pipeline(self):
        from paper2product.services import persistence as db
        db.init_db(":memory:")
        project = run_pipeline(SAMPLE_REQUEST)
        self.assertIsNotNone(project.id)
        self.assertIsNotNone(project.paper_spec)
        self.assertIsNotNone(project.code_scaffold)
        self.assertIsNotNone(project.reproducibility)
        self.assertIsNotNone(project.visual_pack)
        self.assertIsNotNone(project.distillation)
        self.assertGreater(len(project.agent_messages), 5)
        self.assertGreater(len(project.experiment_runs), 0)
        self.assertGreater(project.confidence_score, 0)


if __name__ == "__main__":
    unittest.main()
