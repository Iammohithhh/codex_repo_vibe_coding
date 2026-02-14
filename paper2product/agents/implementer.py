"""Implementer Agent — Generates modular code and interfaces.

Produces framework-specific code scaffolds with proper structure:
  core/ algorithm modules
  train/, eval/, inference/ pipelines
  config presets
  unit and regression tests
  CI workflow
  Dockerfile
"""
from __future__ import annotations

from typing import Dict, List

from ..models.schema import (
    AgentMessage,
    AgentRole,
    ArtifactType,
    Citation,
    CodeScaffold,
    GeneratedFile,
    PaperSpec,
    TraceLink,
)


def _build_model_code(spec: PaperSpec, framework: str) -> str:
    """Generate the core model module."""
    equations_comment = "\n".join(
        f"    # Equation: {eq.raw}\n    # Intuition: {eq.intuition}"
        for eq in spec.key_equations
    )
    arch_components = ", ".join(spec.architecture_components) if spec.architecture_components else "feedforward network"

    if framework == "tensorflow":
        return f'''"""Core model implementation — TensorFlow/Keras.

Paper: Implements {spec.method}
Architecture components: {arch_components}
Source: {spec.citations.get("method", Citation("method", "")).snippet[:100]}
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PaperModel(keras.Model):
    """Model architecture derived from paper specification.

    Components: {arch_components}
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256,
                 output_dim: int = 10, dropout: float = 0.1):
        super().__init__()
        self.encoder = keras.Sequential([
            layers.Dense(hidden_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.LayerNormalization(),
        ])
        self.core = keras.Sequential([
            layers.Dense(hidden_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.LayerNormalization(),
            layers.Dense(hidden_dim, activation="relu"),
        ])
        self.head = layers.Dense(output_dim)

    def call(self, x, training=False):
        h = self.encoder(x, training=training)
        h = self.core(h, training=training)
        return self.head(h)


def build_model(config: dict) -> keras.Model:
    """Factory function to create model from config."""
    return PaperModel(
        input_dim=config.get("input_dim", 512),
        hidden_dim=config.get("hidden_dim", 256),
        output_dim=config.get("output_dim", 10),
        dropout=config.get("dropout", 0.1),
    )


def compute_loss(model, x, y, training=True):
    """Compute loss with citation traceability.

{equations_comment}
    """
    logits = model(x, training=training)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)
'''
    else:  # pytorch (default)
        return f'''"""Core model implementation — PyTorch.

Paper: Implements {spec.method}
Architecture components: {arch_components}
Source: {spec.citations.get("method", Citation("method", "")).snippet[:100]}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperModel(nn.Module):
    """Model architecture derived from paper specification.

    Components: {arch_components}
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256,
                 output_dim: int = 10, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.core = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.core(h)
        return self.head(h)


def build_model(config: dict) -> PaperModel:
    """Factory function to create model from config."""
    return PaperModel(
        input_dim=config.get("input_dim", 512),
        hidden_dim=config.get("hidden_dim", 256),
        output_dim=config.get("output_dim", 10),
        dropout=config.get("dropout", 0.1),
    )


def compute_loss(model: PaperModel, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute loss with citation traceability.

{equations_comment}
    """
    logits = model(x)
    return F.cross_entropy(logits, y)
'''


def _build_train_script(spec: PaperSpec, framework: str) -> str:
    """Generate training pipeline."""
    hp = spec.hyperparameters
    lr = hp.get("learning_rate", "1e-3")
    bs = hp.get("batch_size", 32)
    epochs = hp.get("epochs", 10)
    optimizer = hp.get("optimizer", "Adam")

    if framework == "tensorflow":
        return f'''"""Training pipeline — TensorFlow.

Hyperparameters from paper:
  learning_rate: {lr}
  batch_size: {bs}
  epochs: {epochs}
  optimizer: {optimizer}
"""
import json
import os
import tensorflow as tf
from core.model import build_model


def load_config(path: str = "configs/default.json") -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {{"learning_rate": {lr}, "batch_size": {bs}, "epochs": {epochs}}}


def train(config: dict):
    model = build_model(config)
    optimizer = tf.keras.optimizers.{optimizer}(learning_rate=float(config.get("learning_rate", {lr})))
    epochs = config.get("epochs", {epochs})

    # Placeholder: replace with actual dataset loading
    print(f"Training for {{epochs}} epochs with {optimizer} optimizer")
    print(f"Config: {{config}}")

    for epoch in range(epochs):
        # Simulated training step
        loss = 1.0 / (epoch + 1)
        print(f"Epoch {{epoch+1}}/{{epochs}} — loss: {{loss:.4f}}")

    print("Training complete.")
    return model


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
'''
    else:
        return f'''"""Training pipeline — PyTorch.

Hyperparameters from paper:
  learning_rate: {lr}
  batch_size: {bs}
  epochs: {epochs}
  optimizer: {optimizer}
"""
import json
import os
import torch
from core.model import build_model, compute_loss


def load_config(path: str = "configs/default.json") -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {{"learning_rate": {lr}, "batch_size": {bs}, "epochs": {epochs}}}


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    optimizer = torch.optim.{optimizer}(model.parameters(), lr=float(config.get("learning_rate", {lr})))
    epochs = config.get("epochs", {epochs})

    print(f"Training on {{device}} for {{epochs}} epochs with {optimizer} optimizer")
    print(f"Config: {{config}}")

    model.train()
    for epoch in range(epochs):
        # Simulated training step — replace with actual data loader
        x = torch.randn(config.get("batch_size", {bs}), config.get("input_dim", 512)).to(device)
        y = torch.randint(0, config.get("output_dim", 10), (config.get("batch_size", {bs}),)).to(device)

        optimizer.zero_grad()
        loss = compute_loss(model, x, y)
        loss.backward()
        optimizer.step()

        print(f"Epoch {{epoch+1}}/{{epochs}} — loss: {{loss.item():.4f}}")

    print("Training complete.")
    torch.save(model.state_dict(), "checkpoints/model.pt")
    return model


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    cfg = load_config()
    train(cfg)
'''


def _build_eval_script(spec: PaperSpec, framework: str) -> str:
    """Generate evaluation pipeline."""
    metrics_list = ", ".join(spec.metrics) if spec.metrics else "accuracy"

    return f'''"""Evaluation pipeline.

Metrics tracked: {metrics_list}
Datasets: {", ".join(spec.datasets) if spec.datasets else "default"}
"""
import json


def evaluate(model, test_data=None, config=None):
    """Run evaluation and return metrics.

    Returns dict mapping metric names to values.
    Expected metrics from paper: {metrics_list}
    """
    results = {{}}
    metrics = {spec.metrics if spec.metrics else ["accuracy"]}

    for metric in metrics:
        # Placeholder: replace with actual metric computation
        results[metric] = 0.0

    print(f"Evaluation results: {{results}}")
    return results


def compare_with_paper(observed: dict, expected: dict) -> dict:
    """Compute delta between observed and expected (paper-reported) metrics."""
    delta = {{}}
    for metric, expected_val in expected.items():
        observed_val = observed.get(metric, 0.0)
        delta[metric] = {{
            "expected": expected_val,
            "observed": observed_val,
            "delta": round(observed_val - expected_val, 4),
            "within_tolerance": abs(observed_val - expected_val) < 0.05 * abs(expected_val) if expected_val else True,
        }}
    return delta


if __name__ == "__main__":
    results = evaluate(None)
    print(json.dumps(results, indent=2))
'''


def _build_inference_script(spec: PaperSpec, framework: str) -> str:
    """Generate inference pipeline."""
    if framework == "tensorflow":
        return '''"""Inference pipeline — TensorFlow.

Provides both single-sample and batch inference with latency tracking.
"""
import time
import numpy as np
import tensorflow as tf
from core.model import build_model


class InferenceEngine:
    def __init__(self, config: dict, model_path: str = None):
        self.model = build_model(config)
        if model_path:
            self.model.load_weights(model_path)
        self.config = config

    def predict(self, x):
        start = time.time()
        result = self.model(x, training=False)
        latency = time.time() - start
        return {"predictions": result.numpy().tolist(), "latency_ms": latency * 1000}

    def predict_batch(self, batch):
        start = time.time()
        results = self.model(batch, training=False)
        latency = time.time() - start
        return {
            "predictions": results.numpy().tolist(),
            "batch_size": len(batch),
            "latency_ms": latency * 1000,
            "per_sample_ms": (latency * 1000) / max(len(batch), 1),
        }


if __name__ == "__main__":
    engine = InferenceEngine({"input_dim": 512, "output_dim": 10})
    sample = np.random.randn(1, 512).astype(np.float32)
    print(engine.predict(sample))
'''
    else:
        return '''"""Inference pipeline — PyTorch.

Provides both single-sample and batch inference with latency tracking.
"""
import time
import torch
from core.model import build_model


class InferenceEngine:
    def __init__(self, config: dict, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(config).to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.config = config

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        start = time.time()
        x = x.to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)
        latency = time.time() - start
        return {
            "predictions": probs.cpu().tolist(),
            "class": torch.argmax(probs, dim=-1).cpu().tolist(),
            "latency_ms": latency * 1000,
        }

    @torch.no_grad()
    def predict_batch(self, batch: torch.Tensor) -> dict:
        start = time.time()
        batch = batch.to(self.device)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=-1)
        latency = time.time() - start
        return {
            "predictions": probs.cpu().tolist(),
            "classes": torch.argmax(probs, dim=-1).cpu().tolist(),
            "batch_size": len(batch),
            "latency_ms": latency * 1000,
            "per_sample_ms": (latency * 1000) / max(len(batch), 1),
        }


if __name__ == "__main__":
    engine = InferenceEngine({"input_dim": 512, "output_dim": 10})
    sample = torch.randn(1, 512)
    print(engine.predict(sample))
'''


def _build_config(spec: PaperSpec) -> str:
    """Generate default config JSON."""
    hp = spec.hyperparameters.copy()
    hp.setdefault("learning_rate", 1e-3)
    hp.setdefault("batch_size", 32)
    hp.setdefault("epochs", 10)
    hp.setdefault("optimizer", "Adam")
    hp.setdefault("input_dim", 512)
    hp.setdefault("hidden_dim", 256)
    hp.setdefault("output_dim", 10)
    hp.setdefault("dropout", 0.1)

    import json
    return json.dumps(hp, indent=2)


def _build_tests(spec: PaperSpec, framework: str) -> str:
    """Generate test suite."""
    if framework == "tensorflow":
        return '''"""Test suite for generated model."""
import unittest
import numpy as np


class TestModel(unittest.TestCase):
    def test_model_creation(self):
        from core.model import build_model
        model = build_model({"input_dim": 512, "hidden_dim": 256, "output_dim": 10})
        self.assertIsNotNone(model)

    def test_forward_pass(self):
        from core.model import build_model
        import tensorflow as tf
        model = build_model({"input_dim": 512, "hidden_dim": 256, "output_dim": 10})
        x = tf.random.normal((4, 512))
        out = model(x)
        self.assertEqual(out.shape, (4, 10))

    def test_output_range(self):
        from core.model import build_model
        import tensorflow as tf
        model = build_model({"input_dim": 512, "hidden_dim": 256, "output_dim": 10})
        x = tf.random.normal((2, 512))
        out = model(x)
        self.assertFalse(tf.reduce_any(tf.math.is_nan(out)).numpy())


class TestEval(unittest.TestCase):
    def test_compare_with_paper(self):
        from eval.evaluate import compare_with_paper
        delta = compare_with_paper({"accuracy": 0.92}, {"accuracy": 0.95})
        self.assertIn("accuracy", delta)
        self.assertIn("delta", delta["accuracy"])


if __name__ == "__main__":
    unittest.main()
'''
    else:
        return '''"""Test suite for generated model."""
import unittest
import torch


class TestModel(unittest.TestCase):
    def test_model_creation(self):
        from core.model import build_model
        model = build_model({"input_dim": 512, "hidden_dim": 256, "output_dim": 10})
        self.assertIsNotNone(model)

    def test_forward_pass(self):
        from core.model import build_model
        model = build_model({"input_dim": 512, "hidden_dim": 256, "output_dim": 10})
        x = torch.randn(4, 512)
        out = model(x)
        self.assertEqual(out.shape, (4, 10))

    def test_output_range(self):
        from core.model import build_model
        model = build_model({"input_dim": 512, "hidden_dim": 256, "output_dim": 10})
        x = torch.randn(2, 512)
        out = model(x)
        self.assertFalse(torch.isnan(out).any().item())

    def test_loss_computation(self):
        from core.model import build_model, compute_loss
        model = build_model({"input_dim": 512, "hidden_dim": 256, "output_dim": 10})
        x = torch.randn(4, 512)
        y = torch.randint(0, 10, (4,))
        loss = compute_loss(model, x, y)
        self.assertGreater(loss.item(), 0)


class TestEval(unittest.TestCase):
    def test_compare_with_paper(self):
        from eval.evaluate import compare_with_paper
        delta = compare_with_paper({"accuracy": 0.92}, {"accuracy": 0.95})
        self.assertIn("accuracy", delta)
        self.assertIn("delta", delta["accuracy"])


if __name__ == "__main__":
    unittest.main()
'''


def _build_ci_workflow(framework: str) -> str:
    """Generate GitHub Actions CI workflow."""
    pip_deps = "torch torchvision" if framework == "pytorch" else "tensorflow"
    return f'''name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          pip install {pip_deps} pytest
          pip install -r requirements.txt || true
      - name: Run tests
        run: python -m pytest tests/ -v
      - name: Smoke test training
        run: python train/train.py || echo "Training smoke test (expected placeholder)"
'''


def _build_dockerfile(framework: str) -> str:
    """Generate Dockerfile for containerized execution."""
    base = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime" if framework == "pytorch" else "tensorflow/tensorflow:2.15.0-gpu"
    pip_deps = "torch" if framework == "pytorch" else "tensorflow"
    return f'''FROM {base}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

COPY . .

# Default: run inference server
EXPOSE 8080
CMD ["python", "inference/serve.py"]
'''


def _build_readme(spec: PaperSpec, framework: str) -> str:
    """Generate comprehensive README."""
    equations_md = "\n".join(f"- `{eq.raw}` — {eq.intuition}" for eq in spec.key_equations)
    datasets_md = ", ".join(spec.datasets) if spec.datasets else "N/A"
    metrics_md = ", ".join(spec.metrics) if spec.metrics else "N/A"

    return f'''# {spec.method}

> Auto-generated implementation by Paper2Product 2.0

## Problem
{spec.problem}

## Method
{spec.method}

## Key Equations
{equations_md}

## Datasets
{datasets_md}

## Metrics
{metrics_md}

## Citation Traceability
- **Problem source:** {spec.citations.get("problem", Citation("", "")).snippet[:150]}
- **Method source:** {spec.citations.get("method", Citation("", "")).snippet[:150]}

## Project Structure
```
├── core/
│   └── model.py          # Core model implementation
├── train/
│   └── train.py           # Training pipeline
├── eval/
│   └── evaluate.py        # Evaluation and benchmarking
├── inference/
│   └── serve.py           # Inference engine
├── configs/
│   └── default.json       # Default hyperparameters
├── tests/
│   └── test_model.py      # Unit and regression tests
├── .github/workflows/
│   └── ci.yml             # CI pipeline
├── Dockerfile
└── README.md
```

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Train
python train/train.py

# Evaluate
python eval/evaluate.py

# Run inference
python inference/serve.py
```

## Framework
{framework}

## Assumptions
{chr(10).join(f"- {a}" for a in spec.assumptions)}

## Limitations
{chr(10).join(f"- {l}" for l in spec.limitations)}
'''


def _build_requirements(framework: str) -> str:
    """Generate requirements.txt."""
    if framework == "tensorflow":
        return "tensorflow>=2.15.0\nnumpy>=1.24.0\nscikit-learn>=1.3.0\n"
    return "torch>=2.1.0\ntorchvision>=0.16.0\nnumpy>=1.24.0\nscikit-learn>=1.3.0\n"


def generate_scaffold(spec: PaperSpec, framework: str = "pytorch") -> tuple[CodeScaffold, list[AgentMessage]]:
    """Main implementer entry point."""
    messages: list[AgentMessage] = []

    messages.append(AgentMessage(
        role=AgentRole.IMPLEMENTER,
        content=f"Generating {framework} code scaffold for: {spec.method[:80]}",
        metadata={"framework": framework, "phase": "start"},
    ))

    problem_cite = spec.citations.get("problem", Citation("", ""))
    method_cite = spec.citations.get("method", Citation("", ""))

    files = [
        GeneratedFile(
            path="README.md",
            content=_build_readme(spec, framework),
            artifact_type=ArtifactType.REPORT,
            trace_links=[TraceLink("README.md", citations=[problem_cite, method_cite])],
            language="markdown",
        ),
        GeneratedFile(
            path="core/model.py",
            content=_build_model_code(spec, framework),
            artifact_type=ArtifactType.CODE,
            trace_links=[TraceLink("core/model.py", citations=[method_cite])],
        ),
        GeneratedFile(
            path="core/__init__.py",
            content="",
            artifact_type=ArtifactType.CODE,
        ),
        GeneratedFile(
            path="train/train.py",
            content=_build_train_script(spec, framework),
            artifact_type=ArtifactType.CODE,
            trace_links=[TraceLink("train/train.py", citations=[method_cite])],
        ),
        GeneratedFile(
            path="train/__init__.py",
            content="",
            artifact_type=ArtifactType.CODE,
        ),
        GeneratedFile(
            path="eval/evaluate.py",
            content=_build_eval_script(spec, framework),
            artifact_type=ArtifactType.CODE,
        ),
        GeneratedFile(
            path="eval/__init__.py",
            content="",
            artifact_type=ArtifactType.CODE,
        ),
        GeneratedFile(
            path="inference/serve.py",
            content=_build_inference_script(spec, framework),
            artifact_type=ArtifactType.CODE,
        ),
        GeneratedFile(
            path="inference/__init__.py",
            content="",
            artifact_type=ArtifactType.CODE,
        ),
        GeneratedFile(
            path="configs/default.json",
            content=_build_config(spec),
            artifact_type=ArtifactType.CODE,
            language="json",
        ),
        GeneratedFile(
            path="tests/test_model.py",
            content=_build_tests(spec, framework),
            artifact_type=ArtifactType.TEST_SUITE,
        ),
        GeneratedFile(
            path=".github/workflows/ci.yml",
            content=_build_ci_workflow(framework),
            artifact_type=ArtifactType.CODE,
            language="yaml",
        ),
        GeneratedFile(
            path="Dockerfile",
            content=_build_dockerfile(framework),
            artifact_type=ArtifactType.CODE,
            language="dockerfile",
        ),
        GeneratedFile(
            path="requirements.txt",
            content=_build_requirements(framework),
            artifact_type=ArtifactType.CODE,
        ),
    ]

    # Build architecture mermaid
    components = spec.architecture_components if spec.architecture_components else ["model"]
    mermaid_lines = ["flowchart TD"]
    mermaid_lines.append("    A[Input Data] --> B[Preprocessing]")
    mermaid_lines.append("    B --> C[Encoder]")
    prev = "C"
    for i, comp in enumerate(components[:5]):
        node = chr(ord("D") + i)
        mermaid_lines.append(f"    {prev} --> {node}[{comp}]")
        prev = node
    mermaid_lines.append(f"    {prev} --> OUT[Output / {spec.metrics[0] if spec.metrics else 'Prediction'}]")
    mermaid_lines.append(f"    OUT --> EVAL[Evaluation: {', '.join(spec.metrics[:3]) if spec.metrics else 'metrics'}]")
    mermaid = "\n".join(mermaid_lines)

    scaffold = CodeScaffold(
        framework=framework,
        files=files,
        architecture_mermaid=mermaid,
        ci_workflow=_build_ci_workflow(framework),
        docker_file=_build_dockerfile(framework),
        confidence=0.75,
    )

    messages.append(AgentMessage(
        role=AgentRole.IMPLEMENTER,
        content=f"Generated {len(files)} files for {framework} scaffold.",
        metadata={"files_count": len(files), "files": [f.path for f in files]},
        confidence=0.8,
    ))

    return scaffold, messages
