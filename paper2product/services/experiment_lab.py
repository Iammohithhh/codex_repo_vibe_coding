"""Experiment Lab — Run configs, benchmark dashboard, experiment tracking.

Provides:
- Experiment configuration management
- Benchmark result tracking
- Delta reports (expected vs observed)
- Experiment history and comparison
- Minimal compute experiment planner
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..models.schema import ExperimentRun, PaperSpec
from . import persistence as db


def create_experiment(
    project_id: str,
    config: Dict[str, Any],
    expected_metrics: Optional[Dict[str, float]] = None,
) -> ExperimentRun:
    """Create a new experiment run with given config."""
    run = ExperimentRun(
        config=config,
        metrics_expected=expected_metrics or {},
        status="pending",
        logs=[f"Experiment created with config: {config}"],
    )
    db.save_experiment_run(project_id, run)
    return run


def start_experiment(project_id: str, run_id: str) -> Dict:
    """Mark experiment as running."""
    runs = db.list_experiment_runs(project_id)
    for r in runs:
        if r.get("id") == run_id:
            r["status"] = "running"
            r["logs"] = r.get("logs", []) + [f"Started at {time.time()}"]
            run = ExperimentRun(**{k: v for k, v in r.items() if k in ExperimentRun.__dataclass_fields__})
            db.save_experiment_run(project_id, run)
            return r
    return {"error": "Run not found"}


def complete_experiment(
    project_id: str,
    run_id: str,
    observed_metrics: Dict[str, float],
    logs: Optional[List[str]] = None,
) -> Dict:
    """Complete an experiment with observed metrics."""
    runs = db.list_experiment_runs(project_id)
    for r in runs:
        if r.get("id") == run_id:
            r["status"] = "completed"
            r["metrics_observed"] = observed_metrics
            r["logs"] = r.get("logs", []) + (logs or []) + [f"Completed at {time.time()}"]

            # Compute delta report
            delta = {}
            for metric, expected in r.get("metrics_expected", {}).items():
                observed = observed_metrics.get(metric, 0.0)
                if expected != 0:
                    pct_diff = round((observed - expected) / abs(expected) * 100, 2)
                else:
                    pct_diff = 0.0
                delta[metric] = {
                    "expected": expected,
                    "observed": observed,
                    "delta": round(observed - expected, 4),
                    "pct_diff": pct_diff,
                    "within_5pct": abs(pct_diff) <= 5.0,
                }
            r["delta_report"] = delta

            run = ExperimentRun(**{k: v for k, v in r.items() if k in ExperimentRun.__dataclass_fields__})
            db.save_experiment_run(project_id, run)
            return r
    return {"error": "Run not found"}


def list_experiments(project_id: str) -> List[Dict]:
    """List all experiments for a project."""
    return db.list_experiment_runs(project_id)


def get_benchmark_dashboard(project_id: str) -> Dict[str, Any]:
    """Generate a benchmark dashboard summary."""
    runs = db.list_experiment_runs(project_id)

    total = len(runs)
    completed = [r for r in runs if r.get("status") == "completed"]
    pending = [r for r in runs if r.get("status") == "pending"]
    running = [r for r in runs if r.get("status") == "running"]
    failed = [r for r in runs if r.get("status") == "failed"]

    # Aggregate metrics across completed runs
    all_metrics: Dict[str, List[float]] = {}
    for r in completed:
        for metric, value in r.get("metrics_observed", {}).items():
            if metric not in all_metrics:
                all_metrics[metric] = []
            all_metrics[metric].append(value)

    metric_summary = {}
    for metric, values in all_metrics.items():
        metric_summary[metric] = {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "runs": len(values),
        }

    return {
        "total_runs": total,
        "completed": len(completed),
        "pending": len(pending),
        "running": len(running),
        "failed": len(failed),
        "metric_summary": metric_summary,
        "recent_runs": runs[:10],
    }


def plan_minimal_experiments(spec: PaperSpec) -> List[Dict[str, Any]]:
    """Plan minimal compute experiments for reproducibility verification."""
    plans = []

    base_config = dict(spec.hyperparameters)

    # 1. Smoke test
    smoke = dict(base_config)
    smoke["epochs"] = 1
    smoke["batch_size"] = min(smoke.get("batch_size", 32), 4)
    plans.append({
        "name": "Smoke Test",
        "purpose": "Verify forward/backward pass works without errors.",
        "config": smoke,
        "expected_time": "< 1 minute",
        "priority": 1,
    })

    # 2. Quick convergence
    quick = dict(base_config)
    quick["epochs"] = min(quick.get("epochs", 10), 3)
    plans.append({
        "name": "Quick Convergence Check",
        "purpose": "Verify loss decreases over a few epochs.",
        "config": quick,
        "expected_time": "5-15 minutes",
        "priority": 2,
    })

    # 3. Hyperparameter sensitivity
    if "learning_rate" in base_config:
        lr = float(base_config["learning_rate"])
        for factor, label in [(0.1, "low_lr"), (10.0, "high_lr")]:
            variant = dict(base_config)
            variant["learning_rate"] = lr * factor
            variant["epochs"] = min(variant.get("epochs", 10), 3)
            plans.append({
                "name": f"LR Sensitivity ({label})",
                "purpose": f"Test learning rate sensitivity at {lr * factor}.",
                "config": variant,
                "expected_time": "5-15 minutes",
                "priority": 3,
            })

    # 4. Full replication
    plans.append({
        "name": "Full Replication",
        "purpose": "Replicate paper results with exact hyperparameters.",
        "config": base_config,
        "expected_time": "varies",
        "priority": 4,
    })

    return plans
