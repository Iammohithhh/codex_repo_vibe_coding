"""Productization Layer — Converts research logic into scalable service architecture.

Provides:
- Architecture templates (edge/mobile/server/hybrid)
- Cost estimation for cloud deployment
- Security/privacy checklist
- API hardening templates
- Launch package generation
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from ..models.schema import (
    APIContract,
    DeploymentPlan,
    LaunchPackage,
    PaperSpec,
    CodeScaffold,
)


# ---------------------------------------------------------------------------
# API Contract Generation
# ---------------------------------------------------------------------------
def generate_api_contract(spec: PaperSpec, scaffold: CodeScaffold) -> APIContract:
    """Generate a complete API contract from the paper spec and code scaffold."""
    endpoints = [
        {
            "path": "/api/v1/predict",
            "method": "POST",
            "summary": "Run inference on input data",
            "request_body": {
                "type": "object",
                "properties": {
                    "data": {"type": "array", "items": {"type": "number"}},
                    "config": {"type": "object", "description": "Optional runtime config overrides"},
                },
                "required": ["data"],
            },
            "response": {
                "type": "object",
                "properties": {
                    "predictions": {"type": "array"},
                    "confidence": {"type": "number"},
                    "latency_ms": {"type": "number"},
                },
            },
        },
        {
            "path": "/api/v1/predict/batch",
            "method": "POST",
            "summary": "Run batch inference",
            "request_body": {
                "type": "object",
                "properties": {
                    "batch": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                },
                "required": ["batch"],
            },
            "response": {
                "type": "object",
                "properties": {
                    "predictions": {"type": "array"},
                    "batch_size": {"type": "integer"},
                    "latency_ms": {"type": "number"},
                },
            },
        },
        {
            "path": "/api/v1/model/info",
            "method": "GET",
            "summary": "Get model metadata and paper info",
            "response": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string"},
                    "framework": {"type": "string"},
                    "paper_title": {"type": "string"},
                    "metrics": {"type": "object"},
                    "version": {"type": "string"},
                },
            },
        },
        {
            "path": "/api/v1/health",
            "method": "GET",
            "summary": "Health check endpoint",
            "response": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "model_loaded": {"type": "boolean"},
                    "uptime_seconds": {"type": "number"},
                },
            },
        },
    ]

    openapi_spec = {
        "openapi": "3.0.3",
        "info": {
            "title": f"Paper2Product API — {spec.method[:60]}",
            "version": "1.0.0",
            "description": f"Auto-generated API for: {spec.problem[:200]}",
        },
        "servers": [
            {"url": "http://localhost:8080", "description": "Local development"},
            {"url": "https://api.staging.example.com", "description": "Staging"},
            {"url": "https://api.example.com", "description": "Production"},
        ],
        "paths": {},
        "components": {
            "securitySchemes": {
                "bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"},
                "apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
            },
        },
        "security": [{"bearerAuth": []}, {"apiKey": []}],
    }

    for ep in endpoints:
        path_item: Dict[str, Any] = {}
        method = ep["method"].lower()
        operation: Dict[str, Any] = {"summary": ep["summary"]}
        if "request_body" in ep:
            operation["requestBody"] = {
                "required": True,
                "content": {"application/json": {"schema": ep["request_body"]}},
            }
        operation["responses"] = {
            "200": {
                "description": "Success",
                "content": {"application/json": {"schema": ep["response"]}},
            },
            "400": {"description": "Bad request"},
            "401": {"description": "Unauthorized"},
            "500": {"description": "Internal server error"},
        }
        path_item[method] = operation
        openapi_spec["paths"][ep["path"]] = path_item

    # SDK stubs
    sdk_stubs = _generate_sdk_stubs(spec, endpoints)

    return APIContract(
        openapi_spec=openapi_spec,
        endpoints=endpoints,
        sdk_stubs=sdk_stubs,
        rate_limits={"default": 100, "predict": 60, "batch": 20},
        auth_config={"type": "bearer_jwt", "issuer": "paper2product", "expiry_hours": 24},
    )


def _generate_sdk_stubs(spec: PaperSpec, endpoints: List[Dict]) -> Dict[str, str]:
    """Generate SDK client stubs in Python, JavaScript, and cURL."""
    method_name = spec.method[:30].replace(" ", "_").lower()

    python_sdk = f'''"""Paper2Product Python SDK — {spec.method[:50]}"""
import requests
from typing import Any, Dict, List, Optional


class Paper2ProductClient:
    def __init__(self, base_url: str = "http://localhost:8080", api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {{api_key}}"

    def predict(self, data: List[float], config: Optional[Dict] = None) -> Dict[str, Any]:
        """Run single inference."""
        payload = {{"data": data}}
        if config:
            payload["config"] = config
        resp = self.session.post(f"{{self.base_url}}/api/v1/predict", json=payload)
        resp.raise_for_status()
        return resp.json()

    def predict_batch(self, batch: List[List[float]]) -> Dict[str, Any]:
        """Run batch inference."""
        resp = self.session.post(f"{{self.base_url}}/api/v1/predict/batch", json={{"batch": batch}})
        resp.raise_for_status()
        return resp.json()

    def model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        resp = self.session.get(f"{{self.base_url}}/api/v1/model/info")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        """Health check."""
        resp = self.session.get(f"{{self.base_url}}/api/v1/health")
        resp.raise_for_status()
        return resp.json()


# Usage:
# client = Paper2ProductClient("http://localhost:8080", api_key="your-key")
# result = client.predict([1.0, 2.0, 3.0])
'''

    js_sdk = f'''// Paper2Product JavaScript SDK — {spec.method[:50]}

class Paper2ProductClient {{
  constructor(baseUrl = "http://localhost:8080", apiKey = "") {{
    this.baseUrl = baseUrl.replace(/\\/$/, "");
    this.apiKey = apiKey;
  }}

  async _fetch(path, options = {{}}) {{
    const headers = {{ "Content-Type": "application/json" }};
    if (this.apiKey) headers["Authorization"] = `Bearer ${{this.apiKey}}`;
    const resp = await fetch(`${{this.baseUrl}}${{path}}`, {{ ...options, headers }});
    if (!resp.ok) throw new Error(`HTTP ${{resp.status}}: ${{resp.statusText}}`);
    return resp.json();
  }}

  async predict(data, config = null) {{
    const body = {{ data }};
    if (config) body.config = config;
    return this._fetch("/api/v1/predict", {{ method: "POST", body: JSON.stringify(body) }});
  }}

  async predictBatch(batch) {{
    return this._fetch("/api/v1/predict/batch", {{ method: "POST", body: JSON.stringify({{ batch }}) }});
  }}

  async modelInfo() {{
    return this._fetch("/api/v1/model/info");
  }}

  async health() {{
    return this._fetch("/api/v1/health");
  }}
}}

// Usage:
// const client = new Paper2ProductClient("http://localhost:8080", "your-key");
// const result = await client.predict([1.0, 2.0, 3.0]);
'''

    curl_examples = f'''# Paper2Product cURL Examples — {spec.method[:50]}

# Health check
curl -s http://localhost:8080/api/v1/health | python -m json.tool

# Model info
curl -s -H "Authorization: Bearer YOUR_KEY" \\
  http://localhost:8080/api/v1/model/info | python -m json.tool

# Single prediction
curl -s -X POST http://localhost:8080/api/v1/predict \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_KEY" \\
  -d '{{"data": [1.0, 2.0, 3.0]}}' | python -m json.tool

# Batch prediction
curl -s -X POST http://localhost:8080/api/v1/predict/batch \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_KEY" \\
  -d '{{"batch": [[1.0, 2.0], [3.0, 4.0]]}}' | python -m json.tool
'''

    return {
        "python": python_sdk,
        "javascript": js_sdk,
        "curl": curl_examples,
    }


# ---------------------------------------------------------------------------
# Deployment Planning
# ---------------------------------------------------------------------------
ARCHITECTURE_TEMPLATES = {
    "server": {
        "description": "Standard server-side inference with auto-scaling.",
        "components": ["Load Balancer", "API Gateway", "Inference Server", "Model Store", "Monitoring"],
        "estimated_monthly_cost": {"small": 50, "medium": 200, "large": 800},
        "pros": ["Centralized management", "Easy to update", "Full GPU utilization"],
        "cons": ["Network latency", "Requires server infrastructure", "Scaling costs"],
    },
    "edge": {
        "description": "On-device inference for low-latency applications.",
        "components": ["Model Converter", "ONNX Runtime", "Device SDK", "Telemetry"],
        "estimated_monthly_cost": {"small": 10, "medium": 30, "large": 100},
        "pros": ["Zero network latency", "Offline capability", "Privacy preserving"],
        "cons": ["Limited compute", "Update complexity", "Device fragmentation"],
    },
    "hybrid": {
        "description": "Client-side fast path with server fallback for complex requests.",
        "components": ["Edge Model", "Server Model", "Router", "Sync Service"],
        "estimated_monthly_cost": {"small": 40, "medium": 150, "large": 600},
        "pros": ["Best latency for common cases", "Handles complex requests", "Graceful degradation"],
        "cons": ["Complex architecture", "Two models to maintain", "Sync overhead"],
    },
    "mobile": {
        "description": "Mobile-optimized inference using TFLite/CoreML.",
        "components": ["Model Quantizer", "Mobile Runtime", "Background Sync", "Analytics"],
        "estimated_monthly_cost": {"small": 5, "medium": 20, "large": 80},
        "pros": ["Offline capable", "Low cost at scale", "Native performance"],
        "cons": ["Model size constraints", "Platform-specific optimizations", "Limited compute"],
    },
}


def generate_deployment_plan(
    spec: PaperSpec, architecture_type: str = "server"
) -> DeploymentPlan:
    """Generate a deployment plan with cost estimation."""
    template = ARCHITECTURE_TEMPLATES.get(architecture_type, ARCHITECTURE_TEMPLATES["server"])

    security_checklist = [
        {"item": "Input validation and sanitization", "status": "required", "priority": "high"},
        {"item": "Authentication (JWT/API key)", "status": "required", "priority": "high"},
        {"item": "Rate limiting", "status": "required", "priority": "high"},
        {"item": "HTTPS/TLS encryption", "status": "required", "priority": "high"},
        {"item": "PII detection and handling", "status": "recommended", "priority": "medium"},
        {"item": "Model abuse prevention", "status": "recommended", "priority": "medium"},
        {"item": "Prompt injection safeguards", "status": "conditional", "priority": "medium"},
        {"item": "Audit logging", "status": "recommended", "priority": "medium"},
        {"item": "Data retention policy", "status": "required", "priority": "medium"},
        {"item": "CORS configuration", "status": "required", "priority": "low"},
        {"item": "Container security scanning", "status": "recommended", "priority": "low"},
    ]

    return DeploymentPlan(
        architecture_type=architecture_type,
        estimated_cost_monthly=template["estimated_monthly_cost"]["medium"],
        cloud_provider="aws",
        container_config={
            "base_image": "python:3.11-slim",
            "gpu_required": architecture_type == "server",
            "memory_limit": "4Gi" if architecture_type == "server" else "1Gi",
            "cpu_limit": "2" if architecture_type == "server" else "0.5",
            "replicas": {"min": 1, "max": 10},
        },
        scaling_config={
            "metric": "cpu_utilization",
            "target_percent": 70,
            "scale_up_cooldown": 60,
            "scale_down_cooldown": 300,
        },
        security_checklist=security_checklist,
    )


# ---------------------------------------------------------------------------
# Launch Package
# ---------------------------------------------------------------------------
def generate_launch_package(
    spec: PaperSpec,
    api_contract: APIContract,
    deployment: DeploymentPlan,
) -> LaunchPackage:
    """Generate a launch package for staging and release."""
    return LaunchPackage(
        web_deploy={
            "platform": "Docker + Kubernetes",
            "registry": "ghcr.io/paper2product",
            "staging_url": "https://staging.paper2product.app",
            "production_url": "https://api.paper2product.app",
            "deployment_steps": [
                "Build Docker image",
                "Push to container registry",
                "Deploy to staging cluster",
                "Run integration tests",
                "Promote to production",
            ],
        },
        mobile_release={
            "platform": "Android (Flutter)",
            "package_name": "com.paper2product.app",
            "min_sdk": 21,
            "target_sdk": 34,
            "features": [
                "Browse projects",
                "View visual maps",
                "Trigger artifact builds",
                "Review confidence reports",
                "Approve releases",
            ],
            "integrations": [
                "Firebase Auth",
                "Firebase Crashlytics",
                "Firebase Analytics",
                "Push Notifications (FCM)",
            ],
        },
        staging_config={
            "environment": "staging",
            "auto_deploy": True,
            "branch": "staging",
            "health_check_interval": 30,
            "rollback_on_failure": True,
        },
        release_checklist=[
            {"item": "All tests passing", "category": "quality", "required": True},
            {"item": "API contract validated", "category": "api", "required": True},
            {"item": "Security checklist complete", "category": "security", "required": True},
            {"item": "Performance benchmarks met", "category": "performance", "required": True},
            {"item": "Documentation updated", "category": "docs", "required": True},
            {"item": "Privacy policy reviewed", "category": "legal", "required": True},
            {"item": "Staging smoke tests passed", "category": "quality", "required": True},
            {"item": "Monitoring and alerting configured", "category": "ops", "required": True},
            {"item": "Rollback plan documented", "category": "ops", "required": False},
            {"item": "Load testing completed", "category": "performance", "required": False},
        ],
        telemetry_config={
            "provider": "OpenTelemetry",
            "metrics": ["request_count", "latency_p50", "latency_p99", "error_rate", "model_confidence"],
            "traces": True,
            "logs": True,
            "sampling_rate": 0.1,
        },
    )
