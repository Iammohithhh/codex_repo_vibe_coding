"""Team Collaboration Service — Shared workspaces, reviews, approvals.

Supports:
- Workspace creation and membership
- Review comments on artifacts
- Approval workflows with status tracking
- Change diffing between project versions
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..models.schema import (
    Project,
    ProjectStatus,
    ReviewComment,
    ReviewStatus,
    Workspace,
)
from . import persistence as db


# ---------------------------------------------------------------------------
# Workspace Management
# ---------------------------------------------------------------------------
def create_workspace(name: str, members: Optional[List[str]] = None) -> Workspace:
    """Create a new shared workspace."""
    workspace = Workspace(
        name=name,
        members=members or ["owner"],
    )
    db.save_workspace(workspace)
    return workspace


def add_member(workspace_id: str, member: str) -> Dict:
    """Add a member to a workspace."""
    ws_data = db.get_workspace(workspace_id)
    if not ws_data:
        return {"error": "Workspace not found"}
    if member not in ws_data.get("members", []):
        ws_data["members"].append(member)
        ws = Workspace(**ws_data)
        db.save_workspace(ws)
    return ws_data


def list_workspace_projects(workspace_id: str) -> List[Dict]:
    """List all projects in a workspace."""
    return db.list_projects(workspace_id=workspace_id)


# ---------------------------------------------------------------------------
# Review & Approval Workflows
# ---------------------------------------------------------------------------
def add_review(
    project_id: str,
    author: str,
    content: str,
    artifact_path: Optional[str] = None,
    line_number: Optional[int] = None,
) -> ReviewComment:
    """Add a review comment to a project."""
    review = ReviewComment(
        author=author,
        content=content,
        artifact_path=artifact_path,
        line_number=line_number,
        status=ReviewStatus.PENDING,
    )
    db.save_review(project_id, review)
    return review


def update_review_status(
    project_id: str,
    review_id: str,
    status: str,
) -> Dict:
    """Update the status of a review."""
    reviews = db.list_reviews(project_id)
    for r in reviews:
        if r.get("id") == review_id:
            r["status"] = status
            review = ReviewComment(**{k: v for k, v in r.items() if k in ReviewComment.__dataclass_fields__})
            review.status = ReviewStatus(status)
            db.save_review(project_id, review)
            return r
    return {"error": "Review not found"}


def get_reviews(project_id: str) -> List[Dict]:
    """Get all reviews for a project."""
    return db.list_reviews(project_id)


def approve_project(project_id: str, approver: str) -> Dict:
    """Mark a project as approved for production export."""
    project_data = db.get_project(project_id)
    if not project_data:
        return {"error": "Project not found"}

    # Check for unresolved reviews
    reviews = db.list_reviews(project_id)
    pending = [r for r in reviews if r.get("status") == "pending"]
    rejected = [r for r in reviews if r.get("status") == "rejected"]

    if rejected:
        return {
            "error": "Cannot approve: rejected reviews exist",
            "rejected_count": len(rejected),
        }

    return {
        "project_id": project_id,
        "status": "approved",
        "approver": approver,
        "pending_reviews": len(pending),
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# Change Diffing
# ---------------------------------------------------------------------------
def diff_projects(project_a: Dict, project_b: Dict) -> Dict[str, Any]:
    """Compare two project versions and return differences."""
    changes: Dict[str, Any] = {
        "added": [],
        "removed": [],
        "modified": [],
    }

    spec_a = project_a.get("paper_spec", {})
    spec_b = project_b.get("paper_spec", {})

    # Compare key fields
    for field in ["problem", "method", "datasets", "metrics", "key_equations"]:
        val_a = spec_a.get(field)
        val_b = spec_b.get(field)
        if val_a != val_b:
            changes["modified"].append({
                "field": field,
                "before": val_a,
                "after": val_b,
            })

    # Compare code scaffold files
    scaffold_a = project_a.get("code_scaffold", {})
    scaffold_b = project_b.get("code_scaffold", {})
    files_a = {f.get("path", ""): f for f in scaffold_a.get("files", [])}
    files_b = {f.get("path", ""): f for f in scaffold_b.get("files", [])}

    for path in set(files_b.keys()) - set(files_a.keys()):
        changes["added"].append({"file": path})
    for path in set(files_a.keys()) - set(files_b.keys()):
        changes["removed"].append({"file": path})
    for path in set(files_a.keys()) & set(files_b.keys()):
        if files_a[path].get("content") != files_b[path].get("content"):
            changes["modified"].append({"file": path, "type": "content_changed"})

    # Compare confidence scores
    conf_a = project_a.get("confidence_score", 0)
    conf_b = project_b.get("confidence_score", 0)
    if conf_a != conf_b:
        changes["modified"].append({
            "field": "confidence_score",
            "before": conf_a,
            "after": conf_b,
            "delta": round(conf_b - conf_a, 4),
        })

    return changes
