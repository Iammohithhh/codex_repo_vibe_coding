"""Authentication service — JWT tokens + API key management.

Provides:
- API key generation and validation
- JWT token creation and verification
- Middleware helper for the HTTP server
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode

# Secret key — override via P2P_SECRET env var in production
_SECRET = os.environ.get("P2P_SECRET", "paper2product-dev-secret-change-me")

# In-memory API key store (in production, persist to DB)
_api_keys: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# API Key Management
# ---------------------------------------------------------------------------

def generate_api_key(owner: str = "default", scopes: list[str] | None = None) -> dict:
    """Generate a new API key."""
    raw_key = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    record = {
        "key_hash": key_hash,
        "owner": owner,
        "scopes": scopes or ["read", "write"],
        "created_at": time.time(),
        "active": True,
    }
    _api_keys[key_hash] = record
    return {"api_key": raw_key, "owner": owner, "scopes": record["scopes"]}


def validate_api_key(raw_key: str) -> dict | None:
    """Validate an API key, return the record if valid."""
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    record = _api_keys.get(key_hash)
    if record and record.get("active"):
        return record
    return None


def revoke_api_key(raw_key: str) -> bool:
    """Revoke an API key."""
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    if key_hash in _api_keys:
        _api_keys[key_hash]["active"] = False
        return True
    return False


# ---------------------------------------------------------------------------
# JWT Implementation (stdlib-only, HMAC-SHA256)
# ---------------------------------------------------------------------------

def _b64encode(data: bytes) -> str:
    return urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return urlsafe_b64decode(s)


def create_jwt(payload: dict, expires_in: int = 86400) -> str:
    """Create a JWT token with HMAC-SHA256 signature."""
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {**payload, "iat": now, "exp": now + expires_in}

    segments = [
        _b64encode(json.dumps(header).encode()),
        _b64encode(json.dumps(payload, default=str).encode()),
    ]
    signing_input = ".".join(segments).encode()
    signature = hmac.new(_SECRET.encode(), signing_input, hashlib.sha256).digest()
    segments.append(_b64encode(signature))
    return ".".join(segments)


def verify_jwt(token: str) -> dict | None:
    """Verify a JWT token. Returns payload dict or None if invalid."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        signing_input = f"{parts[0]}.{parts[1]}".encode()
        expected_sig = hmac.new(_SECRET.encode(), signing_input, hashlib.sha256).digest()
        actual_sig = _b64decode(parts[2])

        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        payload = json.loads(_b64decode(parts[1]))

        # Check expiration
        if payload.get("exp", 0) < time.time():
            return None

        return payload
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Auth middleware helper
# ---------------------------------------------------------------------------

def authenticate(headers: dict) -> dict | None:
    """Authenticate a request from headers.

    Supports:
    - Bearer JWT token: Authorization: Bearer <jwt>
    - API key: X-API-Key: <key>

    Returns user info dict or None if unauthenticated.
    """
    # Check JWT Bearer token
    auth_header = headers.get("Authorization") or headers.get("authorization") or ""
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        payload = verify_jwt(token)
        if payload:
            return {"type": "jwt", "user": payload.get("sub", "unknown"), "scopes": payload.get("scopes", ["read", "write"])}

    # Check API key
    api_key = headers.get("X-API-Key") or headers.get("x-api-key") or ""
    if api_key:
        record = validate_api_key(api_key)
        if record:
            return {"type": "api_key", "user": record["owner"], "scopes": record["scopes"]}

    return None


# ---------------------------------------------------------------------------
# Login endpoint helper
# ---------------------------------------------------------------------------

def login(username: str, password: str) -> dict | None:
    """Simple login — in production, check against a user database.

    Default dev credentials: admin/admin123
    Override via P2P_ADMIN_USER and P2P_ADMIN_PASS env vars.
    """
    expected_user = os.environ.get("P2P_ADMIN_USER", "admin")
    expected_pass = os.environ.get("P2P_ADMIN_PASS", "admin123")

    if username == expected_user and password == expected_pass:
        token = create_jwt({"sub": username, "role": "admin", "scopes": ["read", "write", "admin"]})
        return {"token": token, "user": username, "role": "admin"}
    return None
