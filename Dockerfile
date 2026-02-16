# =============================================================================
# Paper2Product 2.0 — Production Dockerfile
# =============================================================================
# Single-process Python server that serves both the API and the SPA frontend.
# No Node.js runtime needed — the SPA (frontend/index.html) is served directly
# by the Python backend. This avoids dual-process issues on PaaS platforms
# (Railway, Render, Fly) that only route to ONE port.
# =============================================================================

FROM python:3.11-slim AS production

# Labels
LABEL maintainer="paper2product-team"
LABEL version="2.0.0"
LABEL description="Paper2Product 2.0 — AI Research OS"

# Security: run as non-root
RUN groupadd -r p2p && useradd -r -g p2p -d /app -s /sbin/nologin p2p

WORKDIR /app

# Copy backend (zero external deps, stdlib only)
COPY paper2product/ ./paper2product/

# Copy SPA frontend (served by Python backend at /)
COPY frontend/index.html ./frontend/index.html

# Copy test files
COPY tests/ ./tests/

# Data directory for SQLite
RUN mkdir -p /app/data && chown -R p2p:p2p /app

# Environment
ENV PYTHONUNBUFFERED=1
ENV P2P_DB_PATH=/app/data/paper2product.db
ENV P2P_SECRET=change-me-in-production
ENV P2P_HOST=0.0.0.0
ENV GROQ_API_KEY=""
ENV GROQ_MODEL=llama-3.3-70b-versatile

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/openapi.json')" || exit 1

USER p2p

# Single process: Python serves API + SPA on one port
# Uses PORT env from platform (Railway/Render/Fly) or defaults to 8000
CMD ["sh", "-c", "python -m paper2product --port=${PORT:-8000}"]
