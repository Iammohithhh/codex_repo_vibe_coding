# =============================================================================
# Paper2Product 2.0 — Multi-stage Production Dockerfile
# =============================================================================
# Stage 1: Build Next.js frontend
# Stage 2: Python backend + built frontend
# =============================================================================

# -- Stage 1: Build frontend --------------------------------------------------
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --production=false
COPY frontend/ ./
RUN npm run build && mkdir -p /app/frontend/public

# -- Stage 2: Production image ------------------------------------------------
FROM python:3.11-slim AS production

# Labels
LABEL maintainer="paper2product-team"
LABEL version="2.0.0"
LABEL description="Paper2Product 2.0 — AI Research OS"

# Install Node.js runtime (needed to serve the Next.js standalone build)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Security: run as non-root
RUN groupadd -r p2p && useradd -r -g p2p -d /app -s /sbin/nologin p2p

WORKDIR /app

# Copy backend (zero external deps, stdlib only)
COPY paper2product/ ./paper2product/

# Copy built frontend
COPY --from=frontend-builder /app/frontend/.next/standalone ./frontend-standalone/
COPY --from=frontend-builder /app/frontend/.next/static ./frontend-standalone/.next/static/
COPY --from=frontend-builder /app/frontend/public ./frontend-standalone/public/

# Copy other project files
COPY frontend/index.html ./frontend/index.html
COPY tests/ ./tests/

# Data directory for SQLite
RUN mkdir -p /app/data && chown -R p2p:p2p /app

# Environment
ENV PYTHONUNBUFFERED=1
ENV P2P_DB_PATH=/app/data/paper2product.db
ENV P2P_SECRET=change-me-in-production
ENV P2P_HOST=0.0.0.0
ENV P2P_PORT=8000
ENV GROQ_API_KEY=""
ENV GROQ_MODEL=llama-3.3-70b-versatile

EXPOSE 8000 3000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/openapi.json')" || exit 1

USER p2p

# Start both backend and frontend
# HOSTNAME=0.0.0.0 forces Next.js to bind all interfaces (not container hostname)
# PORT=3000 prevents Next.js from stealing the platform's PORT env var
CMD ["sh", "-c", "python -m paper2product --port=${P2P_PORT} & HOSTNAME=0.0.0.0 PORT=3000 node frontend-standalone/server.js & wait"]
