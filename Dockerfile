# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.11 slim keeps the image small while supporting all required packages.
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# gcc is required by some langchain transitive deps that compile C extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker layer-caches the install step — rebuilds
# triggered by source-file changes won't re-download packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY . .

# ── Streamlit configuration ───────────────────────────────────────────────────
# Cloud Run injects PORT at runtime (typically 8080). Streamlit must bind to
# that port, not a hardcoded one. We default to 8080 locally so the image
# works without Cloud Run too.
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── Expose port ───────────────────────────────────────────────────────────────
# Cloud Run ignores EXPOSE but it documents intent for local use.
EXPOSE 8080

# ── Health check ──────────────────────────────────────────────────────────────
# Cloud Run has its own health checking; this is for local docker run only.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request, os; urllib.request.urlopen(f\"http://localhost:{os.environ.get('PORT', '8080')}/_stcore/health\")"

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Read $PORT at container start time — Cloud Run sets this before the CMD runs.
CMD streamlit run app.py --server.port=${PORT:-8080}