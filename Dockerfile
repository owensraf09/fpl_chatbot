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
# Disable the browser-open prompt and telemetry; configure the server for
# container environments (no CORS restrictions, accessible on all interfaces).
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────────
# Gives Docker / orchestrators a way to verify the app is running.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py"]