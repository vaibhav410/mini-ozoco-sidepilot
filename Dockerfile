# Mini OZOCO SidePilot -- production image.
#
# Lean by design: uses the Gemini embedding API (no PyTorch), so the
# image stays small and runs in ~512 MB of RAM. Tesseract is included
# so the OCR fallback for screen understanding works out of the box.

FROM python:3.12-slim

# System dependencies: tesseract for the Agent 4 OCR fallback,
# curl for the container health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so code changes don't bust this layer.
COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

COPY app ./app
COPY samples ./samples

# Run as a non-root user; writable dirs owned by it.
RUN useradd --create-home sidepilot \
    && mkdir -p uploads exports data \
    && chown -R sidepilot:sidepilot /app
USER sidepilot

ENV EMBEDDINGS_BACKEND=gemini \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
