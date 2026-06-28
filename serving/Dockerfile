# Underwriting bridge image. Builds from the repo root so the engine files
# (quantum_cortex.py, fourier_optics.py) are available to the serving package.
#   docker build -f serving/Dockerfile -t quantum-cortex-uw:0.1 .
FROM python:3.11-slim

WORKDIR /app

# scikit-image / scipy need a few build/runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

COPY serving/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Engine (root) + serving package.
COPY quantum_cortex.py fourier_optics.py ./
COPY serving/ ./serving/

WORKDIR /app/serving
EXPOSE 8000

# /data is a mounted volume (PVC) holding the persisted cortex weights.
ENV CORTEX_SNAPSHOT_PATH=/data/cortex_state.npz

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
