# syntax=docker/dockerfile:1.6
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System libraries required by EasyOCR (OpenCV bindings) + curl for healthchecks.
# Python 3.12 and torch 2.11.0+cu128 (with sm_120 support) are already in the base.
# The bot itself is CPU-only — GPU passthrough is attached only to tg_ollama in docker-compose.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        build-essential \
        curl \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# --break-system-packages: pytorch/pytorch:2.11.0 base is Ubuntu 24.04 with PEP 668 "externally
# managed" Python. In a disposable container this is safe; the flag lets pip install alongside
# the pre-bundled torch+cu128 without a separate venv layer.
RUN pip install --no-cache-dir --break-system-packages -r /app/requirements.txt

# Pre-download EasyOCR models (~80 MB) so:
#   (a) the first runtime call doesn't pause for ~30 s downloading from craft-ocr.com;
#   (b) tests in containers without internet still pass.
# Models land in /root/.EasyOCR/ inside this layer.
RUN python -c "import easyocr; easyocr.Reader(['ru', 'en'], gpu=False); print('EasyOCR models cached')"

# Pre-download BGE-m3 (~2.2 GB) — used by services.embeddings + services.chunker (tokenizer).
# Same rationale as EasyOCR: fast cold-start, offline-friendly tests.
# Lands in /root/.cache/huggingface/hub/models--BAAI--bge-m3/
RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, devices='cpu'); print('BGE-m3 cached')"

COPY . /app

CMD ["python", "-u", "bot.py"]
