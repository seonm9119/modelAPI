#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${DEEPSEEK_OCR_MODEL_NAME:-deepseek-ai/DeepSeek-OCR-2}"
MODEL_CACHE_DIR="${DEEPSEEK_OCR_MODEL_CACHE_DIR:-${HF_HOME:-/models/pretrained}}"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

model_name = os.environ.get("DEEPSEEK_OCR_MODEL_NAME", "deepseek-ai/DeepSeek-OCR-2")
model_cache_dir = os.environ.get("DEEPSEEK_OCR_MODEL_CACHE_DIR", os.environ.get("HF_HOME", "/models/pretrained"))

print(f"[DeepSeek-OCR-2] downloading or reusing {model_name} in {model_cache_dir}")
snapshot_download(repo_id=model_name, cache_dir=model_cache_dir, resume_download=True)
print(f"[DeepSeek-OCR-2] ready: {model_name}")
PY
