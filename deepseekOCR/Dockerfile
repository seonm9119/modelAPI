ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/pretrained \
    DEEPSEEK_OCR_MODEL_CACHE_DIR=/models/pretrained \
    DEEPSEEK_OCR_OUTPUT_DIR=/models/outputs \
    DEEPSEEK_OCR_MODEL_NAME=deepseek-ai/DeepSeek-OCR-2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

ARG TRANSFORMERS_VERSION=4.46.3
ARG TOKENIZERS_VERSION=0.20.3
ARG FLASH_ATTN_VERSION=2.7.3
ARG INSTALL_FLASH_ATTN=1

RUN python -m pip install --upgrade pip setuptools wheel packaging ninja \
    && python -m pip install \
        "transformers==${TRANSFORMERS_VERSION}" \
        "tokenizers==${TOKENIZERS_VERSION}" \
        PyMuPDF \
        img2pdf \
        einops \
        easydict \
        addict \
        Pillow \
        numpy \
        fastapi \
        uvicorn[standard] \
        python-multipart \
        huggingface_hub \
        requests \
        tqdm \
        matplotlib \
    && if [ "$INSTALL_FLASH_ATTN" = "1" ]; then \
        python -m pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation; \
    fi

COPY download_deepseek_ocr2_model.sh /app/download_deepseek_ocr2_model.sh
COPY app.py config.py /app/
COPY engine/ /app/engine/
COPY routes/ /app/routes/
RUN chmod +x /app/download_deepseek_ocr2_model.sh

EXPOSE 8002

CMD ["bash", "-lc", "/app/download_deepseek_ocr2_model.sh && python -m uvicorn app:app --host 0.0.0.0 --port 8002"]
