ARG BASE_IMAGE=pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLEOCR_HOME=/models/pretrained \
    PADDLEOCR_PRETRAINED_DIR=/models/pretrained \
    PADDLE_PDX_MODEL_SOURCE=BOS

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# GPU runtime by default. This image is intended to run with NVIDIA Container Toolkit.
ARG PADDLE_PACKAGE=paddlepaddle-gpu==3.3.1
ARG PADDLE_INDEX_URL=https://www.paddlepaddle.org.cn/packages/stable/cu118/
ARG PADDLEOCR_VERSION=3.5.0

RUN if [ -n "$PADDLE_INDEX_URL" ]; then \
        python -m pip install "$PADDLE_PACKAGE" -i "$PADDLE_INDEX_URL"; \
    else \
        python -m pip install "$PADDLE_PACKAGE"; \
    fi \
    && python -m pip install \
        "paddleocr==${PADDLEOCR_VERSION}" \
        fastapi \
        uvicorn[standard] \
        python-multipart \
        pillow

COPY download_ppocrv5_models.sh /app/download_ppocrv5_models.sh
COPY app.py config.py /app/
COPY engine/ /app/engine/
COPY routes/ /app/routes/
RUN chmod +x /app/download_ppocrv5_models.sh

EXPOSE 8001

CMD ["bash", "-lc", "/app/download_ppocrv5_models.sh && python -m uvicorn app:app --host 0.0.0.0 --port 8001"]
