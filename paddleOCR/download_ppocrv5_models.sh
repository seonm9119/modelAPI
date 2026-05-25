#!/usr/bin/env bash
set -euo pipefail

PRETRAINED_DIR="${PADDLEOCR_PRETRAINED_DIR:-/models/pretrained}"
DET_MODEL_DIR="${TEXT_DETECTION_MODEL_DIR:-${PRETRAINED_DIR}/PP-OCRv5_server_det_infer}"
REC_MODEL_DIR="${TEXT_RECOGNITION_MODEL_DIR:-${PRETRAINED_DIR}/korean_PP-OCRv5_mobile_rec_infer}"

DET_MODEL_URL="${TEXT_DETECTION_MODEL_URL:-https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar}"
REC_MODEL_URL="${TEXT_RECOGNITION_MODEL_URL:-https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/korean_PP-OCRv5_mobile_rec_infer.tar}"

download_model() {
    local model_name="$1"
    local model_url="$2"
    local model_dir="$3"
    local tar_path="/tmp/${model_name}.tar"

    if [ -d "$model_dir" ] && [ "$(find "$model_dir" -mindepth 1 -print -quit)" ]; then
        echo "[PP-OCRv5] ${model_name} already exists: ${model_dir}"
        return
    fi

    echo "[PP-OCRv5] downloading ${model_name}"
    mkdir -p "$PRETRAINED_DIR"
    curl -fL "$model_url" -o "$tar_path"
    tar -xf "$tar_path" -C "$PRETRAINED_DIR"
    rm -f "$tar_path"
    echo "[PP-OCRv5] ready: ${model_dir}"
}

download_model "PP-OCRv5_server_det_infer" "$DET_MODEL_URL" "$DET_MODEL_DIR"
download_model "korean_PP-OCRv5_mobile_rec_infer" "$REC_MODEL_URL" "$REC_MODEL_DIR"
