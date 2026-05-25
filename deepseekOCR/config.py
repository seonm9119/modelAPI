import os

DEEPSEEK_OCR_MODEL_NAME = os.environ.get("DEEPSEEK_OCR_MODEL_NAME", "deepseek-ai/DeepSeek-OCR-2")
DEEPSEEK_OCR_MODEL_CACHE_DIR = os.environ.get("DEEPSEEK_OCR_MODEL_CACHE_DIR", os.environ.get("HF_HOME", "/models/pretrained"))
DEEPSEEK_OCR_OUTPUT_DIR = os.environ.get("DEEPSEEK_OCR_OUTPUT_DIR", "/models/outputs")

DEEPSEEK_OCR_DEVICE = os.environ.get("DEEPSEEK_OCR_DEVICE", "cuda")
DEEPSEEK_OCR_TORCH_DTYPE = os.environ.get("DEEPSEEK_OCR_TORCH_DTYPE", "bfloat16")
DEEPSEEK_OCR_ATTN_IMPLEMENTATION = os.environ.get("DEEPSEEK_OCR_ATTN_IMPLEMENTATION", "flash_attention_2")
DEEPSEEK_OCR_MAX_NEW_TOKENS = int(os.environ.get("DEEPSEEK_OCR_MAX_NEW_TOKENS", "8192"))
DEEPSEEK_OCR_USE_CACHE = os.environ.get("DEEPSEEK_OCR_USE_CACHE", "true").lower() == "true"
DEEPSEEK_OCR_RELEASE_AFTER_REQUEST = os.environ.get("DEEPSEEK_OCR_RELEASE_AFTER_REQUEST", "true").lower() == "true"

DEFAULT_PROMPT = os.environ.get("DEEPSEEK_OCR_PROMPT", "<image>\n<|grounding|>Convert the document to markdown. ")
BASE_SIZE = int(os.environ.get("DEEPSEEK_OCR_BASE_SIZE", "1024"))
IMAGE_SIZE = int(os.environ.get("DEEPSEEK_OCR_IMAGE_SIZE", "768"))
CROP_MODE = os.environ.get("DEEPSEEK_OCR_CROP_MODE", "true").lower() == "true"
SAVE_RESULTS = os.environ.get("DEEPSEEK_OCR_SAVE_RESULTS", "false").lower() == "true"
KEEP_RESULTS = os.environ.get("DEEPSEEK_OCR_KEEP_RESULTS", "false").lower() == "true"

PREDICT_OPTION_NAMES = [
    "prompt",
    "base_size",
    "image_size",
    "crop_mode",
    "max_new_tokens",
    "use_cache",
    "save_results",
    "keep_results",
]
