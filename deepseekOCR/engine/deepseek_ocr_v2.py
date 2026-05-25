import os
import shutil
import tempfile
from contextlib import contextmanager
import gc
from threading import Lock
import torch
from transformers import AutoModel, AutoTokenizer
from config import (
    BASE_SIZE,
    CROP_MODE,
    DEEPSEEK_OCR_ATTN_IMPLEMENTATION,
    DEEPSEEK_OCR_DEVICE,
    DEEPSEEK_OCR_MAX_NEW_TOKENS,
    DEEPSEEK_OCR_MODEL_CACHE_DIR,
    DEEPSEEK_OCR_MODEL_NAME,
    DEEPSEEK_OCR_OUTPUT_DIR,
    DEEPSEEK_OCR_TORCH_DTYPE,
    DEEPSEEK_OCR_USE_CACHE,
    DEFAULT_PROMPT,
    IMAGE_SIZE,
    KEEP_RESULTS,
    SAVE_RESULTS,
)

tokenizer = None
ocr_model = None
ocr_lock = Lock()


def parse_bool(option_value, default_value=False):
    if isinstance(option_value, bool):
        return option_value
    if isinstance(option_value, str):
        return option_value.lower() in ("1", "true", "yes", "y", "on")
    if option_value is None:
        return default_value
    return bool(option_value)


def get_torch_dtype():
    if DEEPSEEK_OCR_TORCH_DTYPE == "float16":
        return torch.float16
    if DEEPSEEK_OCR_TORCH_DTYPE == "float32":
        return torch.float32
    return torch.bfloat16


def get_ocr_model():
    global tokenizer, ocr_model

    if DEEPSEEK_OCR_DEVICE != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("DeepSeek-OCR-2 inference requires CUDA because the upstream infer() path uses CUDA tensors.")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_OCR_MODEL_NAME, trust_remote_code=True, cache_dir=DEEPSEEK_OCR_MODEL_CACHE_DIR)

    if ocr_model is None:
        model_options = {
            "trust_remote_code": True,
            "use_safetensors": True,
            "cache_dir": DEEPSEEK_OCR_MODEL_CACHE_DIR,
            "torch_dtype": get_torch_dtype(),
        }

        if DEEPSEEK_OCR_ATTN_IMPLEMENTATION:
            model_options["_attn_implementation"] = DEEPSEEK_OCR_ATTN_IMPLEMENTATION

        ocr_model = AutoModel.from_pretrained(DEEPSEEK_OCR_MODEL_NAME, **model_options)
        ocr_model = ocr_model.eval().to(device=DEEPSEEK_OCR_DEVICE, dtype=get_torch_dtype())

    return ocr_model, tokenizer


def release_ocr_model():
    global ocr_model

    with ocr_lock:
        loaded_model = ocr_model
        ocr_model = None

    if loaded_model is not None:
        del loaded_model

    gc.collect()
    release_cuda_cache()


@contextmanager
def limit_generation_options(model, max_new_tokens, use_cache):
    original_generate = model.generate

    def limited_generate(*args, **kwargs):
        kwargs["max_new_tokens"] = max_new_tokens
        kwargs["use_cache"] = use_cache
        return original_generate(*args, **kwargs)

    model.generate = limited_generate
    try:
        yield
    finally:
        model.generate = original_generate


def release_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def build_infer_options(options):
    if options is None:
        options = {}

    prompt = options.get("prompt", DEFAULT_PROMPT)
    if not prompt:
        prompt = DEFAULT_PROMPT

    return {
        "prompt": prompt,
        "base_size": int(options.get("base_size", BASE_SIZE)),
        "image_size": int(options.get("image_size", IMAGE_SIZE)),
        "crop_mode": parse_bool(options.get("crop_mode"), CROP_MODE),
        "max_new_tokens": max(1, int(options.get("max_new_tokens", DEEPSEEK_OCR_MAX_NEW_TOKENS))),
        "use_cache": parse_bool(options.get("use_cache"), DEEPSEEK_OCR_USE_CACHE),
        "save_results": parse_bool(options.get("save_results"), SAVE_RESULTS),
        "keep_results": parse_bool(options.get("keep_results"), KEEP_RESULTS),
    }


def read_saved_text(output_path):
    result_path = os.path.join(output_path, "result.mmd")
    if not os.path.exists(result_path):
        return ""

    with open(result_path, "r", encoding="utf-8") as result_file:
        return result_file.read()


def inference(image_input, options=None):
    infer_options = build_infer_options(options)
    os.makedirs(DEEPSEEK_OCR_OUTPUT_DIR, exist_ok=True)
    output_path = tempfile.mkdtemp(prefix="deepseek_ocr2_", dir=DEEPSEEK_OCR_OUTPUT_DIR)

    try:
        with ocr_lock:
            model, model_tokenizer = get_ocr_model()
            with limit_generation_options(model, infer_options["max_new_tokens"], infer_options["use_cache"]):
                if infer_options["save_results"]:
                    model.infer(
                        model_tokenizer,
                        prompt=infer_options["prompt"],
                        image_file=image_input,
                        output_path=output_path,
                        base_size=infer_options["base_size"],
                        image_size=infer_options["image_size"],
                        crop_mode=infer_options["crop_mode"],
                        save_results=True,
                    )
                    generated_text = read_saved_text(output_path)
                else:
                    generated_text = model.infer(
                        model_tokenizer,
                        prompt=infer_options["prompt"],
                        image_file=image_input,
                        output_path=output_path,
                        base_size=infer_options["base_size"],
                        image_size=infer_options["image_size"],
                        crop_mode=infer_options["crop_mode"],
                        save_results=False,
                        eval_mode=True,
                    )

        return {
            "model": DEEPSEEK_OCR_MODEL_NAME,
            "prompt": infer_options["prompt"],
            "text": generated_text or "",
            "output_path": output_path if infer_options["keep_results"] else None,
        }
    except RuntimeError:
        release_cuda_cache()
        raise
    finally:
        release_cuda_cache()
        if not infer_options["keep_results"]:
            shutil.rmtree(output_path, ignore_errors=True)
