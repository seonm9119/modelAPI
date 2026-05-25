import gc
from threading import Lock

import paddle
from paddleocr import PaddleOCR
from config import (
    PADDLEOCR_DEVICE,
    PREDICT_OPTION_NAMES,
    TEXT_DETECTION_MODEL_DIR,
    TEXT_DETECTION_MODEL_NAME,
    TEXT_RECOGNITION_MODEL_DIR,
    TEXT_RECOGNITION_MODEL_NAME,
    USE_DOC_ORIENTATION_CLASSIFY,
    USE_DOC_UNWARPING,
    USE_TEXTLINE_ORIENTATION,
)

ocr_model = None
ocr_model_lock = Lock()


def get_ocr_model():
    global ocr_model

    if ocr_model is None:
        ocr_model = PaddleOCR(
            text_detection_model_name=TEXT_DETECTION_MODEL_NAME,
            text_detection_model_dir=TEXT_DETECTION_MODEL_DIR,
            text_recognition_model_name=TEXT_RECOGNITION_MODEL_NAME,
            text_recognition_model_dir=TEXT_RECOGNITION_MODEL_DIR,
            use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
            use_doc_unwarping=USE_DOC_UNWARPING,
            use_textline_orientation=USE_TEXTLINE_ORIENTATION,
            device=PADDLEOCR_DEVICE,
        )

    return ocr_model


def release_ocr_model():
    global ocr_model

    with ocr_model_lock:
        ocr_model = None
        gc.collect()

        if not PADDLEOCR_DEVICE.startswith("gpu"):
            return

        try:
            paddle.device.synchronize()
        except Exception:
            pass

        try:
            paddle.device.cuda.empty_cache()
        except Exception:
            pass


def inference(image_input, options=None):
    if options is None:
        options = {}

    predict_options = {}

    for option_name in PREDICT_OPTION_NAMES:
        if option_name in options:
            predict_options[option_name] = options[option_name]

    with ocr_model_lock:
        return get_ocr_model().predict(input=image_input, **predict_options)
