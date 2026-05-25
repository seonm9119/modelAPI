TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_server_det"
TEXT_DETECTION_MODEL_DIR = "/models/pretrained/PP-OCRv5_server_det_infer"
TEXT_RECOGNITION_MODEL_NAME = "korean_PP-OCRv5_mobile_rec"
TEXT_RECOGNITION_MODEL_DIR = "/models/pretrained/korean_PP-OCRv5_mobile_rec_infer"

USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False

PADDLEOCR_DEVICE = "gpu:0"

PREDICT_OPTION_NAMES = [
    "use_doc_orientation_classify",
    "use_doc_unwarping",
    "use_textline_orientation",
    "text_det_limit_side_len",
    "text_det_limit_type",
    "text_det_thresh",
    "text_det_box_thresh",
    "text_det_unclip_ratio",
    "text_rec_score_thresh",
    "return_word_box",
]
