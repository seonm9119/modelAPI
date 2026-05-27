import os
from fastapi import APIRouter, HTTPException, Request
from engine.paddle_ocr_v5 import inference, release_ocr_model
from routes.common import convert_byte_to_img

router = APIRouter()


def summarize_inference_error(error):
    error_text = str(error)
    if "Out of memory" in error_text or "ResourceExhausted" in error_text:
        return 507, "Paddle OCR GPU out of memory"

    if not error_text:
        return 500, "Paddle OCR inference failed"

    return 500, error_text[:1000]


# Request format:
# {
#     "byte_img": "base64_encoded_image_bytes",
#     "release_after_inference": true,
#     "predict_options": {
#         "use_doc_orientation_classify": false,
#         "use_doc_unwarping": false,
#         "use_textline_orientation": false,
#         "text_det_limit_side_len": 64,
#         "text_det_limit_type": "min",
#         "text_det_thresh": 0.3,
#         "text_det_box_thresh": 0.6,
#         "text_det_unclip_ratio": 1.5,
#         "text_rec_score_thresh": 0.0,
#         "return_word_box": false
#     }
# }

@router.post("/inference")
async def run_paddle_ocr_inference(request: Request):
    request_data = await request.json()
    predict_options = request_data.get("predict_options", {})
    release_after_inference = request_data.get("release_after_inference", True)
    image_input = convert_byte_to_img(request_data.get("byte_img"))
    inference_failed = False

    try:
        ocr_results = inference(image_input, predict_options)
        return [ocr_result.json for ocr_result in ocr_results]
    except Exception as error:
        inference_failed = True
        status_code, detail = summarize_inference_error(error)
        raise HTTPException(status_code=status_code, detail=detail) from None
    finally:
        if release_after_inference or inference_failed:
            release_ocr_model()
        if os.path.exists(image_input):
            os.remove(image_input)


@router.post("/release")
def release_paddle_ocr_inference_model():
    release_ocr_model()
    return {"success": True}


# Response example:
# [
#     {
#         "res": {
#             "input_path": "/tmp/tmpabcd.png",
#             "page_index": null,
#             "rec_texts": [
#                 "INVOICE",
#                 "Invoice No: INV-2026-001",
#                 "Total: 125,000 KRW"
#             ],
#             "rec_scores": [
#                 0.9821,
#                 0.9475,
#                 0.9362
#             ],
#             "dt_polys": [
#                 [[42, 31], [155, 31], [155, 59], [42, 59]],
#                 [[42, 92], [286, 92], [286, 118], [42, 118]],
#                 [[42, 138], [254, 138], [254, 164], [42, 164]]
#             ],
#             "rec_polys": [
#                 [[42, 31], [155, 31], [155, 59], [42, 59]],
#                 [[42, 92], [286, 92], [286, 118], [42, 118]],
#                 [[42, 138], [254, 138], [254, 164], [42, 164]]
#             ],
#             "rec_boxes": [
#                 [42, 31, 155, 59],
#                 [42, 92, 286, 118],
#                 [42, 138, 254, 164]
#             ],
#         }
#     }
# ]
