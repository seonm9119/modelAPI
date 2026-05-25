import os
from fastapi import APIRouter, HTTPException, Request
from config import DEEPSEEK_OCR_RELEASE_AFTER_REQUEST
from engine.deepseek_ocr_v2 import inference, release_ocr_model
from routes.common import convert_byte_to_img

router = APIRouter()


# Request format:
# {
#     "byte_img": "base64_encoded_image_bytes",
#     "predict_options": {
#         # Optional. DeepSeek-OCR-2 prompt controls the output style.
#         # Use the markdown prompt for document parsing, or "Free OCR." for plain OCR.
#         "prompt": "<image>\n<|grounding|>Convert the document to markdown. ",
#
#         # Optional. Official DeepSeek-OCR-2 default mode:
#         # (0-6)x768x768 local crops + 1x1024x1024 global view.
#         "base_size": 1024,
#         "image_size": 768,
#         "crop_mode": true,
#
#         # Optional. false returns the generated text directly from model.infer(eval_mode=True).
#         # true asks the upstream model code to write result.mmd/result_with_boxes.jpg first,
#         # then this API reads result.mmd back as text.
#         "save_results": false,
#
#         # Optional. Keep generated result files under /models/outputs and return output_path.
#         "keep_results": false
#     }
# }

@router.post("/inference")
async def run_deepseek_ocr_inference(request: Request):
    request_data = await request.json()
    predict_options = request_data.get("predict_options", {})
    image_input = convert_byte_to_img(request_data.get("byte_img"))

    try:
        return inference(image_input, predict_options)
    except RuntimeError as runtime_error:
        raise HTTPException(status_code=500, detail=str(runtime_error))
    finally:
        if DEEPSEEK_OCR_RELEASE_AFTER_REQUEST:
            release_ocr_model()
        if os.path.exists(image_input):
            os.remove(image_input)


@router.post("/release")
def release_deepseek_ocr_inference_model():
    release_ocr_model()
    return {"success": True}


# Response example:
# {
#     "model": "deepseek-ai/DeepSeek-OCR-2",
#     "prompt": "<image>\n<|grounding|>Convert the document to markdown. ",
#     "text": "# Invoice\n\nInvoice No: INV-2026-001\n\nTotal: 125,000 KRW",
#     "output_path": null
# }
