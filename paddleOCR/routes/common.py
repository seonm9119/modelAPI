import base64
import binascii
import tempfile
from io import BytesIO
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

def convert_byte_to_img(byte_img, suffix=".png"):
    if not byte_img:
        raise HTTPException(status_code=400, detail="byte_img is required")

    try:
        image_bytes = base64.b64decode(byte_img, validate=True)
    except (binascii.Error, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="byte_img must be a base64 encoded image")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise HTTPException(status_code=400, detail="byte_img must be a valid image")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as image_file:
        image_path = image_file.name
        image_file.write(image_bytes)

    return image_path
