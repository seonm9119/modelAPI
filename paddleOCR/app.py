from fastapi import FastAPI
from routes.inference import router as inference_router

app = FastAPI(title="PP-OCRv5 API")

app.include_router(inference_router)


@app.get("/health")
def health():
    return {"status": "ok"}
