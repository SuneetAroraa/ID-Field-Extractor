import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO

from detection import detect_and_crop
from ocr import get_reader, ocr

app = FastAPI()

model = None
reader = None

@app.on_event("startup")
def load_models():
    global model, reader
    model = YOLO("yolov8n.pt")
    reader = get_reader()

@app.post("/extract-id")
async def extract_id(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        cropped = detect_and_crop(model, tmp_path)
        if cropped is None:
            raise HTTPException(status_code=422, detail="No ID document detected")

        result = ocr(cropped, reader)
    finally:
        os.remove(tmp_path)

    return JSONResponse(content={
    "fields": {
        "name": result.name,
        "id_number": result.id_number,
        "date_of_birth": result.date_of_birth,
        "date_of_issue": result.date_of_issue,
        "expiry_date": result.expiry_date,
    },
    "confidence_scores": result.confidence_scores,
    "raw_text": result.raw_text,
})