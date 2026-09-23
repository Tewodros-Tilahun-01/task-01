"""
main.py — FastAPI routes. No model logic lives here.
"""

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.inference import predictor

app = FastAPI(
    title="CIFAR-10 Classifier",
    description="Adversarial-attack target: CNN trained on CIFAR-10.",
    version="1.0.0",
)


def _require_model() -> None:
    if not predictor.ready:
        raise HTTPException(status_code=503, detail="Model not loaded — run train.py first")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    """Liveness check."""
    return {"status": "ok", "model_loaded": predictor.ready}


@app.get("/model/info")
def model_info():
    """Return architecture metadata saved by train.py."""
    _require_model()
    return predictor.info


@app.post("/predict")
async def predict(file: UploadFile = File(default=None), request: Request = None):
    """
    Classify a single CIFAR-10 image.

    Accepts either:
    - multipart/form-data with field `file`
    - raw bytes (Content-Type: application/octet-stream)

    Returns
    -------
    {
        "predicted_class": 3,
        "class_name": "cat",
        "confidence": 0.923,
        "probabilities": {"0": ..., ..., "9": ...}
    }
    """
    _require_model()

    raw = await file.read() if file else await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="No image data received")

    try:
        result = predictor.predict(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return JSONResponse(content={
        "predicted_class": result.predicted_class,
        "class_name":      result.class_name,
        "confidence":      result.confidence,
        "probabilities":   result.probabilities,
    })


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Classify multiple images in one request."""
    _require_model()

    predictions = []
    for f in files:
        raw = await f.read()
        try:
            result = predictor.predict(raw)
        except ValueError as exc:
            predictions.append({"filename": f.filename, "error": str(exc)})
            continue

        predictions.append({
            "filename":        f.filename,
            "predicted_class": result.predicted_class,
            "class_name":      result.class_name,
            "confidence":      result.confidence,
            "probabilities":   result.probabilities,
        })

    return JSONResponse(content={"predictions": predictions})
