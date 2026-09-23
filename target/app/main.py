"""
main.py — FastAPI routes. No model logic lives here.
"""

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.inference import predictor
from app.schemas import ErrorResponse

app = FastAPI(
    title="CIFAR-10 Classifier",
    description="Adversarial-attack target: CNN trained on CIFAR-10.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Override 422 validation errors to match the shared error schema."""
    errors = exc.errors()
    detail = "; ".join(
        f"{' -> '.join(str(loc) for loc in e['loc'])}: {e['msg']}"
        for e in errors
    )
    return JSONResponse(
        status_code=422,
        content={"detail": detail, "type": "validation_error"},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Return all HTTP exceptions in the shared error schema."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "type": "http_error"},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all for any unhandled exception — returns a consistent 500."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

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
        except RuntimeError as exc:
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
