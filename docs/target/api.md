# Target — REST API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

---

## Endpoints

### `GET /`

Health check. Returns whether the server is up and the model is loaded.

**Response `200 OK`**

```json
{
  "status": "ok",
  "model_loaded": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"ok"` if the server is running |
| `model_loaded` | boolean | `false` if weights were not found at startup |

---

### `GET /model/info`

Returns architecture metadata saved by the training script.

**Response `200 OK`**

```json
{
  "architecture": "CifarCNN",
  "num_classes": 10,
  "input_shape": [3, 32, 32],
  "normalize": {
    "mean": [0.4914, 0.4822, 0.4465],
    "std":  [0.2470, 0.2435, 0.2616]
  },
  "best_val_acc": 0.8044,
  "test_acc":     0.8026
}
```

**Error responses**

| Status | Condition |
|--------|-----------|
| `503 Service Unavailable` | Model not loaded |

---

### `POST /predict`

Classify a single image. Accepts two input formats:

**Option A — multipart/form-data**

```
Content-Type: multipart/form-data
Field: file  (image file)
```

**Linux / macOS**
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@image.png"
```

**Windows (PowerShell)**
```powershell
curl.exe -X POST http://localhost:8000/predict `
         -F "file=@C:\path\to\image.png"
```

**Option B — raw bytes**

```
Content-Type: application/octet-stream
Body: raw image bytes
```

**Linux / macOS**
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/octet-stream" \
     --data-binary @image.png
```

**Windows (PowerShell)**
```powershell
curl.exe -X POST http://localhost:8000/predict `
         -H "Content-Type: application/octet-stream" `
         --data-binary "@C:\path\to\image.png"
```


| Field | Type | Description |
|-------|------|-------------|
| `predicted_class` | int | Index of the predicted class (0–9) |
| `class_name` | string | Human-readable class label |
| `confidence` | float | Softmax probability of the predicted class (6 d.p.) |
| `probabilities` | object | Softmax probability for all 10 classes, keyed by index string |

**Error responses**

| Status | Condition |
|--------|-----------|
| `400 Bad Request` | No image data received, or bytes cannot be decoded as an image |
| `422 Unprocessable Entity` | Request body fails FastAPI field validation |
| `503 Service Unavailable` | Model not loaded |

---

### `POST /predict/batch`

Classify multiple images in a single request.

**Request**

```
Content-Type: multipart/form-data
Field: files  (one or more image files)
```

**Linux / macOS**
```bash
curl -X POST http://localhost:8000/predict/batch \
     -F "files=@image1.png" \
     -F "files=@image2.png"
```

**Windows (PowerShell)**
```powershell
curl.exe -X POST http://localhost:8000/predict/batch `
         -F "files=@C:\path\to\image1.png" `
         -F "files=@C:\path\to\image2.png"
```

**Response `200 OK`**

```json
{
  "predictions": [
    {
      "filename": "image1.png",
      "predicted_class": 3,
      "class_name": "cat",
      "confidence": 0.923100,
      "probabilities": { "0": 0.002100, "...": "..." }
    },
    {
      "filename": "image2.png",
      "error": "Cannot decode image: ..."
    }
  ]
}
```

Per-item errors are returned inline — a failed image does not abort the whole batch.

**Error responses**

| Status | Condition |
|--------|-----------|
| `422 Unprocessable Entity` | `files` field missing or fails FastAPI field validation |
| `503 Service Unavailable` | Model not loaded |

All error responses share a consistent shape:

```json
{
  "detail": "Human-readable error message",
  "type": "error_type_string"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `detail` | string | Description of the error |
| `type` | string | `"validation_error"`, `"http_error"`, or the exception class name for 500s |

---
