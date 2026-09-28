# Pipeline Vulnerabilities

Infrastructure and deployment weaknesses identified in the Docker-based inference service.

---

## V-01: No Authentication or Authorization

**Severity:** CRITICAL  
**MITRE ATLAS:** AML.T0040 (ML Model Inference API Access)  
**NIST AI RMF:** GOVERN 2.1 — Roles, responsibilities, and lines of communication (accountability)

### Finding

The API exposes all endpoints without authentication:

```python
# target/app/main.py
@app.post("/predict")
async def predict(file: UploadFile = File(default=None), request: Request = None):
    _require_model()
    raw = await file.read() if file else await request.body()
    result = predictor.predict(raw)
    return JSONResponse(content={...})
```

**No checks for:**
- API keys
- OAuth tokens
- Client identity
- Rate limiting per user

### Impact

- **Model extraction:** Unlimited queries enable model stealing (10,000 queries in 105 seconds observed)
- **Resource abuse:** No cost accounting or query throttling
- **Reconnaissance:** Attackers can profile model behavior without attribution
- **Audit gap:** No logs tying queries to identities

### Evidence

- All 46,572 attack queries (FGSM, PGD, Square, Extraction) accepted without challenge
- No HTTP 401/403 responses observed
- No `Authorization` header required in any request

### Discovery Method

Manual inspection of `target/app/main.py` + automated endpoint enumeration

---

## V-02: Full Probability Distribution Exposure

**Severity:** HIGH  
**MITRE ATLAS:** AML.T0024 (Exfiltration via ML Inference API)  
**NIST AI RMF:** MEASURE 2.7 — AI system security and resilience are evaluated and documented

### Finding

API returns complete softmax distribution for all 10 classes:

```json
{
  "predicted_class": 3,
  "class_name": "cat",
  "confidence": 0.923100,
  "probabilities": {
    "0": 0.002100, "1": 0.001200, "2": 0.003400, "3": 0.923100,
    "4": 0.001000, "5": 0.062000, "6": 0.002100, "7": 0.004000,
    "8": 0.001000, "9": 0.001000
  }
}
```

**Why this matters:**

Full probability vectors leak more information than top-1 predictions:
1. **High-fidelity extraction:** Knowledge distillation works better with soft labels (all class probabilities)
2. **Attack gradient estimation:** Confidence distributions help estimate decision boundaries
3. **Ensemble behavior:** Reveals model uncertainty and second-best predictions

### Impact

- Model extraction agreement rate: 80.3% (high fidelity enabled by probability access)
- Black-box attacks improved: Square Attack uses probabilities to estimate score-based gradients
- No business justification for exposing full distribution to external clients

### Evidence

- `evidence/logs/model_extraction/extraction_summary.json` — 80.3% agreement achieved
- `evidence/logs/square/square_summary.json` — Attack uses probabilities for optimization

### Recommendation

Return only `predicted_class` and `confidence` (top-1 probability). Remove `probabilities` field.

---

## V-03: Verbose Model Metadata Exposure

**Severity:** MEDIUM  
**MITRE ATLAS:** AML.T0007 (Discover ML Artifacts)  
**NIST AI RMF:** MEASURE 2.7 — AI system security and resilience are evaluated and documented

### Finding

`/model/info` endpoint leaks training details:

```bash
$ curl http://localhost:8000/model/info
```

```json
{
  "architecture": "CifarCNN",
  "num_classes": 10,
  "input_shape": [3, 32, 32],
  "normalize": {
    "mean": [0.4914, 0.4822, 0.4465],
    "std": [0.247, 0.2435, 0.2616]
  },
  "best_val_acc": 0.8044,
  "test_acc": 0.8026
}
```

**Information disclosed:**
- Exact architecture name
- Input dimensions and normalization parameters
- Validation and test accuracy metrics

### Impact

- **Extraction efficiency:** Attacker knows optimal surrogate architecture (same input shape, class count, normalization parameters)
- **Transfer attack planning:** Test and validation accuracy metrics indicate model capacity and potential weak points
- **Normalization parameters:** Mean and std values enable attacker to preprocess data identically to victim model

### Evidence

Observed during reconnaissance phase—endpoint accessible without authentication.

### Recommendation

Remove endpoint from production deployment or gate behind authentication. Only expose minimal metadata required for client integration (e.g., input shape, class names).

---

## V-04: No Rate Limiting or Query Budget

**Severity:** CRITICAL  
**MITRE ATLAS:** AML.T0024.002 (Extract ML Model), AML.T0034 (Cost Harvesting)  
**NIST AI RMF:** MANAGE 1.3 — Responses to high-priority AI risks are developed, planned, and documented

### Finding

No query throttling implemented:

- Model extraction: **10,000 queries in 104.8 seconds** (95.4 queries/second)
- Square Attack: **46,572 total queries** across all budgets
- No HTTP 429 (Too Many Requests) responses observed
- No CAPTCHA or challenge-response mechanism

### Impact

- **Model extraction at scale:** Attacker can collect unlimited training data
- **Cost attack:** Compute resources consumed without accountability
- **Reconnaissance:** Unrestricted profiling of model behavior
- **DoS potential:** Malicious actors can exhaust GPU/CPU resources

### Evidence

```json
// evidence/logs/model_extraction/extraction_summary.json
{
  "queries_sent": 10000,
  "query_time_seconds": 104.8,
  "queries_per_second": 95.4
}
```

### Recommendation

Implement rate limiting:
- **All users:** 100 queries/minute

---

## V-05: No Adversarial Input Detection

**Severity:** HIGH  
**MITRE ATLAS:** AML.T0015 (Evade ML Model)  
**NIST AI RMF:** MEASURE 2.7 — AI system security and resilience are evaluated and documented

### Finding

API accepts adversarial inputs without detection:

- All 200 FGSM adversarial images accepted (91% successful misclassifications)
- All 200 PGD adversarial images accepted (100% successful at ε=8/255)
- No anomaly warnings in responses
- No rejection or flagging of suspicious inputs

**No checks for:**
- Statistical outliers (pixel value distributions)
- High-frequency noise patterns
- Adversarial perturbation signatures
- Ensemble prediction variance

### Impact

- **Silent evasion:** Adversarial attacks succeed without detection
- **No forensics:** No logs indicating suspicious inputs
- **Trust erosion:** Users cannot distinguish malicious from legitimate predictions

### Evidence

API responses for adversarial images identical in format to clean images—no status flags or warnings.

### Recommendation

Implement input anomaly detection:
1. **Statistical checks:** Flag images with unusual pixel distributions
2. **JPEG compression filter:** Recompress inputs to remove adversarial noise
3. **Ensemble variance:** Run multiple models/augmentations, flag high disagreement
4. **Logging:** Record suspected adversarial inputs for review

---

## V-06: Container Configuration Weaknesses

**Severity:** MEDIUM  
**MITRE ATLAS:** N/A (Infrastructure layer)  
**NIST AI RMF:** MEASURE 2.7 — AI system security and resilience are evaluated and documented

### Finding

Dockerfile uses non-distroless base with unnecessary packages:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY weights/ ./weights/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Issues:**
- Runs as root (no `USER` directive)
- Binds to 0.0.0.0 (all interfaces) instead of localhost/specific interface
- No resource limits (memory, CPU)
- Model weights copied into image (potential leakage if image pushed to registry)

### Impact

- **Privilege escalation:** Container compromise grants root access
- **Resource exhaustion:** No CPU/memory limits
- **Weight leakage:** Model weights in image layers could be extracted from registries

### Evidence

Manual inspection of `target/Dockerfile`.

### Recommendation

**Hardened Dockerfile:**
```dockerfile
FROM python:3.12-slim
RUN groupadd -r appuser && useradd -r -g appuser appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt
COPY --chown=appuser:appuser app/ ./app/
# Weights mounted at runtime via volume (not copied into image)
# Use: docker run -v /secure/path/weights:/app/weights:ro ...
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Add resource limits in `docker run`:
```bash
docker run --memory=2g --cpus=1.0 -p 127.0.0.1:8000:8000 \
    -e MODEL_SHA256=c83f56d8354266c487c0a537d4c44e56149f27fcf8950f469b279ecf340d5929 \
    -e MODEL_INFO_SHA256=b8435f91a1e1f5a9e96ea0c12c2f1fa29c8857221038e0eddb6759a701cbe6c8 \
    cifar10-target:latest
```

---

## V-07: No Request/Response Logging

**Severity:** MEDIUM  
**MITRE ATLAS:** N/A (Detection gap — logging would help surface AML.T0024.002 extraction activity)  
**NIST AI RMF:** MANAGE 4.1 — Post-deployment monitoring plans, including incident response

### Finding

No logging of:
- Client IP addresses
- Request timestamps
- Prediction outputs
- Query patterns
- Error rates per client

### Impact

- **No audit trail:** Cannot investigate suspicious activity post-incident
- **No anomaly detection:** Cannot identify model extraction or attack patterns
- **Compliance gap:** Regulatory frameworks may require AI decision logging
- **No forensics:** Cannot reconstruct attack timelines

### Evidence

Checked FastAPI logs—only startup messages, no per-request logs with predictions.

### Recommendation

Add structured logging middleware:

```python
@app.middleware("http")
async def log_predictions(request: Request, call_next):
    client_ip = request.client.host
    timestamp = datetime.utcnow()
    response = await call_next(request)
    
    # Log prediction if this was a prediction endpoint
    if request.url.path == "/predict":
        logger.info({
            "timestamp": timestamp,
            "client_ip": client_ip,
            "endpoint": "/predict",
            "predicted_class": response_body.get("predicted_class"),
            "confidence": response_body.get("confidence")
        })
    
    return response
```

Retain logs for 90 days minimum for incident investigation.

---

## Vulnerability Summary Table

| ID | Vulnerability | Severity | ATLAS | Impact |
|----|---------------|----------|-------|--------|
| V-01 | No Authentication | CRITICAL | AML.T0040 | Unrestricted model access |
| V-02 | Full Probability Exposure | HIGH | AML.T0024 | Enables high-fidelity extraction |
| V-03 | Verbose Metadata | MEDIUM | AML.T0007 | Training details leaked |
| V-04 | No Rate Limiting | CRITICAL | AML.T0024.002 / AML.T0034 | Model extraction at scale |
| V-05 | No Adversarial Detection | HIGH | AML.T0015 | Silent evasion attacks |
| V-06 | Container Weaknesses | MEDIUM | N/A | Privilege escalation risk |
| V-07 | No Logging | MEDIUM | N/A | Detection/forensics gap |

**Total: 2 Critical, 2 High, 3 Medium**

---

**Next:** See `mitigations.md` for remediation guidance mapped to MITRE ATLAS defenses.