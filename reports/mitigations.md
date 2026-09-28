# Mitigations

Remediation roadmap for adversarial robustness and pipeline security findings.

---

## ML Model Defenses

### M-01: Adversarial Training

**Maps to Findings:** Attack-01 (FGSM), Attack-02 (PGD), Attack-03 (Square), Attack-04 (Extraction Transfer)  
**MITRE ATLAS:** AML.M0003 (Model Hardening)  
**NIST AI RMF:** MANAGE 1.3 — Responses to high-priority AI risks  
**Priority:** HIGH  
**Status:** Proposed

#### Implementation

Train model on mix of clean and adversarial examples:

```python
# Adversarial training loop (Madry et al. 2017)
def train_epoch_adversarial(model, train_loader, optimizer, epsilon=8/255):
    attack = torchattacks.PGD(model, eps=epsilon, alpha=2*epsilon/7, steps=7)
    
    for images, labels in train_loader:
        # Generate adversarial examples
        adv_images = attack(images, labels)
        
        # Train on both clean and adversarial
        outputs_clean = model(images)
        outputs_adv = model(adv_images)
        
        loss = 0.5 * criterion(outputs_clean, labels) + \
               0.5 * criterion(outputs_adv, labels)
        
        loss.backward()
        optimizer.step()
```

**Parameters:**
- Adversarial ratio: 50% clean, 50% adversarial
- Perturbation budget: ε=8/255 (L∞)
- Attack during training: PGD-7 (7 steps, faster than PGD-20)
- Training epochs: 40 (2× baseline to compensate for harder optimization)

#### Expected Results

**Benefits:**
- Adversarial accuracy at ε=8/255: 40-50% (up from 0%)
- PGD success rate: 50-60% (down from 100%)
- Black-box attack robustness: Improved but not eliminated

**Trade-offs:**
- Clean accuracy drop: 5-8% (expected 75-78% from 83.5%)
- Training time: 2-3× longer
- Model capacity may need increase (more parameters to learn robust features)

#### Validation

After retraining, re-run attack scripts:
```bash
python -m attacks.attack_01  # FGSM
python -m attacks.attack_02  # PGD
python -m attacks.attack_03  # Square
```

**Success criteria:** Adversarial accuracy ≥40% at ε=8/255

---

## Pipeline Security Defenses

### M-05: API Authentication and Authorization

**Maps to Finding:** V-01 (No Authentication)  
**MITRE ATLAS:** AML.M0019 (Control Access to ML Models and Data in Production)  
**NIST AI RMF:** GOVERN 2.1 — Roles, responsibilities, and lines of communication (accountability)  
**Priority:** CRITICAL  
**Status:** Proposed

#### Implementation

Add API key authentication:

```python
# target/app/auth.py
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key not in valid_api_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return x_api_key

# Apply to endpoints
@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(file: UploadFile):
    ...
```

**Key management:**
- Generate unique keys per user/application
- Store hashed keys in database
- Rotate keys quarterly
- Revoke keys on suspicious activity

#### Expected Results

- Prevents anonymous access
- Enables query attribution
- Supports rate limiting per key
- Audit trail per user

---

### M-06: Rate Limiting and Query Budgets

**Maps to Finding:** V-04 (No Rate Limiting)  
**MITRE ATLAS:** AML.M0004 (Restrict Number of ML Model Queries)  
**NIST AI RMF:** MANAGE 1.3 — Responses to high-priority AI risks  
**Priority:** CRITICAL  
**Status:** Proposed

#### Implementation

**Option 1: Application-level (FastAPI middleware)**

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile):
    ...
```

**Option 2: Infrastructure-level (Nginx)**

```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;

location /predict {
    limit_req zone=api_limit burst=10;
    proxy_pass http://backend:8000;
}
```

**Rate limit:**
- All users: 100 queries/minute

#### Expected Results

- Model extraction blocked (requires >10,000 queries)
- DoS mitigation
- Cost control per user

---

### M-07: Minimize API Response Data

**Maps to Finding:** V-02 (Full Probability Exposure)  
**MITRE ATLAS:** AML.M0002 (Passive ML Output Obfuscation)  
**NIST AI RMF:** MEASURE 2.7 — AI system security and resilience are evaluated and documented  
**Priority:** HIGH  
**Status:** Proposed

#### Implementation

Return only top prediction:

```python
@app.post("/predict")
async def predict(file: UploadFile):
    result = predictor.predict(raw)
    
    # Remove full probability distribution
    return JSONResponse(content={
        "predicted_class": result.predicted_class,
        "class_name": result.class_name,
        "confidence": result.confidence,
        # "probabilities": result.probabilities,  # REMOVED
    })
```

**Optional:** Return top-3 predictions for legitimate use cases, but never all 10.

#### Expected Results

- Model extraction harder (less information per query)
- Black-box attacks less effective (no soft labels)
- Minimal impact on legitimate users (top-1 usually sufficient)

**Status:** **IMPLEMENTED** in hardened inference.py (uses SHA-256 integrity checks)

---

### M-08: Remove or Gate Model Metadata Endpoint

**Maps to Finding:** V-03 (Verbose Metadata)  
**MITRE ATLAS:** AML.M0000 (Limit Public Release of Information)  
**NIST AI RMF:** MEASURE 2.7 — AI system security and resilience are evaluated and documented  
**Priority:** MEDIUM  
**Status:** Proposed

#### Implementation

**Option 1: Remove endpoint entirely**
```python
# Delete or comment out /model/info route
```

**Option 2: Gate behind authentication**
```python
@app.get("/model/info", dependencies=[Depends(verify_admin_key)])
def model_info():
    return predictor.info
```

**Option 3: Return minimal metadata**
```python
@app.get("/model/info")
def model_info():
    return {
        "input_shape": [3, 32, 32],  # Required for client integration
        "classes": CIFAR10_CLASSES,  # Public knowledge
        # Remove: architecture, training details, accuracy
    }
```

#### Expected Results

- Attacker cannot tailor extraction strategy to victim architecture
- Reduced reconnaissance value

---

### M-09: Comprehensive Logging and Monitoring

**Maps to Finding:** V-07 (No Logging)  
**MITRE ATLAS:** N/A (Detection/monitoring control; no dedicated ATLAS mitigation — supports AML.M0004)  
**NIST AI RMF:** MANAGE 4.1 — Post-deployment monitoring plans, including incident response  
**Priority:** MEDIUM  
**Status:** Proposed

#### Implementation

**Structured logging:**

```python
import structlog

logger = structlog.get_logger()

@app.post("/predict")
async def predict(request: Request, file: UploadFile, api_key: str = Depends(verify_api_key)):
    result = predictor.predict(raw)
    
    logger.info(
        "prediction",
        api_key=api_key,
        client_ip=request.client.host,
        timestamp=datetime.utcnow().isoformat(),
        predicted_class=result.predicted_class,
        confidence=result.confidence,
    )
    
    return result
```

**Monitoring alerts:**
- High query rate from single IP (>100/hour)
- Low confidence predictions (avg <0.5)
- Repeated same-class predictions (potential model probing)
- Sudden traffic spikes

**Retention:** 90 days minimum, indexed for search

#### Expected Results

- Audit trail for incident investigation
- Early detection of model extraction attempts
- Compliance with AI logging requirements

---

### M-10: Container Hardening

**Maps to Finding:** V-06 (Container Weaknesses)  
**MITRE ATLAS:** N/A (Infrastructure)  
**NIST AI RMF:** MEASURE 2.7 — AI system security and resilience are evaluated and documented  
**Priority:** MEDIUM  
**Status:** Proposed

#### Implementation

**Hardened Dockerfile:**

```dockerfile
FROM python:3.12-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

# Copy application code
COPY --chown=appuser:appuser app/ ./app/

# Drop privileges
USER appuser

# Bind to localhost only (use reverse proxy for external access)
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"]
```

**Runtime resource limits:**

```bash
docker run \
  --memory=2g \
  --cpus=1.0 \
  --read-only \
  --tmpfs /tmp \
  --security-opt=no-new-privileges \
  -p 127.0.0.1:8000:8000 \
  cifar10-target
```

**Model weights as volume (not in image):**

```bash
docker run -v /secure/path/weights:/app/weights:ro \
    -p 127.0.0.1:8000:8000 \
    -e MODEL_SHA256=c83f56d8354266c487c0a537d4c44e56149f27fcf8950f469b279ecf340d5929 \
    -e MODEL_INFO_SHA256=b8435f91a1e1f5a9e96ea0c12c2f1fa29c8857221038e0eddb6759a701cbe6c8 \
    cifar10-target:latest
```

#### Expected Results

- Reduced privilege escalation risk
- Resource exhaustion protection
- Prevents weight extraction from image layers

**Status:** **PARTIALLY IMPLEMENTED** (integrity checks in inference.py via MODEL_SHA256 env var)

---

## Mitigation Priority Matrix

| Priority | Mitigation | Effort | Impact | Timeline |
|----------|-----------|--------|--------|----------|
| **CRITICAL** | M-05 (Authentication) | Low | High | 2 hours |
| **CRITICAL** | M-06 (Rate Limiting) | Low | High | 2 hours |
| **HIGH** | M-01 (Adversarial Training) | High | High | 10 hours |
| **HIGH** | M-07 (Minimize Response Data) | Low | Medium | 1 hour |
| **MEDIUM** | M-08 (Remove Metadata) | Low | Low | 1 hour |
| **MEDIUM** | M-09 (Logging) | Medium | Medium | 4 hours |
| **MEDIUM** | M-10 (Container Hardening) | Low | Medium | 3 hours |

---


## Testing and Validation

After implementing mitigations, re-run full attack suite:

```bash
# Re-test all attacks
python -m attacks.attack_01  # FGSM
python -m attacks.attack_02  # PGD
python -m attacks.attack_03  # Square
python -m attacks.attack_04  # Extraction

# Compare success rates before/after
# Goal: <50% adversarial success at ε=8/255
```

---