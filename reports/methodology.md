# Methodology

## Engagement Scope

**Objective:** Evaluate the security posture of a production-style PyTorch image classifier against adversarial machine learning attacks and infrastructure vulnerabilities.

**Target System:**
- **Model:** Custom CNN (CifarCNN) trained on CIFAR-10 dataset
- **Architecture:** 3 convolutional blocks (3→32→64→128 channels) + 2 fully-connected layers
- **Training:** 20 epochs, Adam optimizer, cross-entropy loss on 45,000 clean images
- **Baseline accuracy:** 83.5% on 10,000-image test set
- **local:** FastAPI REST server in Python 3.12-slim Docker container
- **API Endpoints:** `/predict` (single image), `/predict/batch`, `/model/info`, `/` (health)
- **Input format:** PNG/JPEG images converted to 32×32×3 tensors, normalized per CIFAR-10 statistics

**Test Environment:**
- **Attacker position:** External API access only (black-box scenario)
- **Access:** HTTP queries to `localhost:8000` (simulating remote attacker)
- **Tools:** Python 3.12, PyTorch 2.1, torchattacks library
- **Hardware:** Local workstation (CPU inference)

---

## Threat Model

### Assumptions

**Attacker Capabilities:**
1. Query API with arbitrary images (no access to model weights or training data)
2. Observe predictions, confidence scores, and full probability distributions
3. Craft input perturbations and measure model responses
4. Execute unlimited queries (no rate limiting deployed)

**Attacker Goals:**
1. **Evasion:** Cause misclassification of specific inputs
2. **Model extraction:** Replicate model behavior via query-based learning
3. **Reconnaissance:** Discover architecture, training details, weaknesses via metadata


---

## MITRE ATLAS Mapping

This assessment evaluated tactics from the **MITRE ATLAS** (Adversarial Threat Landscape for Artificial-Intelligence Systems) framework:

| ATLAS Tactic | Technique | Tested |
|--------------|-----------|--------|
| **ML Model Access** | AML.T0024 (Obtain Model Artifacts) | ✓ (model info endpoint) |
| **ML Attack Staging** | AML.T0043 (Craft Adversarial Data) | ✓ (FGSM, PGD, Square) |
| **Evade ML Model** | AML.T0015 (Evade ML Model) | ✓ (white/black-box evasion) |
| **Exfiltration** | AML.T0057 (Model Extraction) | ✓ (query-based stealing) |

---

## NIST AI RMF Functions

Assessment activities aligned with **NIST AI Risk Management Framework** functions:

### Govern
- Evaluated whether model deployment follows secure ML operations practices
- Assessed presence of security governance (auth, logging, monitoring)

### Map
- Identified attack surface: API endpoints, error messages, metadata exposure
- Mapped adversarial threat vectors: evasion, extraction, reconnaissance

### Measure
- Quantified adversarial robustness across perturbation budgets (ε=2/255 to 16/255)
- Measured model extraction efficiency (queries required, agreement rate, transferability)
- Calculated attack success rates per class and per epsilon

### Manage
- Documented mitigations mapped to ATLAS defenses (adversarial training, input validation, monitoring)
- Provided remediation priority based on risk severity

---

## Attack Categories

### 1. White-box Gradient-based Attacks

**Access required:** Model gradients (simulated via local copy for realistic assessment)

**Attacks executed:**
- **FGSM (Fast Gradient Sign Method):** Single-step perturbation in gradient direction
- **PGD (Projected Gradient Descent):** 20-step iterative refinement with random start

**Parameters tested:**
- Epsilon (perturbation budget): 2/255, 4/255, 8/255, 16/255
- Norm: L∞ (max per-pixel change)
- Test set: 200 images (correctly classified subset)

**Success criteria:** Prediction flip on originally correct classification

### 2. Black-box Query-based Attacks

**Access required:** API query access only

**Attack executed:**
- **Square Attack:** Score-based black-box attack using random search and importance sampling

**Parameters tested:**
- Query budgets: 50, 200, 500 queries per image
- Epsilon: 8/255 (L∞ norm)
- Test set: 200 images

**Success criteria:** Misclassification within query budget

### 3. Model Extraction

**Access required:** API query access only

**Attack stages:**
1. **Query collection:** 10,000 CIFAR-10 test images sent to API, predictions recorded
2. **Surrogate training:** Different architecture (RealisticSurrogate) trained via knowledge distillation on API responses
3. **Agreement evaluation:** Surrogate tested against victim API on 1,000 held-out images
4. **Transferability testing:** PGD adversarials crafted on surrogate, tested on victim

**Success criteria:** ≥70% agreement rate, ≥50% adversarial transfer success

### 4. Pipeline Vulnerability Analysis

**Manual inspection of:**
- Dockerfile and container configuration
- API error handling and verbosity
- Authentication and authorization mechanisms
- Input validation and sanitization
- Logging and monitoring coverage
- Metadata exposure via `/model/info` endpoint

---

## Tools Used

| Tool | Version | Purpose |
|------|---------|---------|
| PyTorch | 2.1.0 | Model loading, inference, gradient computation |
| torchattacks | 3.5.1 | Pre-built FGSM, PGD implementations |
| FastAPI | 0.104 | Target API framework |
| Docker | 24.0 | Container runtime for target |
| Python | 3.12 | Attack scripts |
| curl | 8.5 | API testing |

---

## Success Measurement

### Adversarial Attacks
- **Attack success rate:** Percentage of correctly classified images that flipped after perturbation
- **Adversarial accuracy:** Model accuracy on perturbed images (lower = more successful attack)
- **Accuracy drop:** Clean accuracy − adversarial accuracy

### Model Extraction
- **Agreement rate:** Percentage of test images where surrogate and victim produce same prediction
- **Transfer success rate:** Percentage of surrogate-crafted adversarials that fool victim
- **Query efficiency:** Queries required per percentage point of agreement

### Pipeline Vulnerabilities
- **Severity classification:** Critical / High / Medium / Low based on exploitability and impact
- **ATLAS control mapping:** Each finding mapped to relevant ATLAS defense

---

## Reproducibility

attacks used fixed random seeds (seed=42) and deterministic PyTorch settings to ensure reproducible results:

```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
```

Evidence artifacts (images, logs, summaries) stored in `evidence/` directory with naming convention: `<index>_eps<epsilon>_true<true_label>_clean<clean_pred>_adv<adv_pred>.png`.
