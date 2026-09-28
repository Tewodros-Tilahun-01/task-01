# Attack 04 — Model Extraction

## Overview

Model Extraction is a multi-phase black-box attack that steals a model's behavior by querying its API, training a surrogate model on the responses, and testing whether adversarial examples crafted on the surrogate transfer to the victim. This attack demonstrates intellectual property theft and enables subsequent white-box attacks without direct model access.

---

## Attack Type

**Black-box (query-based)** — Requires only API access to query model predictions. No knowledge of victim model architecture, weights, or gradients needed. The surrogate model uses a different architecture to simulate realistic black-box conditions.

---

## When to Use

- Demonstrating model intellectual property theft risks
- Testing API rate limiting and abuse prevention
- Evaluating transferability of adversarial examples across models
- Simulating sophisticated attacker scenarios (multi-stage attacks)
- Measuring information leakage through probability distributions

---

## How It Works

The attack executes in four phases: (1) Query the victim API with 10,000 test images and collect predicted probability distributions. (2) Train a surrogate model using knowledge distillation on the collected responses (different architecture than victim). (3) Evaluate agreement rate between surrogate and victim predictions on new test data. (4) Generate adversarial examples on the surrogate using PGD and test whether they transfer to the victim API.

---

## Running the Attack

**Prerequisites:**
- Python virtual environment active (`source .venv/bin/activate`)
- Dependencies installed (`pip install -r requirements.txt`)
- **Target API must be running** at `http://localhost:8000`

**Start the target first:**

```bash
docker run -p 127.0.0.1:8000:8000 -e MODEL_SHA256=c83f56d8354266c487c0a537d4c44e56149f27fcf8950f469b279ecf340d5929 -e MODEL_INFO_SHA256=b8435f91a1e1f5a9e96ea0c12c2f1fa29c8857221038e0eddb6759a701cbe6c8 cifar10-target:latest
```

**Command:**

From the project root:

```bash
python -m attacks.attack_04
```

**Configuration:**
- **Query dataset:** 10,000 images from CIFAR-10 test set
- **Surrogate training:** 20 epochs, batch size 128, lr=0.001
- **Surrogate architecture:** `RealisticSurrogate` (different from victim's `CifarCNN`)
- **Transfer attack:** PGD with epsilon=8/255, tested on 200 images
- **Random seed:** 42 (for reproducible results)

**Output:**

The attack generates evidence in:
- `evidence/adversarial/model_extraction/` — Surrogate model weights, training loss plot, transfer attack examples
- `evidence/logs/model_extraction/` — JSON summary with query count, agreement rate, transfer success rate

**Attack phases:**
1. **Query collection:** ~5-10 minutes for 10,000 images
2. **Surrogate training:** ~2-5 minutes for 20 epochs (CPU)
3. **Agreement evaluation:** ~30 seconds for 1,000 images
4. **Transfer testing:** ~1 minute for 200 images

**Model Extraction Module:**

The attack uses a modular implementation in `attacks/model_extraction/`:
- `query.py` — Collects training data by querying victim API
- `training.py` — Trains surrogate using knowledge distillation
- `evaluation.py` — Measures agreement and tests transferability
- `surrogate_arch.py` — Defines `RealisticSurrogate` CNN (different from victim)

**Note:** This attack makes 10,000+ API calls and demonstrates why rate limiting is critical for production ML APIs.
