# Attack 03 — Square Attack (Black-box)

## Overview

Square Attack is a query-efficient black-box adversarial attack that uses random search over square-shaped perturbations to fool the classifier. It requires only API access to query the model's predictions, making it practical for attacking real-world deployed systems without internal model knowledge.

---

## Attack Type

**Black-box (score-based)** — Requires only API access to query model predictions. No knowledge of model architecture, weights, or gradients needed.

---

## When to Use

- Testing robustness in realistic threat scenarios (attacker has API access only)
- Evaluating production models without exposing internal details
- Measuring query-efficiency of black-box attacks
- When gradient computation is unavailable or impractical
- Demonstrating risks of unlimited API access without rate limiting

---

## How It Works

Square Attack randomly samples square-shaped perturbations in the image and queries the API to measure whether each perturbation increases the attack loss. Successful perturbations are kept, and the process repeats until the query budget is exhausted or the attack succeeds. The attack uses random restarts and a progressive refinement schedule to balance exploration and exploitation within the epsilon constraint.

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
python -m attacks.attack_03
```

**Configuration:**
- **Epsilon:** 8/255 (fixed L∞ perturbation budget)
- **Query budgets:** 50, 200, 500 queries per image
- **Random seed:** 42 (for reproducible results)

**Output:**

The attack generates evidence in:
- `evidence/adversarial/square/` — Side-by-side comparison images for each query budget
- `evidence/logs/square/` — JSON logs with query counts, timing, and success rates

**Test images:** 200 samples from CIFAR-10 test set  
**Saved examples:** 10 comparison images per query budget (most query-efficient attacks prioritized)

**Note:** This attack is slower than FGSM/PGD because it makes many sequential API calls. Expect several minutes for 200 images at 500 queries/image.
