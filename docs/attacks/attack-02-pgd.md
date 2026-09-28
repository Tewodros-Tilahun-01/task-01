# Attack 02 — PGD (Projected Gradient Descent)

## Overview

PGD is an iterative gradient-based adversarial attack that applies FGSM multiple times with smaller step sizes, projecting the result back into the valid perturbation budget after each step. It produces stronger adversarial examples than FGSM and is considered the gold standard for evaluating adversarial robustness.

---

## Attack Type

**White-box** — Requires full access to model architecture, weights, and gradients.

---

## When to Use

- Rigorous adversarial robustness evaluation (stronger than FGSM)
- Generating challenging adversarial examples for adversarial training
- Benchmarking defense mechanisms (PGD is the standard evaluation method)
- When computational budget allows iterative optimization
- Testing model robustness against sophisticated attackers

---

## How It Works

PGD starts with a random perturbation within the epsilon ball, then iteratively applies small gradient steps (alpha) in the direction that maximizes loss, projecting back into the epsilon constraint after each step. This process repeats for a fixed number of iterations (20 steps by default), allowing the attack to find stronger adversarial examples than single-step FGSM. The step size alpha is set to 2.5 × epsilon / steps.

---

## Running the Attack

**Prerequisites:**
- Python virtual environment active (`source .venv/bin/activate`)
- Dependencies installed (`pip install -r requirements.txt`)
- Trained model weights at `target/weights/cifar_cnn.pth`

**Command:**

From the project root:

```bash
python -m attacks.attack_02
```

**Configuration:**
- **Steps:** 20 iterations (tuned for this model — no accuracy gain beyond 20)
- **Alpha:** 2.5 × epsilon / 20 (scaled per epsilon)
- **Random seed:** 42 (for reproducible results)

**Testing with API (optional):**

If the target container is running at `http://localhost:8000`, the script will also send adversarial examples to the API and log responses.

**Output:**

The attack generates evidence in:
- `evidence/adversarial/pgd/` — Side-by-side comparison images for each epsilon
- `evidence/logs/pgd/` — JSON logs with attack results and API responses

**Tested epsilon values:** 2/255, 4/255, 8/255, 16/255  
**Test images:** 200 samples from CIFAR-10 test set  
**Saved examples:** 10 comparison images per epsilon (successful attacks prioritized)
