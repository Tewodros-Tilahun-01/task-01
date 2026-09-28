# Attack 01 — FGSM (Fast Gradient Sign Method)

## Overview

FGSM is a single-step gradient-based adversarial attack that creates imperceptible perturbations by adding the sign of the loss gradient to input images. It provides a fast baseline for evaluating model robustness against adversarial examples with minimal computational cost.

---

## Attack Type

**White-box** — Requires full access to model architecture, weights, and gradients.

---

## When to Use

- Quick adversarial robustness baseline testing
- Initial vulnerability assessment of a new model
- Generating adversarial examples for adversarial training datasets
- Comparing robustness across different epsilon values
- When computational resources are limited (faster than iterative methods)

---

## How It Works

FGSM computes the gradient of the loss with respect to the input image, takes the sign of each gradient element, and scales it by epsilon (perturbation budget). The perturbation is added to the original image to create an adversarial example that looks nearly identical to humans but causes misclassification. The attack is "fast" because it requires only one gradient computation per image.

---

## Running the Attack

**Prerequisites:**
- Python virtual environment active (`source .venv/bin/activate`)
- Dependencies installed (`pip install -r requirements.txt`)
- Trained model weights at `target/weights/cifar_cnn.pth`

**Command:**

From the project root:

```bash
python -m attacks.attack_01
```

**Testing with API (optional):**

If the target container is running at `http://localhost:8000`, the script will also send adversarial examples to the API and log responses. This is optional — the attack runs against the local model by default.

**Output:**

The attack generates evidence in:
- `evidence/adversarial/fgsm/` — Side-by-side comparison images for each epsilon
- `evidence/logs/fgsm/` — JSON logs with attack results and API responses

**Tested epsilon values:** 2/255, 4/255, 8/255, 16/255  
**Test images:** 200 samples from CIFAR-10 test set  
**Saved examples:** 10 comparison images per epsilon (successful attacks prioritized)
