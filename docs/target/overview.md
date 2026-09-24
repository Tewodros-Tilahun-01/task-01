# Target — Overview

## What it is

The target is a CIFAR-10 image classifier served as a REST API inside a Docker container.
It acts as the **system under attack** for this red team assessment — a realistic stand-in
for a production ML inference service.

It consists of two parts:

- **A trained CNN** (`CifarCNN`) — a custom PyTorch model trained on the CIFAR-10 dataset
  to classify 32×32 RGB images into 10 categories.
- **A FastAPI inference server** — exposes the model over HTTP, accepting image uploads and
  returning class predictions with confidence scores.

---

## Why it was built this way

| Decision | Reasoning |
|----------|-----------|
| CIFAR-10 dataset | Freely available, well-understood, no licensing issues. Realistic enough for adversarial work. |
| Custom CNN (not a pretrained model) | Gives full control over the architecture and training process. Weights are reproducible from scratch. |
| FastAPI | Minimal, fast, produces automatic interactive docs at `/docs`. Easy to probe. |
| Docker deployment | Isolates the target from the attack environment. Attacks interact with it over HTTP only, as a real attacker would. |
| Intentionally undefended | No adversarial training, no input sanitisation, no rate limiting. Makes it a clean baseline victim for demonstrating attack impact. |

---

## CIFAR-10 classes

The model classifies images into one of 10 categories:

| Index | Class |
|-------|-------|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

---

## Directory structure

```
target/
├── app/
│   ├── main.py           # FastAPI routes
│   ├── inference.py      # Model loading and prediction logic
│   ├── model.py          # CifarCNN architecture
│   ├── preprocessing.py  # Image bytes → normalised tensor
│   └── schemas.py        # Pydantic error response schema
├── training/
│   ├── train.py          # Training loop, saves weights
│   └── dataset.py        # CIFAR-10 DataLoaders with augmentation
├── weights/
│   ├── cifar_cnn.pth     # Saved model weights (best val accuracy)
│   └── model_info.json   # Architecture metadata consumed by inference.py
├── data/                 # CIFAR-10 raw data (auto-downloaded by train.py)
├── requirements.txt      # Python dependencies for the target
└── Dockerfile            # Container image definition
```

---

## Related documents

- [architecture.md](architecture.md) — CifarCNN layer-by-layer breakdown and training details
- [api.md](api.md) — Full REST API reference
- [preprocessing.md](preprocessing.md) — Image preprocessing pipeline
