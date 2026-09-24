# Target — Model Architecture

## Overview

`CifarCNN` is a custom convolutional neural network trained on CIFAR-10. It follows a
classic feature-extractor + classifier pattern with three convolutional blocks and a
two-layer fully connected head.

- **Input:** `(N, 3, 32, 32)` — batch of RGB images
- **Output:** `(N, 10)` — raw logits, one per class

---

## Layer breakdown

### Feature extractor

The feature extractor is three identical-pattern blocks. Each block doubles the channel
count, maintains spatial size with `padding=1`, then halves it with `MaxPool2d(2)`.

```
Input: (N, 3, 32, 32)
│
├── Block 1
│   ├── Conv2d(3 → 32,  kernel=3, padding=1)   → (N, 32, 32, 32)
│   ├── BatchNorm2d(32)
│   ├── ReLU
│   ├── Conv2d(32 → 32, kernel=3, padding=1)   → (N, 32, 32, 32)
│   ├── BatchNorm2d(32)
│   ├── ReLU
│   ├── MaxPool2d(2)                            → (N, 32, 16, 16)
│   └── Dropout2d(0.25)
│
├── Block 2
│   ├── Conv2d(32 → 64, kernel=3, padding=1)   → (N, 64, 16, 16)
│   ├── BatchNorm2d(64)
│   ├── ReLU
│   ├── Conv2d(64 → 64, kernel=3, padding=1)   → (N, 64, 16, 16)
│   ├── BatchNorm2d(64)
│   ├── ReLU
│   ├── MaxPool2d(2)                            → (N, 64, 8, 8)
│   └── Dropout2d(0.25)
│
└── Block 3
    ├── Conv2d(64 → 128,  kernel=3, padding=1) → (N, 128, 8, 8)
    ├── BatchNorm2d(128)
    ├── ReLU
    ├── Conv2d(128 → 128, kernel=3, padding=1) → (N, 128, 8, 8)
    ├── BatchNorm2d(128)
    ├── ReLU
    ├── MaxPool2d(2)                            → (N, 128, 4, 4)
    └── Dropout2d(0.25)

Feature map after extractor: (N, 128, 4, 4)  →  2048 values per sample
```

### Classifier head

```
├── Flatten                                    → (N, 2048)
├── Linear(2048 → 512)
├── ReLU
├── Dropout(0.5)
└── Linear(512 → 10)                          → (N, 10)  raw logits
```

---

## Design choices

| Choice | Reason |
|--------|--------|
| Double conv per block | Deeper feature extraction without immediately reducing spatial size |
| BatchNorm after every conv | Stabilises training, allows higher learning rate |
| Dropout2d(0.25) in conv blocks | Regularises spatial feature maps, reduces overfitting |
| Dropout(0.5) in classifier | Standard regularisation before the final linear layer |
| Cosine annealing LR scheduler | Smooth decay without needing manual step tuning |
| Adam optimiser, LR=1e-3 | Good default for CIFAR-scale models |

---

## Training configuration

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 (50,000 train / 10,000 test) |
| Validation split | 5,000 held out from training set (fixed seed=42) |
| Epochs | 20 |
| Batch size | 128 |
| Optimiser | Adam (lr=1e-3) |
| LR scheduler | CosineAnnealingLR (T_max=20) |
| Loss | CrossEntropyLoss |
| Training augmentation | RandomHorizontalFlip, RandomCrop(32, padding=4) |
| Eval normalisation | Mean=(0.4914, 0.4822, 0.4465), Std=(0.2470, 0.2435, 0.2616) |

---

## Training data augmentation

Only the training set is augmented. Validation and test sets receive normalisation only.

```
Train transform:
  RandomHorizontalFlip()
  RandomCrop(32, padding=4)
  ToTensor()
  Normalize(mean, std)

Val / Test transform:
  ToTensor()
  Normalize(mean, std)
```

---

## Saved outputs

After training, two files are written to `target/weights/`:

| File | Contents |
|------|----------|
| `cifar_cnn.pth` | `state_dict` of the best checkpoint (highest val accuracy) |
| `model_info.json` | Architecture metadata: num_classes, input_shape, normalisation stats, val/test accuracy |

`model_info.json`:

```json
{
  "architecture": "CifarCNN",
  "num_classes": 10,
  "input_shape": [3, 32, 32],
  "normalize": {
    "mean": [0.4914, 0.4822, 0.4465],
    "std":  [0.247, 0.2435, 0.2616]
  },
  "best_val_acc": 0.8044,
  "test_acc":     0.8026
}
```

---

## Expected performance

| Split | Accuracy |
|-------|----------|
| Validation | 80.44% |
| Test | 80.26% |

These are the actual values from `target/weights/model_info.json`. They reflect clean
(unperturbed) inputs. Adversarial attacks are expected to significantly reduce confidence
and flip predictions.

---

## Related documents

- [overview.md](overview.md) — target purpose and design decisions
- [preprocessing.md](preprocessing.md) — how raw image bytes are converted to tensors at inference time
