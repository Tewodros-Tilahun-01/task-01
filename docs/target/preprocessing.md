# Target — Image Preprocessing Pipeline

Every image sent to the inference server goes through the same preprocessing pipeline
before it reaches the model. This document describes each step, the values used, and
why they matter for adversarial attacks.

---

## Pipeline overview

```
Raw bytes (HTTP request)
        │
        ▼
  PIL Image decode
  └── convert to RGB
        │
        ▼
  Resize to 32×32
        │
        ▼
  ToTensor()
  └── [0, 255] uint8  →  [0.0, 1.0] float32
  └── shape: (H, W, C)  →  (C, H, W)
        │
        ▼
  Normalize(mean, std)
  └── per-channel: (pixel - mean) / std
        │
        ▼
  unsqueeze(0)
  └── (3, 32, 32)  →  (1, 3, 32, 32)  — add batch dimension
        │
        ▼
  Tensor on device (CPU or CUDA)
```

---

## Steps in detail

### 1. Decode image bytes

```python
img = Image.open(io.BytesIO(raw)).convert("RGB")
```

- Accepts any image format PIL supports (PNG, JPEG, BMP, etc.)
- Forces conversion to RGB — grayscale and RGBA inputs are handled transparently
- Raises `ValueError` if the bytes cannot be decoded

### 2. Resize

```python
transforms.Resize((32, 32))
```

- Resizes to exactly 32×32 pixels regardless of the original dimensions
- Uses PIL's default resampling filter (bilinear)
- Required because the model only accepts `(3, 32, 32)` input

### 3. ToTensor

```python
transforms.ToTensor()
```

- Converts the PIL image from `uint8 [0, 255]` to `float32 [0.0, 1.0]`
- Transposes axes from `(H, W, C)` to `(C, H, W)`

### 4. Normalise

```python
transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465),
    std =(0.2470, 0.2435, 0.2616)
)
```

Per-channel normalisation using CIFAR-10 training set statistics:

| Channel | Mean | Std |
|---------|------|-----|
| Red | 0.4914 | 0.2470 |
| Green | 0.4822 | 0.2435 |
| Blue | 0.4465 | 0.2616 |

Formula applied per channel:

```
output = (input - mean) / std
```

After this step pixel values are no longer in `[0, 1]` — they are centred around zero
with unit-ish variance. The model was trained on data normalised with these exact values,
so they must match at inference time.

### 5. Add batch dimension

```python
tensor.unsqueeze(0)   # (3, 32, 32) → (1, 3, 32, 32)
```

The model expects a batch of inputs `(N, C, H, W)`. A single image gets a batch size
of 1.

---

## Adversarial attack implications

| Property | Implication |
|----------|-------------|
| Normalisation stats are public via `/model/info` | An attacker can apply the exact same normalisation when crafting adversarial perturbations — no reverse engineering needed |
| Fixed resize to 32×32 | Large images are downsampled; perturbations added to the original resolution may be weakened after resize |
| No input validation beyond PIL decode | Any file that PIL can open is accepted — oversized inputs, unusual formats, and edge cases pass through |
| No pixel clamping after normalisation | Adversarial tensors with out-of-range values are not rejected |

---

## Source

`target/app/preprocessing.py` — `bytes_to_tensor(raw, device)`

The full transform pipeline is defined as a module-level constant `_transform` and
applied the same way on every request.

---

## Related documents

- [architecture.md](architecture.md) — model input shape and training normalisation
- [api.md](api.md) — how images are submitted to the server
