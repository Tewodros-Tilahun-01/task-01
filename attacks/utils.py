"""
utils.py — helper functions shared by all attack scripts.
"""

import io
import json
import pathlib
import sys
import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from target.app.model import CifarCNN                       # noqa: E402
from target.training.dataset import MEAN, STD, _DATA_DIR   # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WEIGHTS_PATH = _ROOT / "target" / "weights" / "cifar_cnn.pth"

EVIDENCE_ADV  = _ROOT / "evidence" / "adversarial"
EVIDENCE_LOGS = _ROOT / "evidence" / "logs"

# ---------------------------------------------------------------------------
# CIFAR-10 class names
# ---------------------------------------------------------------------------
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: torch.device) -> CifarCNN:
    """Load the trained model and set it to eval mode."""
    model = CifarCNN(num_classes=10).to(device)
    state = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"[utils] Model loaded from {WEIGHTS_PATH}")
    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_test_loader(batch_size: int = 64) -> DataLoader:
    """Load the CIFAR-10 test set with the same normalization used in training."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_set = datasets.CIFAR10(
        _DATA_DIR, train=False, download=True, transform=transform
    )
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Tensor to image helpers
# ---------------------------------------------------------------------------

_mean_t = torch.tensor(MEAN).view(3, 1, 1)
_std_t  = torch.tensor(STD).view(3, 1, 1)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo the normalization so pixel values go back to [0, 1]."""
    t = tensor.clone().cpu()
    return (t * _std_t + _mean_t).clamp(0, 1)


def tensor_to_pil(tensor: torch.Tensor, scale: int = 256) -> Image.Image:
    """Convert a normalized tensor to a viewable PIL image, scaled up from 32×32."""
    img = denormalize(tensor)
    img = (img * 255).byte()
    img = img.permute(1, 2, 0).numpy()
    pil = Image.fromarray(img)
    return pil.resize((scale, scale), Image.NEAREST)


def tensor_to_png_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a tensor to PNG bytes so it can be sent to the API."""
    pil = tensor_to_pil(tensor, scale=32)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------

def send_to_api(
    tensor: torch.Tensor,
    url: str = "http://localhost:8000/predict",
    filename: str = "image.png",
) -> dict:
    """Send an image to the prediction API and return the response."""
    png_bytes = tensor_to_png_bytes(tensor)
    try:
        resp = requests.post(
            url,
            files={"file": (filename, png_bytes, "image/png")},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Connection refused — is the target container running?"}
    except requests.exceptions.RequestException as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Evidence saving
# ---------------------------------------------------------------------------

def save_comparison(
    clean: torch.Tensor,
    adv: torch.Tensor,
    true_label: int,
    clean_pred: int,
    adv_pred: int,
    epsilon: float,
    index: int,
    out_dir: pathlib.Path,
    scale: int = 256,
) -> pathlib.Path:
    """Save a side-by-side image showing the clean and adversarial version."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    true_name  = CIFAR10_CLASSES[true_label]
    clean_name = CIFAR10_CLASSES[clean_pred]
    adv_name   = CIFAR10_CLASSES[adv_pred]

    clean_pil = tensor_to_pil(clean, scale=scale)
    adv_pil   = tensor_to_pil(adv,   scale=scale)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    axes[0].imshow(np.array(clean_pil))
    axes[0].set_title(f"Clean\nTrue: {true_name}\nPred: {clean_name}", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(np.array(adv_pil))
    axes[1].set_title(f"Adversarial (ε={epsilon})\nTrue: {true_name}\nPred: {adv_name}", fontsize=9)
    axes[1].axis("off")

    changed = clean_pred != adv_pred
    fig.suptitle(
        "Attack SUCCESS ✓" if changed else "Attack failed ✗",
        fontsize=10,
        color="red" if changed else "gray",
    )
    fig.tight_layout()

    fname = (
        f"{index:04d}_eps{epsilon}"
        f"_true{true_label}"
        f"_clean{clean_pred}"
        f"_adv{adv_pred}.png"
    )
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_log(data: dict, path: pathlib.Path) -> None:
    """Save data as a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def plot_results(summary: list[dict], attack_name: str, out_dir: pathlib.Path) -> None:
    """Save two plots: success rate vs epsilon and accuracy vs epsilon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    epsilons      = [r["epsilon"]      for r in summary]
    success_rates = [r["success_rate"] for r in summary]
    clean_accs    = [r["clean_acc"]    for r in summary]
    adv_accs      = [r["adv_acc"]      for r in summary]

    prefix = attack_name.lower()

    # plot 1 — success rate
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epsilons, success_rates, marker="o", color="red", linewidth=2)
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Attack success rate")
    ax.set_title(f"{attack_name} — Attack Success Rate vs Epsilon")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    for x, y in zip(epsilons, success_rates):
        ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    fig.tight_layout()
    path1 = out_dir / f"{prefix}_success_rate.png"
    fig.savefig(path1, dpi=120)
    plt.close(fig)
    print(f"\n  Plot saved: {path1}")

    # plot 2 — clean vs adversarial accuracy
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epsilons, clean_accs, marker="s", label="Clean accuracy",
            color="steelblue", linewidth=2)
    ax.plot(epsilons, adv_accs, marker="o", label="Adversarial accuracy",
            color="red", linewidth=2)
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{attack_name} — Clean vs Adversarial Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path2 = out_dir / f"{prefix}_accuracy.png"
    fig.savefig(path2, dpi=120)
    plt.close(fig)
    print(f"  Plot saved: {path2}")
