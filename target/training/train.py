"""
train.py — Train CifarCNN on CIFAR-10 and save the best weights.

Run from the project root (with .venv active):
    python -m target.training.train

Output
------
target/weights/cifar_cnn.pth     model weights (best val accuracy)
target/weights/model_info.json   metadata consumed by app/inference.py
"""

import hashlib
import json
import pathlib
import sys

import torch
import torch.nn as nn
import torch.optim as optim

# Allow `from app.model import …` when run as a module from the repo root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from app.model import CifarCNN              # noqa: E402
from training.dataset import get_loaders, MEAN, STD  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TARGET_DIR   = pathlib.Path(__file__).parent.parent   # target/
WEIGHTS_DIR  = TARGET_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = WEIGHTS_DIR / "cifar_cnn.pth"
INFO_PATH    = WEIGHTS_DIR / "model_info.json"

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
BATCH_SIZE  = 128
EPOCHS      = 20
LR          = 1e-3
NUM_CLASSES = 10
INPUT_SHAPE = (3, 32, 32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, criterion, device, *, train: bool):
    model.train(train)
    total_loss, correct = 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / n, correct / n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    train_loader, val_loader, test_loader = get_loaders(batch_size=BATCH_SIZE)

    model     = CifarCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # Cosine annealing works well for CIFAR-10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Training for {EPOCHS} epochs …\n")
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )
        va_loss, va_acc = run_epoch(
            model, val_loader, None, criterion, device, train=False
        )
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"train  loss={tr_loss:.4f}  acc={tr_acc:.4f}  |  "
            f"val    loss={va_loss:.4f}  acc={va_acc:.4f}"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), WEIGHTS_PATH)

    print(f"\nBest val accuracy : {best_acc:.4f}")
    print(f"Weights saved to  : {WEIGHTS_PATH}")

    # Load best weights and evaluate on test set once
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    te_loss, te_acc = run_epoch(model, test_loader, None, criterion, device, train=False)
    print(f"Test accuracy     : {te_acc:.4f}")

    info = {
        "architecture": "CifarCNN",
        "num_classes":  NUM_CLASSES,
        "input_shape":  list(INPUT_SHAPE),
        "normalize":    {
            "mean": list(MEAN),
            "std":  list(STD),
        },
        "best_val_acc": round(best_acc, 6),
        "test_acc":     round(te_acc, 6),
    }
    INFO_PATH.write_text(json.dumps(info, indent=2))
    print(f"Metadata saved to : {INFO_PATH}")

    model_sha = hashlib.sha256(WEIGHTS_PATH.read_bytes()).hexdigest()
    info_sha  = hashlib.sha256(INFO_PATH.read_bytes()).hexdigest()
    print("\n# ── Copy these into your docker run command ──────────────────────")
    print(f"  -e MODEL_SHA256={model_sha}")
    print(f"  -e MODEL_INFO_SHA256={info_sha}")
    print("# ──────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
