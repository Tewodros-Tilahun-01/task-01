"""
inference.py — Load the trained model weights and run predictions.

The module-level `predictor` singleton is created once at import time so
all FastAPI workers share a single loaded model.
"""

import json
import pathlib
import sys
from dataclasses import dataclass

import torch

from app.model import CifarCNN
from app.preprocessing import bytes_to_tensor

# ---------------------------------------------------------------------------
# Paths — weights/ sits next to app/ inside target/
# ---------------------------------------------------------------------------
_WEIGHTS_DIR = pathlib.Path(__file__).parent.parent / "weights"
_WEIGHTS     = _WEIGHTS_DIR / "cifar_cnn.pth"
_INFO_FILE   = _WEIGHTS_DIR / "model_info.json"

# ---------------------------------------------------------------------------
# CIFAR-10 class labels (index → name)
# ---------------------------------------------------------------------------
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class Prediction:
    predicted_class: int
    class_name: str
    confidence: float
    probabilities: dict[str, float]


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class Predictor:
    """Wraps the loaded model and exposes a single `predict` method."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: CifarCNN | None = None
        self.info: dict = {}
        self._load()

    def _load(self) -> None:
        if not _WEIGHTS.exists():
            print(
                f"[WARNING] Weights not found at {_WEIGHTS}\n"
                "         Run:  python -m target.training.train\n"
                "         Then restart the server.",
                file=sys.stderr,
            )
            return

        info = json.loads(_INFO_FILE.read_text()) if _INFO_FILE.exists() else {}
        num_classes = info.get("num_classes", 10)

        model = CifarCNN(num_classes=num_classes).to(self.device)
        model.load_state_dict(torch.load(_WEIGHTS, map_location=self.device))
        model.eval()

        self.model = model
        self.info  = info
        print(f"[INFO] Model loaded from {_WEIGHTS} on {self.device}", file=sys.stderr)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(self, image_bytes: bytes) -> Prediction:
        """
        Run inference on raw image bytes.

        Raises
        ------
        RuntimeError  – model not loaded
        ValueError    – image bytes cannot be decoded
        """
        if not self.ready:
            raise RuntimeError("Model not loaded")

        tensor = bytes_to_tensor(image_bytes, self.device)

        with torch.no_grad():
            logits = self.model(tensor)               # (1, 10)
            probs  = torch.softmax(logits, dim=1)[0]  # (10,)

        predicted  = int(probs.argmax().item())
        confidence = float(probs[predicted].item())
        all_probs  = {
            str(i): round(float(p), 6) for i, p in enumerate(probs)
        }

        return Prediction(
            predicted_class=predicted,
            class_name=CIFAR10_CLASSES[predicted],
            confidence=round(confidence, 6),
            probabilities=all_probs,
        )


# Single shared instance — loaded once when the module is first imported.
predictor = Predictor()
