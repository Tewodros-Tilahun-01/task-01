"""
inference.py — Load the trained model weights and run predictions.

The module-level `predictor` singleton is created once at import time so
all FastAPI workers share a single loaded model.
"""

import hashlib
import hmac
import io
import json
import os
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
# Integrity helper
# ---------------------------------------------------------------------------

def _verified_bytes(path: pathlib.Path, env_var: str) -> bytes:
    """
    Read *path* once, hash the bytes with SHA-256, and compare the digest
    to the hex value stored in environment variable *env_var*.

    Returns the raw bytes on success.
    Raises RuntimeError on any mismatch or missing env var.
    The error message intentionally omits the path and the expected hash.
    """
    expected = os.environ.get(env_var)
    if not expected:
        raise RuntimeError(
            f"Required environment variable {env_var!r} is not set. "
            "Set it to the SHA-256 hex digest of the file."
        )

    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()

    if not hmac.compare_digest(actual, expected.lower()):
        raise RuntimeError("File integrity check failed.")  # no path, no hash

    return data


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
        # --- read and verify both files; fail closed on any problem ---
        weights_bytes = _verified_bytes(_WEIGHTS,   "MODEL_SHA256")
        info_bytes    = _verified_bytes(_INFO_FILE, "MODEL_INFO_SHA256")

        # --- parse metadata from the verified bytes, not by re-reading ---
        info = json.loads(info_bytes)
        num_classes = info.get("num_classes")
        if not isinstance(num_classes, int):
            raise RuntimeError("model_info.json: 'num_classes' must be an integer.")
        if num_classes != len(CIFAR10_CLASSES):
            raise RuntimeError(
                f"model_info.json: num_classes={num_classes} does not match "
                f"the {len(CIFAR10_CLASSES)} known classes."
            )

        # --- load weights from the already-verified bytes ---
        model = CifarCNN(num_classes=num_classes).to(self.device)
        model.load_state_dict(
            torch.load(io.BytesIO(weights_bytes), map_location=self.device, weights_only=True)
        )
        model.eval()

        self.model = model
        self.info  = info
        print(f"[INFO] Model loaded on {self.device}", file=sys.stderr)  # no path

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
