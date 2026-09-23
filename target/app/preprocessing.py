"""
preprocessing.py — Convert raw image bytes into a model-ready tensor.
"""

import io

import torch
from PIL import Image
from torchvision import transforms

# CIFAR-10 training statistics (RGB)
_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2470, 0.2435, 0.2616)

_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def bytes_to_tensor(raw: bytes, device: torch.device) -> torch.Tensor:
    """
    Decode image bytes and return a (1, 3, 32, 32) tensor on *device*.

    The image is converted to RGB so grayscale or RGBA inputs are handled
    transparently. No grayscale conversion is applied.

    Raises
    ------
    ValueError
        If the bytes cannot be decoded as an image.
    """
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    return _transform(img).unsqueeze(0).to(device)   # add batch dimension
