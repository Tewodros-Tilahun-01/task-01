"""
model.py — CifarCNN architecture.

Shared by train.py (training) and inference.py (serving).
"""

import torch
import torch.nn as nn


class CifarCNN(nn.Module):
    """
    Deeper CNN for CIFAR-10 image classification.

    Architecture
    ------------
    Block 1: Conv(3→32)  → BN → ReLU → Conv(32→32)  → BN → ReLU → MaxPool2d → Dropout
    Block 2: Conv(32→64) → BN → ReLU → Conv(64→64)  → BN → ReLU → MaxPool2d → Dropout
    Block 3: Conv(64→128)→ BN → ReLU → Conv(128→128)→ BN → ReLU → MaxPool2d → Dropout
    Classifier: Flatten → Linear(128*4*4 → 512) → ReLU → Dropout → Linear(512 → 10)

    Input : (N, 3, 32, 32)  — RGB CIFAR-10 images
    Output: (N, 10)         — raw logits
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 32×32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 32×32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 16×16
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 16×16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 16×16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 8×8
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 8×8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # 8×8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # 4×4
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
