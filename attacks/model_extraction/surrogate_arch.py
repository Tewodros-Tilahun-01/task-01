"""Surrogate architecture based on leaked metadata hints.

Attacker's best guess using only: input shape, num classes, performance hint.
Different from victim's actual architecture (realistic black-box scenario).
"""

import torch.nn as nn


class RealisticSurrogate(nn.Module):
    """Surrogate CNN designed without exact victim architecture.
    
    Uses common CIFAR-10 patterns: 3 conv blocks, BatchNorm, dropout.
    Different from victim's actual architecture.
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 32×32 → 16×16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            # Block 2: 16×16 → 8×8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            # Block 3: 8×8 → 4×4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
