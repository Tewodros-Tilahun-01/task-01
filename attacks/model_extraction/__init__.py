"""
Model extraction attack components.

This module provides functionality for stealing ML models via API queries:
- query: Collect labeled dataset from victim API
- training: Train surrogate model using knowledge distillation
- evaluation: Measure agreement and test transferability
- surrogate_arch: Realistic surrogate architecture (attacker's guess)
"""

from .query import collect_query_data
from .surrogate_arch import RealisticSurrogate
from .training import train_surrogate, save_training_results
from .evaluation import (
    evaluate_agreement,
    craft_surrogate_adversarials,
    test_transferability,
)

__all__ = [
    "collect_query_data",
    "RealisticSurrogate",
    "train_surrogate",
    "save_training_results",
    "evaluate_agreement",
    "craft_surrogate_adversarials",
    "test_transferability",
]
