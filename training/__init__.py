"""
JustRL Training Module

This module implements the simple RL recipe from JustRL:
- Single-stage training with fixed hyperparameters
- GRPO (Group Relative Policy Optimization) algorithm
- Binary outcome rewards from lightweight DAPO verifier
"""

from .train import JustRLConfig, GRPOTrainer, MathDataset
from .verifier import grade_answer_binary, extract_boxed_answer, normalize_answer

__all__ = [
    "JustRLConfig",
    "GRPOTrainer",
    "MathDataset",
    "grade_answer_binary",
    "extract_boxed_answer",
    "normalize_answer",
]

