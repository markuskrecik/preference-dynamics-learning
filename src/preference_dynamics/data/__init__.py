"""
Data pipeline and validation for preference dynamics.

This module provides:
- Pydantic schemas for data validation
- Dataset loading and saving utilities
- PyTorch Dataset classes
- Data preprocessing and normalization
"""

from preference_dynamics.data.manager import DataManager
from preference_dynamics.data.schemas import DataConfig

__all__ = [
    "DataConfig",
    "DataManager",
]
