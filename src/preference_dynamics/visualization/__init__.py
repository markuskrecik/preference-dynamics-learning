"""
Visualization module for preference dynamics parameter learning.

This module provides plotting utilities for:
- Time series visualization (desires and efforts)
- Parameter comparison plots (true vs predicted)
- Training curves (loss over epochs)
- Performance metrics (MAE per parameter)
"""

from preference_dynamics.visualization.metrics import (
    plot_metrics,
    plot_training_curves,
)
from preference_dynamics.visualization.parameters import (
    plot_parameter_comparison,
)
from preference_dynamics.visualization.timeseries import (
    plot_time_series,
)

__all__ = [
    "plot_time_series",
    "plot_parameter_comparison",
    "plot_training_curves",
    "plot_metrics",
]
