"""
Training metrics visualization utilities.

This module provides functions to visualize training progress and model performance metrics.
"""

from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from numpydantic import NDArray
from plotly.subplots import make_subplots

from preference_dynamics.schemas import TrainingHistory


def plot_training_curves(
    history: TrainingHistory,
    show_val: bool = True,
    **kwargs: Any,
) -> go.Figure:
    """
    Plot training curves (loss over epochs) using plotly express.

    Creates line plots showing training loss and optionally validation loss
    over epochs. Also shows epoch times if available.

    Args:
        history: Training history dictionary with keys:
            - 'train_loss': list[float] (required)
            - 'val_loss': list[float] (optional)
            - 'epoch_times': list[float] (optional)
            - 'best_epoch': int (optional) - epoch index (0-based) of best validation loss
        show_val: Whether to show validation loss if available
        **kwargs: Additional keyword arguments for plotly.graph_objects.Figure.update_layout
    """

    train_loss = history.train_loss
    epochs = np.arange(1, len(train_loss) + 1)

    n_plots = 1
    if history.epoch_times:
        n_plots += 1

    fig = make_subplots(
        rows=n_plots,
        cols=1,
        subplot_titles=(
            ["Training and Validation Loss"] + (["Epoch Duration"] if n_plots > 1 else [])
        ),
        vertical_spacing=0.25,
    )

    # Plot loss curves
    fig.add_trace(
        go.Scatter(
            x=epochs, y=train_loss, name="Train Loss", mode="lines+markers", marker={"size": 6}
        ),
        row=1,
        col=1,
    )
    if show_val and history.val_loss:
        val_loss = history.val_loss
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=val_loss,
                name="Validation Loss",
                mode="lines+markers",
                marker={"size": 6, "symbol": "square"},
            ),
            row=1,
            col=1,
        )

    # Mark best epoch if available
    if history.best_epoch:
        best_epoch = history.best_epoch
        fig.add_vline(
            x=best_epoch + 1,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            annotation_text="Best Epoch",
            row=1,
            col=1,
        )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)

    # Plot epoch times if available
    if history.epoch_times:
        epoch_times = history.epoch_times
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=epoch_times,
                name="Epoch Time",
                mode="lines+markers",
                marker={"size": 6, "symbol": "triangle-up", "color": "green"},
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Time (seconds)", row=2, col=1)

    fig.update_layout(
        title_text="Training Progress",
        showlegend=True,
        **kwargs,
    )

    fig.show()


def plot_metrics(
    metrics: dict[str, float | NDArray[Literal["*"], float]],
    target_names: list[str] | None = None,
    *,
    width: int = 700,
    height: int = 1000,
    **kwargs: Any,
) -> None:
    """
    Plot all performance metrics from compute_metrics().

    Creates visualizations showing:
    1. All scalar metrics (MSE, MAE, RMSE, R², MAPE, MRE, NRMSE, correlations, etc.) as bar chart
    2. All per-target metrics (MAE, MSE, MRE, NRMSE) as bar charts

    Args:
        metrics: Metrics dictionary from compute_metrics() with keys:
            - 'mse': float (optional)
            - 'mae': float (optional)
            - 'rmse': float (optional)
            - 'r2': float (optional)
            - 'mre': float (optional)
            - 'nrmse_range': float (optional)
            - 'nrmse_std': float (optional)
            - 'pearson_r': float (optional)
            - 'spearman_r': float (optional)
            - 'medae': float (optional)
            - 'per_target_mse': array (optional)
            - 'per_target_mae': array (optional)
            - 'per_target_mre': array (optional)
            - 'per_target_nrmse': array (optional)
        target_names: List of target names
        **kwargs: Additional keyword arguments for plotly.graph_objects.Figure.update_layout

    Example:
        >>> from preference_dynamics.training.metrics import compute_metrics
        >>> metrics = compute_metrics(y_true, y_pred)
        >>> plot_metrics(metrics)
    """

    if target_names is None:
        target_names = [f"Target {i}" for i in range(len(metrics["per_target_mae"]))]  # type: ignore

    scalar_metrics = {
        "MSE": "mse",
        "RMSE": "rmse",
        "NRMSE (range)": "nrmse_range",
        "NRMSE (std)": "nrmse_std",
        "MAE": "mae",
        "MedAE": "medae",
        "MRE": "mre",
        "R²": "r2",
        "Pearson r": "pearson_r",
        "Spearman r": "spearman_r",
    }

    metric_names = []
    metric_values = []
    for display_name, key in scalar_metrics.items():
        if key in metrics:
            metric_names.append(display_name)
            metric_values.append(float(metrics[key]))

    per_target_metrics = {
        "RMSE": "per_target_rmse",
        "NRMSE": "per_target_nrmse",
        "MAE": "per_target_mae",
        "MRE": "per_target_mre",
    }

    available_per_target = {
        display_name: metrics[key]
        for display_name, key in per_target_metrics.items()
        if key in metrics
    }

    n_per_target = len(available_per_target)
    n_cols = 2
    n_rows = 1 + ((n_per_target + 1) // 2)

    subplot_titles = ["Overall Performance Metrics", ""]
    for display_name in available_per_target:
        subplot_titles.append(f"Per-Target {display_name}")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Overall scalar metrics bar chart
    if metric_names:
        fig.add_trace(
            go.Bar(
                x=metric_names,
                y=metric_values,
                text=[f"{v:.4f}" for v in metric_values],
                textposition="outside",
                name="Overall Metrics",
            ),
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text="Metric", row=1, col=1, tickangle=-45)
        fig.update_yaxes(title_text="Value", row=1, col=1, range=[0, 1.1])

    row_idx = 2
    col_idx = 1

    for display_name, per_target_values in available_per_target.items():
        per_target_array = np.asarray(per_target_values)
        if len(per_target_array) != len(target_names):
            raise ValueError(
                f"Incorrect number of target names:  {len(target_names)}, expected {len(per_target_array)}."
            )
        else:
            x_labels = target_names

        tick_indices = list(range(len(x_labels)))
        tick_labels = x_labels

        fig.add_trace(
            go.Bar(
                x=list(range(len(per_target_array))),
                y=per_target_array,
                name=f"Per-Target {display_name}",
                text=[f"{v:.4f}" if abs(v) > 1e-3 else f"{v:.2e}" for v in per_target_array],
                textposition="outside",
                showlegend=False,
            ),
            row=row_idx,
            col=col_idx,
        )

        fig.update_xaxes(
            title_text="Target",
            row=row_idx,
            col=col_idx,
            tickvals=tick_indices,
            ticktext=tick_labels,
            tickangle=-45,
        )
        fig.update_yaxes(title_text=display_name, row=row_idx, col=col_idx, range=[0, 1.1])

        col_idx += 1
        if col_idx > n_cols:
            col_idx = 1
            row_idx += 1

    fig.update_layout(
        title_text="Model Performance Metrics",
        showlegend=False,
        width=width,
        height=height,
        **kwargs,
    )

    fig.show()
