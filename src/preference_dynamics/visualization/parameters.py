"""
Parameter comparison visualization utilities.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from numpydantic import NDArray


def plot_parameter_comparison(
    y_true: torch.Tensor | NDArray[Literal["*, *"], float] | pd.DataFrame,
    y_pred: torch.Tensor | NDArray[Literal["*, *"], float] | pd.DataFrame,
    col_names: list[str] | None = None,
    *,
    title: str = "Predictions vs True Values",
    render_mode: str = "webgl",
    width: int = 700,  # fake aspect ratio ~ 1, not even sure if it's possible otherwise for facets
    **kwargs: Any,
) -> None:
    """
    Plot facet of scatter plots of predicted vs true values for each parameter
    and 45-degree reference line (perfect prediction).

    Args:
        y_true: True values of shape (n_samples, n_params)
        y_pred: Predicted values of shape (n_samples, n_params)
        col_names: Column names for parameters (default: "Target 0", "Target 1", ...)
        render_mode: Rendering mode, either "webgl" or "svg" (default: "webgl")
        **kwargs: Additional arguments for px.scatter

    Raises:
        ValueError: If y_true and y_pred have different shapes

    Returns:
        None, displays figure using fig.show()
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have same shape, got {y_true.shape} and {y_pred.shape}"
        )
    dfs = []
    y_np = {}
    for n, y in {"true": y_true, "pred": y_pred}.items():
        y_np[n] = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
        y_df = pd.DataFrame(y_np[n])
        if col_names:
            y_df.columns = col_names
        y_df = y_df.melt(value_name=n)
        dfs.append(y_df)
    df = dfs[0].join(dfs[1], lsuffix="_ignore")

    fig = px.scatter(
        df,
        x="true",
        y="pred",
        title=title,
        labels={
            "true": "True",
            "pred": "Predicted",
        },
        facet_col="variable",
        facet_col_wrap=4,
        facet_row_spacing=0.12,
        facet_col_spacing=0.05,
        opacity=0.3,
        render_mode=render_mode,
        width=width,
        **kwargs,
    )
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(matches=None, showticklabels=True)

    # 45-degree reference line
    # plotly is a joke, this took me 3h
    # yes, we need to specify ranges for all plots manually
    mins = np.min(y_np["true"], axis=0)
    maxs = np.max(y_np["true"], axis=0)
    mins_df = pd.DataFrame(np.concatenate([[mins, mins]])).T
    maxs_df = pd.DataFrame(np.concatenate([[maxs, maxs]])).T
    ref_df = pd.concat([mins_df, maxs_df], axis=0)
    ref_df["variable"] = ref_df.index

    # Don't use render_mode="auto", it will draw the line behind the data
    ref_fig = px.line(
        ref_df, x=0, y=1, facet_col="variable", facet_col_wrap=4, render_mode=render_mode
    )
    ref_fig.update_traces(line={"dash": "dash", "color": "red"})
    for t in ref_fig.data:
        fig.add_trace(t)

    fig.add_trace(ref_fig.data[0], row="all", col="all")

    fig.show()
