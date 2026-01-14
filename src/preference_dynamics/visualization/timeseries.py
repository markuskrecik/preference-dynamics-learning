"""
Time series visualization functions.
"""

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from preference_dynamics.schemas import TimeSeriesSample


def _plot_time_series_single(sample: TimeSeriesSample, **kwargs: Any) -> go.Figure:
    """
    Plot time series for a single sample showing desires and efforts in separate subplots.

    Args:
        sample: TimeSeriesSample object containing time series data
        **kwargs: Additional keyword arguments for plotly.express.line

    Returns:
        plotly.graph_objects.Figure with time series plots
    """

    n = sample.n_actions
    col_names = [
        ["Desire"] * n + ["Effort"] * n,
        [f"Desire u_{i}" for i in range(n)] + [f"Effort a_{i}" for i in range(n)],
    ]

    df = pd.DataFrame(sample.time_series.T, columns=col_names)
    df[("Time", "")] = sample.time_points
    df_long = df.melt(id_vars=[("Time", "")])
    df_long.columns = ["Time", "Type", "Variable", "Value"]
    fig = px.line(
        df_long,
        x="Time",
        y="Value",
        color="Variable",
        facet_row="Type",
        **kwargs,
    )
    fig.update_yaxes(matches=None, showticklabels=True)

    return fig


def _plot_time_series_batch(samples: list[TimeSeriesSample], **kwargs: Any) -> go.Figure:
    """
    Plot time series for multiple samples showing desires and efforts in separate subplots.
    samples are overlaid on the same plots with labels indicating sample index.

    Args:
        samples: list of TimeSeriesSample objects containing time series data
        **kwargs: Additional keyword arguments for plotly.express.line

    Returns:
        plotly.graph_objects.Figure with time series plots
    """
    for s_i, sample in enumerate(samples):
        n = sample.n_actions
        col_names = [
            ["Desire"] * n + ["Effort"] * n,
            [f"Desire u_{i}, s={s_i}" for i in range(n)]
            + [f"Effort a_{i}, s={s_i}" for i in range(n)],
        ]

        df = pd.DataFrame(sample.time_series.T, columns=col_names)
        df[("Time", "")] = sample.time_points
        df_long = df.melt(id_vars=[("Time", "")])
        df_long.columns = ["Time", "Type", "Variable", "Value"]
        fig = px.line(
            df_long,
            x="Time",
            y="Value",
            color="Variable",
            facet_row="Type",
            **kwargs,
        )
        fig.update_yaxes(matches=None, showticklabels=True)
        if s_i == 0:
            final_fig = fig
        else:
            final_fig.add_traces(fig.data)

    return final_fig


def plot_time_series(sample: TimeSeriesSample | list[TimeSeriesSample], **kwargs: Any) -> None:
    """
    Plot time series for desires and efforts using plotly.

    Plots either a single sample or multiple samples overlaid. Creates subplots
    for desires and efforts.

    Args:
        sample: TimeSeriesSample or list of TimeSeriesSample objects
        **kwargs: Additional keyword arguments for plotly.express.line

    Returns:
        None, displays figure using fig.show()

    Raises:
        ValueError: If input type is invalid
    """
    if isinstance(sample, TimeSeriesSample):
        fig = _plot_time_series_single(sample, **kwargs)
    elif isinstance(sample, list):
        fig = _plot_time_series_batch(sample, **kwargs)
    else:
        raise ValueError(
            f"Invalid input type: {type(sample)}, expected TimeSeriesSample or list[TimeSeriesSample]"
        )
    fig.show()
