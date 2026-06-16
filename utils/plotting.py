"""Reusable Plotly chart builders.

These helpers return :class:`plotly.graph_objects.Figure` objects so they can
be embedded directly in the Streamlit dashboard. Plotly is imported lazily so
the core library has no hard dependency on it.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "fan_chart",
    "convergence_plot",
    "payoff_diagram",
    "bar_comparison",
]

_TEMPLATE = "plotly_dark"


def _go():
    import plotly.graph_objects as go

    return go


def fan_chart(fan_data: dict, sample_paths: Optional[np.ndarray] = None, title: str = "Path Fan Chart"):
    """Build a percentile fan chart with optional raw-path overlay.

    Parameters
    ----------
    fan_data : dict
        Output of :meth:`PathVisualizer.fan_chart_data`.
    sample_paths : numpy.ndarray, optional
        Subset of paths to overlay.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _go()
    t = fan_data["time_index"]
    fig = go.Figure()
    band_pairs = [("p5", "p95"), ("p25", "p75")]
    for lo, hi in band_pairs:
        if lo in fan_data and hi in fan_data:
            fig.add_trace(go.Scatter(x=t, y=fan_data[hi], line=dict(width=0), showlegend=False))
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=fan_data[lo],
                    fill="tonexty",
                    fillcolor="rgba(80,160,255,0.2)",
                    line=dict(width=0),
                    name=f"{lo}-{hi}",
                )
            )
    if "p50" in fan_data:
        fig.add_trace(go.Scatter(x=t, y=fan_data["p50"], line=dict(color="#50a0ff"), name="median"))
    if sample_paths is not None:
        for path in sample_paths:
            fig.add_trace(
                go.Scatter(
                    x=t, y=path, line=dict(width=0.5, color="rgba(200,200,200,0.25)"),
                    showlegend=False, hoverinfo="skip",
                )
            )
    fig.update_layout(template=_TEMPLATE, title=title, xaxis_title="Time step", yaxis_title="Price")
    return fig


def convergence_plot(df, analytical: Optional[float] = None, title: str = "Convergence"):
    """Plot price with a 95% CI band against the path count.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``n_paths``, ``price``, ``ci_lower`` and ``ci_upper``.
    analytical : float, optional
        Reference value drawn as a horizontal line.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _go()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["n_paths"], y=df["ci_upper"], line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=df["n_paths"], y=df["ci_lower"], fill="tonexty",
            fillcolor="rgba(80,160,255,0.2)", line=dict(width=0), name="95% CI",
        )
    )
    fig.add_trace(go.Scatter(x=df["n_paths"], y=df["price"], mode="lines+markers", name="MC price"))
    if analytical is not None:
        fig.add_hline(y=analytical, line_dash="dash", line_color="orange", annotation_text="analytical")
    fig.update_layout(
        template=_TEMPLATE, title=title, xaxis_title="n_paths", yaxis_title="Price",
        xaxis_type="log",
    )
    return fig


def payoff_diagram(S_T: Sequence[float], payoff: Sequence[float], title: str = "Payoff at Expiry"):
    """Line chart of payoff versus terminal spot.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _go()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_T, y=payoff, mode="lines", line=dict(color="#50a0ff")))
    fig.update_layout(template=_TEMPLATE, title=title, xaxis_title="S_T", yaxis_title="Payoff")
    return fig


def bar_comparison(labels: Sequence[str], values: Sequence[float], title: str = "", y_title: str = ""):
    """Simple bar chart for comparing methods.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _go()
    fig = go.Figure(go.Bar(x=list(labels), y=list(values), marker_color="#50a0ff"))
    fig.update_layout(template=_TEMPLATE, title=title, yaxis_title=y_title)
    return fig
