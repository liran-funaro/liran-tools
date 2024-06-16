"""
Author: Liran Funaro <liran.funaro@gmail.com>
"""
import dataclasses
import sys
import typing
from typing import Optional

import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
from IPython.core.display_functions import display

from liran_tools import plot_utils
from liran_tools.results import Experiment, ExperimentGroup

from liran_tools.results import analyze

DataFrameModifier = typing.Callable[[pd.DataFrame], pd.DataFrame]


@dataclasses.dataclass
class PlotConfig(analyze.DataConfig):
    throughput_label: str | None = None
    latency_label: str | None = None
    hue: str | None = None
    phase: str | None = None
    n_cols: int = 3
    limit: tuple[int, int] = (None, None)
    filter_func: DataFrameModifier | None = None
    map_func: DataFrameModifier | None = None


DEFAULT_CONFIG = PlotConfig(**dataclasses.asdict(analyze.DEFAULT_CONFIG))


def _get_ax(ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    return ax


def plot_phase_boundaries(df: pd.DataFrame, conf: PlotConfig = DEFAULT_CONFIG, ax: Optional[plt.Axes] = None):
    # Plot boundaries
    ax = _get_ax(ax)

    for k, color in zip(df[conf.phase].unique(), ("red", "blue", "black", "green")):
        try:
            l, h = analyze.get_min_max_time(df[df[conf.phase] == k])
            ax.axvline(l, color=color, linewidth=1, linestyle="--", label=k.title())
            ax.axvline(h, color=color, linewidth=1, linestyle="--")
        except Exception as ex:
            print(f"Failed plotting {k} boundaries:", ex, file=sys.stderr)


def plot_workload_begin(
        e: Experiment, df: pd.DataFrame,
        conf: PlotConfig = DEFAULT_CONFIG, ax: Optional[plt.Axes] = None,
):
    if len(df) == 0:
        return

    if conf.phase is not None:
        return plot_phase_boundaries(df, conf, ax)

    first_time_seconds = df[analyze.TimeField].iloc[0]
    first_timestamp = analyze.get_timestamp_col(df).iloc[0]
    w_start = e.min_time_no_tz
    w_end = e.max_time_no_tz

    workload_start = (w_start - first_timestamp).total_seconds() + first_time_seconds
    workload_end = (w_end - first_timestamp).total_seconds() + first_time_seconds

    ax = _get_ax(ax)
    ax.axvline(workload_start, color="black", linestyle="--", linewidth=1, label="Workload")
    ax.axvline(workload_end, color="black", linestyle="--", linewidth=1)


def filter_df(pdf: pd.DataFrame, conf: PlotConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    if conf.filter_func:
        pdf = conf.filter_func(pdf)

    l_lim, h_lim = conf.limit
    if h_lim is not None and h_lim < 0:
        _, h_lim = analyze.get_min_max_time(pdf)
    if l_lim is not None and h_lim is not None:
        pdf = pdf[(pdf[analyze.TimeField] > l_lim) & (pdf[analyze.TimeField] < h_lim)]
    elif l_lim is not None:
        pdf = pdf[pdf[analyze.TimeField] > l_lim]
    elif h_lim is not None:
        pdf = pdf[pdf[analyze.TimeField] < h_lim]

    pdf = pdf.reset_index()
    if conf.map_func:
        pdf = conf.map_func(pdf)
    return pdf


def _experiment_plot(
        e: Experiment, df: pd.DataFrame, y_label: Optional[str],
        conf: PlotConfig, ax: plt.Axes
):
    plot_workload_begin(e, df, conf=conf, ax=ax)
    ax.set_xlabel("Experiment time")
    if y_label:
        ax.set_ylabel(y_label)
    ax.set_ylim(0, None)
    plot_utils.large_number_y_ticks(ax=ax)
    plot_utils.nicer_plot(ax=ax)
    plot_utils.top_legend(n_cols=conf.n_cols, ax=ax)
    plot_utils.seconds_xticks(ax=ax)


def plot(e: Experiment, conf: PlotConfig = DEFAULT_CONFIG, ax: Optional[plt.Axes] = None):
    df = analyze.get(e, conf=conf)
    df = filter_df(df, conf=conf)
    ax = _get_ax(ax)
    sns.lineplot(df, x=analyze.TimeField, y=conf.value_field, hue=conf.hue, style=conf.hue, ax=ax)
    _experiment_plot(e, df, conf.throughput_label, conf, ax)


def plot_hist(e: Experiment, fields: list[str], conf: PlotConfig = DEFAULT_CONFIG):
    dfs = analyze.get_hist(e, fields, conf=conf)
    for k, df in dfs.items():
        sns.barplot(df, x="le", y=conf.value_field)
        plot_utils.nicer_plot()
        ax = plt.gca()
        t = ax.get_xticklabels()
        ax.set_xticklabels(t, rotation=90)
        plt.xlabel(k.replace("_", " "))
        plt.show()


def plot_all_exp(eg: ExperimentGroup, conf: list[PlotConfig] = (DEFAULT_CONFIG,)):
    for row, e in eg.iter_exp():
        display(row.to_frame().T)
        fig, axes = plt.subplots(1, len(conf), figsize=(3 * len(conf), 3))
        for ax, c in zip(axes, conf):
            try:
                plot(e, conf=dataclasses.replace(c, n_cols=1), ax=ax)
            except Exception as ex:
                print(e, ex, file=sys.stderr)
        plt.tight_layout()
        plt.show()


def plt_bar_df(
        pdf: pd.DataFrame, x: str, y: str, conf: PlotConfig = DEFAULT_CONFIG,
        ax: Optional[plt.Axes] = None, **kwargs,
):
    pdf = filter_df(pdf, conf=conf)
    ax = _get_ax(ax)
    sns.barplot(pdf, x=x, y=y, hue=conf.hue, ax=ax, **kwargs)
    plot_utils.nicer_plot(ax=ax)
    plot_utils.large_number_y_ticks(ax=ax)
    if conf.hue is not None:
        plot_utils.top_legend(n_cols=conf.n_cols, ax=ax)
    return pdf


def plt_line_df(
        pdf: pd.DataFrame, x: str, y: str, conf: PlotConfig = DEFAULT_CONFIG,
        ax: Optional[plt.Axes] = None, **kwargs,
):
    pdf = filter_df(pdf, conf=conf)
    ax = _get_ax(ax)
    sns.lineplot(pdf, x=x, y=y, hue=conf.hue, style=conf.hue, markers=True, ax=ax, **kwargs)
    ax.set_ylim(0, None)
    plot_utils.nicer_plot(ax=ax)
    plot_utils.large_number_y_ticks(ax=ax)
    if conf.hue is not None:
        plot_utils.top_legend(n_cols=conf.n_cols, ax=ax)
    return pdf
