"""
Author: Liran Funaro <liran.funaro@gmail.com>

Copyright (C) 2006-2023 Liran Funaro

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import itertools
import datetime

import matplotlib
import numpy as np
from matplotlib import pylab as plt
from matplotlib.patches import FancyBboxPatch

from typing import Union, List, Optional


def nicer_plot(ax: Optional[plt.Axes] = None, grid_x=True, grid_y=True):
    if ax is None:
        for i, ax in enumerate(plt.gcf().get_axes()):
            nicer_plot(ax, grid_x=grid_x, grid_y=grid_y if i == 0 else False)
        return

    if grid_x and grid_y:
        ax.grid(True, linestyle=':', alpha=0.8, which='both')
    else:
        if grid_y:
            ax.yaxis.grid(True, linestyle=':', alpha=0.8, which='both')
        else:
            ax.yaxis.grid(False, which='both')
        if grid_x:
            ax.xaxis.grid(True, linestyle=':', alpha=0.8, which='both')
        else:
            ax.xaxis.grid(False, which='both')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def top_legend(ax: Union[None, plt.Axes, List[plt.Axes]] = None, n_cols=None, t=0, mode="expand",
               handles=None, labels=None, **kwargs):
    """
    See :func:`plt.Axes.legend`
    :param ax: Axis
    :param n_cols: number of columns
    :param t: headroom at the top
    :param mode: "expand" or None
    :param handles: :func:`plt.Axes.legend`
    :param labels: :func:`plt.Axes.legend`
    :param kwargs: other arguments
    :return:
    """
    if n_cols is None:
        for k in ('cols', 'n', 'ncols', 'ncol'):
            n_cols = kwargs.get(k, None)
            if n_cols is not None:
                break
        if n_cols is None:
            n_cols = 4

    if ax is None:
        return top_legend(plt.gcf().get_axes(),
                          n_cols=n_cols, t=t, mode=mode, handles=handles, labels=labels)

    if isinstance(ax, (list, tuple)):
        c_handles, c_labels = map(
            list,
            map(itertools.chain.from_iterable, zip(*[a.get_legend_handles_labels() for a in ax]))
        )
        if handles is None:
            handles = c_handles
        if labels is None:
            labels = c_labels
        ax = ax[0]

    loc = 'lower left' if mode == "expand" else 'lower center'

    return ax.legend(loc=loc, bbox_to_anchor=(0, 1 + t, 1, 100),
                     ncol=n_cols, frameon=False,
                     mode=mode, borderaxespad=0.,
                     handles=handles, labels=labels, **kwargs)


def bottom_legend(ax=None, n_cols=10, t=1, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05 * t),
              ncol=n_cols, frameon=False, **kwargs)


def apply_xticks(func):
    ticks, labels = plt.xticks()
    plt.xticks(ticks, map(func, ticks))


def apply_yticks(func):
    ticks, labels = plt.yticks()
    plt.yticks(ticks, map(func, ticks))


def seconds_xticks(ax=None):
    if ax is None:
        ax = plt.gca()

    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _pos: str(datetime.timedelta(seconds=x))))
    ax.tick_params(axis="x", labelrotation=45)


def nice_patch(ax=None, mutation_aspect=2):
    if ax is None:
        ax = plt.gca()
    y_min, y_max = ax.get_ylim()
    y_sz = y_max - y_min
    x_min, x_max = ax.get_xlim()
    x_sz = x_max - x_min

    ma = mutation_aspect * y_sz / x_sz

    for patch in list(ax.patches):
        bb = patch.get_bbox()
        color = patch.get_facecolor()
        width = abs(bb.width)
        height = abs(bb.height)
        pad = width / 4

        p_bbox = FancyBboxPatch((bb.xmin + pad * 1.1, bb.ymin),
                                width - pad * 1.1 * 2, height - pad * ma,
                                boxstyle=f"round,pad={pad}",
                                ec="none", fc=color,
                                mutation_aspect=ma,
                                )
        patch.remove()
        ax.add_patch(p_bbox)


ABBREVIATIONS = {
    -4: "p",  # pico  - 1e-12
    -3: "n",  # nano  - 1e-9
    -2: "μ",  # micro - 1e-6
    -1: "m",  # milli - 1e-3
    0: "",
    1: "K",  # kilo  - 1e3
    2: "M",  # mega  - 1e6
    3: "G",  # giga  - 1e9
    4: "T",  # tera  - 1e12
    5: "P",  # peta  - 1e15
    6: "E",  # exa   - 1e18
}


def _get_number_level(value: int | float) -> int:
    if value == 0:
        return 0

    abs_value = abs(value)
    for level in sorted(ABBREVIATIONS):
        if abs_value < 10 ** (3 * level):
            return level - 1
    return max(ABBREVIATIONS)


def _large_number_ticks(value: int | float) -> str:
    level = _get_number_level(value)
    adjusted_value = round(value / 10 ** (3 * level), 2)
    return f"{adjusted_value:g}{ABBREVIATIONS[level]}"


def large_number_y_ticks(ax: Optional[plt.Axes] = None):
    if ax is None:
        ax = plt.gca()
    ticks = ax.get_yticks()
    abs_ticks = np.abs(ticks)
    if ((abs_ticks == 0) | ((abs_ticks >= 1) & (abs_ticks < 1e3))).all():
        return
    ax.set_yticks(ticks)
    ax.set_yticklabels(list(map(_large_number_ticks, ticks)))
