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
# noinspection PyUnresolvedReferences
import os
# noinspection PyUnresolvedReferences
import numpy as np
# noinspection PyUnresolvedReferences
import matplotlib.pylab as plt
# noinspection PyUnresolvedReferences
import matplotlib.ticker as ticker
# noinspection PyUnresolvedReferences
import pstats
# noinspection PyUnresolvedReferences
import cProfile

import sys
import resource

import pandas as pd
import seaborn as sns
from tabulate import tabulate
import matplotlib as mpl
from IPython import get_ipython

# noinspection PyUnresolvedReferences
from liran_tools import plot_utils, results, text, timeformat
from liran_tools.plot_utils import *


def init_notebook(dpi=300, figsize=(8, 3), bg='none'):
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (1000, -1))
    except ValueError:
        pass

    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_colwidth', 200)
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['figure.facecolor'] = bg
    plt.rcParams['figure.figsize'] = figsize


def notebook_fig(fig_height=4, fig_width=8, **kwargs):
    return plt.figure(figsize=(fig_width, fig_height), **kwargs)


def init_paper_plot(font_size=13, factor=1.3, acm=True):
    # Maybe add fonttype, or maybe remove T1 fontenc (or switch to T3)
    # See issue: https://github.com/matplotlib/matplotlib/issues/11303
    # mpl.rcParams['pdf.fonttype'] = 42

    plt.rcParams['font.size'] = font_size
    plt.rcParams['text.usetex'] = True

    default_dpi = mpl.rcParamsDefault['figure.dpi']
    mpl.rcParams['figure.dpi'] = default_dpi * factor

    if acm:
        plt.rcParams['text.latex.preamble'] = "\n".join([
            r'\usepackage[tt=false, type1=true]{libertine}',
            r'\usepackage[libertine]{newtxmath}',
        ])


def fix_tex_comment():
    if plt.rcParams['text.usetex']:
        axes = plt.gcf().get_axes()
        for ax in axes:
            l = ax.get_xlabel()
            ax.set_xlabel(l.replace("%", "\\%"))

            l = ax.get_ylabel()
            ax.set_ylabel(l.replace("%", "\\%"))


def save_fig(filename, lgd=None, fig_format='pdf', cwd=None, **kwargs):
    fix_tex_comment()

    if cwd is None:
        cwd = os.getcwd()

    filepath = os.path.join(cwd, 'figs', f"{filename}.{fig_format}")
    if lgd is None:
        try:
            lgd = plt.gca().legend_
        except:
            pass

    if lgd is None:
        bbox_extra_artists = None
    elif type(lgd) in (list, tuple):
        bbox_extra_artists = lgd
    else:
        bbox_extra_artists = (lgd,)

    plt.savefig(filepath, format=fig_format, bbox_extra_artists=bbox_extra_artists,
                bbox_inches='tight', dpi=300, **kwargs)


def show_colors(color_list, title=None):
    sns.palplot(color_list)
    if title:
        plt.title(title)
    plt.show()


def print_table(t):
    print(tabulate(t, floatfmt=",.2f"))


def goto_main_project_folder(project_name: str):
    cwd = os.getcwd()
    while os.path.basename(cwd) != project_name:
        if cwd in ('/', ''):
            raise RuntimeError("Cannot find main project path.")
        cwd = os.path.dirname(cwd)

    os.chdir(cwd)

    if cwd not in sys.path:
        sys.path.append(cwd)


def goto_main_repo_folder():
    cwd = os.getcwd()
    while '.git' not in cwd:
        if cwd in ('/', ''):
            raise RuntimeError("Cannot find main repo path.")
        cwd = os.path.dirname(cwd)

    os.chdir(cwd)

    if cwd not in sys.path:
        sys.path.append(cwd)
