import os
import platform
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from toolz import merge
from os.path import join

COLORS = {
    "green": (0.21647058823529414, 0.64, 0.28),
    "gray": (0.5077342149936128, 0.5077349468656874, 0.5077399378226443)
}


def fg(width, height):
    return plt.figure(figsize=(width, height))


def subplots(nrows, ncols, width, height, **kwargs):
    return plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)


def add_legend(ax=None, **kwargs):
    if ax is None:
        plt.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1), **kwargs)
    else:
        ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1), **kwargs)


def save_factory(folder, backgrounds=('white',), tight_layout=True):
    folder = os.path.expanduser(folder)
    os.makedirs(folder, exist_ok=True)
    def save(fig, name, savefig=True, tight_layout=tight_layout):
        if tight_layout:
            fig.tight_layout()
        if savefig:
            for bg in backgrounds:
                ext = '' if len(backgrounds) == 1 else f'_{bg}'
                for ax in fig.axes:
                    ax.set_facecolor(bg)
                fig.savefig(join(folder, name + ext + '.png'),
                            dpi=150, facecolor=bg)
                fig.savefig(join(folder, name + ext + '.pdf'), facecolor=bg)
        return fig
    return save



def setup_plotting_env():
    all_fig_dct = {
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': 'black',
        "axes.labelcolor": "black",
        "text.color": "black",
        'xtick.color': 'black',
        'ytick.color': 'black',
        'svg.fonttype': 'none',
        'lines.linewidth': 1,
    }

    # all in points
    font_dct = {
        "axes.labelpad": 3.5,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 5,
        "xtick.major.size": 3.6,
        "ytick.major.size": 3.6,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5
    }

    plot_config = merge(all_fig_dct, font_dct)

    plt.style.use('default')
    sns.set_style('white', merge(sns.axes_style('ticks'), plot_config))
    sns.set_context('paper', rc=plot_config)

    if platform.system() != 'Darwin':
        plt.rcParams['ps.usedistiller'] = 'xpdf'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['figure.dpi'] = 200


def aceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10 ** precision), 10 ** precision)


def afloor(a, precision=0):
    return np.true_divide(np.floor(a * 10 ** precision), 10 ** precision)


def clean_ticks(
    ax, axis="x", precision=2, zero_symmetry=True, include_zero=True, eps=np.spacing(1.0), dtype=float,
):
    # call self if we want all axes
    if axis.lower()[0] == "a":
        clean_ticks(
            ax,
            axis="x",
            precision=precision,
            zero_symmetry=zero_symmetry,
            include_zero=include_zero,
        )
        clean_ticks(
            ax,
            axis="y",
            precision=precision,
            zero_symmetry=zero_symmetry,
            include_zero=include_zero,
        )

    # if it's a single axis make a list
    if isinstance(ax, (plt.matplotlib.axes._subplots.Axes)):
        ax = np.array([ax])

    limits = []
    for _ax in ax.ravel():
        if axis.lower()[0] == "x":
            limits.append(_ax.get_xlim())
        else:
            limits.append(_ax.get_ylim())

    limits = np.array(limits)

    # do the axes span zero?
    span_zero = ((limits[:, 0] < 0) & (limits[:, 1] > 0)).any()

    # if axes span zero, force symmetry
    if span_zero and zero_symmetry:
        new_limits = [aceil(np.max(np.abs(limits)), precision)] * 2
        new_limits[0] = -new_limits[0]
    else:
        new_limits = [
            afloor(np.min(limits[:, 0]), precision),
            aceil(np.max(limits[:, 1]), precision),
        ]

    if span_zero and include_zero:
        new_ticks = [new_limits[0], 0, new_limits[1]]
    else:
        new_ticks = [new_limits[0], new_limits[1]]

    new_limits[0] = new_limits[0] - eps
    new_limits[-1] = new_limits[-1] + eps

    new_ticks = [dtype(_) for _ in new_ticks]

    if axis.lower()[0] == "x":
        for _ax in ax.ravel():
            _ax.set_xlim(new_limits)
            _ax.set_xticks(new_ticks)
            _ax.set_xticklabels(new_ticks)
    else:
        for _ax in ax.ravel():
            _ax.set_ylim(new_limits)
            _ax.set_yticks(new_ticks)
            _ax.set_yticklabels(new_ticks)


def plot_bootstrap_matrix(
    x, y, ci=99, label=None, color=None, alpha=0.5, mu=None, linewidth=1, ax=None, **plt_kwargs
):
    '''Expects a tidy dataframe that has already been bootstrapped. I.e., each trial is the
    average of a different sample (with replacement) of the original data'''
    if mu is None:
        _mu = np.nanmean(y, axis=0)
        mu = _mu
    if ci == "sd":
        lo = mu - 2 * np.nanstd(y, axis=0)
        hi = mu + 2 * np.nanstd(y, axis=0)
    else:
        lo = np.nanquantile(y, 1 - (ci / 100), axis=0)
        hi = np.nanquantile(y, ci / 100, axis=0)

    plotter = plt if ax is None else ax

    lines = plotter.plot(x, mu, label=label, color=color, linewidth=linewidth, **plt_kwargs)
    color = lines[-1].get_color()
    plotter.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, **plt_kwargs)

    return plt.gca() if ax is None else ax


def bootstrap_lineplot(
    df, x, y, ci=99, label=None, color=None, alpha=0.5, mu=None, linewidth=1, ax=None, **plt_kwargs
):
    '''Expects a tidy dataframe that has already been bootstrapped. I.e., each trial is the
    average of a different sample (with replacement) of the original data'''
    gb = df.groupby(x)
    _mu = gb[y].mean()
    if mu is None:
        mu = _mu
    else:
        import pandas as pd
        mu = pd.Series(mu, index=_mu.index)
    if ci == "sd":
        lo = mu - (2 * gb[y].std())
        hi = mu + (2 * gb[y].std())
    else:
        lo = gb[y].quantile(1 - (ci / 100))
        hi = gb[y].quantile(ci / 100)

    plotter = plt if ax is None else ax

    lines = plotter.plot(mu.index, mu, label=label, color=color, linewidth=linewidth, **plt_kwargs)
    color = lines[-1].get_color()
    plotter.fill_between(mu.index, lo, hi, color=color, alpha=alpha, linewidth=0, **plt_kwargs)

    return plt.gca() if ax is None else ax
