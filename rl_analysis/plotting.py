import os
import platform
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from toolz import merge
from os.path import join
from collections import namedtuple


COLORS = {
    "green": (0.21647058823529414, 0.64, 0.28),
    "gray": (0.5077342149936128, 0.5077349468656874, 0.5077399378226443)
}

snc_learn = sns.blend_palette(["#8AFF9C", "#00660F"], n_colors=6)
vta_learn = sns.blend_palette(["#E4C2FF", "#5A009E"], n_colors=6)
ctrl_learn = sns.dark_palette("#DEDEDE", 7, reverse=True)[:-1]
palette_map = {"snc": snc_learn, "vta": vta_learn, "ctrl": ctrl_learn}

# compares snc, vta, and control conditions
GenoPalette = namedtuple("GenoPalette", ("snc", "vta", "ctrl"))
geno_palette = GenoPalette(snc_learn[3], vta_learn[3], ctrl_learn[3])

StimPalette = namedtuple("StimPalette", ("stim", "catch", "ctrl"))
stim_palette = StimPalette("#00AAFF", "#ABABAB", geno_palette.ctrl)

syllable_aliases = {
	20: "Run",
	27: "Pause",
	17: "Scrunch",
	76: "Rear up, turn right",
	30: "Pause, turn left", 
	59: "Rear",
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


# Jeff's saving function
def savefig(fig=None, basename=None, fmts=["pdf", "png"], **kwargs):

	if basename is None:
		return RuntimeError("Need filename to save!")
	if fig is None:
		fig = plt.gcf()

	bground_color = plt.rcParams["figure.facecolor"]
	suffix = f"{bground_color}_bground"

	for _fmt in fmts:
		try:
			fig.savefig(f"{basename}_{suffix}.{_fmt}", **kwargs)
		except FileNotFoundError:
			import os
			os.makedirs(os.path.dirname(basename))
			fig.savefig(f"{basename}_{suffix}.{_fmt}", **kwargs)


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


default_label_map = {
    "change_count": "Counts above baseline",
    "fold_change_count": "Counts above baseline (ratio)",
    "log2_fold_change_count": "Counts above baseline (log2-ratio)",
    "out_p_adj_entropy": "Outgoing entropy (adj.)",
    "in_p_adj_entropy": "Incoming entropy (adj.)",
    "03-gram adj2 entropy": "Block entropy (3-gram)",
    "distance of similar syllables (z correlation)": "Syllable uniqueness",
    "velocity_2d_mm (mean)": "2D velocity (mm/s, average)",
    "velocity_2d_mm": "2D velocity (mm/s)",
    "velocity_3d_mm": "3D velocity (mm/s)",
    "height_ave_mm (mean)": "Ave. height (mm)",
    "height_ave_mm": "Ave. height (mm)",
    "pca scaled trace (cv) robust_minmax_with_clipping": "Kin. variability",
    "rle": "Usage",
    "acceleration_2d_mm": "2D acceleration",
    "velocity_angle": "Ang. velocity (degrees/s)",
    "velocity_height": "Z velocity (mm/s)",
    "block_entropy": "Block entropy",
    "scalar trace (mean) cv robust minmax": "Kinematic var.",
    "duration (mean)": "Duration (average)",
    "total_mutual_information_hutter": "Mutual Info.",
    "transition": "Syll. transitions",
    "predicted_syllable": "Syll.",
}


def clean_plot_labels(ax=None, label_map={}):
    label_map = merge(default_label_map, label_map)
    if ax is None:
        ax = plt.gca()

    fig = plt.gcf()
    fig.canvas.draw()

    # for text_obj in ax.findobj(plt.matplotlib.text.Text):
    #     for k, v in label_map.items():
    #         if text_obj.get_text() == k:
    #             text_obj.set_text(v)

    xlabels = ax.get_xticklabels()
    ylabels = ax.get_yticklabels()

    # print(xlabels)
    # print(ylabels)
    new_xlabels = [
        label_map[_.get_text()] if _.get_text() in label_map.keys() else _ for _ in xlabels
    ]
    new_ylabels = [
        label_map[_.get_text()] if _.get_text() in label_map.keys() else _ for _ in ylabels
    ]

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    # need to fix this first...
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels(new_xlabels)
    ax.set_yticklabels(new_ylabels)

    # # trigger an update
    # ax.set_xticklabels([_.get_text() for _ in ax.get_xticklabels()])
    # ax.set_yticklabels([_.get_text() for _ in ax.get_yticklabels()])


def clean_ticks(
    ax,
    axis="x",
    precision=2,
    zero_symmetry=True,
    include_zero=True,
    eps=np.spacing(1.0),
    dtype=float,
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


def format_pval(pval, thresholds=[-np.inf, 0.001, 0.01, 0.05, np.inf]):
    import pandas as pd

    try:
        iter(pval)
    except TypeError:
        pval = [pval]
    # make bins, assume last bin defines non-significance
    labels = [f"p < {_}" for _ in thresholds[1:]]
    labels[-1] = "n.s."
    bin_labels = pd.cut(pval, bins=thresholds, right=False, labels=labels)
    return bin_labels.astype("str")


def plot_pval(
    pval_df,
    keys,
    key_levels,
    ax=None,
    colors=None,
    alpha_threshold=0.05,
    continuity_threshold=0,
    position="top",
    offset=0.9,
    height=0.05,
    min_width=0,
    spacing=1.5,
    **kwargs,
):
    import pandas as pd
    if ax is None:
        ax = plt.gca()
    
    if colors is None:
        colors = sns.color_palette()
    
    if position == "top":
        offset = ax.get_ylim()[1] * offset
        height = (ax.get_ylim()[1] - ax.get_ylim()[0]) * height
    elif position == "bottom":
        offset = ax.get_ylim()[0] * offset
        height = -(ax.get_ylim()[1] - ax.get_ylim()[0]) * height
    
    for _key, _color in zip(keys, colors):
        try:
            significance = pval_df.xs(_key, level=key_levels) < alpha_threshold
        except TypeError:
            significance = pval_df[_key] < alpha_threshold

        significance_group = (significance.diff() != 0).cumsum().rename("group")
        significance_df = pd.concat([significance_group, significance.rename("alpha")], axis=1).set_index("group", append=True)
        
        group_sz = significance_df.groupby("group").size()
        group_alpha = significance_df.groupby("group")["alpha"].all()
        plt_groups = group_sz.loc[(group_sz >= continuity_threshold) & (group_alpha)].index
        
        for _group in plt_groups:
            plt_index = significance_df.xs(_group, level="group").index.tolist()
            if (plt_index[-1] - plt_index[0]) < min_width:
                plt_index.append(plt_index[-1] + min_width)
            # if len(plt_index) < min_width:
                # plt_index
            ax.fill_between(plt_index, offset, offset + height, fc=_color, ec=None, lw=0, clip_on=False, **kwargs)
        
        if len(keys) > 0:
            offset += height * spacing


def lighten_color(color, amount=0.5):
    """
        Lightens the given color by multiplying (1-luminosity) by the given amount.
        Input can be matplotlib color string, hex string, or RGB tuple.
    ​
        Examples:
        >> lighten_color('g', 0.3)
        >> lighten_color('#F034A3', 0.6)
        >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])