import matplotlib.pyplot as plt
import platform
import seaborn as sns
from rl_analysis.util import merge_dict



all_fig_dct = {
	"pdf.fonttype": 42,
	"font.sans-serif": "Arial",
	"font.family": "sans-serif",
	"mathtext.fontset": "custom",
	"mathtext.rm": "Liberation Sans",
	"mathtext.it": "Liberation Sans:italic",
	"mathtext.bf": "Liberation Sans:bold",
	"savefig.facecolor": "white",
	"figure.facecolor": "white",
	"lines.linewidth": 1,
	"axes.edgecolor": "black",
	"axes.labelcolor": "black",
	"xtick.color": "black",
	"ytick.color": "black",
}

if platform.system() != "Darwin":
	all_fig_dct["ps.usedistiller"] = "xpdf"

# all in points
font_dct = {
	"axes.labelpad": 3.5,
	"font.size": 7,
	"figure.titlesize": 7,
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
	"ytick.major.pad": 1.5,
}

plot_config = merge_dict(all_fig_dct, font_dct)


def get_default_env():
	config1 = merge_dict(sns.axes_style("white"), merge_dict(sns.axes_style("ticks"), plot_config))
	config2 = merge_dict(sns.plotting_context("paper"), config1)
	return config2


def setup_plotting_env():
	plt.style.use("default")
	sns.set_style("white", merge_dict(sns.axes_style("ticks"), plot_config))
	sns.set_context("paper", rc=plot_config)
	if platform.system() != "Darwin":
		plt.rcParams["ps.usedistiller"] = "xpdf"
	plt.rcParams["pdf.fonttype"] = 42