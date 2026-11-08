import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.formatter.use_locale": False,
})

RUN_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def decimal_comma(ax, axes="xy"):
    """Ukrainian decimal separator on the given axes."""
    from matplotlib.ticker import FuncFormatter

    def fmt(x, _):
        s = f"{x:g}"
        return s.replace(".", ",")

    if "x" in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(fmt))
    if "y" in axes:
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))
