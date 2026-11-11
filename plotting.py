"""Small plotting helpers shared by the data-analysis scripts."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ingarch_core import acf, cum_periodogram, pacf

C_OBS, C_A, C_B, C_C = "#222222", "#1f6fb4", "#d9731a", "#3a9a5c"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "font.family": "serif",
})


def stem_acf(ax, values, n, title="", ylabel="ACF", start=1):
    lags = np.arange(start, start + len(values))
    ax.vlines(lags, 0, values, color=C_OBS, lw=1.1)
    ax.axhline(0, color=C_OBS, lw=0.5)
    b = 1.96 / np.sqrt(n)
    ax.axhline(b, color=C_A, ls="--", lw=0.7)
    ax.axhline(-b, color=C_A, ls="--", lw=0.7)
    ax.set_xlabel("Lag")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def plot_acf(ax, z, nlags, title="", partial=False):
    v = (pacf(z, nlags) if partial else acf(z, nlags))[1:]
    stem_acf(ax, v, len(z), title, "PACF" if partial else "ACF")


def plot_cpgram(ax, z, title=""):
    f, cp, crit = cum_periodogram(z)
    ax.plot(f, cp, color=C_OBS, lw=0.9)
    line = (f - f[0]) / (f[-1] - f[0])
    ax.plot(f, line + crit, color=C_A, ls="--", lw=0.7)
    ax.plot(f, line - crit, color=C_A, ls="--", lw=0.7)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Cum. periodogram")
    if title:
        ax.set_title(title)


def plot_pit(ax, heights, title=""):
    b = len(heights)
    ax.bar((np.arange(b) + 0.5) / b, heights, width=1 / b, color="#b9c7d6",
           edgecolor="white", lw=0.6)
    ax.axhline(1, color=C_OBS, ls="--", lw=0.7)
    ax.set_xlabel("PIT")
    ax.set_ylabel("Relative frequency")
    ax.set_xlim(0, 1)
    if title:
        ax.set_title(title)


def marginal_bands(sims, edges):
    """Bin-wise mean and 95% band of the relative frequencies of simulated
    paths (rows of `sims`)."""
    H = np.array([np.histogram(s, bins=edges)[0] / len(s) for s in sims])
    return H.mean(0), np.quantile(H, 0.025, 0), np.quantile(H, 0.975, 0)


def marginal_summary(x):
    x = np.asarray(x, float)
    m = x.mean()
    return np.array([m, x.var(ddof=1), x.var(ddof=1) / m, np.mean(x == 0),
                     np.quantile(x, 0.5), np.quantile(x, 0.95), x.max()])
