"""Figure 1 (fig:intro) of the paper: (a) distribution function of each p-value under lambda = 0 at (k,d) = (3,8),
(b) power at alpha = 0.05 at (3,8), as -log(1 - power) against lambda^2, (c) risk E sin^2(angle) of the argmax and of
the unfolding estimator divided by the lower bound of Theorem 4(i), at (3,8) and (5,3).
Reads figure1_data.json and figure1_null_ecdf.npz (written by reduce_raw.py from the field simulation).
Writes fig_intro.pdf (the file of the paper) and fig_intro.png.

usage: python3 make_figure.py [--power log|linear] [--out DIR]
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))


def lower_bound_risk(lam, d, k):
    """Smallest R with R >= ((d-1) - d R)^2 / ((d-1) k lam^2) (Theorem 4(i)): the smaller root (as in tests_lib.py)."""
    a = d * d
    b = -(2 * d * (d - 1) + (d - 1) * k * lam ** 2)
    c = (d - 1) ** 2
    return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)


ALPHA = 0.05
LOG_XMAX = 16

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "lines.linewidth": 1.0,
    "lines.markersize": 3, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})

# test name -> (label, colour, linestyle, marker)
STYLE = {
    "T":       (r"$t$-spacing test (closed form)", "C3", "-", "o"),
    "studmax": (r"studentised maximum $\sup Z/\hat\sigma$ (Monte Carlo)", "C0", "-", "s"),
    "F":       (r"locally optimal invariant test (exact)", "C2", "-", "^"),
    "unfold":  (r"unfolding spectral test (Monte Carlo)", "C1", "-", "D"),
    "oracle":  (r"most powerful invariant test (Monte Carlo)", "k", ":", None),
    "plugin":  (r"$\sup Z/\hat\sigma$ with the quantile of $\sup Z/\sigma$", "0.55", "-", None),
    "S":       (r"spacing test, $\sigma$ known (closed form)", "C3", "--", "o"),
    "max":     (r"maximum test $\sup Z/\sigma$, $\sigma$ known (Monte Carlo)", "C0", "--", "s"),
}
# panel (c): colour = estimator (red = argmax, the estimator of the t-spacing test; orange = unfolding), marker = pair
EST_COL = {"argmax": "C3", "svd": "C1"}
PAIR_MK = {(3, 3): "o", (3, 5): "^", (3, 8): "s", (5, 3): "D"}
ESTIM_LAMS = (0.125, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15)


def panel_h0(ax, ecdf, zoom=False):
    n = int(ecdf["n"])
    u = ecdf["rank"] / n
    order = ["plugin", "oracle", "unfold", "F", "studmax", "S", "max", "T"]
    for name in order:
        lab, col, ls, mk = STYLE[name]
        lw = 1.4 if name == "plugin" else 0.9
        ax.plot(ecdf["p_" + name], u, ls, color=col, lw=lw, label=lab)
    ax.plot([0, 1], [0, 1], color="0.75", lw=0.6, zorder=0)
    ax.set_xlabel(r"$p$-value"); ax.set_ylabel("distribution function under $\\lambda=0$")
    lim = 0.1 if zoom else 1.0
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(r"(a) level, $(k,d)=(3,8)$")


def panel_power(ax, power, scale):
    rows = {r["lam"]: r for r in power}
    keys = sorted(l for l in rows if (scale == "linear" or l * l <= LOG_XMAX + 1e-9))
    lams = np.array(keys, float)
    order = ["oracle", "studmax", "F", "unfold", "T", "S", "max"]
    for name in order:
        lab, col, ls, mk = STYLE[name]
        y = np.array([rows[l].get(name, np.nan) for l in keys])
        if scale == "linear":
            ax.plot(lams, y, ls, color=col, marker=mk, label=lab, markerfacecolor="white" if ls == "--" else col, mew=0.7)
        else:
            dfc = 1 - y
            ok = (dfc > 0) & (lams ** 2 <= LOG_XMAX)
            ax.plot(lams[ok] ** 2, -np.log(dfc[ok]), ls, color=col, marker=mk, label=lab,
                    markerfacecolor="white" if ls == "--" else col, mew=0.7)
    if scale == "linear":
        ax.axhline(ALPHA, color="0.75", lw=0.6, zorder=0)
        ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"power at $\alpha=0.05$"); ax.set_ylim(0, 1.02)
    else:
        ax.set_xlabel(r"$\lambda^2$"); ax.set_ylabel(r"$-\log(1-\mathrm{power})$")
        ax.set_xlim(0, LOG_XMAX); ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_title(r"(b) power, $(k,d)=(3,8)$")


def panel_estim(ax, est, pairs=((3, 8), (5, 3))):
    """Risk E sin^2(angle) of the two estimators divided by the lower bound of Theorem 4(i)."""
    for (k, d) in pairs:
        mk = PAIR_MK[(k, d)]
        pts = sorted((r["lam"], r["mean_sin2_argmax"], r["mean_sin2_svd"]) for r in est
                     if r["k"] == k and r["d"] == d and r["lam"] in ESTIM_LAMS)
        if not pts:
            continue
        lam = np.array([p[0] for p in pts]); ra = np.array([p[1] for p in pts]); rs = np.array([p[2] for p in pts])
        lb = lower_bound_risk(lam, d, k)
        ax.plot(lam, ra / lb, "-", color=EST_COL["argmax"], lw=0.8, marker=mk, ms=3.0)
        ax.plot(lam, rs / lb, "--", color=EST_COL["svd"], lw=0.8, marker=mk, ms=3.0, markerfacecolor="white", mew=0.8)
    ax.axhline(1.0, color="k", lw=0.7)
    ax.set_xscale("log"); ax.set_xlim(0.1, 17)
    ax.set_xticks([0.1, 0.3, 1, 3, 10]); ax.set_xticklabels(["0.1", "0.3", "1", "3", "10"]); ax.minorticks_off()
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("risk / lower bound")
    ax.set_title(r"(c) estimation of the direction")
    h = [Line2D([], [], color=EST_COL["argmax"], ls="-", lw=1.2, label="argmax"),
         Line2D([], [], color=EST_COL["svd"], ls="--", lw=1.2, label="unfolding")]
    h += [Line2D([], [], color="k", marker=PAIR_MK[kd], ls="", ms=3.0, label=r"$(%d,%d)$" % kd) for kd in pairs]
    ax.set_ylim(0.85, 6.0)
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(-0.02, 1.02), frameon=False, handlelength=1.6,
              borderaxespad=0.1, ncol=2, columnspacing=0.8)


def build(data, ecdf, scale="log"):
    fig, ax = plt.subplots(1, 3, figsize=(5.62, 2.45))
    panel_h0(ax[0], ecdf)
    panel_power(ax[1], data["power_k3d8"], scale)
    panel_estim(ax[2], data["estimation"])
    for axi in ax:
        axi.set_box_aspect(1)
    order = ["T", "studmax", "F", "unfold", "oracle", "plugin", "S", "max"]
    handles = []
    for name in order:
        lab, col, ls, mk = STYLE[name]
        handles.append(Line2D([], [], color=col, ls=ls, marker=mk, ms=3, lw=1.4 if name == "plugin" else 1.0,
                              markerfacecolor="white" if ls == "--" else col, mew=0.7, label=lab))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, columnspacing=1.0, handlelength=2.2)
    fig.tight_layout(pad=0.3, w_pad=0.9, rect=(0, 0.25, 1, 1.02))
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", default="log", choices=["log", "linear"])
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    data = json.load(open(os.path.join(HERE, "figure1_data.json")))
    ecdf = dict(np.load(os.path.join(HERE, "figure1_null_ecdf.npz")))
    fig = build(data, ecdf, a.power)
    name = "fig_intro" if a.power == "log" else "fig_intro_linear"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print(f"wrote {name}.pdf and {name}.png")


if __name__ == "__main__":
    main()
