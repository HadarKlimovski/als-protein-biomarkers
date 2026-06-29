"""Figure 2 Panel B — Oxford adjusted-univariate Cox volcano (broken y-axis for NEFL).

Exposes `plot_volcano_oxford_adjusted(fig, subplot_spec)` for use by the assembler.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec as _gridspec
import seaborn as sns
from statsmodels.stats.multitest import multipletests

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG1 = os.path.join(_HERE, "..", "..", "figure_1")
if _FIG1 not in sys.path:
    sys.path.insert(0, os.path.abspath(_FIG1))
from label_utils import resolve_label_overlaps  # noqa: E402

DATA_DIR = os.environ.get("ALS_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OXF_CSV = os.path.join(DATA_DIR, "serum_cox_adjusted.csv")
TLV_CSV = os.path.join(DATA_DIR, "univariate_unadj_vs_adj.csv")

REPLICATED = ["GSN", "IGFBP2", "MEGF10"]
LABEL_PROTEINS = REPLICATED + ["NEFL"]

RISK_COLOR = "#BC4D43"
PROTECTIVE_COLOR = "#2f7e85"
NONSIG_COLOR = "#BBBBBB"
REFLINE_COLOR = "0.4"
REFLINE_STYLE = "--"
REFLINE_WIDTH = 0.6
SPINE_WIDTH = 0.5

FDR_THRESHOLD = 0.05
XLIM = (-2, 2)
Y_LOW = (0, 5)
Y_HIGH = (22, 26)
AXIS_LABEL_FONTSIZE = 5
TICK_FONTSIZE = 5
TITLE_FONTSIZE = 6
LABEL_FONTSIZE = 4


def _load():
    df_full = pd.read_csv(OXF_CSV).dropna(subset=["p.value", "HR"]).copy()
    tlv = pd.read_csv(TLV_CSV).dropna(subset=["adj_p"])
    overlap = set(df_full["protein"]) & set(tlv["protein"])
    df = df_full[df_full["protein"].isin(overlap)].copy()
    df["FDR_overlap"] = multipletests(df["p.value"], method="fdr_bh")[1]

    nefl_row = df_full[df_full["protein"] == "NEFL"].copy()
    if len(nefl_row) and "NEFL" not in df["protein"].values:
        nefl_row["FDR_overlap"] = nefl_row["q"].values
        df = pd.concat([df, nefl_row], ignore_index=True)

    df["log2HR"] = np.log2(df["HR"].astype(float))
    df["neglog10FDR"] = -np.log10(df["FDR_overlap"].astype(float).clip(lower=1e-300))
    return df


def plot_volcano_oxford_adjusted(fig, subplot_spec):
    df = _load()
    sig_line = -np.log10(FDR_THRESHOLD)
    is_label = df["protein"].isin(LABEL_PROTEINS)
    df_other = df[~is_label]
    df_red   = df[is_label & (df["HR"].astype(float) > 1)]
    df_blue  = df[is_label & (df["HR"].astype(float) < 1)]

    inner = _gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=subplot_spec,
        height_ratios=[1, 3], hspace=0.08,
    )
    ax_hi = fig.add_subplot(inner[0])
    ax_lo = fig.add_subplot(inner[1], sharex=ax_hi)

    for ax in (ax_hi, ax_lo):
        ax.scatter(df_other["log2HR"], df_other["neglog10FDR"],
                   color=NONSIG_COLOR, alpha=0.7, edgecolors="none", s=7, zorder=2)
        ax.scatter(df_red["log2HR"], df_red["neglog10FDR"],
                   color=RISK_COLOR, alpha=0.85, edgecolors="none", s=7, zorder=3)
        ax.scatter(df_blue["log2HR"], df_blue["neglog10FDR"],
                   color=PROTECTIVE_COLOR, alpha=0.85, edgecolors="none", s=7, zorder=3)
        ax.set_xlim(*XLIM)

    ax_lo.axhline(sig_line, color=REFLINE_COLOR, linestyle=REFLINE_STYLE,
                  linewidth=REFLINE_WIDTH, zorder=1)
    ax_hi.set_ylim(*Y_HIGH)
    ax_lo.set_ylim(*Y_LOW)

    ax_hi.spines["bottom"].set_visible(False)
    ax_hi.spines["top"].set_visible(False)
    ax_hi.spines["right"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_lo.spines["right"].set_visible(False)
    ax_hi.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    df_labels = df[df["protein"].isin(LABEL_PROTEINS)].copy()
    texts_lo, texts_hi = [], []
    for _, row in df_labels.iterrows():
        y_val = float(row["neglog10FDR"])
        x_val = float(row["log2HR"])
        name  = str(row["protein"])
        fw    = "bold" if name == "NEFL" else "normal"
        if Y_LOW[0] <= y_val <= Y_LOW[1]:
            t = ax_lo.text(x_val, y_val, name, ha="center", va="bottom",
                           fontsize=LABEL_FONTSIZE, fontweight=fw)
            texts_lo.append(t)
        elif Y_HIGH[0] <= y_val <= Y_HIGH[1]:
            t = ax_hi.text(x_val, y_val, name, ha="center", va="bottom",
                           fontsize=LABEL_FONTSIZE, fontweight=fw)
            texts_hi.append(t)
    if adjust_text is not None:
        if texts_lo:
            adjust_text(texts_lo, ax=ax_lo,
                        only_move={"text": "xy", "points": "xy"},
                        expand_text=(1.8, 2.0), expand_points=(1.8, 2.0),
                        force_text=(3.0, 3.0), force_points=(1.5, 1.5), lim=5000,
                        arrowprops=dict(arrowstyle="-", color="0.5", lw=0.4))
            resolve_label_overlaps(ax_lo, texts_lo, max_iter=300, step_px=4)
        if texts_hi:
            adjust_text(texts_hi, ax=ax_hi,
                        arrowprops=dict(arrowstyle="-", color="0.5", lw=0.4))
            resolve_label_overlaps(ax_hi, texts_hi, max_iter=100, step_px=4)

    ax_hi.set_title("Oxford", fontsize=TITLE_FONTSIZE)
    ax_lo.set_xlabel("log2(Hazard Ratio)", fontsize=AXIS_LABEL_FONTSIZE)
    # Shared y-label
    bbox_hi = ax_hi.get_position()
    bbox_lo = ax_lo.get_position()
    y_mid = 0.5 * (bbox_hi.y1 + bbox_lo.y0)
    x_lab = bbox_lo.x0 - 0.045
    fig.text(x_lab, y_mid, "−log10(FDR-adjusted P value)",
             va="center", ha="center", rotation="vertical",
             fontsize=AXIS_LABEL_FONTSIZE)

    ax_lo.set_xticks(range(int(XLIM[0]), int(XLIM[1]) + 1))
    ax_hi.yaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
    ax_lo.set_yticks(np.arange(int(Y_LOW[0]), int(Y_LOW[1]) + 1, 1))
    for ax in (ax_hi, ax_lo):
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=SPINE_WIDTH)
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_WIDTH)

    # Re-assert hidden spines after layout
    ax_hi.spines["bottom"].set_visible(False)
    ax_hi.spines["top"].set_visible(False)
    ax_hi.spines["right"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_lo.spines["right"].set_visible(False)

    fig.canvas.draw()
    spine_lw = ax_lo.spines["left"].get_linewidth()
    x_spine  = ax_lo.get_position().x0
    y_bot_hi = ax_hi.get_position().y0
    y_top_lo = ax_lo.get_position().y1
    dx, dy = 0.006, 0.006
    for ycenter in (y_bot_hi, y_top_lo):
        mark = Line2D([x_spine - dx, x_spine + dx],
                      [ycenter - dy, ycenter + dy],
                      transform=fig.transFigure, color="k",
                      linewidth=spine_lw, clip_on=False)
        fig.add_artist(mark)

    ax_lo.axvline(0, color=REFLINE_COLOR, linestyle=REFLINE_STYLE,
                  linewidth=REFLINE_WIDTH, zorder=1)
    ax_hi.axvline(0, color=REFLINE_COLOR, linestyle=REFLINE_STYLE,
                  linewidth=REFLINE_WIDTH, zorder=1)
    x_zero_fig = ax_lo.transData.transform((0, 0))[0]
    x_zero_fig = fig.transFigure.inverted().transform((x_zero_fig, 0))[0]
    gap = Line2D([x_zero_fig, x_zero_fig], [y_top_lo, y_bot_hi],
                 transform=fig.transFigure, color=REFLINE_COLOR,
                 linestyle=REFLINE_STYLE, linewidth=REFLINE_WIDTH,
                 clip_on=False, zorder=1)
    fig.add_artist(gap)
    return ax_hi, ax_lo


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 400,
    })
    fig = plt.figure(figsize=(2.4, 2.2))
    gs = _gridspec.GridSpec(1, 1, figure=fig)
    plot_volcano_oxford_adjusted(fig, gs[0])
    out = os.path.join(OUT_DIR, "volcano_oxford_adjusted")
    fig.savefig(out + ".png", bbox_inches="tight", dpi=400)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}.{{png,pdf}}")


if __name__ == "__main__":
    main()
