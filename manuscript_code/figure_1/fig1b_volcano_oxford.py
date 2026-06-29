"""Figure 1 Panel B — Oxford univariate Cox volcano (broken y-axis for NEFL).

Exposes `plot_volcano_oxford(fig, subplot_spec)` which carves the given
GridSpec cell into a nested 2-row layout for the broken-axis effect. Running
as a script renders a standalone PNG/PDF.
"""
import os
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

from adjustText import adjust_text

DATA_DIR = os.environ.get("ALS_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OXF_CSV = os.path.join(DATA_DIR, "serum_cox_unadjusted.csv")
TLV_CSV = os.path.join(DATA_DIR, "significant_uniivariate_fdr_for_paper_new.csv")

REPLICATED = [
    "GSN", "APOL1", "PEPD", "KIT", "GHR", "PTX3", "TNFRSF12A", "FGL1",
    "LTBP2", "LRG1", "ITIH3", "REG1A", "EFNA1", "EPHB4", "APOF", "CTSB",
    "CD14", "CHGB", "IGFBP2",
]
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
Y_LOW = (0, 10)
Y_HIGH = (24, 26)
AXIS_LABEL_FONTSIZE = 5
TICK_FONTSIZE = 5
TITLE_FONTSIZE = 6
LABEL_FONTSIZE = 4


def _load():
    oxf = pd.read_csv(OXF_CSV)
    oxf = oxf.rename(columns={"protein": "Protein", "p.value": "p_value", "q": "q"})
    oxf["hazard"] = np.exp(oxf["beta"].astype(float))
    oxf = oxf.dropna(subset=["p_value", "hazard"]).copy()

    tlv = pd.read_csv(TLV_CSV).rename(columns={"protein": "Protein"})
    overlap = set(tlv["Protein"]) & set(oxf["Protein"])
    df = oxf[oxf["Protein"].isin(overlap)].copy()
    df["FDR_overlap"] = multipletests(df["p_value"], method="fdr_bh")[1]

    nefl_row = oxf[oxf["Protein"] == "NEFL"].copy()
    if len(nefl_row) and "NEFL" not in df["Protein"].values:
        nefl_row["FDR_overlap"] = nefl_row["q"].values
        df = pd.concat([df, nefl_row], ignore_index=True)

    df["log2HR"] = np.log2(df["hazard"].astype(float))
    df["neglog10FDR"] = -np.log10(df["FDR_overlap"].astype(float).clip(lower=1e-300))
    return df


def plot_volcano_oxford(fig, subplot_spec):
    df = _load()
    sig_line = -np.log10(FDR_THRESHOLD)
    is_sig = df["FDR_overlap"] < FDR_THRESHOLD
    is_labeled = df["Protein"].isin(LABEL_PROTEINS)
    df_nonsig   = df[~is_sig & ~is_labeled]
    df_red      = df[is_sig & (df["hazard"] > 1) & ~is_labeled]
    df_blue     = df[is_sig & (df["hazard"] < 1) & ~is_labeled]
    df_lab_red  = df[is_labeled & (df["hazard"] > 1)]
    df_lab_blue = df[is_labeled & (df["hazard"] < 1)]

    inner = _gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=subplot_spec,
        height_ratios=[1, 3], hspace=0.08,
    )
    ax_hi = fig.add_subplot(inner[0])
    ax_lo = fig.add_subplot(inner[1], sharex=ax_hi)

    for ax in (ax_hi, ax_lo):
        ax.scatter(df_nonsig["log2HR"], df_nonsig["neglog10FDR"],
                   color=NONSIG_COLOR, alpha=0.5, edgecolors="none", s=7, zorder=2)
        ax.scatter(df_red["log2HR"], df_red["neglog10FDR"],
                   color=RISK_COLOR, alpha=0.5, edgecolors="none", s=7, zorder=3)
        ax.scatter(df_blue["log2HR"], df_blue["neglog10FDR"],
                   color=PROTECTIVE_COLOR, alpha=0.5, edgecolors="none", s=7, zorder=3)
        ax.scatter(df_lab_red["log2HR"], df_lab_red["neglog10FDR"],
                   color=RISK_COLOR, alpha=1.0, edgecolors="none", s=7, zorder=4)
        ax.scatter(df_lab_blue["log2HR"], df_lab_blue["neglog10FDR"],
                   color=PROTECTIVE_COLOR, alpha=1.0, edgecolors="none", s=7, zorder=4)
        ax.set_xlim(*XLIM)

    ax_lo.axhline(sig_line, color=REFLINE_COLOR, linestyle=REFLINE_STYLE,
                  linewidth=REFLINE_WIDTH, zorder=1)

    ax_hi.set_ylim(*Y_HIGH)
    ax_lo.set_ylim(*Y_LOW)

    # Hide shared spines
    ax_hi.spines["bottom"].set_visible(False)
    ax_hi.spines["top"].set_visible(False)
    ax_hi.spines["right"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_lo.spines["right"].set_visible(False)
    ax_hi.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    df_labels = df[df["Protein"].isin(LABEL_PROTEINS)].copy()
    texts_lo, texts_hi = [], []
    for _, row in df_labels.iterrows():
        y_val = float(row["neglog10FDR"])
        x_val = float(row["log2HR"])
        name  = str(row["Protein"])
        fw = "bold" if name == "NEFL" else "normal"
        if Y_LOW[0] <= y_val <= Y_LOW[1]:
            texts_lo.append(ax_lo.text(x_val, y_val, name,
                                       fontsize=LABEL_FONTSIZE,
                                       fontweight=fw, color="black", zorder=4))
        elif Y_HIGH[0] <= y_val <= Y_HIGH[1]:
            texts_hi.append(ax_hi.text(x_val, y_val, name,
                                       fontsize=LABEL_FONTSIZE,
                                       fontweight=fw, color="black", zorder=4))

    adjust_text(
        texts_lo, ax=ax_lo,
        expand_text=(1.2, 1.4), expand_points=(1.4, 1.6),
        force_text=(0.5, 0.7), force_points=(0.3, 0.5),
        arrowprops=dict(arrowstyle="-", color="0.55", lw=0.4, shrinkA=4),
    )
    if texts_hi:
        adjust_text(
            texts_hi, ax=ax_hi,
            expand_text=(1.2, 1.4), expand_points=(1.4, 1.6),
            force_text=(0.5, 0.7), force_points=(0.3, 0.5),
            arrowprops=dict(arrowstyle="-", color="0.55", lw=0.4, shrinkA=4),
        )

    ax_hi.set_title("Oxford", fontsize=TITLE_FONTSIZE)
    ax_lo.set_xlabel("log2(Hazard Ratio)", fontsize=AXIS_LABEL_FONTSIZE)
    # Shared y-label centered on the panel
    bbox_hi = ax_hi.get_position()
    bbox_lo = ax_lo.get_position()
    y_mid = 0.5 * (bbox_hi.y1 + bbox_lo.y0)
    x_lab = bbox_lo.x0 - 0.045
    fig.text(x_lab, y_mid, "−log10(FDR-adjusted P value)",
             va="center", ha="center", rotation="vertical",
             fontsize=AXIS_LABEL_FONTSIZE)

    ax_lo.set_xticks(range(int(XLIM[0]), int(XLIM[1]) + 1))
    ax_hi.set_yticks([24, 26])
    ax_lo.set_yticks(np.arange(0, 11, 2))
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

    # Break marks + x=0 dashed reference across the broken-axis gap
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
        "font.family": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 400,
    })
    fig = plt.figure(figsize=(2.4, 2.2))
    gs = _gridspec.GridSpec(1, 1, figure=fig)
    plot_volcano_oxford(fig, gs[0])
    out = os.path.join(OUT_DIR, "volcano_oxford")
    fig.savefig(out + ".png", bbox_inches="tight", dpi=400)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}.{{png,pdf}}")


if __name__ == "__main__":
    main()
