"""Figure 2 Panel A — Tel Aviv adjusted-univariate Cox volcano.

Exposes `plot_volcano_tlv_adjusted(ax)` for use by the figure assembler.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
TLV_CSV = os.path.join(DATA_DIR, "univariate_unadj_vs_adj.csv")
OXF_CSV = os.path.join(DATA_DIR, "serum_cox_adjusted.csv")

REPLICATED = ["GSN", "IGFBP2", "MEGF10"]

RISK_COLOR = "#BC4D43"
PROTECTIVE_COLOR = "#2f7e85"
NONSIG_COLOR = "#BBBBBB"
REFLINE_COLOR = "0.4"
REFLINE_STYLE = "--"
REFLINE_WIDTH = 0.6
SPINE_WIDTH = 0.5

FDR_THRESHOLD = 0.05
XLIM = (-2, 2)
YLIM = (0, 4)
AXIS_LABEL_FONTSIZE = 5
TICK_FONTSIZE = 5
TITLE_FONTSIZE = 6
LABEL_FONTSIZE = 4


def _load():
    df = pd.read_csv(TLV_CSV).dropna(subset=["adj_p", "adj_hazard_ratio"]).copy()
    oxf = pd.read_csv(OXF_CSV).dropna(subset=["p.value"])
    overlap = set(df["protein"]) & set(oxf["protein"])
    df = df[df["protein"].isin(overlap)].copy()
    df["FDR_overlap"] = multipletests(df["adj_p"], method="fdr_bh")[1]
    df["log2HR"] = np.log2(df["adj_hazard_ratio"].astype(float))
    df["neglog10FDR"] = -np.log10(df["FDR_overlap"].astype(float))
    return df


def plot_volcano_tlv_adjusted(ax):
    df = _load()
    sig_line = -np.log10(FDR_THRESHOLD)
    is_label = df["protein"].isin(REPLICATED)
    df_other = df[~is_label]
    df_red   = df[is_label & (df["adj_hazard_ratio"] > 1)]
    df_blue  = df[is_label & (df["adj_hazard_ratio"] < 1)]

    ax.scatter(df_other["log2HR"], df_other["neglog10FDR"],
               color=NONSIG_COLOR, alpha=0.7, edgecolors="none", s=7, zorder=2)
    ax.scatter(df_red["log2HR"], df_red["neglog10FDR"],
               color=RISK_COLOR, alpha=0.85, edgecolors="none", s=7, zorder=3)
    ax.scatter(df_blue["log2HR"], df_blue["neglog10FDR"],
               color=PROTECTIVE_COLOR, alpha=0.85, edgecolors="none", s=7, zorder=3)

    ax.axhline(sig_line, color=REFLINE_COLOR, linestyle=REFLINE_STYLE,
               linewidth=REFLINE_WIDTH, zorder=1)
    ax.axvline(0, color=REFLINE_COLOR, linestyle=REFLINE_STYLE,
               linewidth=REFLINE_WIDTH, zorder=1)

    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xticks(range(int(XLIM[0]), int(XLIM[1]) + 1))
    ax.set_yticks(np.arange(int(YLIM[0]), int(YLIM[1]) + 1, 1))

    texts = []
    for _, row in df[is_label].iterrows():
        texts.append(ax.text(
            float(row["log2HR"]), float(row["neglog10FDR"]),
            str(row["protein"]),
            fontsize=LABEL_FONTSIZE, ha="center", va="bottom",
        ))
    if adjust_text is not None and texts:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.4),
            only_move={"text": "xy", "points": "xy"},
            expand_text=(1.8, 2.0), expand_points=(1.8, 2.0),
            force_text=(3.0, 3.0), force_points=(1.5, 1.5), lim=5000,
        )
        resolve_label_overlaps(ax, texts, max_iter=300, step_px=4)

    ax.set_xlabel("log2(Hazard Ratio)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("−log10(FDR-adjusted P value)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Tel Aviv", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=SPINE_WIDTH)
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 400,
    })
    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    plot_volcano_tlv_adjusted(ax)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "volcano_tlv_adjusted")
    fig.savefig(out + ".png", bbox_inches="tight", dpi=400)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}.{{png,pdf}}")


if __name__ == "__main__":
    main()
