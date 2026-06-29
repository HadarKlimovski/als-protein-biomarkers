"""Figure 1 Panel A — Tel Aviv univariate Cox volcano (unadjusted, overlap-FDR).

Exposes `plot_volcano_tlv(ax)` for use by the figure assembler. Running as a
script renders a standalone PNG/PDF.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

from adjustText import adjust_text

DATA_DIR = os.environ.get("ALS_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TLV_CSV = os.path.join(DATA_DIR, "significant_uniivariate_fdr_for_paper_new.csv")
OXF_CSV = os.path.join(DATA_DIR, "serum_cox_unadjusted.csv")

REPLICATED = [
    "GSN", "APOL1", "PEPD", "KIT", "GHR", "PTX3", "TNFRSF12A", "FGL1",
    "LTBP2", "LRG1", "ITIH3", "REG1A", "EFNA1", "EPHB4", "APOF", "CTSB",
    "CD14", "CHGB", "IGFBP2",
]

RISK_COLOR = "#BC4D43"
PROTECTIVE_COLOR = "#2f7e85"
NONSIG_COLOR = "#BBBBBB"
REFLINE_COLOR = "0.4"
REFLINE_STYLE = "--"
REFLINE_WIDTH = 0.6
SPINE_WIDTH = 0.5

FDR_THRESHOLD = 0.05
XLIM = (-2, 2)
YLIM = (0, 5)
AXIS_LABEL_FONTSIZE = 5
TICK_FONTSIZE = 5
TITLE_FONTSIZE = 6
LABEL_FONTSIZE = 4


def _load():
    tlv = pd.read_csv(TLV_CSV).rename(columns={"protein": "Protein"})
    tlv = tlv.dropna(subset=["p_value", "hazard_ratio"]).copy()
    oxf = pd.read_csv(OXF_CSV).rename(columns={"protein": "Protein"})
    overlap = set(tlv["Protein"]) & set(oxf["Protein"])
    df = tlv[tlv["Protein"].isin(overlap)].copy()
    df["FDR_overlap"] = multipletests(df["p_value"], method="fdr_bh")[1]
    df["log2HR"] = np.log2(df["hazard_ratio"].astype(float))
    df["neglog10FDR"] = -np.log10(df["FDR_overlap"].astype(float).clip(lower=1e-300))
    return df


def plot_volcano_tlv(ax):
    df = _load()
    sig_line = -np.log10(FDR_THRESHOLD)
    is_sig = df["FDR_overlap"] < FDR_THRESHOLD
    is_labeled = df["Protein"].isin(REPLICATED)
    df_nonsig    = df[~is_sig & ~is_labeled]
    df_red       = df[is_sig & (df["hazard_ratio"] > 1) & ~is_labeled]
    df_blue      = df[is_sig & (df["hazard_ratio"] < 1) & ~is_labeled]
    df_lab_red   = df[is_labeled & (df["hazard_ratio"] > 1)]
    df_lab_blue  = df[is_labeled & (df["hazard_ratio"] < 1)]

    # Background dots: alpha 0.5. Labeled dots: full opacity, drawn on top.
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

    ax.axhline(sig_line, color=REFLINE_COLOR, linestyle=REFLINE_STYLE,
               linewidth=REFLINE_WIDTH, zorder=1)
    ax.axvline(0, color=REFLINE_COLOR, linestyle=REFLINE_STYLE,
               linewidth=REFLINE_WIDTH, zorder=1)

    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xticks(range(int(XLIM[0]), int(XLIM[1]) + 1))
    ax.set_yticks(np.arange(int(YLIM[0]), int(YLIM[1]) + 1, 1))

    df_labels = df[df["Protein"].isin(REPLICATED)].copy()
    # Place labels at their dot, then let adjust_text push them apart and
    # draw short leader lines back to dots only for labels that had to move.
    texts = [
        ax.text(float(r["log2HR"]), float(r["neglog10FDR"]),
                str(r["Protein"]), fontsize=LABEL_FONTSIZE,
                color="black", zorder=4)
        for _, r in df_labels.iterrows()
    ]
    adjust_text(
        texts, ax=ax,
        expand_text=(1.2, 1.4), expand_points=(1.4, 1.6),
        force_text=(0.5, 0.7), force_points=(0.3, 0.5),
        arrowprops=dict(arrowstyle="-", color="0.55", lw=0.4, shrinkA=4),
    )

    ax.set_xlabel("log2(Hazard Ratio)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("−log10(FDR-adjusted P value)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Tel Aviv", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=SPINE_WIDTH)
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)


def main():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 400,
    })
    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    plot_volcano_tlv(ax)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "volcano_tlv")
    fig.savefig(out + ".png", bbox_inches="tight", dpi=400)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}.{{png,pdf}}")


if __name__ == "__main__":
    main()
