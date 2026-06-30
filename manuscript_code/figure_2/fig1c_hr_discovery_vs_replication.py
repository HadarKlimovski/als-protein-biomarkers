"""Figure 1 Panel C — Discovery vs Replication log2HR scatter for 19 replicated proteins.

Exposes `plot_hr_correlation(ax)` for use by the figure assembler.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

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

XLIM = (-2, 2)
YLIM = (-2, 2)
AXIS_LABEL_FONTSIZE = 5
TICK_FONTSIZE = 5
TITLE_FONTSIZE = 6
LABEL_FONTSIZE = 4
ANNOT_FONTSIZE = 4.5


def _format_sci(p):
    if not np.isfinite(p):
        return "NA"
    if p == 0:
        return "<1 × 10⁻³⁰⁰"
    exponent = int(np.floor(np.log10(abs(p))))
    mantissa = p / (10 ** exponent)
    sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return f"{mantissa:.1f} × 10{str(exponent).translate(sup)}"


def _load():
    tlv = pd.read_csv(TLV_CSV).rename(columns={"protein": "Protein"})
    oxf = pd.read_csv(OXF_CSV).rename(columns={"protein": "Protein"})
    oxf["hazard"] = np.exp(oxf["beta"].astype(float))
    tlv = tlv.rename(columns={"hazard_ratio": "hazard"})
    overlap = set(tlv["Protein"]) & set(oxf["Protein"])
    merged = pd.merge(
        tlv[tlv["Protein"].isin(overlap)][["Protein", "hazard", "p_value"]],
        oxf[oxf["Protein"].isin(overlap)][["Protein", "hazard", "p.value"]].rename(columns={"p.value": "p_value"}),
        on="Protein", suffixes=("_discovery", "_replication"), how="inner",
    )
    merged = merged[(merged["hazard_discovery"] > 0) & (merged["hazard_replication"] > 0)].copy()
    merged["log2HR_discovery"]  = np.log2(merged["hazard_discovery"])
    merged["log2HR_replication"] = np.log2(merged["hazard_replication"])
    merged_19 = merged[merged["Protein"].isin(REPLICATED)].copy()
    return merged_19


def plot_hr_correlation(ax):
    merged_19 = _load()
    r19, p19 = pearsonr(merged_19["log2HR_discovery"], merged_19["log2HR_replication"])

    pos_both = (merged_19["log2HR_discovery"] > 0) & (merged_19["log2HR_replication"] > 0)
    neg_both = (merged_19["log2HR_discovery"] < 0) & (merged_19["log2HR_replication"] < 0)
    color_19 = pd.Series([NONSIG_COLOR] * len(merged_19), index=merged_19.index)
    color_19[pos_both] = RISK_COLOR
    color_19[neg_both] = PROTECTIVE_COLOR

    sns.regplot(data=merged_19, x="log2HR_discovery", y="log2HR_replication",
                scatter=False, ci=None,
                line_kws={"color": "0.85", "linewidth": 0.9}, ax=ax)
    # All 19 dots in this panel are labeled, so all are fully opaque.
    is_colored = pos_both | neg_both
    if (~is_colored).any():
        ax.scatter(merged_19.loc[~is_colored, "log2HR_discovery"],
                   merged_19.loc[~is_colored, "log2HR_replication"],
                   c=color_19[~is_colored].values, edgecolors="none",
                   s=8, alpha=1.0, zorder=3)
    if is_colored.any():
        ax.scatter(merged_19.loc[is_colored, "log2HR_discovery"],
                   merged_19.loc[is_colored, "log2HR_replication"],
                   c=color_19[is_colored].values, edgecolors="none",
                   s=8, alpha=1.0, zorder=3)
    ax.axhline(0, color=REFLINE_COLOR, linestyle=REFLINE_STYLE,
               linewidth=REFLINE_WIDTH, zorder=1)
    ax.axvline(0, color=REFLINE_COLOR, linestyle=REFLINE_STYLE,
               linewidth=REFLINE_WIDTH, zorder=1)

    # Manually anchor a few labels with longer leaders so the cluster in
    # the upper-right is readable. The rest are placed by adjust_text near
    # their dots.
    # Manually pull a few crowded labels out of the diagonal cluster with
    # leader lines; the rest sit next to their dots without leaders.
    manual_labels = {
        "LTBP2":   (-0.30,  1.50),
        "IGFBP2":  ( 0.65,  1.25),
        "LRG1":    ( 1.30,  1.30),
        "EPHB4":   (-0.60,  0.90),
        "PTX3":    (-0.50,  0.55),
        "CHGB":    (-0.40,  0.25),
    }
    for name, (lx, ly) in manual_labels.items():
        row = merged_19[merged_19["Protein"] == name].iloc[0]
        ax.annotate(
            name, xy=(float(row["log2HR_discovery"]), float(row["log2HR_replication"])),
            xytext=(lx, ly), fontsize=LABEL_FONTSIZE, color="black",
            ha="center", va="center", zorder=4,
            arrowprops=dict(arrowstyle="-", color="0.55", lw=0.4,
                            shrinkA=2.5, shrinkB=2.5),
        )

    # Small per-label nudges so the text sits beside the dot rather than on
    # top of it. adjust_text will further refine to remove overlaps. Picked
    # so each label moves into nearby whitespace.
    near_offsets = {
        "APOF":      ( 0.08,  0.05),
        "TNFRSF12A": ( 0.20,  0.00),
        "EFNA1":     (-0.08,  0.05),
        "ITIH3":     ( 0.10,  0.04),
        "CD14":      ( 0.10,  0.00),
        "REG1A":     ( 0.10, -0.05),
        "FGL1":      ( 0.00, -0.15),
        "CTSB":      ( 0.05, -0.08),
        "GSN":       ( 0.05, -0.10),
        "APOL1":     (-0.10,  0.05),
        "GHR":       ( 0.08,  0.05),
        "KIT":       (-0.10, -0.04),
        "PEPD":      (-0.10, -0.07),
    }
    other = merged_19[~merged_19["Protein"].isin(manual_labels)]
    texts = []
    for _, row in other.iterrows():
        name = str(row["Protein"])
        x0 = float(row["log2HR_discovery"])
        y0 = float(row["log2HR_replication"])
        dx, dy = near_offsets.get(name, (0.08, 0.05))
        texts.append(ax.text(x0 + dx, y0 + dy, name,
                             fontsize=LABEL_FONTSIZE, color="black", zorder=4))
    adjust_text(
        texts, ax=ax,
        expand_text=(1.2, 1.4), expand_points=(1.4, 1.6),
        force_text=(0.5, 0.7), force_points=(0.3, 0.5),
        arrowprops=dict(arrowstyle="-", color="0.55", lw=0.4, shrinkA=4),
    )

    ax.text(0.98, 0.04, f"r = {r19:.2f}, P = {_format_sci(p19)}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=ANNOT_FONTSIZE)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xticks([-2, -1, 0, 1, 2]); ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlabel("log2(Hazard Ratio) - Tel Aviv", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("log2(Hazard Ratio) - Oxford", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Tel Aviv vs Oxford", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=SPINE_WIDTH)
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
    print(f"Panel C: r = {r19:.3f}, P = {p19:.2e}")


def main():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 400,
    })
    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    plot_hr_correlation(ax)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "hr_correlation_19sig_pearson")
    fig.savefig(out + ".png", bbox_inches="tight", dpi=400)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}.{{png,pdf}}")


if __name__ == "__main__":
    main()
