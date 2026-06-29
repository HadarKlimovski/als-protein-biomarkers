"""Panel D — TLV + Oxford clinical correlation as wide horizontal band.

Layout: 2 stacked heatmaps (TLV on top, Oxford on bottom), each 3 clinical
variables x 19 proteins. Shared x-axis (proteins, rotated). Single colorbar.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pointbiserialr

DATA_DIR = os.environ.get("ALS_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TLV_DATA = os.path.join(DATA_DIR, "proteomics_survival_from_enrollment.csv")
OXF_CSV = os.path.join(DATA_DIR, "oxford_supp_replicated_clinical_corr(in).csv")
STYLE = os.path.join(DATA_DIR, "nature_medicine.mplstyle")

CLINICAL_VARS = {
    "Age Onset (years)": "Age at onset",
    "Sex female": "Sex (female)",
}
DISPLAY_VARS = list(CLINICAL_VARS.values())
VMAX = 0.45


def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def compute_tlv(proteins):
    df = pd.read_csv(TLV_DATA)
    avail = [p for p in proteins if p in df.columns]
    r = pd.DataFrame(index=avail, columns=DISPLAY_VARS, dtype=float)
    p = pd.DataFrame(index=avail, columns=DISPLAY_VARS, dtype=float)
    for prot in avail:
        for col, label in CLINICAL_VARS.items():
            mask = df[[prot, col]].notna().all(axis=1)
            x = df.loc[mask, prot].astype(float).values
            y = df.loc[mask, col].astype(float).values
            if label == "Sex (female)":
                rv, pv = pointbiserialr(y, x)
            else:
                rv, pv = spearmanr(x, y)
            r.loc[prot, label] = rv
            p.loc[prot, label] = pv
    return r, p


def load_oxford():
    df = pd.read_csv(OXF_CSV)
    df["Clinical Variable"] = df["Clinical Variable"].replace({"Sex": "Sex (female)"})
    r = df.pivot(index="Protein", columns="Clinical Variable", values="Correlation (r)")
    p = df.pivot(index="Protein", columns="Clinical Variable", values="P-value")
    return r.reindex(columns=DISPLAY_VARS), p.reindex(columns=DISPLAY_VARS)


def draw_block(ax, cax, r, p, cohort_label, cmap, show_xticks):
    """r,p indexed by protein (rows) x var (cols). We transpose so vars are
    rows on the heatmap and proteins are columns."""
    rT = r.T  # vars x proteins
    pT = p.T
    sns.heatmap(rT, ax=ax, cmap=cmap, center=0, vmin=-VMAX, vmax=VMAX,
                linewidths=0.5, linecolor="white",
                cbar=cax is not None, cbar_ax=cax,
                cbar_kws={"label": "Correlation (r)"} if cax is not None else None,
                xticklabels=show_xticks, yticklabels=True)
    for i, var in enumerate(rT.index):
        for j, prot in enumerate(rT.columns):
            s = stars(pT.loc[var, prot])
            if s:
                ax.text(j + 0.5, i + 0.5, s, ha="center", va="center",
                        fontsize=5, color="black")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", length=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    if show_xticks:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    # cohort label on the far left, outside y-tick labels, rotated vertically
    ax.text(-0.18, 0.5, cohort_label, transform=ax.transAxes,
            ha="center", va="center", fontsize=7, fontweight="bold", rotation=90)


def main():
    if os.path.exists(STYLE):
        plt.style.use(STYLE)
    plt.rcParams.update({
        "font.size": 6, "axes.labelsize": 6,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    oxf_r, oxf_p = load_oxford()
    proteins = list(oxf_r.index)
    tlv_r, tlv_p = compute_tlv(proteins)
    common = [p for p in proteins if p in tlv_r.index]
    tlv_r, tlv_p = tlv_r.loc[common], tlv_p.loc[common]
    oxf_r, oxf_p = oxf_r.loc[common], oxf_p.loc[common]
    print(f"Plotting {len(common)} proteins x {len(DISPLAY_VARS)} variables (stacked)")

    cmap = sns.diverging_palette(180, 30, s=60, l=55, as_cmap=True)

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(6.8, 2.4))
    gs = GridSpec(2, 2, width_ratios=[1.0, 0.018],
                  height_ratios=[1.0, 1.0],
                  hspace=0.18, wspace=0.02,
                  left=0.16, right=0.92, top=0.94, bottom=0.30)
    ax_tlv = fig.add_subplot(gs[0, 0])
    ax_oxf = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[:, 1])

    draw_block(ax_tlv, None, tlv_r, tlv_p, "Tel Aviv\n(n=349)", cmap, show_xticks=False)
    draw_block(ax_oxf, cax, oxf_r, oxf_p, "Oxford\n(n=392)", cmap, show_xticks=True)

    cbar = ax_oxf.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5, length=1.5)
    cbar.set_label("Correlation (r)", fontsize=6)
    cbar.outline.set_linewidth(0.4)

    out = os.path.join(OUT_DIR, "protein_clinical_correlation_stacked")
    fig.savefig(out + ".png", bbox_inches="tight", dpi=400)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}.{{png,pdf}}")


if __name__ == "__main__":
    main()
