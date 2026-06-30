"""Figure 2 — hollow-border Venn of FDR<0.05 adjusted-Cox proteins (overlap-FDR).
Equal-size circles, hollow (border only), protein names inside, no title.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from statsmodels.stats.multitest import multipletests

DATA_DIR = os.environ.get("ALS_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
TLV_CSV = os.path.join(DATA_DIR, "univariate_unadj_vs_adj.csv")
OXF_CSV = os.path.join(DATA_DIR, "serum_cox_adjusted.csv")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

TLV_COLOR = "#A42050"
OXF_COLOR = "#009092"

RADIUS = 1.75
SEP = 2.20  # distance between centers (smaller => more overlap)
BORDER_WIDTH = 1.4
NAME_FONTSIZE = 7
SETLABEL_FONTSIZE = 9

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 400,
})


def _adjusted_overlap_sig():
    tlv = pd.read_csv(TLV_CSV).dropna(subset=["adj_p", "adj_hazard_ratio"]).copy()
    oxf = pd.read_csv(OXF_CSV).dropna(subset=["p.value", "HR"]).copy()
    overlap = sorted(set(tlv["protein"]) & set(oxf["protein"]))
    t = tlv[tlv["protein"].isin(overlap)].copy()
    o = oxf[oxf["protein"].isin(overlap)].copy()
    t["q"] = multipletests(t["adj_p"], method="fdr_bh")[1]
    o["q"] = multipletests(o["p.value"], method="fdr_bh")[1]
    return (set(t.loc[t["q"] < 0.05, "protein"]),
            set(o.loc[o["q"] < 0.05, "protein"]))


def _column_block(names, max_per_col=8):
    """Lay out names in up to `max_per_col` rows; multiple columns if needed."""
    n = len(names)
    n_cols = max(1, int(np.ceil(n / max_per_col)))
    n_rows = int(np.ceil(n / n_cols))
    cols = [names[i*n_rows:(i+1)*n_rows] for i in range(n_cols)]
    return cols


def _draw_block(ax, cx, cy, names, fontsize, max_per_col=8, col_spacing=0.55,
                line_spacing=0.18):
    if not names:
        return
    cols = _column_block(names, max_per_col=max_per_col)
    n_cols = len(cols)
    x0 = cx - (n_cols - 1) * col_spacing / 2
    for ci, col in enumerate(cols):
        x = x0 + ci * col_spacing
        n_rows = len(col)
        y_top = cy + (n_rows - 1) * line_spacing / 2
        for ri, name in enumerate(col):
            y = y_top - ri * line_spacing
            ax.text(x, y, name, ha="center", va="center",
                    fontsize=fontsize, color="0.15")


def plot_hollow_venn(ax, tlv_sig, oxf_sig):
    tlv_only = sorted(tlv_sig - oxf_sig)
    oxf_only = sorted(oxf_sig - tlv_sig)
    both     = sorted(tlv_sig & oxf_sig)

    cx_t, cx_o = -SEP/2, SEP/2
    cy = 0
    ax.add_patch(Circle((cx_t, cy), RADIUS, fill=False,
                        edgecolor=TLV_COLOR, linewidth=BORDER_WIDTH))
    ax.add_patch(Circle((cx_o, cy), RADIUS, fill=False,
                        edgecolor=OXF_COLOR, linewidth=BORDER_WIDTH))

    # Centroid x-coords for each region
    x_tlv_only_center = cx_t - 0.45
    x_oxf_only_center = cx_o + 0.45
    x_overlap_center  = 0.0

    _draw_block(ax, x_tlv_only_center, cy, tlv_only, NAME_FONTSIZE,
                max_per_col=6, col_spacing=0.50, line_spacing=0.26)
    # 22 Oxford names: 2 columns x 11 rows
    _draw_block(ax, x_oxf_only_center, cy, oxf_only, NAME_FONTSIZE,
                max_per_col=11, col_spacing=0.85, line_spacing=0.26)
    _draw_block(ax, x_overlap_center, cy, both, NAME_FONTSIZE,
                max_per_col=4, col_spacing=0.30, line_spacing=0.26)

    # Set labels under each circle
    ax.text(cx_t, -RADIUS - 0.18, "Tel Aviv",
            ha="center", va="top", fontsize=SETLABEL_FONTSIZE, color=TLV_COLOR)
    ax.text(cx_o, -RADIUS - 0.18, "Oxford",
            ha="center", va="top", fontsize=SETLABEL_FONTSIZE, color=OXF_COLOR)

    ax.set_xlim(cx_t - RADIUS - 0.4, cx_o + RADIUS + 0.4)
    ax.set_ylim(-RADIUS - 0.6, RADIUS + 0.4)
    ax.set_aspect("equal")
    ax.set_axis_off()


def main():
    tlv_sig, oxf_sig = _adjusted_overlap_sig()
    print(f"TLV sig: {len(tlv_sig)}  Oxford sig: {len(oxf_sig)}  both: {len(tlv_sig & oxf_sig)}")
    print(f"Replicated: {sorted(tlv_sig & oxf_sig)}")

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    plot_hollow_venn(ax, tlv_sig, oxf_sig)
    fig.tight_layout()
    base = os.path.join(OUT_DIR, "venn_adjusted_hollow")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"saved: {base}.{{pdf,png}}")


if __name__ == "__main__":
    main()
