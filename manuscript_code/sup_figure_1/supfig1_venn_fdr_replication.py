"""Supplementary Fig 1 — Venn of FDR-significant proteins, Tel Aviv vs Oxford.

Each cohort's FDR<0.05 set is counted independently over its full measured panel
(Tel Aviv: 1218 proteins; Oxford: 5416 proteins). Overlap = proteins FDR-sig in both.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn2_circles

DATA_DIR = os.environ.get("ALS_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))

STYLE_PATH = os.path.join(DATA_DIR, "nature_medicine.mplstyle")
plt.style.use(STYLE_PATH)
sns.set_theme(context="paper", style="ticks")

SPINE_WIDTH = 0.3
mpl.rcParams.update({
    "font.size": 5,
    "axes.labelsize": 6,
    "axes.titlesize": 6,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    "legend.frameon": False,
    "lines.linewidth": 0.5,
    "axes.linewidth": SPINE_WIDTH,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

TLV_PATH = os.path.join(DATA_DIR, "significant_uniivariate_fdr_for_paper_new.csv")
OXFORD_PATH = os.path.join(DATA_DIR, "serum_cox_unadjusted.csv")

FDR_THR = 0.05

tlv = pd.read_csv(TLV_PATH).rename(columns={"protein": "Protein"})
oxford = pd.read_csv(OXFORD_PATH).rename(columns={"q": "FDR_adjusted_pval"})
oxford = oxford.rename(columns={oxford.columns[0]: "Protein"}) if "Protein" not in oxford.columns else oxford

tlv_sig = set(tlv.loc[tlv["FDR_adjusted_pval"] < FDR_THR, "Protein"])
ox_sig = set(oxford.loc[oxford["FDR_adjusted_pval"] < FDR_THR, "Protein"])
both = tlv_sig & ox_sig

print(f"Tel Aviv FDR<{FDR_THR}: {len(tlv_sig)} (of {len(tlv)})")
print(f"Oxford   FDR<{FDR_THR}: {len(ox_sig)} (of {len(oxford)})")
print(f"Overlap: {len(both)}")
print("Overlap proteins:", sorted(both))

COLOR_TLV = "#7D8FA8"
COLOR_OX = "#B79977"

fig, ax = plt.subplots(figsize=(3.35, 2.8))
v = venn2(
    subsets=(len(tlv_sig - ox_sig), len(ox_sig - tlv_sig), len(both)),
    set_labels=(f"Tel Aviv\n(n={len(tlv_sig)})", f"Oxford\n(n={len(ox_sig)})"),
    set_colors=(COLOR_TLV, COLOR_OX),
    alpha=0.55,
    ax=ax,
)
c = venn2_circles(
    subsets=(len(tlv_sig - ox_sig), len(ox_sig - tlv_sig), len(both)),
    linestyle="-",
    linewidth=0.5,
    color="0.3",
    ax=ax,
)

for lbl in v.set_labels:
    if lbl is not None:
        lbl.set_fontsize(6)
for sid in ("10", "01", "11"):
    lbl = v.get_label_by_id(sid)
    if lbl is not None:
        lbl.set_fontsize(6)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "venn_fdr_replication.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "venn_fdr_replication.png"), bbox_inches="tight", dpi=400)
print(f"Saved to {OUT_DIR}")

pd.DataFrame({"Protein": sorted(both)}).to_csv(
    os.path.join(OUT_DIR, "overlap_proteins.csv"), index=False
)
