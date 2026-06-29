"""Assemble Figure 2 (Nature Medicine 3-panel, full-width).

Real matplotlib subplots — text stays vector, no imshow of pre-rendered PNGs.
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from fig2a_volcano_telaviv_adjusted import plot_volcano_tlv_adjusted
from fig2b_volcano_oxford_adjusted import plot_volcano_oxford_adjusted
from fig2c_slope_unadjusted_vs_adjusted import plot_slope_chart

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 400,
    "axes.unicode_minus": False,
})

FIGSIZE = (7.2, 2.2)
PANEL_LETTER_SIZE = 6

fig = plt.figure(figsize=FIGSIZE)
gs = gridspec.GridSpec(
    1, 3, figure=fig, wspace=0.40,
    left=0.06, right=0.985, top=0.90, bottom=0.18,
)

ax_a = fig.add_subplot(gs[0, 0])
plot_volcano_tlv_adjusted(ax_a)

ax_b_hi, ax_b_lo = plot_volcano_oxford_adjusted(fig, gs[0, 1])

ax_c = fig.add_subplot(gs[0, 2])
plot_slope_chart(ax_c)

for letter, ax in [("a", ax_a), ("b", ax_b_hi), ("c", ax_c)]:
    ax.text(-0.18, 1.10, letter, transform=ax.transAxes,
            fontsize=PANEL_LETTER_SIZE, fontweight="bold",
            va="top", ha="left")

fig.savefig(os.path.join(BASE_PATH, "fig2_main.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(BASE_PATH, "fig2_main.png"), bbox_inches="tight", dpi=400)
plt.close(fig)
print(f"saved: {BASE_PATH}/fig2_main.{{pdf,png}}")
