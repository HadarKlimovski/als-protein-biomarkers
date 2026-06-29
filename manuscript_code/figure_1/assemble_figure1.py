"""Assemble Figure 1 (Nature Medicine 3-panel, full-width).

Uses real matplotlib subplots (NOT imshow-of-PNGs) so all text stays vector and
at face-value font size in the final PDF.

Layout: 1 row × 3 columns at 7.2" × 2.2" (full publication width).
Panel B carves its cell into a nested 2-row gridspec for the broken-axis effect.
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

from fig1a_volcano_telaviv import plot_volcano_tlv
from fig1b_volcano_oxford import plot_volcano_oxford
from fig1c_hr_discovery_vs_replication import plot_hr_correlation

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Publication settings: Arial/Helvetica (fall back to Nimbus Sans), vector PDF.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 400,
    "axes.unicode_minus": False,
})

FIGSIZE = (7.2, 2.2)  # Nature Medicine full publication width, equal panels
PANEL_LETTER_SIZE = 6

fig = plt.figure(figsize=FIGSIZE)
gs = gridspec.GridSpec(
    1, 3, figure=fig, wspace=0.40,
    left=0.06, right=0.985, top=0.90, bottom=0.18,
)

ax_a = fig.add_subplot(gs[0, 0])
plot_volcano_tlv(ax_a)

# Panel B: broken-axis; the plotter creates 2 nested axes
ax_b_hi, ax_b_lo = plot_volcano_oxford(fig, gs[0, 1])

ax_c = fig.add_subplot(gs[0, 2])
plot_hr_correlation(ax_c)

# Panel letters
for letter, ax in [("a", ax_a), ("b", ax_b_hi), ("c", ax_c)]:
    ax.text(-0.18, 1.10, letter, transform=ax.transAxes,
            fontsize=PANEL_LETTER_SIZE, fontweight="bold",
            va="top", ha="left")

fig.savefig(os.path.join(BASE_PATH, "fig1_main.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(BASE_PATH, "fig1_main.png"), bbox_inches="tight", dpi=400)
plt.close(fig)
print(f"saved: {BASE_PATH}/fig1_main.{{pdf,png}}")
