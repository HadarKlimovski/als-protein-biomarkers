"""Cell-type origin heatmap for GSN, IGFBP2, MEGF10 from Human Protein Atlas.

Outputs: panel_celltype_origin.{pdf,png}
"""
import io
import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.environ.get("ALS_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
URL = "https://www.proteinatlas.org/download/tsv/rna_single_cell_type.tsv.zip"
CACHE = os.path.join(DATA_DIR, "hpa_single_cell_type.tsv.zip")
PROTEINS = ["GSN", "IGFBP2", "MEGF10"]

CELL_GROUPS = [
    ("Neurons", [
        ("brain excitatory neurons", "Excitatory neurons"),
        ("brain inhibitory neurons", "Inhibitory neurons"),
        ("other brain neurons", "Other brain neurons"),
    ]),
    ("Glia", [
        ("astrocytes", "Astrocytes"),
        ("oligodendrocytes", "Oligodendrocytes"),
        ("oligodendrocyte progenitor cells", "OPCs"),
        ("ependymal cells", "Ependymal"),
        ("schwann cells", "Schwann"),
        ("müller glia", "Müller glia"),
    ]),
    ("Pituitary", [
        ("pituicytes/fscs", "Pituicytes"),
        ("pituitary stem cells", "Pituitary stem"),
        ("thyrotrophs", "Thyrotrophs"),
    ]),
    ("Muscle", [
        ("myonuclei", "Myonuclei"),
        ("myosatellite cells", "Satellite"),
        ("smooth muscle cells", "Smooth muscle"),
    ]),
    ("Liver", [
        ("hepatocytes", "Hepatocytes"),
    ]),
    ("Digestive", [
        ("parietal cells", "Parietal"),
        ("gastric chief cells", "Gastric chief"),
        ("goblet cells", "Goblet"),
        ("esophageal apical cells", "Esophageal apical"),
    ]),
    ("Respiratory", [
        ("respiratory ciliated cells", "Resp. ciliated"),
    ]),
    ("Other", [
        ("fibroblasts", "Fibroblasts"),
        ("melanocytes", "Melanocytes"),
        ("urothelial cells", "Urothelial"),
        ("megakaryocytes", "Megakaryocytes"),
    ]),
]

GROUP_COLORS = {
    "Neurons": "#9C7DC2",
    "Glia": "#5B9279",
    "Pituitary": "#7DB1B5",
    "Eye": "#F4A460",
    "Muscle": "#D9A05B",
    "Vascular": "#E58A4E",
    "Liver": "#A37F4F",
    "Digestive": "#ED7D31",
    "Respiratory": "#4F81BD",
    "Reproductive": "#C77FA8",
    "Other": "#7F7F7F",
}

CELL_TYPES = [c for _, items in CELL_GROUPS for c, _ in items]
PRETTY = {c: p for _, items in CELL_GROUPS for c, p in items}

STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "figure_style.mplstyle")


def load_hpa():
    if os.path.exists(CACHE):
        print(f"Reading cached HPA file: {CACHE}")
        with zipfile.ZipFile(CACHE) as z:
            with z.open(z.namelist()[0]) as f:
                return pd.read_csv(f, sep="\t")
    print(f"Downloading {URL} ...")
    raw = urllib.request.urlopen(URL).read()
    with open(CACHE, "wb") as f:
        f.write(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(f, sep="\t")


def main():
    if os.path.exists(STYLE):
        plt.style.use(STYLE)
    sns.set_theme(context="paper", style="ticks")
    # Nature Medicine sizing: full width = 180 mm (~7.1"), max height 240 mm.
    # Fonts 5–7 pt for axis ticks/labels.
    plt.rcParams.update({
        "font.size": 6, "axes.labelsize": 6,
        "xtick.labelsize": 5, "ytick.labelsize": 6,
        "savefig.dpi": 400, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    df = load_hpa()
    val_col = "nCPM" if "nCPM" in df.columns else "nTPM"
    # Compute z-score across ALL HPA cell types (not just the curated subset),
    # so z reflects enrichment relative to the entire human single-cell atlas.
    full = (df[df["Gene name"].isin(PROTEINS)]
              .pivot_table(index="Gene name", columns="Cell type",
                           values=val_col, fill_value=0).reindex(PROTEINS))
    full_log = np.log10(full + 1)
    full_z = full_log.sub(full_log.mean(axis=1), axis=0).div(
        full_log.std(axis=1).replace(0, 1), axis=0)

    # Restrict to the curated cell list, using the full-HPA z-scores
    cell_subset = [c for c in CELL_TYPES if c in full.columns]
    mat = full[cell_subset].copy()
    mat_z = full_z[cell_subset].copy()
    mat.columns = [PRETTY[c] for c in mat.columns]
    mat_z.columns = [PRETTY[c] for c in mat_z.columns]
    # Drop cell types where all 3 proteins have z < 1.5 (full-HPA z-score)
    keep_cells = (mat_z >= 1.5).any(axis=0)
    mat = mat.loc[:, keep_cells]
    mat_z = mat_z.loc[:, keep_cells]
    # Rebuild group structure with only kept cells (drop empty groups)
    kept_pretty = set(mat.columns)
    filtered_groups = []
    for grp, items in CELL_GROUPS:
        kept_items = [(raw, pretty) for raw, pretty in items if pretty in kept_pretty]
        if kept_items:
            filtered_groups.append((grp, kept_items))
    cell_order = [pretty for _, items in filtered_groups for _, pretty in items]
    mat = mat[cell_order]
    mat_z = mat_z[cell_order]

    # Build figure: group-bar above (spans full content width incl. colorbar
    # gutter so the last "Other" rectangle can extend past the heatmap),
    # heatmap below, colorbar in its own column.
    from matplotlib.gridspec import GridSpec
    # Nature Medicine full-width: 180 mm = 7.087", landscape heatmap.
    fig = plt.figure(figsize=(7.1, 2.1))
    gs = GridSpec(2, 2, height_ratios=[0.12, 1.0], width_ratios=[1.0, 0.020],
                  hspace=0.05, wspace=0.04,
                  left=0.08, right=0.94, top=0.93, bottom=0.34)
    ax_grp = fig.add_subplot(gs[0, :])  # spans both heatmap + colorbar columns
    ax = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    # Group bar at top — placeholder; rectangles are drawn after we figure out
    # the exact width that lines up with the heatmap and ends at the colorbar.
    n = len(cell_order)
    ax_grp.set_ylim(0, 1)
    ax_grp.axis("off")

    # Mask cells with zero raw expression so they render as white (not blue)
    mask_zero = (mat == 0)
    sns.heatmap(mat_z, ax=ax, cmap="vlag", center=0, vmin=-2.5, vmax=2.5,
                cbar_ax=cax,
                cbar_kws={"label": "z-score (log nCPM)\nvs. all HPA cell types"},
                linewidths=0.3, linecolor="white",
                mask=mask_zero)
    cax.tick_params(labelsize=5, width=0.4)
    cax.yaxis.label.set_size(5)
    ax.set_facecolor("white")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    # Align ax_grp so 1 unit = 1 cell of the heatmap, and its right edge sits
    # at the colorbar's left edge. Then DRAW the rectangles with the last one
    # extending into the gutter.
    fig.canvas.draw()
    pos_ax = ax.get_position()
    pos_grp = ax_grp.get_position()
    pos_cax = cax.get_position()
    new_grp_width = pos_cax.x0 - pos_ax.x0
    ax_grp.set_position([pos_ax.x0, pos_grp.y0, new_grp_width, pos_grp.height])
    cell_width_fig = pos_ax.width / n
    xlim_right = new_grp_width / cell_width_fig
    ax_grp.set_xlim(0, xlim_right)

    # Now draw the rectangles & labels — first n-1 groups exactly fit their
    # cells; last group fills the remaining space (its single cell + gutter).
    SHORT = {"Immune / blood": "Immune"}
    LEAK = {"Respiratory": 0.9}  # narrow groups extend rightward over next group
    # Compute base positions first.
    group_pos = []
    pos = 0
    for i, (grp, items) in enumerate(filtered_groups):
        is_last = (i == len(filtered_groups) - 1)
        w = (xlim_right - pos) if is_last else len(items)
        group_pos.append((grp, items, pos, w))
        pos += w
    # Draw non-leaking groups first.
    for grp, items, p, w in group_pos:
        if grp in LEAK:
            continue
        ax_grp.add_patch(plt.Rectangle((p, 0), w, 1,
                                         color=GROUP_COLORS[grp], alpha=0.85,
                                         linewidth=0, zorder=1))
        fs = 5 if w < 2 else (5.5 if w < 3 else 6)
        label = SHORT.get(grp, grp) if w < 3 else grp
        ax_grp.text(p + w / 2, 0.5, label, ha="center", va="center",
                     fontsize=fs, color="white", fontweight="bold", zorder=3)
    # Draw leaking groups on top with a thin white edge.
    for grp, items, p, w in group_pos:
        if grp not in LEAK:
            continue
        draw_w = w + LEAK[grp]
        ax_grp.add_patch(plt.Rectangle((p, 0), draw_w, 1,
                                         color=GROUP_COLORS[grp], alpha=0.95,
                                         linewidth=0,
                                         zorder=2))
        ax_grp.text(p + draw_w / 2, 0.5, grp, ha="center", va="center",
                     fontsize=6, color="white", fontweight="bold", zorder=3)

    # Wide white vertical dividers spanning both panels (skip the right edge of
    # leaking groups so their colored bar stays continuous).
    pos = 0
    for grp, items in filtered_groups[:-1]:
        pos += len(items)
        ax.axvline(pos, color="white", linewidth=2.5, zorder=10)
        if grp not in LEAK:
            ax_grp.axvline(pos, color="white", linewidth=2.5, zorder=10)

    out = os.path.join(OUT_DIR, "panel_celltype_origin")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"saved: {out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
