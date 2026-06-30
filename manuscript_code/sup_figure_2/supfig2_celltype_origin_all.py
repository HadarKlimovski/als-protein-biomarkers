"""Supplementary panel — HPA single-cell expression for GSN, IGFBP2, MEGF10
across ALL 154 HPA cell types (no z-score filter), grouped by tissue.

2-column flipped layout: cell types on the y-axis, proteins on the x-axis.
Split into two side-by-side panels (somatic / non-reproductive on the left,
reproductive + connective + immune on the right) so it fits one Nature
Medicine page (~7.2" × 9.5") with readable cell-type labels.
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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch

DATA_DIR = os.environ.get("ALS_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
URL = "https://www.proteinatlas.org/download/tsv/rna_single_cell_type.tsv.zip"
CACHE = os.path.join(DATA_DIR, "hpa_single_cell_type.tsv.zip")

PROTEINS = ["GSN", "IGFBP2", "MEGF10"]

# Full tissue grouping for all 154 HPA single-cell types.
# (raw HPA name, display label)
CELL_GROUPS = [
    ("Brain", [
        ("brain excitatory neurons",         "Excitatory neurons"),
        ("brain inhibitory neurons",         "Inhibitory neurons"),
        ("other brain neurons",              "Other brain neurons"),
        ("astrocytes",                       "Astrocytes"),
        ("oligodendrocytes",                 "Oligodendrocytes"),
        ("oligodendrocyte progenitor cells", "OPCs"),
        ("ependymal cells",                  "Ependymal"),
        ("microglia",                        "Microglia"),
        ("bergmann glia",                    "Bergmann glia"),
        ("schwann cells",                    "Schwann"),
        ("müller glia",                      "Müller glia"),
        ("choroid plexus epithelial cells",  "Choroid plexus"),
    ]),
    ("Eye", [
        ("rod photoreceptor cells",          "Rod photoreceptors"),
        ("cone photoreceptor cells",         "Cone photoreceptors"),
        ("retinal amacrine cells",           "Retinal amacrine"),
        ("retinal bipolar cells",            "Retinal bipolar"),
        ("retinal ganglion cells",           "Retinal ganglion"),
        ("retinal horizontal cells",         "Retinal horizontal"),
        ("retinal pigment epithelial cells", "RPE"),
        ("ocular epithelial cells",          "Ocular epithelial"),
        ("conjunctival goblet cells",        "Conjunctival goblet"),
        ("lacrimal acinar cells",            "Lacrimal acinar"),
    ]),
    ("Pituitary", [
        ("pituicytes/fscs",       "Pituicytes"),
        ("pituitary stem cells",  "Pituitary stem"),
        ("somatotrophs",          "Somatotrophs"),
        ("thyrotrophs",           "Thyrotrophs"),
        ("corticotrophs",         "Corticotrophs"),
        ("gonadotrophs",          "Gonadotrophs"),
        ("lactotrophs",           "Lactotrophs"),
    ]),
    ("Endocrine", [
        ("adrenal cortex cells",   "Adrenal cortex"),
        ("adrenal medulla cells",  "Adrenal medulla"),
        ("pancreatic islet cells", "Pancreatic islet"),
        ("neuroendocrine cells",   "Neuroendocrine"),
    ]),
    ("Muscle", [
        ("myonuclei",                    "Myonuclei"),
        ("myosatellite cells",           "Satellite"),
        ("smooth muscle cells",          "Smooth muscle"),
        ("cardiomyocytes",               "Cardiomyocytes"),
        ("fibro-adipogenic progenitors", "FAPs"),
    ]),
    ("Liver", [
        ("hepatocytes",            "Hepatocytes"),
        ("kupffer cells",          "Kupffer"),
        ("hepatic stellate cells", "Hepatic stellate"),
        ("cholangiocytes",         "Cholangiocytes"),
    ]),
    ("Digestive", [
        ("parietal cells",                    "Parietal"),
        ("gastric chief cells",               "Gastric chief"),
        ("gastric progenitor cells",          "Gastric progenitor"),
        ("foveolar cells",                    "Foveolar"),
        ("mucous neck cells",                 "Mucous neck"),
        ("goblet cells",                      "Goblet"),
        ("esophageal apical cells",           "Esophageal apical"),
        ("esophageal basal cells",            "Esophageal basal"),
        ("esophageal suprabasal cells",       "Esophageal suprabasal"),
        ("enterocytes",                       "Enterocytes"),
        ("enteric stem cells",                "Enteric stem"),
        ("enteric transient amplifying cells","Enteric TA"),
        ("paneth cells",                      "Paneth"),
        ("tuft cells",                        "Tuft"),
        ("colonocytes",                       "Colonocytes"),
    ]),
    ("Pancreas", [
        ("pancreatic acinar cells", "Pancreatic acinar"),
        ("pancreatic duct cells",   "Pancreatic duct"),
    ]),
    ("Salivary", [
        ("salivary acinar cells",       "Salivary acinar"),
        ("salivary basal cells",        "Salivary basal"),
        ("salivary duct cells",         "Salivary duct"),
        ("salivary ionocytes",          "Salivary ionocytes"),
        ("salivary myoepithelial cells","Salivary myoepi."),
        ("submucosal glandular cells",  "Submucosal gland"),
    ]),
    ("Respiratory", [
        ("respiratory ciliated cells",     "Resp. ciliated"),
        ("respiratory basal cells",        "Resp. basal"),
        ("respiratory deuterosomal cells", "Resp. deuterosomal"),
        ("respiratory ionocytes",          "Resp. ionocytes"),
        ("respiratory secretory cells",    "Resp. secretory"),
        ("alveolar cells type 1",          "Alveolar type 1"),
        ("alveolar cells type 2",          "Alveolar type 2"),
        ("transitional alveolar cells",    "Transitional alveolar"),
    ]),
    ("Renal/Urinary", [
        ("proximal tubule cells",                    "Proximal tubule"),
        ("loop of henle epithelial cells",           "Loop of Henle"),
        ("distal convoluted tubule cells",           "DCT"),
        ("renal connecting tubule cells",            "Connecting tubule"),
        ("renal collecting duct intercalated cells", "CD intercalated"),
        ("renal collecting duct principal cells",    "CD principal"),
        ("podocytes",                                "Podocytes"),
        ("papillary tip epithelial cells",           "Papillary tip"),
        ("urothelial cells",                         "Urothelial"),
    ]),
    ("Reproductive (F)", [
        ("granulosa cells",                "Granulosa"),
        ("oocytes",                        "Oocytes"),
        ("ovarian stromal cells",          "Ovarian stromal"),
        ("breast hormone-responsive cells","Breast hormone-resp."),
        ("breast lactating cells",         "Breast lactating"),
        ("breast myoepithelial cells",     "Breast myoepi."),
        ("breast secretory cells",         "Breast secretory"),
        ("endometrial ciliated cells",     "Endometrial ciliated"),
        ("endometrial glandular cells",    "Endometrial glandular"),
        ("endometrial luminal cells",      "Endometrial luminal"),
        ("endometrial secretory cells",    "Endometrial secretory"),
        ("endometrial stromal cells",      "Endometrial stromal"),
        ("fallopian secretory cells",      "Fallopian secretory"),
        ("fallopian tube ciliated cells",  "Fallopian ciliated"),
        ("decidual stromal cells",         "Decidual stromal"),
        ("cytotrophoblasts",               "Cytotrophoblasts"),
        ("syncytiotrophoblasts",           "Syncytiotrophoblasts"),
        ("migrating cytotrophoblasts",     "Migrating CTB"),
        ("extravillous trophoblasts",      "EVT"),
        ("hofbauer cells",                 "Hofbauer"),
    ]),
    ("Reproductive (M)", [
        ("spermatogonia",                             "Spermatogonia"),
        ("early primary spermatocytes",               "Early prim. spermatocytes"),
        ("late primary spermatocytes",                "Late prim. spermatocytes"),
        ("early spermatids",                          "Early spermatids"),
        ("late spermatids",                           "Late spermatids"),
        ("sertoli cells",                             "Sertoli"),
        ("leydig cells",                              "Leydig"),
        ("peritubular myoid cells",                   "Peritubular myoid"),
        ("basal prostatic cells",                     "Prostatic basal"),
        ("prostatic glandular cells",                 "Prostatic glandular"),
        ("prostatic club cells",                      "Prostatic club"),
        ("prostatic hillock cells",                   "Prostatic hillock"),
        ("epididymal basal cells",                    "Epididymal basal"),
        ("epididymal clear cells",                    "Epididymal clear"),
        ("epididymal principal cells",                "Epididymal principal"),
        ("epididymal efferent duct absorptive cells", "Epi. efferent absorptive"),
        ("epididymal efferent duct ciliated cells",   "Epi. efferent ciliated"),
    ]),
    ("Skin", [
        ("basal keratinocytes",      "Basal keratinocytes"),
        ("suprabasal keratinocytes", "Suprabasal keratinocytes"),
        ("melanocytes",              "Melanocytes"),
    ]),
    ("Connective", [
        ("fibroblasts",   "Fibroblasts"),
        ("adipocytes",    "Adipocytes"),
        ("mesothelial cells", "Mesothelial"),
    ]),
    ("Vascular", [
        ("vascular endothelial cells",  "Vascular endothelial"),
        ("vascular smooth muscle cells","Vascular SMC"),
        ("lymphatic endothelial cells", "Lymphatic endothelial"),
        ("pericytes",                   "Pericytes"),
        ("epicardial cells",            "Epicardial"),
    ]),
    ("Immune/Blood", [
        ("hematopoietic stem cells",         "HSC"),
        ("erythrocyte progenitors",          "Erythrocyte prog."),
        ("erythrocytes",                     "Erythrocytes"),
        ("megakaryocyte-erythroid progenitors","MEP"),
        ("megakaryocyte progenitors",        "Megakaryocyte prog."),
        ("megakaryocytes",                   "Megakaryocytes"),
        ("platelets",                        "Platelets"),
        ("monocyte progenitors",             "Monocyte prog."),
        ("monocytes",                        "Monocytes"),
        ("macrophages",                      "Macrophages"),
        ("neutrophil progenitors",           "Neutrophil prog."),
        ("neutrophils",                      "Neutrophils"),
        ("mast cells",                       "Mast"),
        ("innate lymphoid cells",            "ILCs"),
        ("b-cells",                          "B-cells"),
        ("t-cells",                          "T-cells"),
        ("plasma cells",                     "Plasma"),
        ("nk-cells",                         "NK-cells"),
        ("cdc",                              "cDCs"),
        ("pdcs",                             "pDCs"),
        ("thymocytes",                       "Thymocytes"),
        ("thymic myoid cells",               "Thymic myoid"),
        ("medullary thymic epithelial cells","mTEC"),
    ]),
]

GROUP_COLORS = {
    "Brain":              "#7A5BAA",  # merged Neurons+Glia (deep purple)
    "Eye":                "#F4A460",
    "Pituitary":          "#7DB1B5",
    "Endocrine":          "#8FC2BB",
    "Muscle":             "#D9A05B",
    "Liver":              "#A37F4F",
    "Digestive":          "#ED7D31",
    "Pancreas":           "#E0A06B",
    "Salivary":           "#C99860",
    "Respiratory":        "#4F81BD",
    "Renal/Urinary":      "#6FA9CF",
    "Reproductive (F)":   "#C77FA8",
    "Reproductive (M)":   "#9B5C99",
    "Skin":               "#B58A6C",
    "Connective":         "#A9A9A9",
    "Vascular":           "#E58A4E",
    "Immune/Blood":       "#7F7F7F",
}

STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "figure_style.mplstyle")


def load_hpa():
    if os.path.exists(CACHE):
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
    plt.rcParams.update({
        "font.size": 5, "axes.labelsize": 5,
        "xtick.labelsize": 4.5, "ytick.labelsize": 6,
        "savefig.dpi": 400, "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })

    df = load_hpa()
    val_col = "nCPM" if "nCPM" in df.columns else "nTPM"

    # Merge undiff. + diff. spermatogonia into one "spermatogonia" entry
    # (average nCPM/nTPM across the two HPA subtypes) so the supplementary
    # heatmap shows a single spermatogonia row.
    sperm_mask = df["Cell type"].isin(
        ["undifferentiated spermatogonia", "differentiating spermatogonia"])
    if sperm_mask.any():
        merged = (df[sperm_mask]
                    .groupby(["Gene", "Gene name"], as_index=False)[val_col]
                    .mean())
        merged["Cell type"] = "spermatogonia"
        df = pd.concat([df[~sperm_mask], merged], ignore_index=True)

    # z-score per protein across all HPA cell types
    full = (df[df["Gene name"].isin(PROTEINS)]
              .pivot_table(index="Gene name", columns="Cell type",
                           values=val_col, fill_value=0)
              .reindex(PROTEINS))
    full_log = np.log10(full + 1)
    full_z = full_log.sub(full_log.mean(axis=1), axis=0).div(
        full_log.std(axis=1).replace(0, 1), axis=0)

    # Build cell order from the curated grouping
    mapped_raw = [raw for _, items in CELL_GROUPS for raw, _ in items]
    all_cells = set(full.columns)
    missing = sorted(all_cells - set(mapped_raw))
    if missing:
        print(f"  WARNING: {len(missing)} cell types not in CELL_GROUPS:")
        for m in missing:
            print(f"    - {m}")
        # add unmapped cells under an "Other" group so figure still has all 154
        CELL_GROUPS.append(("Other", [(m, m.title()) for m in missing]))
        GROUP_COLORS.setdefault("Other", "#BBBBBB")

    cell_order_raw = [raw for _, items in CELL_GROUPS for raw, _ in items
                      if raw in all_cells]
    pretty_map = {raw: pretty for _, items in CELL_GROUPS for raw, pretty in items}

    mat = full[cell_order_raw].copy()
    mat_z = full_z[cell_order_raw].copy()
    mat.columns = [pretty_map[c] for c in mat.columns]
    mat_z.columns = [pretty_map[c] for c in mat_z.columns]
    # Transpose: rows = cells, cols = proteins
    mat = mat.T
    mat_z = mat_z.T
    print(f"\nFinal heatmap: {mat.shape[0]} cell types x {mat.shape[1]} proteins")

    filtered_groups = []
    for grp, items in CELL_GROUPS:
        kept = [(raw, pretty) for raw, pretty in items if raw in cell_order_raw]
        if kept:
            filtered_groups.append((grp, kept))

    # Pick split point so column 1 and column 2 are as balanced as possible.
    total = sum(len(items) for _, items in filtered_groups)
    cumulative = 0
    best_split = 0
    best_diff = total
    for i, (_, items) in enumerate(filtered_groups):
        cumulative += len(items)
        diff = abs(total - 2 * cumulative)
        if diff < best_diff:
            best_diff = diff
            best_split = i + 1
    groups_left  = filtered_groups[:best_split]
    groups_right = filtered_groups[best_split:]
    n_left  = sum(len(items) for _, items in groups_left)
    n_right = sum(len(items) for _, items in groups_right)
    n_max   = max(n_left, n_right)
    rows_left  = [pretty for _, items in groups_left  for _, pretty in items]
    rows_right = [pretty for _, items in groups_right for _, pretty in items]
    print(f"Split: column1={n_left} (groups: {[g for g,_ in groups_left]})")
    print(f"       column2={n_right} (groups: {[g for g,_ in groups_right]})")

    # Portrait supplementary panel. Taller than Nature Medicine main figure
    # height so each system gets enough vertical room for its label.
    fig_w = 7.2
    fig_h = 11.0
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Manual axes layout in figure-fraction coordinates. Per panel:
    #   [system text label | colored stripe | tick-label gutter | heatmap]
    # The system text sits OUTSIDE the colored stripe so it can be horizontal
    # and uniform regardless of how few cells a system has.
    top    = 0.970
    bottom = 0.030
    plot_h = top - bottom
    # Per panel:
    #   [sys-label gutter | thin colored stripe | tick-label gutter | heatmap]
    # The vertical system label lives in its own gutter (NOT inside the stripe)
    # so it isn't constrained by the stripe's vertical extent — small systems
    # (Pancreas, Endocrine) can render their full name without being clipped.
    sys_w   = 0.040   # vertical system labels (~0.29" wide)
    grp_w   = 0.014   # thin colored stripe (no text inside)
    label_w = 0.225   # cell-type tick labels (full room for longest names)
    heat_w  = 0.135   # 3 protein cols
    gap     = 0.010   # gap between left panel and right panel (tight; the sys gutter separates them)
    cbar_w  = 0.012
    cbar_pad = 0.018
    left_margin = 0.008

    x_sys_L   = left_margin
    x_grp_L   = x_sys_L + sys_w
    x_lab_L   = x_grp_L + grp_w + 0.005
    x_heat_L  = x_lab_L + label_w
    x_sys_R   = x_heat_L + heat_w + gap
    x_grp_R   = x_sys_R + sys_w
    x_lab_R   = x_grp_R + grp_w + 0.005
    x_heat_R  = x_lab_R + label_w
    x_cbar    = x_heat_R + heat_w + cbar_pad

    h_L = plot_h
    h_R = plot_h
    y_L = top - h_L
    y_R = top - h_R

    ax_sys_L = fig.add_axes([x_sys_L, y_L, sys_w, h_L])
    ax_grp_L = fig.add_axes([x_grp_L, y_L, grp_w, h_L])
    ax_L     = fig.add_axes([x_heat_L, y_L, heat_w, h_L])
    ax_sys_R = fig.add_axes([x_sys_R, y_R, sys_w, h_R])
    ax_grp_R = fig.add_axes([x_grp_R, y_R, grp_w, h_R])
    ax_R     = fig.add_axes([x_heat_R, y_R, heat_w, h_R])
    cax      = fig.add_axes([x_cbar, y_L + h_L * 0.35, cbar_w, h_L * 0.30])

    # Use full system names (no abbreviations) so the rotated vertical
    # labels read clearly. Long compound names are wrapped to keep their
    # rendered height comparable to the shorter labels.
    FULL_LABEL = {
        "Reproductive (F)": "Reproductive (F)",
        "Reproductive (M)": "Reproductive (M)",
        "Renal/Urinary":    "Renal / Urinary",
        "Immune/Blood":     "Immune / Blood",
    }

    def draw_panel(ax, ax_grp, ax_sys, rows, groups, show_cbar):
        sub = mat_z.loc[rows]
        sub_raw = mat.loc[rows]
        mask_zero = (sub_raw == 0)
        sns.heatmap(sub, ax=ax, cmap="vlag", center=0, vmin=-2.5, vmax=2.5,
                    cbar=show_cbar, cbar_ax=cax if show_cbar else None,
                    cbar_kws={"label": "z-score (log nCPM)\nvs. all HPA cell types"}
                               if show_cbar else None,
                    linewidths=0.6, linecolor="white",
                    mask=mask_zero)
        ax.set_facecolor("white")
        ax.set_xlabel(""); ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 fontsize=7.5, fontweight="normal")
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7.5)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.yaxis.tick_left()

        n = len(rows)
        for a in (ax_grp, ax_sys):
            a.set_xlim(0, 1)
            a.set_ylim(0, n)
            a.invert_yaxis()
            a.axis("off")

        UNIFORM_FS = 7.5
        GAP = 0.55  # row-units; total white-band thickness between systems
        # Per-row physical height (inches) for this panel; used to convert
        # rotated-text widths into row-unit "label heights" for collision math.
        ax_pos = ax_sys.get_position()
        panel_h_inches = ax_pos.height * fig_h
        per_row_inches = panel_h_inches / n
        # Bold sans-serif rotated text is wider than plain estimates; we
        # over-estimate slightly so the collision pass leaves real breathing room.
        char_w_inches  = 0.78 * (UNIFORM_FS / 72.0)

        # Pass 1: target centers (group centers) and rendered label heights
        pos = 0
        stripes = []
        for i, (grp, items) in enumerate(groups):
            is_last = (i == len(groups) - 1)
            h = (n - pos) if is_last else len(items)
            y0 = pos + (0 if i == 0 else GAP / 2)
            y1 = (pos + h) - (0 if is_last else GAP / 2)
            label = FULL_LABEL.get(grp, grp)
            label_h_rows = char_w_inches * len(label) / per_row_inches
            target_y = (y0 + y1) / 2
            stripes.append({
                "grp": grp, "items": items, "y0": y0, "y1": y1,
                "target_y": target_y, "label": label, "label_h": label_h_rows,
            })
            pos += h

        # Pass 2: resolve collisions by pushing labels down (forward pass)
        PAD = 0.6  # row-units extra padding between label edges
        for j in range(1, len(stripes)):
            prev = stripes[j - 1]
            cur  = stripes[j]
            min_center = prev.get("actual_y", prev["target_y"]) \
                         + prev["label_h"] / 2 + cur["label_h"] / 2 + PAD
            cur["actual_y"] = max(cur["target_y"], min_center)
        stripes[0]["actual_y"] = stripes[0]["target_y"]
        # Backward pass: if last labels were pushed below panel, pull the
        # whole chain up by the same amount.
        overflow = stripes[-1]["actual_y"] + stripes[-1]["label_h"] / 2 - n
        if overflow > 0:
            for s in stripes:
                s["actual_y"] -= overflow

        # Draw colored stripes and labels
        for s in stripes:
            grp = s["grp"]
            ax_grp.add_patch(plt.Rectangle((0, s["y0"]), 1, s["y1"] - s["y0"],
                                           color=GROUP_COLORS[grp],
                                           alpha=0.9, linewidth=0, zorder=1))
            ax_sys.text(0.5, s["actual_y"], s["label"],
                        ha="center", va="center",
                        fontsize=UNIFORM_FS, color=GROUP_COLORS[grp],
                        fontweight="bold", rotation=90, clip_on=False)
            # Leader line from label position to the stripe center (only when
            # the label had to be displaced from its target).
            if abs(s["actual_y"] - s["target_y"]) > 0.05:
                con = ConnectionPatch(
                    xyA=(1.0, s["actual_y"]),  xyB=(0.0, s["target_y"]),
                    coordsA="data", coordsB="data",
                    axesA=ax_sys, axesB=ax_grp,
                    color=GROUP_COLORS[grp], linewidth=0.6, alpha=0.7,
                    zorder=2)
                fig.add_artist(con)

        # White bands across the heatmap at every system boundary
        pos = 0
        for grp, items in groups[:-1]:
            pos += len(items)
            ax.add_patch(plt.Rectangle((0, pos - GAP / 2),
                                       ax.get_xlim()[1], GAP,
                                       color="white", linewidth=0, zorder=10))

    draw_panel(ax_L, ax_grp_L, ax_sys_L, rows_left,  groups_left,  show_cbar=True)
    draw_panel(ax_R, ax_grp_R, ax_sys_R, rows_right, groups_right, show_cbar=False)
    cax.tick_params(labelsize=5, width=0.4)
    cax.yaxis.label.set_size(5.5)

    out = os.path.join(OUT_DIR, "supp_celltype_origin_all")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"saved: {out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
