"""Supplementary Figure 1 — HR-correlation (TLV vs Oxford, weighted) + Venns.

4-panel layout (2 rows × 2 cols), real matplotlib subplots, vector PDF/PNG.
Same architecture and font sizes as the main Figure 1/2 assemblies.

Panels:
  a — HR correlation, unadjusted (TLV-weighted + Oxford-weighted lines)
  b — HR correlation, adjusted (same)
  c — Venn of FDR<0.05 proteins (unadjusted)
  d — Venn of FDR<0.05 proteins (adjusted)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from matplotlib_venn import venn2, venn2_circles
from matplotlib.patches import Circle
from statsmodels.stats.multitest import multipletests

DATA_DIR = os.environ.get("ALS_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TLV_UNADJ_CSV = os.path.join(DATA_DIR, "significant_uniivariate_fdr_for_paper_new.csv")
OXF_UNADJ_CSV = os.path.join(DATA_DIR, "serum_cox_unadjusted.csv")
TLV_ADJ_CSV = os.path.join(DATA_DIR, "univariate_unadj_vs_adj.csv")
OXF_ADJ_CSV = os.path.join(DATA_DIR, "serum_cox_adjusted.csv")

RISK_COLOR = "#BC4D43"
PROTECTIVE_COLOR = "#2f7e85"
NONSIG_COLOR = "#BBBBBB"
TLV_LINE_COLOR = "#A42050"  # Tel Aviv (matches COLOR_TLV_DOT)
OXF_LINE_COLOR = "#009092"  # Oxford (matches COLOR_OXF_DOT)
VENN_DISC_COLOR    = "#A42050"  # Tel Aviv
VENN_REP_COLOR     = "#009092"  # Oxford
VENN_OVERLAP_COLOR = "#525871"  # blend of TLV + OXF
REFLINE_COLOR = "0.4"
REFLINE_STYLE = "--"
REFLINE_WIDTH = 0.6
SPINE_WIDTH = 0.5

AXIS_LABEL_FONTSIZE = 7
TICK_FONTSIZE = 6
TITLE_FONTSIZE = 7
LEGEND_FONTSIZE = 6
VENN_TEXT_FONTSIZE = 7
PANEL_LETTER_SIZE = 11  # match main figure (main_fig_v2.py add_panel_letter)

# Named-protein hollow Venn (panel e) — mirrors figure_2/scripts/venn_adjusted_hollow.py
VENN_RADIUS = 1.75
VENN_SEP = 2.20
VENN_BORDER_WIDTH = 1.4
VENN_NAME_FONTSIZE = 8
VENN_SETLABEL_FONTSIZE = 9


def weighted_pearson(x, y, w):
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[m], y[m], w[m]
    if len(x) < 3:
        return np.nan
    w = w / np.sum(w)
    mx, my = np.sum(w * x), np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    vx, vy = np.sum(w * (x - mx) ** 2), np.sum(w * (y - my) ** 2)
    if vx <= 0 or vy <= 0:
        return np.nan
    return cov / np.sqrt(vx * vy)


def weighted_pearson_perm_p(x, y, w, n_perm=5000, seed=42):
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[m], y[m], w[m]
    if len(x) < 3:
        return np.nan
    obs = weighted_pearson(x, y, w)
    if not np.isfinite(obs):
        return np.nan
    rng = np.random.default_rng(seed)
    permuted = np.array([weighted_pearson(x, rng.permutation(y), w) for _ in range(n_perm)])
    return (np.sum(np.abs(permuted) >= abs(obs)) + 1) / (n_perm + 1)


def format_sci(p):
    if not np.isfinite(p):
        return "NA"
    if p == 0:
        return "<1 × 10⁻³⁰⁰"
    exp = int(np.floor(np.log10(abs(p))))
    mant = p / (10 ** exp)
    sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return f"{mant:.1f} × 10{str(exp).translate(sup)}"


def _load_unadj():
    tlv = pd.read_csv(TLV_UNADJ_CSV).rename(columns={"protein": "Protein", "hazard_ratio": "hazard"})
    oxf = pd.read_csv(OXF_UNADJ_CSV).rename(columns={"protein": "Protein", "p.value": "p_value"})
    oxf["hazard"] = np.exp(oxf["beta"].astype(float))
    overlap = sorted(set(tlv["Protein"]) & set(oxf["Protein"]))
    merged = pd.merge(
        tlv[tlv["Protein"].isin(overlap)][["Protein", "hazard", "p_value"]],
        oxf[oxf["Protein"].isin(overlap)][["Protein", "hazard", "p_value"]],
        on="Protein", suffixes=("_d", "_r"), how="inner",
    )
    merged = merged[(merged["hazard_d"] > 0) & (merged["hazard_r"] > 0)].dropna()
    merged["log2HR_d"] = np.log2(merged["hazard_d"])
    merged["log2HR_r"] = np.log2(merged["hazard_r"])
    return merged, tlv, oxf


def _load_adj():
    tlv = pd.read_csv(TLV_ADJ_CSV).rename(
        columns={"protein": "Protein", "adj_p": "p_value", "adj_hazard_ratio": "hazard"})
    oxf = pd.read_csv(OXF_ADJ_CSV).rename(columns={"protein": "Protein", "p.value": "p_value", "HR": "hazard"})
    oxf["hazard"] = pd.to_numeric(oxf["hazard"], errors="coerce")
    overlap = sorted(set(tlv["Protein"]) & set(oxf["Protein"]))
    merged = pd.merge(
        tlv[tlv["Protein"].isin(overlap)][["Protein", "hazard", "p_value"]],
        oxf[oxf["Protein"].isin(overlap)][["Protein", "hazard", "p_value"]],
        on="Protein", suffixes=("_d", "_r"), how="inner",
    )
    merged = merged[(merged["hazard_d"] > 0) & (merged["hazard_r"] > 0)].dropna()
    merged["log2HR_d"] = np.log2(merged["hazard_d"])
    merged["log2HR_r"] = np.log2(merged["hazard_r"])
    return merged, tlv, oxf


def plot_correlation(ax, merged, title):
    pos_both = (merged["log2HR_d"] > 0) & (merged["log2HR_r"] > 0)
    neg_both = (merged["log2HR_d"] < 0) & (merged["log2HR_r"] < 0)
    color = pd.Series([NONSIG_COLOR] * len(merged), index=merged.index)
    color[pos_both] = RISK_COLOR
    color[neg_both] = PROTECTIVE_COLOR

    w_tlv = -np.log10(merged["p_value_d"].clip(lower=1e-300))
    w_oxf = -np.log10(merged["p_value_r"].clip(lower=1e-300))

    r_tlv = weighted_pearson(merged["log2HR_d"], merged["log2HR_r"], w_tlv)
    p_tlv = weighted_pearson_perm_p(merged["log2HR_d"], merged["log2HR_r"], w_tlv, n_perm=5000)
    r_oxf = weighted_pearson(merged["log2HR_d"], merged["log2HR_r"], w_oxf)
    p_oxf = weighted_pearson_perm_p(merged["log2HR_d"], merged["log2HR_r"], w_oxf, n_perm=5000)

    slope_t, intercept_t = np.polyfit(merged["log2HR_d"], merged["log2HR_r"], 1, w=w_tlv)
    slope_o, intercept_o = np.polyfit(merged["log2HR_d"], merged["log2HR_r"], 1, w=w_oxf)

    ax.scatter(merged["log2HR_d"], merged["log2HR_r"],
               c=color.values, edgecolors="none", s=7, alpha=0.5, zorder=1)
    ax.axhline(0, color=REFLINE_COLOR, linestyle=REFLINE_STYLE,
               linewidth=REFLINE_WIDTH, zorder=1)
    ax.axvline(0, color=REFLINE_COLOR, linestyle=REFLINE_STYLE,
               linewidth=REFLINE_WIDTH, zorder=1)

    xs = np.linspace(-2, 2, 100)
    ax.plot(xs, slope_t * xs + intercept_t, color=TLV_LINE_COLOR, linewidth=1.2,
            label=f"Tel Aviv (r = {r_tlv:.2f}, P = {format_sci(p_tlv)})",
            zorder=4)
    ax.plot(xs, slope_o * xs + intercept_o, color=OXF_LINE_COLOR, linewidth=1.2,
            label=f"Oxford (r = {r_oxf:.2f}, P = {format_sci(p_oxf)})",
            zorder=3)

    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.set_xticks([-2, -1, 0, 1, 2]); ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlabel("log2(Hazard Ratio) - Tel Aviv", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("log2(Hazard Ratio) - Oxford", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=SPINE_WIDTH)
    ax.legend(loc="lower left", fontsize=LEGEND_FONTSIZE,
              frameon=False, bbox_to_anchor=(0.5, 0.04),
              handlelength=1.5, handletextpad=0.4, borderaxespad=0.2)
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)


def plot_venn(ax, tlv_sig, oxf_sig, title):
    both = tlv_sig & oxf_sig
    subsets = (len(tlv_sig - oxf_sig), len(oxf_sig - tlv_sig), len(both))
    v = venn2(subsets=subsets, set_labels=("Tel Aviv", "Oxford"), ax=ax)
    patch_map = {"10": VENN_DISC_COLOR, "01": VENN_REP_COLOR, "11": VENN_OVERLAP_COLOR}
    for sid, col in patch_map.items():
        p = v.get_patch_by_id(sid)
        if p is not None:
            p.set_facecolor(col)
            p.set_alpha(0.35)
            p.set_edgecolor("none")
            p.set_linewidth(0)
    for lbl in v.set_labels:
        if lbl is not None:
            lbl.set_fontsize(VENN_TEXT_FONTSIZE)
    for sid in ("10", "01", "11"):
        lbl = v.get_label_by_id(sid)
        if lbl is not None:
            lbl.set_fontsize(VENN_TEXT_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=2)
    ax.set_aspect("equal")
    # Let matplotlib_venn size the axes around its circles, then add a small
    # margin so neither circle is clipped (works for both balanced and
    # very-asymmetric subsets).
    ax.relim(); ax.autoscale_view()
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    mx = max(abs(x0), abs(x1)) * 1.10
    my = max(abs(y0), abs(y1)) * 1.10
    ax.set_xlim(-mx, mx); ax.set_ylim(-my, my)


def _venn_column_block(names, max_per_col=8):
    n = len(names)
    n_cols = max(1, int(np.ceil(n / max_per_col)))
    n_rows = int(np.ceil(n / n_cols))
    return [names[i * n_rows:(i + 1) * n_rows] for i in range(n_cols)]


def _venn_draw_block(ax, cx, cy, names, fontsize, max_per_col=8,
                     col_spacing=0.55, line_spacing=0.18, fontweight="normal"):
    if not names:
        return
    cols = _venn_column_block(names, max_per_col=max_per_col)
    n_cols = len(cols)
    x0 = cx - (n_cols - 1) * col_spacing / 2
    for ci, col in enumerate(cols):
        x = x0 + ci * col_spacing
        n_rows = len(col)
        y_top = cy + (n_rows - 1) * line_spacing / 2
        for ri, name in enumerate(col):
            y = y_top - ri * line_spacing
            ax.text(x, y, name, ha="center", va="center",
                    fontsize=fontsize, color="0.15", fontweight=fontweight)


def plot_named_venn(ax, tlv_sig, oxf_sig):
    """Hollow (border-only) Venn with the actual protein names inside each region.
    Same geometry/colours as figure_2/scripts/venn_adjusted_hollow.py."""
    tlv_only = sorted(tlv_sig - oxf_sig)
    oxf_only = sorted(oxf_sig - tlv_sig)
    both     = sorted(tlv_sig & oxf_sig)

    cx_t, cx_o, cy = -VENN_SEP / 2, VENN_SEP / 2, 0
    ax.add_patch(Circle((cx_t, cy), VENN_RADIUS, fill=False,
                        edgecolor=VENN_DISC_COLOR, linewidth=VENN_BORDER_WIDTH))
    ax.add_patch(Circle((cx_o, cy), VENN_RADIUS, fill=False,
                        edgecolor=VENN_REP_COLOR, linewidth=VENN_BORDER_WIDTH))

    _venn_draw_block(ax, cx_t - 0.45, cy, tlv_only, VENN_NAME_FONTSIZE,
                     max_per_col=6, col_spacing=0.50, line_spacing=0.26)
    _venn_draw_block(ax, cx_o + 0.45, cy, oxf_only, VENN_NAME_FONTSIZE,
                     max_per_col=11, col_spacing=0.85, line_spacing=0.26)
    _venn_draw_block(ax, 0.0, cy, both, VENN_NAME_FONTSIZE,
                     max_per_col=4, col_spacing=0.30, line_spacing=0.26,
                     fontweight="bold")

    ax.text(cx_t, -VENN_RADIUS - 0.18, "Tel Aviv", ha="center", va="top",
            fontsize=VENN_SETLABEL_FONTSIZE, color="black")
    ax.text(cx_o, -VENN_RADIUS - 0.18, "Oxford", ha="center", va="top",
            fontsize=VENN_SETLABEL_FONTSIZE, color="black")

    ax.set_xlim(cx_t - VENN_RADIUS - 0.3, cx_o + VENN_RADIUS + 0.3)
    ax.set_ylim(-VENN_RADIUS - 0.45, VENN_RADIUS + 0.18)
    ax.set_aspect("equal")
    ax.set_axis_off()


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 400,
        "axes.unicode_minus": False,
    })

    # Data
    merged_un, tlv_un, oxf_un = _load_unadj()
    merged_ad, tlv_ad, oxf_ad = _load_adj()

    # Inject p-value columns expected by plot_correlation
    merged_un = merged_un.rename(columns={"p_value_d": "p_value_d", "p_value_r": "p_value_r"})
    merged_ad = merged_ad.rename(columns={"p_value_d": "p_value_d", "p_value_r": "p_value_r"})

    # Venn sets — significance within the SHARED proteome (BH re-corrected
    # restricted to the overlap). Matches the original venn_shared.png /
    # venn_shared_adjusted.png logic (yields ~42+19+28 unadj, ~9+2+57 adj).
    def _overlap_sig(tlv, oxf, p_col_t, p_col_o):
        overlap = sorted(set(tlv["Protein"]) & set(oxf["Protein"]))
        t_ov = tlv[tlv["Protein"].isin(overlap)].dropna(subset=[p_col_t]).copy()
        o_ov = oxf[oxf["Protein"].isin(overlap)].dropna(subset=[p_col_o]).copy()
        t_ov["_q"] = multipletests(t_ov[p_col_t], method="fdr_bh")[1]
        o_ov["_q"] = multipletests(o_ov[p_col_o], method="fdr_bh")[1]
        return (set(t_ov.loc[t_ov["_q"] < 0.05, "Protein"]),
                set(o_ov.loc[o_ov["_q"] < 0.05, "Protein"]))

    tlv_un_sig, oxf_un_sig = _overlap_sig(tlv_un, oxf_un, "p_value", "p_value")
    tlv_ad_sig, oxf_ad_sig = _overlap_sig(tlv_ad, oxf_ad, "p_value", "p_value")

    print(f"Unadj TLV sig={len(tlv_un_sig)}  OXF sig={len(oxf_un_sig)}  both={len(tlv_un_sig & oxf_un_sig)}")
    print(f"Adj   TLV sig={len(tlv_ad_sig)}  OXF sig={len(oxf_ad_sig)}  both={len(tlv_ad_sig & oxf_ad_sig)}")

    # Figure — Nature Medicine full publication width (180 mm ≈ 7.0"),
    # within the 240 mm (≈9.45") supplementary height limit.
    # 3 rows: a/b (HR corr), c/d (count Venns), e (named-protein Venn, full width).
    fig = plt.figure(figsize=(7.0, 9.4))
    gs = gridspec.GridSpec(
        3, 2, figure=fig, height_ratios=[1.0, 1.0, 2.0],
        wspace=0.30, hspace=0.33,
        left=0.07, right=0.985, top=0.965, bottom=0.04,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    # Row 1 = unadjusted (HR corr | Venn); Row 2 = adjusted (HR corr | Venn);
    # Row 3 = adjusted named Venn. Groups the adjusted story (c/d/e) together.
    plot_correlation(ax_a, merged_un, "Tel Aviv vs Oxford (unadjusted)")
    plot_venn(ax_b, tlv_un_sig, oxf_un_sig, "Significant proteins (unadjusted)")
    plot_correlation(ax_c, merged_ad, "Tel Aviv vs Oxford (adjusted)")
    plot_venn(ax_d, tlv_ad_sig, oxf_ad_sig, "Significant proteins (adjusted)")
    plot_named_venn(ax_e, tlv_ad_sig, oxf_ad_sig)

    for letter, ax in [("a", ax_a), ("b", ax_b), ("c", ax_c), ("d", ax_d)]:
        ax.text(-0.10, 1.10, letter, transform=ax.transAxes,
                fontsize=PANEL_LETTER_SIZE, fontweight="bold",
                va="top", ha="left")
    ax_e.text(0.02, 0.98, "e", transform=ax_e.transAxes,
              fontsize=PANEL_LETTER_SIZE, fontweight="bold",
              va="top", ha="left")

    out = os.path.join(OUT_DIR, "fig1_supp_alt_weights_venn")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"saved: {out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
