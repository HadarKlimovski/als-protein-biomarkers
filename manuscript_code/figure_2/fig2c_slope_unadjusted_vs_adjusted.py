"""Figure 2 Panel C — Unadjusted vs Adjusted slope chart with Tel Aviv | Oxford sub-titles.

Exposes `plot_slope_chart(ax)` for use by the figure assembler.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

DATA_DIR = os.environ.get("ALS_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

TLV_ADJ_PATH = os.path.join(DATA_DIR, "univariate_unadj_vs_adj.csv")
TLV_UNADJ_PATH = os.path.join(DATA_DIR, "significant_uniivariate_fdr_for_paper_new.csv")
OX_ADJ_PATH = os.path.join(DATA_DIR, "serum_cox_adjusted.csv")
OX_UNADJ_PATH = os.path.join(DATA_DIR, "serum_cox_unadjusted.csv")

RISK_COLOR = "#BC4D43"
PROTECTIVE_COLOR = "#2f7e85"
NONSIG_COLOR = "#BBBBBB"
REFLINE_COLOR = "0.4"
REFLINE_STYLE = "--"
REFLINE_WIDTH = 0.6
SPINE_WIDTH = 0.5

AXIS_LABEL_FONTSIZE = 5
TICK_FONTSIZE = 5
TITLE_FONTSIZE = 6
LABEL_FONTSIZE = 4

REPLICATED = ["GSN", "IGFBP2", "MEGF10"]


def _fdr05_logp(pvals):
    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) == 0:
        return None
    fdr = multipletests(p, method="fdr_bh")[1]
    if (fdr < 0.05).sum() == 0:
        return None
    return -np.log10(p[fdr < 0.05].max())


def _place_labels(ax, label_entries, x_dot, x_text, forbidden_ys=None):
    if not label_entries:
        return
    label_entries = sorted(label_entries, key=lambda t: t[0])
    _min_gap = 0.30
    _line_pad = 0.18
    orig_ys = [y for y, _, _ in label_entries]
    ys = list(orig_ys)
    for k in range(1, len(ys)):
        if ys[k] - ys[k - 1] < _min_gap:
            ys[k] = ys[k - 1] + _min_gap
    if forbidden_ys:
        for zone_y in forbidden_ys:
            if zone_y is None:
                continue
            for k in range(len(ys)):
                if abs(ys[k] - zone_y) < _line_pad:
                    orig_y = orig_ys[k]
                    ys[k] = zone_y + _line_pad if orig_y >= zone_y else zone_y - _line_pad
            for k in range(1, len(ys)):
                if ys[k] - ys[k - 1] < _min_gap:
                    ys[k] = ys[k - 1] + _min_gap
    for (orig_y, name, color), y in zip(label_entries, ys):
        ax.plot([x_dot, x_text - 0.01], [orig_y, y],
                color=color, linewidth=0.4, alpha=0.7, zorder=4,
                solid_capstyle="round")
        ax.text(x_text, y, name, fontsize=LABEL_FONTSIZE, color=color,
                va="center", ha="left", zorder=5, fontweight="bold")


def plot_slope_chart(ax):
    tlv_adj = pd.read_csv(TLV_ADJ_PATH).dropna(subset=["adj_p"]).copy()
    tlv_unadj = pd.read_csv(TLV_UNADJ_PATH).dropna(subset=["p_value"]).copy()
    ox_adj = pd.read_csv(OX_ADJ_PATH).dropna(subset=["p.value"]).copy()
    ox_unadj = pd.read_csv(OX_UNADJ_PATH).dropna(subset=["p.value"]).copy()

    overlap = set(tlv_adj["protein"]) & set(ox_adj["protein"])
    tlv_unadj_ov = tlv_unadj[tlv_unadj["protein"].isin(overlap)].copy()
    tlv_adj_ov   = tlv_adj[tlv_adj["protein"].isin(overlap)].copy()
    ox_unadj_ov  = ox_unadj[ox_unadj["protein"].isin(overlap)].copy()
    ox_adj_ov    = ox_adj[ox_adj["protein"].isin(overlap)].copy()

    _tlv_unadj_cut = _fdr05_logp(tlv_unadj_ov["p_value"])
    _tlv_adj_cut   = _fdr05_logp(tlv_adj_ov["adj_p"])
    _ox_unadj_cut  = _fdr05_logp(ox_unadj_ov["p.value"])
    _ox_adj_cut    = _fdr05_logp(ox_adj_ov["p.value"])

    def _sig_set(df, col):
        d = df.dropna(subset=[col]).copy()
        d = d[d[col] > 0]
        if len(d) == 0:
            return set()
        d["_q"] = multipletests(d[col], method="fdr_bh")[1]
        return set(d.loc[d["_q"] < 0.05, "protein"])

    tlv_universe = _sig_set(tlv_unadj_ov, "p_value") | _sig_set(tlv_adj_ov, "adj_p")
    ox_universe  = _sig_set(ox_unadj_ov, "p.value") | _sig_set(ox_adj_ov, "p.value")
    all_proteins = sorted(tlv_universe | ox_universe)
    sig_both = set(REPLICATED)

    rows = []
    for prot in all_proteins:
        u_row = tlv_unadj[tlv_unadj["protein"] == prot]
        a_row = tlv_adj[tlv_adj["protein"] == prot]
        oxa_row = ox_adj[ox_adj["protein"] == prot]
        oxu_row = ox_unadj[ox_unadj["protein"] == prot]
        def _logp(p):
            if pd.isna(p) or not (p > 0):
                return np.nan
            return -np.log10(max(p, 1e-300))
        rows.append({
            "Protein": prot,
            "tlv_unadj": _logp(u_row["p_value"].values[0]) if len(u_row) else np.nan,
            "tlv_adj":   _logp(a_row["adj_p"].values[0])   if len(a_row) else np.nan,
            "ox_unadj":  _logp(oxu_row["p.value"].values[0]) if len(oxu_row) else np.nan,
            "ox_adj":    _logp(oxa_row["p.value"].values[0]) if len(oxa_row) else np.nan,
            "in_tlv":    prot in tlv_universe,
            "in_ox":     prot in ox_universe,
        })
    df = pd.DataFrame(rows)

    unadj_hr_map = dict(zip(tlv_unadj["protein"], tlv_unadj["hazard_ratio"]))
    def dir_color(p):
        return RISK_COLOR if unadj_hr_map.get(p, 1.0) >= 1 else PROTECTIVE_COLOR

    cols = ["tlv_unadj", "tlv_adj", "ox_unadj", "ox_adj"]
    _BG_S, _KEY_S = 3, 7
    _tlv_label_entries, _ox_label_entries = [], []

    for _, row in df.iterrows():
        vals = [row[c] for c in cols]
        is_repl = row["Protein"] in sig_both
        if row["in_tlv"] and not is_repl and np.isfinite(vals[0]) and np.isfinite(vals[1]):
            ax.plot([0, 1], vals[:2], color=NONSIG_COLOR, linewidth=0.5, alpha=0.7, zorder=1)
            ax.scatter([0, 1], vals[:2], s=_BG_S, color=NONSIG_COLOR, alpha=0.7, zorder=1)
        if row["in_ox"] and not is_repl and np.isfinite(vals[2]) and np.isfinite(vals[3]):
            ax.plot([2, 3], vals[2:4], color=NONSIG_COLOR, linewidth=0.5, alpha=0.7, zorder=1)
            ax.scatter([2, 3], vals[2:4], s=_BG_S, color=NONSIG_COLOR, alpha=0.7, zorder=1)

    for _, row in df.iterrows():
        if row["Protein"] not in sig_both:
            continue
        vals = [row[c] for c in cols]
        color = dir_color(row["Protein"])
        if np.isfinite(vals[0]) and np.isfinite(vals[1]):
            ax.plot([0, 1], vals[:2], color=color, linewidth=1.2, alpha=0.95, zorder=3)
            ax.scatter([0, 1], vals[:2], s=_KEY_S, color=color, zorder=4)
            _tlv_label_entries.append((vals[1], row["Protein"], color))
        if np.isfinite(vals[2]) and np.isfinite(vals[3]):
            ax.plot([2, 3], vals[2:4], color=color, linewidth=1.2, alpha=0.95, zorder=3)
            ax.scatter([2, 3], vals[2:4], s=_KEY_S, color=color, zorder=4)
            _ox_label_entries.append((vals[3], row["Protein"], color))

    def _safe_min(*vals):
        vs = [v for v in vals if v is not None]
        return min(vs) if vs else None
    _tlv_line = _safe_min(_tlv_unadj_cut, _tlv_adj_cut)
    _ox_line  = _safe_min(_ox_unadj_cut, _ox_adj_cut)
    if _tlv_line is not None:
        ax.hlines(_tlv_line, -0.3, 1.3, colors=REFLINE_COLOR, linestyles=REFLINE_STYLE,
                  linewidth=REFLINE_WIDTH, zorder=0)
    if _ox_line is not None:
        ax.hlines(_ox_line, 1.7, 3.3, colors=REFLINE_COLOR, linestyles=REFLINE_STYLE,
                  linewidth=REFLINE_WIDTH, zorder=0)
    ax.axvline(1.5, color=REFLINE_COLOR, linestyle=REFLINE_STYLE,
               linewidth=REFLINE_WIDTH, zorder=0)

    _place_labels(ax, _tlv_label_entries, x_dot=1.0, x_text=1.12,
                  forbidden_ys=[_tlv_adj_cut])
    _place_labels(ax, _ox_label_entries,  x_dot=3.0, x_text=3.12,
                  forbidden_ys=[_ox_adj_cut])

    ax.text(0.5, 1.02, "Tel Aviv", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=TITLE_FONTSIZE)
    ax.text(2.5, 1.02, "Oxford", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=TITLE_FONTSIZE)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Unadjusted", "Adjusted", "Unadjusted", "Adjusted"],
                       fontsize=TICK_FONTSIZE)
    ax.set_ylabel(r"$-\log_{10}$(p-value)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=SPINE_WIDTH)
    ax.set_xlim(-0.3, 3.8)
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 400,
    })
    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    plot_slope_chart(ax)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "slope_chart")
    fig.savefig(out + ".png", bbox_inches="tight", dpi=400)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}.{{png,pdf}}")


if __name__ == "__main__":
    main()
