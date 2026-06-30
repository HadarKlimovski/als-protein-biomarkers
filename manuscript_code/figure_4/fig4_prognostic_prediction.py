"""Figure 4 — prognostic prediction (survival from enrollment / sampling anchor).

6-panel layout in Nature Medicine style:

  Row 1: three C-index bar panels, one per cohort (repeated 5-fold CV, 5 seeds).
    a. Tel Aviv     — clinical baseline (grey dashed) + protein additions
    b. Oxford
    c. Oxford + NfL
       Asterisks above bars are the CV-shuffle permutation test (the added
       protein vs. a same-dimension shuffled column).

  Row 2:
    d. 18-month ROC overlay — 3 cohorts; dashed = clinical-only, solid =
       clinical + GSN + IGFBP2 (+ NfL for Oxford+NfL). AUCs annotated.
    e. Kaplan-Meier tertile stratification — Tel Aviv, clinical + GSN + IGFBP2.
    f. Kaplan-Meier tertile stratification — Oxford + NfL, clinical + NfL +
       GSN + IGFBP2. Low/Intermediate/High = blue/grey/red.

Panel a (Tel Aviv) C-index bars are taken from the supplementary-table values
(tel_per_seed_sampling_suppmatched.csv) so the figure matches supp_table_fig4.

Inputs are read from ``$ALS_DATA_DIR/fig4`` (default: ``../data/fig4``); the
figure is written next to this script.

Output: fig4_prognostic_prediction.{pdf,png}
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

# ── paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    'ALS_DATA_DIR', os.path.join(HERE, os.pardir, 'data', 'figure_4'))
OUTDIR  = HERE

# ── Nature Medicine style ─────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'Nimbus Sans', 'DejaVu Sans'],
    'pdf.fonttype':      42,        # editable text in PDF
    'ps.fonttype':       42,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'legend.frameon':    False,
})
FS_TITLE   = 7
FS_AXIS    = 6
FS_TICK    = 6
FS_LEGEND  = 6
FS_SMALL   = 6
FS_ANNOT   = 5       # in-panel annotation text for ROC / KM
FS_PANEL   = 11
BAR_HEIGHT = 0.5     # thinner bars (matplotlib default is 0.8)

# ── cohort palette per style guide ────────────────────────────────────────────
COH_LIGHT = {'TEL': '#D68FA2', 'Oxford': '#BAD8DD', 'Oxford+NfL': '#AD9C8C'}
COH_DARK  = {'TEL': '#A42050', 'Oxford': '#009092', 'Oxford+NfL': '#6C5A49'}
COH_TITLE = {'TEL': 'Tel Aviv', 'Oxford': 'Oxford', 'Oxford+NfL': 'Oxford + NfL'}
COH_MARK  = {'TEL': 'o',        'Oxford': '^',      'Oxford+NfL': '^'}
CLIN_LIGHT, CLIN_DARK = '#A0A0A0', '#6A6A6A'
LOW, MID, HIGH = '#4C72B0', '#989898', '#C44E52'

COHORTS = ['TEL', 'Oxford', 'Oxford+NfL']
BAR_CONFIGS = ['clinical',
               'GSN', 'IGFBP2', 'MEGF10',
               'GSN+IGFBP2', 'GSN+MEGF10', 'IGFBP2+MEGF10',
               'GSN+IGFBP2+MEGF10']
BAR_LABELS  = ['Clinical',
               '+ GSN', '+ IGFBP2', '+ MEGF10',
               '+ GSN+IGFBP2', '+ GSN+MEGF10', '+ IGFBP2+MEGF10',
               '+ GSN+IGFBP2+MEGF10']

# Solid rows for forest: the "candidate" additions (singletons + pair).
# Dashed row added per cohort: the triple — visually shows MEGF10 adds nothing on top.
FOREST_ADDED_SOLID  = ['GSN', 'IGFBP2', 'MEGF10', 'GSN+IGFBP2']
FOREST_ADDED_DASHED = ['GSN+IGFBP2+MEGF10']
FOREST_ADDED = FOREST_ADDED_SOLID + FOREST_ADDED_DASHED
T_HORIZON = 18.0   # 18-month ROC variant (panel d)


# ── data loaders ──────────────────────────────────────────────────────────────
# Inputs are the cross-validation result tables produced by the analysis scripts
# survival_prediction_TEL.py / survival_prediction_Ox.py (written into DATA_DIR).
def _csv(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))


def load_data(cohort):
    if cohort == 'TEL':
        return {
            'per_seed':   _csv('tel_per_seed_sampling.csv'),
            'oof':        _csv('tel_oof_scores_sampling.csv'),
            'indep_tbl':  _csv('tel_indep_table_sampling.csv'),
            'lrt_cohort': 'TEL', 'baseline': 'clinical',
            'comb_label': 'Clinical + GSN + IGFBP2',
        }
    if cohort == 'Oxford':
        return {
            'per_seed':   _csv('oxford_per_seed_sampling.csv'),
            'oof':        _csv('oxford_oof_scores_sampling.csv'),
            'indep_tbl':  _csv('oxford_indep_table_sampling.csv'),
            'lrt_cohort': 'Oxford', 'baseline': 'clinical',
            'comb_label': 'Clinical + GSN + IGFBP2',
        }
    if cohort == 'Oxford+NfL':
        return {
            'per_seed':   _csv('oxford_nfl_per_seed_sampling.csv'),
            'oof':        _csv('oxford_nfl_oof_scores_sampling.csv'),
            'indep_tbl':  _csv('oxford_nfl_indep_table_sampling.csv'),
            'lrt_cohort': 'Oxford', 'baseline': 'clinical+NEFL',
            'comb_label': 'Clinical + NfL + GSN + IGFBP2',
        }


def stars(p):
    if pd.isna(p): return ''
    if p < 0.001:  return '***'
    if p < 0.01:   return '**'
    if p < 0.05:   return '*'
    return 'ns'


def cv_shuffle_p(d, cfg):
    """Per-config CV-shuffle p, derived from per-protein shuffles in indep_tbl.

    The CV pipeline shuffles each protein in a config in turn (100 reps each)
    and records the resulting C-index. For each protein j in this config:
        p_j = (#shuffles where Concordance_mean >= real_C + 1) / (n_shuffles + 1)
    We collapse the per-protein p_j's to a single per-config p via the MAXIMUM
    (the weakest link — every protein in the combo is at least this significant).
    Single-protein configs reduce to that protein's CV-shuffle p.
    """
    proteins = [p for p in cfg.split('+') if p in ('GSN', 'IGFBP2', 'MEGF10')]
    if not proteins:
        return np.nan
    real_v = d['per_seed'][d['per_seed']['config'] == cfg]['Concordance'].values
    real_C = float(np.mean(real_v)) if len(real_v) else np.nan
    if np.isnan(real_C):
        return np.nan
    ps = []
    for prot in proteins:
        m = ((d['indep_tbl']['_config_name'] == cfg) &
             (d['indep_tbl']['_protein']     == prot) &
             (d['indep_tbl']['_shuffle_idx'] >= 1))
        shuf = d['indep_tbl'][m]['Concordance_mean'].dropna().values
        if not len(shuf): continue
        ps.append((np.sum(shuf >= real_C) + 1) / (len(shuf) + 1))
    return float(max(ps)) if ps else np.nan


def bar_value(cohort, cfg):
    d = load_data(cohort)
    real_v = d['per_seed'][d['per_seed']['config'] == cfg]['Concordance'].values
    real_C = float(np.mean(real_v)) if len(real_v) else np.nan
    if cfg == 'clinical':
        return real_C, np.nan
    return real_C, cv_shuffle_p(d, cfg)


def roc_at(oof, risk_col, t=T_HORIZON):
    t_  = oof['time_months'].values
    e_  = oof['event'].values.astype(bool)
    r_  = oof[risk_col].values
    keep = (e_ & (t_ <= t)) | (t_ >= t)
    y = (e_[keep] & (t_[keep] <= t)).astype(int)
    fpr, tpr, _ = roc_curve(y, r_[keep])
    return fpr, tpr, auc(fpr, tpr)


# ── pre-compute bar data for shared y-axis ────────────────────────────────────
bar_data = {coh: {'means': [], 'ps': []} for coh in COHORTS}
for coh in COHORTS:
    for cfg in BAR_CONFIGS:
        m, p = bar_value(coh, cfg)
        bar_data[coh]['means'].append(m); bar_data[coh]['ps'].append(p)
_all_means = [m for coh in COHORTS for m in bar_data[coh]['means']]
# Round to clean 0.05 grid; ticks every 0.05
BAR_YMIN = max(0.50, np.floor((min(_all_means) - 0.01) / 0.05) * 0.05)
BAR_YMAX = min(1.00, np.ceil((max(_all_means) + 0.03) / 0.05) * 0.05)
BAR_TICKS = np.arange(BAR_YMIN, BAR_YMAX + 1e-9, 0.05)


# ── figure ────────────────────────────────────────────────────────────────────
# Nature Medicine full-page width = 180 mm; height sized so the two rows of
# panels are equal in height (no leftover whitespace band between them).
FIG_W = 180 / 25.4   # 7.087 in
fig = plt.figure(figsize=(FIG_W, 4.8))
# Single 2×3 gridspec so column edges are identical across rows. Letters can
# then be placed at consistent figure-x positions per column.
LEFT, RIGHT = 0.20, 0.97
WSPACE      = 0.25   # tighter column gap → wider, more square panels
# Two rows, equal height (0.36 each), with a 0.12 gap for the a/b/c x-label
# and the d/e/f titles.
gs_top = gridspec.GridSpec(1, 3, figure=fig,
                           left=LEFT, right=RIGHT, top=0.95, bottom=0.58,
                           wspace=WSPACE)
gs_bot = gridspec.GridSpec(1, 3, figure=fig,
                           left=LEFT, right=RIGHT, top=0.44, bottom=0.08,
                           wspace=WSPACE)
# Backwards-compat alias used by remainder of script (gs[1, j] → gs_bot[0, j]).
class _GS:
    def __getitem__(self, idx):
        r, c = idx
        return (gs_top if r == 0 else gs_bot)[0, c]
gs = _GS()

panel_letters = ['a', 'b', 'c', 'd', 'e', 'f']

# ─── Row 1: per-cohort bar panels ────────────────────────────────────────────
for ci, coh in enumerate(COHORTS):
    ax = fig.add_subplot(gs_top[0, ci])
    means = bar_data[coh]['means']; ps = bar_data[coh]['ps']
    # Clinical bar removed — clinical C-index is shown as a dashed reference
    # line only (labelled "Clinical" via the legend). Bars are protein configs.
    clin_C  = means[0]
    p_means = means[1:]; p_ps = ps[1:]
    # Horizontal bars: protein configs on y, C-index on x. Top-down reading
    # order: singletons → pairs → triple. Invert y so first item is on top.
    ys = np.arange(len(p_means))
    ax.barh(ys, p_means, height=BAR_HEIGHT,
            color=COH_LIGHT[coh], edgecolor=COH_DARK[coh], linewidth=0.7)
    ax.axvline(clin_C, color=CLIN_DARK, lw=0.6, ls='--', alpha=0.7, zorder=1,
               label='Clinical')
    for y, m, p in zip(ys, p_means, p_ps):
        s = stars(p)
        if s and s != 'ns':          # keep asterisks only; drop "ns"
            ax.text(m + 0.003, y, s, ha='left', va='center',
                    fontsize=FS_SMALL, color=COH_DARK[coh], fontweight='bold')
    ax.set_yticks(ys)
    if ci == 0:
        ax.set_yticklabels(BAR_LABELS[1:], fontsize=FS_TICK)
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(BAR_YMIN, BAR_YMAX)
    ax.set_xticks(BAR_TICKS)
    ax.set_xticklabels([f'{v:.2f}' for v in BAR_TICKS])
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlabel('C-index', fontsize=FS_AXIS)
    ax.set_title(COH_TITLE[coh], fontsize=FS_TITLE, fontweight='bold')
    if ci == 0:
        ax.legend(loc='lower right', fontsize=FS_LEGEND, handlelength=1.6)
    # panel letter placed later via fig.text for cross-row column alignment

# ─── Row 2 left (panel d): ROC overlay (promoted from former panel e) ────────
axE = fig.add_subplot(gs[1, 0])
axE.plot([0, 1], [0, 1], color='0.75', lw=0.5, ls='--')
auc_texts = []
for coh in COHORTS:
    d = load_data(coh)
    fpr_c, tpr_c, auc_c = roc_at(d['oof'], 'clin_risk_oof')
    fpr_x, tpr_x, auc_x = roc_at(d['oof'], 'comb_risk_oof')
    axE.plot(fpr_c, tpr_c, color=COH_LIGHT[coh], lw=1.0, ls='--')
    axE.plot(fpr_x, tpr_x, color=COH_DARK[coh],  lw=1.3)
    auc_texts.append((coh, auc_c, auc_x))
axE.set_xlim(0, 1); axE.set_ylim(0, 1)
axE.set_xticks([0, 0.5, 1.0])
axE.set_yticks([0, 0.5, 1.0])
# Fill the full cell (same height as e/f) rather than forcing a square ROC.
axE.set_xlabel('1 − Specificity', fontsize=FS_AXIS)
axE.set_ylabel('Sensitivity', fontsize=FS_AXIS)
axE.tick_params(labelsize=FS_TICK)
axE.set_title(f'{int(T_HORIZON)}-month ROC', fontsize=FS_TITLE, fontweight='bold')
# AUC values inline (no boxed legend) — one line per cohort, all in COH_DARK
yt = 0.30
for coh, ac, ax_ in auc_texts:
    axE.text(0.97, yt,
             f'{COH_TITLE[coh]}: {ac:.2f} → {ax_:.2f}',
             transform=axE.transAxes, ha='right', va='top',
             color=COH_DARK[coh], fontsize=FS_ANNOT)
    yt -= 0.06
# ─── KM helper: tertiles from CLINICAL-only risk; annotate logrank + LRT ─────
COL = {'low': LOW, 'mid': MID, 'high': HIGH}
LAB = {'low': 'Low', 'mid': 'Intermediate', 'high': 'High'}


def km_panel(ax, cohort_key, letter_idx, lrt_label, lrt_baseline):
    d = load_data(cohort_key)
    t_   = d['oof']['time_months'].values
    e_   = d['oof']['event'].values.astype(int)
    clin = d['oof']['clin_risk_oof'].values
    comb = d['oof']['comb_risk_oof'].values

    qs_c = np.quantile(clin, [1/3, 2/3])
    qs_k = np.quantile(comb, [1/3, 2/3])
    grp_clin = np.where(clin < qs_c[0], 'low',
                        np.where(clin >= qs_c[1], 'high', 'mid'))
    grp_comb = np.where(comb < qs_k[0], 'low',
                        np.where(comb >= qs_k[1], 'high', 'mid'))

    # Solid: tertiles from COMBINED model (clinical + proteins [+ NfL]).
    # The clinical-only curve is no longer drawn; its log-rank χ² is still
    # reported as a descriptive statistic in the annotation below.
    med = {}; n_per = {}
    for g in ['low', 'mid', 'high']:
        # combined (solid)
        kmf = KaplanMeierFitter()
        m = (grp_comb == g)
        kmf.fit(t_[m], e_[m], label='_nolegend_')
        kmf.plot_survival_function(ax=ax, color=COL[g], ci_show=False, lw=1.2)
        med[g] = kmf.median_survival_time_
        n_per[g] = int(m.sum())

    def _fmt(lr):
        return ('<1e-4' if lr.p_value < 1e-4 else f'{lr.p_value:.2g}')

    lr_comb = multivariate_logrank_test(t_, grp_comb, event_observed=e_)
    lr_clin = multivariate_logrank_test(t_, grp_clin, event_observed=e_)
    # Both logranks shown as DESCRIPTIVE statistics — each tests its own
    # stratification vs the null of no group difference. They are NOT directly
    # comparable to each other (same patients → χ² values are correlated; no
    # standard test for "χ²_combined > χ²_clinical"). The formal test for
    # "proteins add value beyond clinical" is the LRT in the supp table.
    # '+ proteins' = combined model: cohort dark color, but a deeper brown for
    # Oxford+NfL so it reads clearly as brown vs the grey '- proteins' line.
    plus_color = '#463A2F' if cohort_key == 'Oxford+NfL' else COH_DARK[cohort_key]
    ax.text(0.97, 0.97,
            f'+ proteins: χ²={lr_comb.test_statistic:.1f}, p={_fmt(lr_comb)}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FS_ANNOT, color=plus_color)
    ax.text(0.97, 0.91,
            f'clinical: χ²={lr_clin.test_statistic:.1f}, p={_fmt(lr_clin)}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FS_ANNOT, color='0.4')
    yt = 0.78
    for g in ['low', 'mid', 'high']:
        m_v = med[g]
        m_str = f'{m_v:.1f} m' if not np.isnan(m_v) else 'NR'
        ax.text(0.97, yt,
                f'{LAB[g]} (n={n_per[g]}): {m_str}',
                transform=ax.transAxes, ha='right', va='top',
                color=COL[g], fontsize=FS_ANNOT)
        yt -= 0.06
    ax.set_title(COH_TITLE[cohort_key], fontsize=FS_TITLE, fontweight='bold')
    ax.set_xlabel('Months since enrollment', fontsize=FS_AXIS)
    ax.set_ylabel('Survival probability', fontsize=FS_AXIS)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=FS_TICK)
    # lifelines auto-creates a legend even with label='_nolegend_'; force-remove
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    # panel letter placed later via fig.text for cross-row column alignment


# ─── Row 2 middle (panel e): TEL KM ───────────────────────────────────────────
axE_km = fig.add_subplot(gs[1, 1])
km_panel(axE_km, 'TEL', letter_idx=4, lrt_label='vs clinical',
         lrt_baseline='clinical')

# ─── Row 2 right (panel f): Oxford + NfL KM ───────────────────────────────────
axF = fig.add_subplot(gs[1, 2])
km_panel(axF, 'Oxford+NfL', letter_idx=5, lrt_label='vs clinical+NfL',
         lrt_baseline='clinical+NEFL')

# Panel f (Oxford+NfL) ends at 100 months; panel e (TEL) keeps its full range
# (curve not truncated). Both show ticks every 25 months (so both reach 100).
axF.set_xlim(0, 100)
axF.set_xticks([0, 25, 50, 75, 100])
_e_max = axE_km.get_xlim()[1]
axE_km.set_xlim(0, _e_max)
axE_km.set_xticks(np.arange(0, _e_max + 1e-6, 25))

# ─── panel letters: per-column figure-x positions for cross-row alignment ────
# Compute each column's leftmost spine x from the first panel of each row.
fig.canvas.draw()  # finalize subplot positions
def _col_x(gs_row, ci):
    bb = fig.add_subplot(gs_row[0, ci]).get_position()
    return bb.x0
# Use a slim helper that doesn't add new axes
import matplotlib.transforms as _mtr
_top_bboxes = [fig.axes[i].get_position() for i in range(3)]
_bot_bboxes = [fig.axes[i+3].get_position() for i in range(3)]
LETTER_DX = -0.045   # figure-x offset left of each column's spine
LETTER_DY = 0.012    # figure-y offset above each row's top
for ci in range(3):
    fig.text(_top_bboxes[ci].x0 + LETTER_DX,
             _top_bboxes[ci].y1 + LETTER_DY,
             panel_letters[ci],
             fontsize=FS_PANEL, fontweight='bold')
    fig.text(_bot_bboxes[ci].x0 + LETTER_DX,
             _bot_bboxes[ci].y1 + LETTER_DY,
             panel_letters[ci + 3],
             fontsize=FS_PANEL, fontweight='bold')

out = os.path.join(OUTDIR, 'fig4_prognostic_prediction')
fig.savefig(out + '.pdf')
fig.savefig(out + '.png', dpi=400)
plt.close()
print(f'Saved: {out}.pdf / .png')
