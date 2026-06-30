"""Extended Data Figure 3 — CV-shuffle permutation null for the C-index gains.

Three rows, one per cohort (survival from enrollment / sampling anchor):
  1. TEL          — sampling-anchored
  2. Oxford       — sampling-anchored
  3. Oxford + NfL — sampling-anchored

For each protein addition, the coloured line is the real cross-validated
C-index (median across seeds) and the grey raincloud is the null distribution
obtained by shuffling that protein's column (same dimension); the annotated p
is the one-sided permutation p-value. All rows use the SAME clinical baseline
(age + sex + delay(onset→sampling) + alsfrs_slope) so the C-index axes are
comparable.

Inputs are read from ``$ALS_DATA_DIR/fig4`` (default: ``../data/fig4``); the
figure is written next to this script.

Output: supfig3_cindex_shuffle.{pdf,png}
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('ALS_DATA_DIR', HERE.parent / 'data' / 'figure_4'))
OUTDIR  = HERE
STYLE   = HERE.parent / 'figure_style.mplstyle'

CONFIGS = [
    'GSN',
    'IGFBP2',
    'MEGF10',
    'GSN+MEGF10',
    'GSN+IGFBP2',
    'IGFBP2+MEGF10',
    'GSN+IGFBP2+MEGF10',
]
PROTEINS = ['GSN', 'IGFBP2', 'MEGF10']
COL_WIDTHS = [cfg.count('+') + 1 for cfg in CONFIGS]

C_CLIN  = '#444444'
C_NULL  = '0.70'

# Cohort colors — match the main figure (fig4_layoutB_v2)
COH_DARK = {'TEL': '#A42050', 'Oxford': '#009092', 'Oxford + NfL': '#6C5A49'}

METRIC = 'Concordance'


# ── beeswarm + histogram (half-raincloud) ─────────────────────────────────────
def half_raincloud(ax, xpos, values, color, half_width=0.38, n_bins=15):
    if len(values) == 0:
        return
    counts, edges = np.histogram(values, bins=n_bins)
    mids  = (edges[:-1] + edges[1:]) / 2
    bar_h = (edges[1] - edges[0]) * 0.88
    scale = half_width / counts.max() if counts.max() > 0 else 1.0
    ax.barh(mids, counts * scale, height=bar_h,
            left=xpos, color=color, alpha=0.30, linewidth=0, zorder=2)
    bin_idx = np.clip(np.digitize(values, edges) - 1, 0, n_bins - 1)
    dot_x   = np.zeros(len(values))
    for b in range(n_bins):
        idx = np.where(bin_idx == b)[0]
        n = len(idx)
        if n == 0:
            continue
        bar_len = counts[b] * scale
        positions = np.linspace(-bar_len, 0, n) if n > 1 else np.array([0.0])
        dot_x[idx] = positions
    ax.scatter(xpos + dot_x, values,
               color=color, s=2.5, alpha=0.55, linewidths=0, zorder=2)


# ── data helpers ──────────────────────────────────────────────────────────────
def real_median(per_seed, cfg):
    v = per_seed[per_seed['config'] == cfg][METRIC].values
    return np.median(v) if len(v) else np.nan


def clin_median(per_seed):
    return real_median(per_seed, 'clinical')


def shuffle_strip(table, cfg, prot):
    return table[
        (table['_config_name'] == cfg) &
        (table['_protein'] == prot) &
        (table['_shuffle_idx'] >= 1)
    ][f'{METRIC}_mean'].dropna().values


def pvalue(real_med, shuf):
    n = len(shuf)
    return float((np.sum(shuf >= real_med) + 1) / (n + 1)) if n else np.nan


# ── data sources ──────────────────────────────────────────────────────────────
# CV result tables produced by survival_prediction_TEL.py / survival_prediction_Ox.py.
def _load(stem):
    return (
        pd.read_csv(DATA_DIR / f'{stem}_per_seed_sampling.csv'),
        pd.read_csv(DATA_DIR / f'{stem}_indep_table_sampling.csv'),
    )


# Sampling-anchored only (onset-anchored rows removed). One color per cohort.
ROWS = [
    ('TEL',          'sampling', COH_DARK['TEL'],          'TEL — sampling-anchored',
     lambda: _load('tel')),
    ('Oxford',       'sampling', COH_DARK['Oxford'],       'Oxford — sampling-anchored',
     lambda: _load('oxford')),
    ('Oxford + NfL', 'sampling', COH_DARK['Oxford + NfL'], 'Oxford + NfL — sampling-anchored',
     lambda: _load('oxford_nfl')),
]


# ── draw one row ──────────────────────────────────────────────────────────────
def draw_row(axes, per_seed, table, color, label):
    clin_med = clin_median(per_seed)

    all_y = []
    for cfg in CONFIGS:
        all_y.append(real_median(per_seed, cfg))
        for p in [pp for pp in cfg.split('+') if pp in PROTEINS]:
            all_y.extend(shuffle_strip(table, cfg, p))
    all_y.append(clin_med)
    ylo = max(0.50, np.nanmin(all_y) - 0.012)
    yhi = min(1.00, np.nanmax(all_y) + 0.022)

    last = len(CONFIGS) - 1
    for ci, (ax, cfg) in enumerate(zip(axes, CONFIGS)):
        prots = [p for p in cfg.split('+') if p in PROTEINS]
        med   = real_median(per_seed, cfg)
        n_sub = len(prots)

        ax.axhline(clin_med, color=C_CLIN, lw=0.85, ls='--', zorder=2,
                   label='Clinical' if ci == 0 else None)
        ax.axhline(med, color=color, lw=1.0, ls='-', zorder=3,
                   label='Real model median' if ci == 0 else None)

        for j, prot in enumerate(prots):
            shuf = shuffle_strip(table, cfg, prot)
            if len(shuf):
                half_raincloud(ax, xpos=j, values=shuf, color=C_NULL)
                p = pvalue(med, shuf)
                p_txt = f'p={p:.2g}' if p >= 0.001 else 'p<0.001'
                ax.text(j, yhi - 0.002, p_txt, ha='center', va='top',
                        fontsize=6, color=color, fontweight='bold')

        ax.set_xlim(-0.5, n_sub - 0.5)
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(range(n_sub))
        ax.set_xticklabels(prots, fontsize=6)
        ax.set_title(cfg, fontsize=6, fontweight='normal', pad=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=6)

        if ci == 0:
            ax.set_ylabel('C-index', fontsize=6)
        else:
            ax.set_yticklabels([])
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', left=False)

        if ci == last:
            # small inline label at the right end of the dashed clinical line
            ax.text(n_sub - 0.5 + 0.08, clin_med, 'clinical',
                    ha='left', va='center', fontsize=6, color=C_CLIN,
                    clip_on=False)

    # centered row title, lifted above the per-panel protein labels
    fig0 = axes[0].figure
    p0 = axes[0].get_position(); p1 = axes[-1].get_position()
    fig0.text((p0.x0 + p1.x1) / 2, p0.y1 + 0.028, label,
              ha='center', va='bottom', fontsize=7, fontweight='bold')


# ── figure ────────────────────────────────────────────────────────────────────
plt.style.use(str(STYLE))

fig, axes = plt.subplots(
    len(ROWS), len(CONFIGS),
    figsize=(9.0, 5.7),
    gridspec_kw={'width_ratios': COL_WIDTHS, 'wspace': 0.06, 'hspace': 0.85},
)

for row_i, (_cohort, _anchor, color, label, loader) in enumerate(ROWS):
    per_seed, table = loader()
    draw_row(axes[row_i], per_seed, table, color, label)

out_base = OUTDIR / 'supfig3_cindex_shuffle'
fig.savefig(str(out_base) + '.pdf', bbox_inches='tight')
fig.savefig(str(out_base) + '.png', bbox_inches='tight', dpi=400)
plt.close()
print(f'Saved: {out_base}.pdf / .png')
