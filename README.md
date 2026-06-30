# Klimovski et al. — serum proteomics in ALS

Code for the survival-association and tissue-specificity analyses and figures in
the manuscript.

```
figure_2/       Figure 2 — unadjusted univariate Cox: analysis + panels + assembler
figure_3/       Figure 3 — adjusted univariate Cox: analysis + panels + assembler
figure_4/       Figure 4 — prognostic prediction (cross-validated Cox, enrollment anchor)
sup_figure_1/   Supplementary Figure 1 — HR correlation + FDR-replication Venns
sup_figure_2/   Supplementary Figure 2 — Human Protein Atlas cell-type specificity
sup_figure_3/   Supplementary Figure 3 — CV-shuffle permutation null for the C-index gains
```

Each folder is self-contained: the Cox analysis script that generates the
figure's result table lives alongside the plotting scripts.

## Environment

Python ≥ 3.8. Install dependencies:

```bash
pip install -r requirements.txt
```


> **Cox analysis is Tel Aviv (discovery) only.** The Oxford replication Cox was
> run by collaborators with the same model.

## `figure_2/` — Figure 2 (unadjusted univariate Cox)

| Script | Role | Description |
|---|---|---|
| `cox_univariate_unadjusted.py` | analysis | Tel Aviv per-protein **unadjusted** univariate Cox (survival from enrollment), BH-FDR → Fig. 2 table. |
| `fig1a_volcano_telaviv.py` | panel A | Tel Aviv unadjusted Cox volcano. |
| `fig1b_volcano_oxford.py` | panel B | Oxford unadjusted Cox volcano (broken y-axis for NEFL). |
| `fig1c_hr_discovery_vs_replication.py` | panel C | Discovery vs replication log2(HR) scatter (19 replicated proteins). |
| `fig1d_protein_clinical_correlation.py` | panel D | TLV + Oxford protein–clinical correlation heatmaps. |
| `assemble_figure1.py` | assembler | Assembles panels A–C into `fig1_main.pdf`. |

```bash
cd figure_2 && python assemble_figure1.py
```

## `figure_3/` — Figure 3 (adjusted univariate Cox)

| Script | Role | Description |
|---|---|---|
| `cox_univariate_adjusted.py` | analysis | Tel Aviv per-protein **age+sex adjusted** univariate Cox vs. unadjusted, BH-FDR → Fig. 3 table. |
| `fig2a_volcano_telaviv_adjusted.py` | panel A | Tel Aviv adjusted Cox volcano. |
| `fig2b_volcano_oxford_adjusted.py` | panel B | Oxford adjusted Cox volcano. |
| `fig2c_slope_unadjusted_vs_adjusted.py` | panel C | Unadjusted vs adjusted significance slope chart (TLV \| Oxford). |
| `fig2_venn_replicated_proteins.py` | — | Hollow Venn of FDR<0.05 adjusted proteins (overlap-FDR). |
| `label_utils.py` | — | Label-overlap helper used by the adjusted volcanoes. |
| `assemble_figure2.py` | A–C | Assembles panels A–C into `fig2_main.pdf`. |

```bash
cd figure_3 && python assemble_figure2.py
```

Discovery-significant proteins are declared at **overlap-FDR < 0.05** (BH
re-applied to the proteins shared with the replication panel); the overlap step
is applied inside the volcano/slope/Venn scripts from the `cox/` result table.

## `sup_figure_1/` — Supplementary Figure 1

| Script | Panel | Description |
|---|---|---|
| `supfig1_venn_fdr_replication.py` | — | Venn of FDR<0.05 proteins, Tel Aviv vs Oxford (each over its full panel). |
| `assemble_supfig1.py` | a–d | 4-panel layout: HR correlation unadjusted (a) & adjusted (b); FDR Venn unadjusted (c) & adjusted (d). |

```bash
cd sup_figure_1 && python assemble_supfig1.py
```

## `sup_figure_2/` — HPA cell-type specificity (Supplementary Figure 2)

| Script | Produces | Description |
|---|---|---|
| `fetch_hpa_annotations.py` | — | Fetches per-gene HPA annotations for the discovery proteins → `hpa_protein_summary.csv`. |
| `supfig2_celltype_origin.py` | Sup. | HPA single-cell expression for GSN, IGFBP2, MEGF10. |
| `supfig2_celltype_origin_all.py` | Sup. | HPA single-cell expression across all 154 cell types. |

HPA tissue data are downloaded from `proteinatlas.org` and cached in `data/`.
The plot stylesheet (`figure_style.mplstyle`) ships with the repo at the root.


## `figure_4/` — Figure 4 (prognostic prediction)

Cross-validated Cox prognostic models (Royston–Parmar flexible parametric
survival, 10-fold CV × 5 repeats), **survival from enrollment / sampling anchor
only**. Self-contained: two analysis pipelines generate the result tables, one
plotting script renders the 6-panel figure (Supplementary Figure 3 reads the
same tables).

```
survival_prediction_TEL.py ─┐
survival_prediction_Ox.py  ─┴─► data/figure_4/*.csv ─► fig4_prognostic_prediction.py        (Figure 4)
                                                    └─► ../sup_figure_3/supfig3_cindex_shuffle.py (Sup. Fig. 3)
```

| Script | Role | Description |
|---|---|---|
| `survival_prediction_TEL.py` | analysis | Tel Aviv pipeline → per-seed C-index/AUC, per-protein shuffle null, OOF risk scores, LRT. |
| `survival_prediction_Ox.py` | analysis | Oxford pipeline (clinical and clinical+NfL baselines) → same outputs for Oxford and Oxford+NfL. |
| `fig4_prognostic_prediction.py` | plotting | C-index bars per cohort (a–c), 18-month ROC overlay (d), Kaplan–Meier tertile stratification for Tel Aviv (e) and Oxford+NfL (f). Stars = CV-shuffle permutation test. |

```bash
# 1. generate result tables (needs raw patient data — see below)
cd figure_4
ALS_RAW_TEL=/path/to/Tel_aviv_full_data.csv   python survival_prediction_TEL.py
ALS_RAW_OXFORD=/path/to/oxford_patient_data.csv python survival_prediction_Ox.py
# 2. render the figure from those tables
python fig4_prognostic_prediction.py
```

## `sup_figure_3/` — Supplementary Figure 3 (CV-shuffle null)

| Script | Produces | Description |
|---|---|---|
| `supfig3_cindex_shuffle.py` | Sup. | Real cross-validated C-index (coloured line, median over seeds) vs. the same-dimension column-shuffle null (grey raincloud) with one-sided permutation p, for TEL / Oxford / Oxford+NfL (sampling anchor). Reads the same `data/figure_4/` tables as Figure 4. |

```bash
cd sup_figure_3 && python supfig3_cindex_shuffle.py
```

### Figure 4 / Sup. Figure 3 data

The two analysis pipelines read **raw patient-level CSVs** (not distributed; set
via env var) and write their result tables into `data/figure_4/`:

| Env var | Raw input |
|---|---|
| `ALS_RAW_TEL` | Tel Aviv patient data (`Tel_aviv_full_data.csv`) |
| `ALS_RAW_OXFORD` | Oxford patient data (`oxford_patient_data.csv`, incl. NfL) |

Result tables (consumed by both plotting scripts, written under
`$ALS_DATA_DIR`, default `data/figure_4/`):

```
data/figure_4/
  tel_{per_seed,oof_scores,indep_table}_sampling.csv, tel_lrt.csv
  oxford_{per_seed,oof_scores,indep_table}_sampling.csv, oxford_lrt.csv
  oxford_nfl_{per_seed,oof_scores,indep_table}_sampling.csv
```

`per_seed` = one row per (seed, config): C-index + six AUC horizons.
`oof_scores` = per-patient out-of-fold risk (`time_months, event,
clin_risk_oof, comb_risk_oof`) for ROC/KM. `indep_table` = real + per-protein
shuffle C-index for the permutation null. `lrt` = continuous-Cox likelihood
ratio test (χ² and permutation p).
