# Klimovski et al. — serum proteomics in ALS

Code for the survival-association and tissue-specificity analyses and figures in
the manuscript.

```
figure_1/       Figure 1 — unadjusted univariate Cox: analysis + panels + assembler
figure_2/       Figure 2 — adjusted univariate Cox: analysis + panels + assembler
sup_figure_1/   Supplementary Figure 1 — HR correlation + FDR-replication Venns
sup_figure_2/   Supplementary Figure 2 — Human Protein Atlas cell-type specificity
```

Each folder is self-contained: the Cox analysis script that generates the
figure's result table lives alongside the plotting scripts.

## Environment

Python ≥ 3.8. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

Scripts read their inputs from a `data/` folder at the repo root (override with
the `ALS_DATA_DIR` environment variable); figures are written next to each
script. **Patient-level data are not distributed** — place your own copies of the
required files in `data/`:

- `proteomics_survival_from_enrollment.csv` — Tel Aviv proteomics + survival
- `significant_uniivariate_fdr_for_paper_new.csv` — Tel Aviv univariate Cox table
- `serum_cox_unadjusted.csv`, `serum_cox_adjusted.csv` — Oxford univariate Cox (collaborator-provided)
- `univariate_unadj_vs_adj.csv` — adjusted-vs-unadjusted Cox table


HPA tissue data are downloaded from `proteinatlas.org` and cached in `data/`.

> **Cox analysis is Tel Aviv (discovery) only.** The Oxford replication Cox was
> run by collaborators with the same model; its tables (`serum_cox_unadjusted.csv`,
> `serum_cox_adjusted.csv`) are consumed by the plotting scripts but not generated
> here. The two analysis scripts are clean distillations of the original
> exploratory code and are **validated** to reproduce the published Tel Aviv
> tables to rounding precision (max Δ ≈ 5×10⁻⁷).

## `figure_1/` — Figure 1 (unadjusted univariate Cox)

| Script | Role | Description |
|---|---|---|
| `cox_univariate_unadjusted.py` | analysis | Tel Aviv per-protein **unadjusted** univariate Cox (survival from enrollment), BH-FDR → Fig. 1 table. |
| `fig1a_volcano_telaviv.py` | panel A | Tel Aviv unadjusted Cox volcano. |
| `fig1b_volcano_oxford.py` | panel B | Oxford unadjusted Cox volcano (broken y-axis for NEFL). |
| `fig1c_hr_discovery_vs_replication.py` | panel C | Discovery vs replication log2(HR) scatter (19 replicated proteins). |
| `fig1d_protein_clinical_correlation.py` | panel D | TLV + Oxford protein–clinical correlation heatmaps. |
| `assemble_figure1.py` | assembler | Assembles panels A–C into `fig1_main.pdf`. |

```bash
cd figure_1 && python assemble_figure1.py
```

## `figure_2/` — Figure 2 (adjusted univariate Cox)

| Script | Role | Description |
|---|---|---|
| `cox_univariate_adjusted.py` | analysis | Tel Aviv per-protein **age+sex adjusted** univariate Cox vs. unadjusted, BH-FDR → Fig. 2 table. |
| `fig2a_volcano_telaviv_adjusted.py` | panel A | Tel Aviv adjusted Cox volcano. |
| `fig2b_volcano_oxford_adjusted.py` | panel B | Oxford adjusted Cox volcano. |
| `fig2c_slope_unadjusted_vs_adjusted.py` | panel C | Unadjusted vs adjusted significance slope chart (TLV \| Oxford). |
| `fig2_venn_replicated_proteins.py` | — | Hollow Venn of FDR<0.05 adjusted proteins (overlap-FDR). |
| `label_utils.py` | — | Label-overlap helper used by the adjusted volcanoes. |
| `assemble_figure2.py` | A–C | Assembles panels A–C into `fig2_main.pdf`. |

```bash
cd figure_2 && python assemble_figure2.py
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

HPA data are downloaded from `proteinatlas.org` (cached in `data/` on first run);
outbound HTTPS required.
