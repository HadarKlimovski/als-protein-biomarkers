"""
Univariate Cox proportional-hazards survival analysis (unadjusted) — Figure 1.
==============================================================================
For every serum protein, fits a univariate Cox model of survival from
enrollment (protein abundance as the sole covariate) on the full cohort, then
applies a Benjamini-Hochberg FDR correction across proteins.

Pipeline (one cohort):
  1. Load the protein x sample matrix with merged clinical columns.
  2. For each protein: univariate Cox; store p, HR, 95% CI, C-index.
  3. BH-FDR across proteins; flag FDR < 0.05.

Protein abundances are the platform-normalized (limma) values; no further
scaling is applied (hazard ratios are per unit of normalized abundance).

Run for the Tel Aviv (discovery) cohort; the Oxford replication cohort is
analysed by collaborators with the same model (see methods).

Reproduces the analysis behind `volcano_tlv.py` (Fig. 1A). Distilled from the
original exploratory notebook `survival_analysis/cox/cox_univariant.ipynb`.

Usage:
    python univariate_cox.py
"""
import os
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Place the input file in a local `data/` folder at the repo root, or set the
# ALS_DATA_DIR environment variable. Patient-level data are not distributed.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("ALS_DATA_DIR", os.path.join(HERE, os.pardir, "data"))
DATA_CSV = os.path.join(DATA_DIR, "proteomics_survival_from_enrollment.csv")
OUT_CSV = "univariate_cox_telaviv_from_enrollment.csv"

TIME_COL = "Survival_from_enrollment"
EVENT_COL = "Status dead=1"



def load_cohort(path):
    """Load matrix, keep survival columns; return df and protein list."""
    df = pd.read_csv(path)
    df = df[df[TIME_COL] > 0].copy()
    protein_cols = [c for c in df.columns]
    return df, protein_cols


def univariate_cox(df, protein):
    """Univariate Cox for one protein on the full cohort; return statistics."""
    d = df.dropna(subset=[protein])[[protein, TIME_COL, EVENT_COL]]
    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(d, duration_col=TIME_COL, event_col=EVENT_COL)
    s = cph.summary.loc[protein]
    return {"p_value": s["p"], "concordance_index": cph.concordance_index_,
            "hazard_ratio": cph.hazard_ratios_[protein],
            "CI_lower": s["exp(coef) lower 95%"],
            "CI_upper": s["exp(coef) upper 95%"]}


def run(path, out_csv):
    df, protein_cols = load_cohort(path)
    rows = {prot: univariate_cox(df, prot) for prot in protein_cols}

    res = pd.DataFrame(rows).T
    res = res.sort_values("p_value")
    res["FDR_adjusted_pval"] = multipletests(res["p_value"], method="fdr_bh")[1]
    res["FDR_significant"] = res["FDR_adjusted_pval"] < 0.05
    res = res.round(6).reset_index().rename(columns={"index": "protein"})
    res.to_csv(out_csv, index=False)

    n_sig = int(res["FDR_significant"].sum())
    print(f"{len(res)} proteins tested | {n_sig} FDR-significant (q<0.05)")
    print(f"wrote {out_csv}")
    return res


if __name__ == "__main__":
    run(DATA_CSV, OUT_CSV)
