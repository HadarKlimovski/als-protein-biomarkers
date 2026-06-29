"""
Adjusted univariate Cox survival analysis — Figure 2.
=====================================================
For each serum protein, fits two Cox models of survival from enrollment and
compares them:
  1. Unadjusted:  Cox(survival ~ protein)
  2. Adjusted:    Cox(survival ~ protein + age at onset + sex)

A Benjamini-Hochberg FDR correction is applied across proteins separately for
the unadjusted and adjusted p-values, and the change in log(HR) after
adjustment (effect attenuation) is reported.

Outputs a single results table (`univariate_unadj_vs_adj.csv`) that feeds the
Fig. 2 volcano panels (`volcano_tlv_adjusted.py`). Discovery-significant
proteins are declared at overlap-FDR < 0.05 (BH re-applied to the proteins
shared with the replication panel; see methods) — that overlap step is done
in the plotting scripts from this table.

Distilled from `visit_in_sheffield/tlv/univariate_adjusted_vs_unadjusted.py`
(inline exploratory plots removed; see the figure_2 volcano scripts).

Usage:
    python univariate_cox_adjusted.py
"""
import os
import warnings
warnings.filterwarnings("ignore")

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
OUT_CSV = "univariate_unadj_vs_adj.csv"

TIME_COL = "Survival_from_enrollment"
EVENT_COL = "Status dead=1"
AGE_COL = "Age Onset (years)"
SEX_COL = "Sex female"



def fit_one(df, cols, protein):
    """Fit a Cox model on the given columns; return (p, HR, CI_lo, CI_hi, c)."""
    d = df[cols + [protein]].dropna()
    cph = CoxPHFitter()
    cph.fit(d, duration_col=TIME_COL, event_col=EVENT_COL)
    s = cph.summary.loc[protein]
    return (s["p"], cph.hazard_ratios_[protein],
            s["exp(coef) lower 95%"], s["exp(coef) upper 95%"],
            cph.concordance_index_)


def main():
    df = pd.read_csv(DATA_CSV)
    df = df[df[TIME_COL] > 0].copy()
    protein_cols = [c for c in df.columns]
    print(f"Patients: {len(df)}, Events: {int(df[EVENT_COL].sum())}, "
          f"Proteins: {len(protein_cols)}")

    rows = []
    for protein in protein_cols:
        try:
            up, uhr, ulo, uhi, uc = fit_one(df, [TIME_COL, EVENT_COL], protein)
            ap, ahr, alo, ahi, ac = fit_one(
                df, [TIME_COL, EVENT_COL, AGE_COL, SEX_COL], protein)
        except Exception:
            continue
        rows.append({
            "protein": protein,
            "unadj_p": up, "unadj_hazard_ratio": uhr,
            "unadj_CI_lower": ulo, "unadj_CI_upper": uhi,
            "unadj_concordance_index": uc,
            "adj_p": ap, "adj_hazard_ratio": ahr,
            "adj_CI_lower": alo, "adj_CI_upper": ahi,
            "adj_concordance_index": ac,
        })

    res = pd.DataFrame(rows)

    # BH-FDR across proteins, separately for unadjusted and adjusted
    res["unadj_FDR"] = multipletests(res["unadj_p"], method="fdr_bh")[1]
    res["adj_FDR"] = multipletests(res["adj_p"], method="fdr_bh")[1]
    res["unadj_FDR_significant"] = res["unadj_FDR"] < 0.05
    res["adj_FDR_significant"] = res["adj_FDR"] < 0.05

    # Effect attenuation after adjustment
    res["delta_logHR"] = np.log(res["adj_hazard_ratio"]) - np.log(res["unadj_hazard_ratio"])

    res = res.sort_values("unadj_p").reset_index(drop=True)
    res.round(6).to_csv(OUT_CSV, index=False)

    print(f"Unadjusted FDR<0.05: {int(res['unadj_FDR_significant'].sum())} | "
          f"Adjusted FDR<0.05: {int(res['adj_FDR_significant'].sum())}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
