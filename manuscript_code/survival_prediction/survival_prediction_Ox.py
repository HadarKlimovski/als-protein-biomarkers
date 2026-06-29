"""
survival_prediction_Ox.py
==========================
End-to-end survival-prediction pipeline for the Oxford cohort.

Runs the same cross-validation, time-dependent AUC, per-protein shuffle and
continuous-Cox likelihood-ratio analyses as `survival_prediction_TEL.py`,
producing two complete sets of outputs:

  - `oxford_*`     : clinical baseline = age + sex + delay + ALSFRS-R slope.
                     Adds GSN, IGFBP2, MEGF10 (alone and in combinations) and
                     serum neurofilament light chain (NfL) as standalone
                     "+NEFL" and "+GSN+IGFBP2+NEFL" configurations.
  - `oxford_nfl_*` : clinical baseline extended with NfL
                     (age + sex + delay + slope + NEFL). Adds the same
                     {GSN, IGFBP2, MEGF10} subsets on top, so the per-protein
                     shuffle test answers "does this protein add anything
                     beyond clinical + NfL?".

Model: Royston-Parmar flexible parametric survival (proportional-odds form,
two internal baseline knots, L2 penalty = 0.10), 10-fold CV repeated five
times, AUC at 6, 12, 18, 24, 30, 36 months, 100 train-fold shuffles per
protein, 1000 joint permutations for the LRT null.

Outputs (in ALS_DATA_DIR, default `../data/figure_4`):

  oxford_per_seed_sampling.csv,     oxford_nfl_per_seed_sampling.csv
  oxford_indep_table_sampling.csv,  oxford_nfl_indep_table_sampling.csv
  oxford_oof_scores_sampling.csv,   oxford_nfl_oof_scores_sampling.csv
  oxford_lrt.csv                    # joint LRT table covering both baselines

Run:
    ALS_DATA_DIR=../data/figure_4 \
    ALS_RAW_OXFORD=/path/to/oxford_patient_data.csv \
    python pipelines/survival_prediction_Ox.py
"""

from __future__ import annotations
import os
import time
import warnings
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines import CoxPHFitter
from lifelines import utils as ll_utils
from lifelines.fitters.coxph_fitter import ParametricSplinePHFitter
from lifelines.utils import concordance_index
from scipy.stats import chi2
from sklearn.model_selection import RepeatedKFold
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# Configuration
# ============================================================================
HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.environ.get("ALS_DATA_DIR", os.path.join(HERE, os.pardir, "data", "figure_4"))
RAW_OX    = os.environ.get(
    "ALS_RAW_OXFORD",
    os.path.join(HERE, os.pardir, os.pardir, "raw", "oxford_patient_data.csv"),
)
os.makedirs(DATA_DIR, exist_ok=True)

PENALIZER        = 0.10
N_BASELINE_KNOTS = 2
N_REPEATS        = 5
N_SPLITS         = 10
N_SHUFFLES       = 100
N_LRT_PERM       = 1000
N_JOBS           = -1
SEED             = 2026
AUC_HORIZONS     = (6.0, 12.0, 18.0, 24.0, 30.0, 36.0)
AUC_KEYS         = tuple(f"AUC_t{int(h)}" for h in AUC_HORIZONS)

PROTEINS = ("GSN", "IGFBP2", "MEGF10")
NEFL     = "NEFL"

# Configurations for the standard run (clinical = no NfL in baseline).
CONFIGS_NO_NFL = (
    "clinical",
    "GSN", "IGFBP2", "MEGF10",
    "GSN+IGFBP2", "GSN+MEGF10", "IGFBP2+MEGF10",
    "GSN+IGFBP2+MEGF10",
    "NEFL", "NEFL_only", "GSN+IGFBP2+NEFL",   # NfL-related additions
)

# Configurations for the NfL-clinical run (clinical already includes NfL).
CONFIGS_WITH_NFL = (
    "clinical",
    "GSN", "IGFBP2", "MEGF10",
    "GSN+IGFBP2", "GSN+MEGF10", "IGFBP2+MEGF10",
    "GSN+IGFBP2+MEGF10",
)

# Combined risk score used downstream for ROC + KM panels.
COMBO_FOR_OOF = ("GSN", "IGFBP2")


# ============================================================================
# Data loading
# ============================================================================
RAW = {
    "age_dead":     "AGE_AT_DEAD_OR_CENSORED",   # days
    "age_sampling": "AGE_AT_SAMPLING",           # years
    "age_onset":    "AGE_AT_SYMPTOM_ONSET",      # years
    "event_str":    "DEAD_OR_CENSORED",          # 'DEAD' / 'CENSORED'
    "gender":       "GENDER",                    # 'MALE' / 'FEMALE'
    "alsfrs_rate":  "ALSFRSR_RATE",              # ALSFRS-R progression slope
    "delay_days":   "ONSET_TO_SAMPLING_DELTA",   # days from onset to sampling
}
CLINICAL_COLS_NO_NFL   = ["age", "sex", "slope", "delay"]
CLINICAL_COLS_WITH_NFL = CLINICAL_COLS_NO_NFL + ["NEFL"]


def load_oxford(path: str = RAW_OX) -> Tuple[pd.DataFrame, bool]:
    """Load Oxford CSV, return (clean_dataframe, has_nfl_flag).

    Returned columns:
        time, event, age, sex, slope, delay, GSN, IGFBP2, MEGF10[, NEFL]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Oxford raw data not found at {path}. "
            "Set ALS_RAW_OXFORD to point at oxford_patient_data.csv."
        )
    raw = pd.read_csv(path)
    needed = list(RAW.values()) + list(PROTEINS)
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise KeyError(f"Required columns missing from Oxford CSV: {missing}")

    age_dead_yr = raw[RAW["age_dead"]] / 365.25
    df = pd.DataFrame({
        "time":  (age_dead_yr - raw[RAW["age_sampling"]]) * 12.0,
        "event": raw[RAW["event_str"]].map({"DEAD": 1, "CENSORED": 0}).astype(int),
        "age":   raw[RAW["age_sampling"]],
        "sex":   (raw[RAW["gender"]].astype(str).str.upper() == "FEMALE").astype(int),
        "slope": raw[RAW["alsfrs_rate"]],
        "delay": raw[RAW["delay_days"]] / 30.4375,   # days -> months
    })
    for p in PROTEINS:
        df[p] = raw[p]

    has_nfl = NEFL in raw.columns
    if has_nfl:
        df[NEFL] = raw[NEFL]
    df = df.dropna().reset_index(drop=True)
    df = df[df["time"] > 0].reset_index(drop=True)
    # NOTE: omics columns (proteins + NfL) are left at their raw scale;
    # standardisation is performed per fold inside `_fit_fold` (training-set
    # mean / SD applied to both training and held-out folds) to avoid
    # information leakage.
    return df, has_nfl


# ============================================================================
# Model: Royston-Parmar spline survival (proportional-odds, scale="odds")
# ============================================================================
# These two helper classes implement a Royston-Parmar flexible parametric
# survival model on the proportional-odds scale by overriding lifelines'
# CoxPHFitter._fit_model_spline. The internal lifelines spline fitter
# (ParametricSplinePHFitter) is reused under the hood; the override threads
# `scale="odds"` and `n_baseline_knots` through so the baseline log cumulative
# odds is modelled by a restricted cubic spline as in Royston & Parmar (2002).
class _CustomSplineFitter(ParametricSplinePHFitter):
    _scipy_fit_method  = "SLSQP"
    _scipy_fit_options = {"maxiter": 1000}
    _FAST_MEDIAN_PREDICT = False
    fit_intercept = True

    def __init__(self, strata, strata_values, n_baseline_knots=N_BASELINE_KNOTS,
                 knots=None, scale="hazard", **kw):
        self.strata        = ll_utils._to_list_or_singleton(strata)
        self.strata_values = strata_values
        self.scale         = scale
        self.n_baseline_knots = n_baseline_knots
        self.knots         = knots
        # Skip ParametricSplinePHFitter in the MRO so its strict positional
        # `strata`/`strata_values` signature does not reject the kwargs we
        # pass through; the spline-specific attributes are already set above.
        super(ParametricSplinePHFitter, self).__init__(**kw)


class SplineCox(CoxPHFitter):
    """Royston-Parmar flexible parametric survival model.

    A thin subclass of `CoxPHFitter` whose `_fit_model_spline` builds a
    `_CustomSplineFitter` instead of the default lifelines spline fitter so
    that `scale="odds"` (proportional-odds form) and the requested number of
    internal baseline knots take effect.
    """

    def __init__(self, scale: str = "odds",
                 n_baseline_knots: int = N_BASELINE_KNOTS,
                 penalizer: float = PENALIZER, **kw):
        super().__init__(penalizer=penalizer, **kw)
        self.scale            = scale
        self.n_baseline_knots = n_baseline_knots
        self.baseline_estimation_method = "spline"

    def _fit_model_spline(self, *args, **kwargs):
        df      = args[0].copy()
        formula = kwargs.pop("formula")
        kwargs.pop("cluster_col", None); kwargs.pop("batch_mode", None)
        strata  = ll_utils._to_list_or_singleton(kwargs.pop("strata", None))
        if strata is None:
            regressors = {"beta_": formula,
                          **{f"phi{i}_": "1"
                             for i in range(1, self.n_baseline_knots + 1)}}
            strata_values = None
        else:
            strata        = ll_utils._to_list_or_singleton(ll_utils._to_list(strata))
            df            = df.set_index(strata).sort_index()
            strata_values = df.groupby(strata).size().index.tolist()
            regressors    = {"beta_": formula}
            for sv in strata_values:
                regressors.update(
                    {_CustomSplineFitter._strata_labeler(sv, i): "1"
                     for i in range(1, self.n_baseline_knots + 1)})
        model = _CustomSplineFitter(
            strata=strata, strata_values=strata_values,
            penalizer=self.penalizer, l1_ratio=self.l1_ratio,
            n_baseline_knots=self.n_baseline_knots, knots=None,
            alpha=self.alpha, label=self._label, scale=self.scale)
        if ll_utils.CensoringType.is_right_censoring(self):
            model.fit_right_censoring(df, *args[1:], regressors=regressors, **kwargs)
        return model


def _make_model() -> SplineCox:
    return SplineCox(scale="odds",
                     n_baseline_knots=N_BASELINE_KNOTS,
                     penalizer=PENALIZER)


def _spline_lp(m: SplineCox, X_z: pd.DataFrame) -> np.ndarray:
    beta = m._model.params_["beta_"].drop("Intercept", errors="ignore")
    cols = [c for c in beta.index if c in X_z.columns]
    return X_z[cols].values @ beta[cols].values


# ============================================================================
# Fold-level helpers
# ============================================================================
def feature_columns(config: str, clinical_cols: List[str]) -> List[str]:
    feats = list(clinical_cols)
    if config == "clinical":
        return feats
    if config == "NEFL_only":
        return [NEFL]                       # NfL alone, no clinical covariates
    for token in config.split("+"):
        if token in PROTEINS:
            feats.append(token)
        elif token == "NEFL":
            if NEFL not in feats:           # avoid duplicate if NfL already in baseline
                feats.append(NEFL)
        else:
            raise ValueError(f"Unknown token '{token}' in config '{config}'.")
    return feats


def _fit_fold(df_train: pd.DataFrame, df_test: pd.DataFrame,
              feats: Iterable[str]) -> np.ndarray:
    feats = list(feats)
    X_tr, X_te = df_train[feats].copy(), df_test[feats].copy()
    mu = X_tr.mean()
    sd = X_tr.std(ddof=1).replace(0, 1)
    Xtr_z, Xte_z = (X_tr - mu) / sd, (X_te - mu) / sd
    fit_df = pd.concat([
        Xtr_z,
        pd.DataFrame({"time": df_train["time"].values,
                      "event": df_train["event"].values}),
    ], axis=1)
    try:
        m = _make_model()
        m.fit(fit_df, duration_col="time", event_col="event", show_progress=False)
        return _spline_lp(m, Xte_z)
    except Exception:
        return np.full(len(df_test), np.nan)


def _auc_at(time_all, event_all, time_te, event_te, lp_te, horizon):
    e_bool = event_te.astype(bool)
    if e_bool.sum() < 2:
        return np.nan
    h = min(horizon, time_te[e_bool].max() - 1e-3)
    if h <= time_te[e_bool].min():
        return np.nan
    surv_tr = Surv.from_arrays(event=event_all.astype(bool), time=time_all)
    surv_te = Surv.from_arrays(event=e_bool, time=time_te)
    try:
        auc, _ = cumulative_dynamic_auc(surv_tr, surv_te, lp_te, [h])
        return float(auc[0])
    except Exception:
        return np.nan


def _safe_mean(x):
    x = np.asarray(list(x), dtype=float)
    x = x[~np.isnan(x)]
    return float(np.mean(x)) if len(x) else np.nan


# ============================================================================
# Real + shuffle fold workers
# ============================================================================
def _run_real_fold(df, train_idx, test_idx, config, clinical_cols):
    d_tr = df.iloc[train_idx].reset_index(drop=True)
    d_te = df.iloc[test_idx].reset_index(drop=True)
    lp_te = _fit_fold(d_tr, d_te, feature_columns(config, clinical_cols))
    return config, np.asarray(test_idx), lp_te


def _run_shuffle_fold(df, train_idx, test_idx, config, protein, shuffle_idx,
                      clinical_cols):
    d_tr = df.iloc[train_idx].reset_index(drop=True)
    d_te = df.iloc[test_idx].reset_index(drop=True)
    rng = np.random.default_rng(SEED * 1000 + shuffle_idx * 31 + abs(hash(protein)) % 997)
    d_tr[protein] = rng.permutation(d_tr[protein].values)
    lp_te = _fit_fold(d_tr, d_te, feature_columns(config, clinical_cols))
    return config, protein, shuffle_idx, np.asarray(test_idx), lp_te


# ============================================================================
# Cross-validation pipeline
# ============================================================================
def run_cv_pipeline(df, configs, clinical_cols, combo_cfg):
    """Run CV + shuffle pipeline; return (per_seed_df, indep_table_df, oof_df)."""
    splits = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                random_state=SEED).split(df))
    n_folds_per_seed = N_SPLITS

    # ------------------------------------------------------------------ Real
    print(f"[real] {len(configs)} configs x {len(splits)} folds = "
          f"{len(configs) * len(splits)} fits", flush=True)
    t0 = time.time()
    real_results = Parallel(n_jobs=N_JOBS, batch_size=8)(
        delayed(_run_real_fold)(df, tr, te, cfg, clinical_cols)
        for (tr, te) in splits for cfg in configs
    )
    print(f"[real] done in {time.time() - t0:.1f}s", flush=True)

    real_lp = {}
    for fold_id, _ in enumerate(splits):
        for cfg_idx, cfg in enumerate(configs):
            real_lp[(fold_id, cfg)] = real_results[fold_id * len(configs) + cfg_idx]

    t_all = df["time"].values
    e_all = df["event"].values.astype(int)

    # ---------------------------- per-seed C-index + multi-horizon AUC ----
    per_seed_rows = []
    for seed_id in range(N_REPEATS):
        folds = range(seed_id * n_folds_per_seed, (seed_id + 1) * n_folds_per_seed)
        for cfg in configs:
            lp_oof = np.full(len(df), np.nan)
            for fold_id in folds:
                _, te_idx, lp_te = real_lp[(fold_id, cfg)]
                lp_oof[te_idx] = lp_te
            valid = ~np.isnan(lp_oof)
            row = {"seed": seed_id, "config": cfg, "Concordance": np.nan}
            for k in AUC_KEYS:
                row[k] = np.nan
            if valid.sum() >= 10:
                row["Concordance"] = concordance_index(
                    t_all[valid], -lp_oof[valid], e_all[valid])
                for key, h in zip(AUC_KEYS, AUC_HORIZONS):
                    row[key] = _auc_at(
                        t_all, e_all, t_all[valid], e_all[valid], lp_oof[valid], h)
            per_seed_rows.append(row)
    per_seed_df = pd.DataFrame(per_seed_rows)[
        ["seed", "config", "Concordance", *AUC_KEYS]
    ]

    # ---------------------------- OOF risk scores ------------------------
    def _pool_oof(cfg):
        sums = np.zeros(len(df))
        cnts = np.zeros(len(df))
        for fold_id, _ in enumerate(splits):
            _, te_idx, lp_te = real_lp[(fold_id, cfg)]
            valid = ~np.isnan(lp_te)
            sums[te_idx[valid]] += lp_te[valid]
            cnts[te_idx[valid]] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnts > 0, sums / cnts, np.nan)

    oof_df = pd.DataFrame({
        "time_months":   t_all,
        "event":         e_all,
        "clin_risk_oof": _pool_oof("clinical"),
        "comb_risk_oof": _pool_oof(combo_cfg),
    })

    # ---------------------------- indep table: real rows -----------------
    table_rows = []
    for cfg in configs:
        sub = per_seed_df[per_seed_df["config"] == cfg]
        row = {
            "label":             "clinical" if cfg == "clinical" else f"real_{cfg}",
            "Concordance_mean":  float(sub["Concordance"].mean()),
            "_config_name":      cfg,
            "_protein":          "",
            "_shuffle_idx":      -1,
        }
        for key in AUC_KEYS:
            row[f"{key}_mean"] = float(sub[key].mean())
        table_rows.append(row)

    # ---------------------------- Shuffles -------------------------------
    proteins_in_cfg = lambda c: [
        p for p in c.split("+") if p in PROTEINS or p == NEFL
    ]
    shuffle_configs = [c for c in configs if c not in ("clinical", "NEFL_only")]
    shuffle_tasks = []
    for cfg in shuffle_configs:
        for protein in proteins_in_cfg(cfg):
            # Skip shuffling NEFL inside clinical+NEFL run (NEFL is baseline already).
            if protein == NEFL and NEFL in clinical_cols:
                continue
            for shuffle_idx in range(1, N_SHUFFLES + 1):
                for (tr, te) in splits:
                    shuffle_tasks.append(
                        (df, tr, te, cfg, protein, shuffle_idx, clinical_cols))

    print(f"[shuffles] {len(shuffle_tasks)} tasks", flush=True)
    t0 = time.time()
    shuf_results = Parallel(n_jobs=N_JOBS, batch_size=16)(
        delayed(_run_shuffle_fold)(*a) for a in shuffle_tasks
    )
    print(f"[shuffles] done in {time.time() - t0:.1f}s", flush=True)

    grouped = {}
    for cfg, protein, shuffle_idx, te_idx, lp_te in shuf_results:
        grouped.setdefault((cfg, protein, shuffle_idx), []).append((te_idx, lp_te))

    seed_to_folds = {}
    for fold_id, _ in enumerate(splits):
        seed_to_folds.setdefault(fold_id // n_folds_per_seed, []).append(fold_id)

    for (cfg, protein, shuffle_idx), folds in grouped.items():
        cs, aucs = [], {h: [] for h in AUC_HORIZONS}
        for seed_id in range(N_REPEATS):
            lp_oof = np.full(len(df), np.nan)
            for fold_id in seed_to_folds[seed_id]:
                te_idx, lp_te = folds[fold_id]
                valid = ~np.isnan(lp_te)
                lp_oof[te_idx[valid]] = lp_te[valid]
            valid_oof = ~np.isnan(lp_oof)
            if valid_oof.sum() < 10:
                continue
            cs.append(concordance_index(
                t_all[valid_oof], -lp_oof[valid_oof], e_all[valid_oof]))
            for h in AUC_HORIZONS:
                aucs[h].append(_auc_at(
                    t_all, e_all,
                    t_all[valid_oof], e_all[valid_oof], lp_oof[valid_oof], h))
        row = {
            "label":             f"shuffle_{cfg}_{protein}_{shuffle_idx}",
            "Concordance_mean":  _safe_mean(cs),
            "_config_name":      cfg,
            "_protein":          protein,
            "_shuffle_idx":      shuffle_idx,
        }
        for key, h in zip(AUC_KEYS, AUC_HORIZONS):
            row[f"{key}_mean"] = _safe_mean(aucs[h])
        table_rows.append(row)
    indep_df = pd.DataFrame(table_rows)
    return per_seed_df, indep_df, oof_df


# ============================================================================
# LRT analysis (continuous Cox) + joint protein-block permutation null
# ============================================================================
def _fit_continuous_cox(df, covs):
    cph = CoxPHFitter(penalizer=PENALIZER)
    cph.fit(df[["time", "event"] + covs], duration_col="time",
            event_col="event", show_progress=False)
    return cph


def _lr_statistic(df, reduced, added):
    full = _fit_continuous_cox(df, reduced + added)
    base = _fit_continuous_cox(df, reduced)
    return 2.0 * (full.log_likelihood_ - base.log_likelihood_)


def run_lrt_analysis(df, has_nfl):
    """Run LRT (chi^2 + permutation p) for every (baseline, added) comparison.

    Two baselines are tested:
      - clinical (age + sex + slope + delay)
      - clinical + NEFL (only when NfL is present in the cohort).
    """
    df = df.copy()
    # Global z-scoring of omics columns for the LRT analysis (mirrors original
    # LRT script). Independent of the per-fold standardisation in the CV path.
    for c in list(PROTEINS) + ([NEFL] if has_nfl else []):
        df[c] = (df[c] - df[c].mean()) / df[c].std(ddof=1)

    comparisons = []
    candidate_added = (
        ("GSN",), ("IGFBP2",), ("MEGF10",),
        ("GSN", "IGFBP2"), ("GSN", "MEGF10"), ("IGFBP2", "MEGF10"),
        ("GSN", "IGFBP2", "MEGF10"),
    )
    for added in candidate_added:
        comparisons.append(("clinical", list(added)))
    if has_nfl:
        comparisons.append(("clinical", [NEFL]))
        for added in candidate_added:
            comparisons.append(("clinical+NEFL", list(added)))

    rng_master = np.random.default_rng(SEED)
    rows = []
    for baseline, added in comparisons:
        baseline_cols = (CLINICAL_COLS_WITH_NFL
                         if baseline == "clinical+NEFL" else CLINICAL_COLS_NO_NFL)
        sub = df[["time", "event"] + baseline_cols + added].dropna().reset_index(drop=True)
        observed = _lr_statistic(sub, baseline_cols, added)
        added_block = sub[added].values.copy()
        n = len(sub)
        nulls = []
        for _ in range(N_LRT_PERM):
            perm = sub.copy()
            perm[added] = added_block[rng_master.permutation(n)]
            try:
                nulls.append(_lr_statistic(perm, baseline_cols, added))
            except Exception:
                pass
        nulls = np.asarray(nulls)
        p_perm = float((np.sum(nulls >= observed) + 1) / (len(nulls) + 1)) \
                 if len(nulls) else np.nan
        rows.append({
            "baseline":   baseline,
            "added":      "+".join(added),
            "df":         len(added),
            "n":          n,
            "events":     int(sub["event"].sum()),
            "lrt_stat":   observed,
            "lrt_p_chi2": float(chi2.sf(observed, df=len(added))),
            "lrt_p_perm": p_perm,
            "n_perm_used": len(nulls),
        })
    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    print(f"[load] Oxford data: {RAW_OX}", flush=True)
    df, has_nfl = load_oxford()
    print(f"  N = {len(df)}, events = {int(df['event'].sum())}, "
          f"has_nfl = {has_nfl}", flush=True)

    # ----------------------------------------------------- Standard run --
    print("\n[run 1/2] standard baseline (no NfL in clinical)", flush=True)
    per_seed_n, indep_n, oof_n = run_cv_pipeline(
        df, CONFIGS_NO_NFL, CLINICAL_COLS_NO_NFL, "+".join(COMBO_FOR_OOF))
    per_seed_n.to_csv(os.path.join(DATA_DIR, "oxford_per_seed_sampling.csv"), index=False)
    indep_n.to_csv(  os.path.join(DATA_DIR, "oxford_indep_table_sampling.csv"), index=False)
    oof_n.to_csv(    os.path.join(DATA_DIR, "oxford_oof_scores_sampling.csv"), index=False)
    print("  saved: oxford_per_seed_sampling.csv / indep_table / oof_scores")

    # ----------------------------------------------------- NfL-clinical run
    if has_nfl:
        print("\n[run 2/2] NfL-in-clinical baseline", flush=True)
        per_seed_x, indep_x, oof_x = run_cv_pipeline(
            df, CONFIGS_WITH_NFL, CLINICAL_COLS_WITH_NFL, "+".join(COMBO_FOR_OOF))
        per_seed_x.to_csv(os.path.join(DATA_DIR, "oxford_nfl_per_seed_sampling.csv"), index=False)
        indep_x.to_csv(  os.path.join(DATA_DIR, "oxford_nfl_indep_table_sampling.csv"), index=False)
        oof_x.to_csv(    os.path.join(DATA_DIR, "oxford_nfl_oof_scores_sampling.csv"), index=False)
        print("  saved: oxford_nfl_per_seed_sampling.csv / indep_table / oof_scores")
    else:
        print("\n[run 2/2] NfL column not present in raw CSV; skipping NfL-clinical run.")

    # ----------------------------------------------------- LRT analysis --
    print("\n[lrt] running likelihood-ratio tests + permutation null ...", flush=True)
    lrt_df = run_lrt_analysis(df, has_nfl)
    out_lrt = os.path.join(DATA_DIR, "oxford_lrt.csv")
    lrt_df.to_csv(out_lrt, index=False)
    print(f"  saved: {out_lrt}")


if __name__ == "__main__":
    main()
