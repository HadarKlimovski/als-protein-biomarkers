"""
survival_prediction_TEL.py
===========================
End-to-end survival-prediction pipeline for the Tel Aviv cohort.

For each protein configuration (clinical baseline, plus every non-empty subset
of {GSN, IGFBP2, MEGF10}) the script

  1. fits a Royston-Parmar flexible parametric survival model
     (proportional-odds form, 2 internal baseline knots, L2 penalty = 0.10)
     across 10-fold cross-validation repeated five times;
  2. pools the out-of-fold (OOF) linear predictors within each repeat to
     estimate Harrell's concordance index and time-dependent AUC at six
     horizons (6, 12, 18, 24, 30, 36 months);
  3. runs a per-protein train-fold permutation test (100 shuffles of each
     protein in turn, refitting the model on the shuffled training data) to
     produce an empirical null distribution of cross-validated C-index for
     each protein within every configuration;
  4. saves the per-patient OOF linear predictors for the clinical-only and
     the combined clinical + GSN + IGFBP2 models (used downstream for ROC and
     Kaplan-Meier visualisation);
  5. computes the continuous Cox likelihood-ratio test, calibrated against an
     empirical null built from 1000 joint permutations of the added protein
     block.

Outputs (written into the directory pointed to by ALS_DATA_DIR, default
`../data/figure_4`):

  tel_per_seed_sampling.csv     # one row per (seed, config): C-index + six AUC values
  tel_indep_table_sampling.csv  # one row per real config and per
                                #   (config, protein, shuffle index)
  tel_oof_scores_sampling.csv   # one row per patient: time, event,
                                #   clinical risk, combined risk
  tel_lrt.csv                   # one row per protein-addition comparison
                                #   (LR statistic, df, chi^2 p, permutation p)

Run:
    ALS_DATA_DIR=../data/figure_4 \
    ALS_RAW_TEL=/path/to/Tel_aviv_full_data.csv \
    python pipelines/survival_prediction_TEL.py
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
HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("ALS_DATA_DIR", os.path.join(HERE, os.pardir, "data", "figure_4"))
RAW_TEL  = os.environ.get(
    "ALS_RAW_TEL",
    os.path.join(HERE, os.pardir, os.pardir, "raw", "Tel_aviv_full_data.csv"),
)
os.makedirs(DATA_DIR, exist_ok=True)

PENALIZER         = 0.10
N_BASELINE_KNOTS  = 2
N_REPEATS         = 5
N_SPLITS          = 10
N_SHUFFLES        = 100
N_LRT_PERM        = 1000
N_JOBS            = -1
SEED              = 2026
AUC_HORIZONS      = (6.0, 12.0, 18.0, 24.0, 30.0, 36.0)
AUC_KEYS          = tuple(f"AUC_t{int(h)}" for h in AUC_HORIZONS)

PROTEINS  = ("GSN", "IGFBP2", "MEGF10")
CONFIGS   = (
    "clinical",
    "GSN", "IGFBP2", "MEGF10",
    "GSN+IGFBP2", "GSN+MEGF10", "IGFBP2+MEGF10",
    "GSN+IGFBP2+MEGF10",
)
COMBO_FOR_OOF = ("GSN", "IGFBP2")  # combined risk score used downstream for ROC / KM

# LRT comparisons: every non-empty subset of {GSN, IGFBP2, MEGF10} added to
# the clinical baseline.
LRT_ADDED_SETS = (
    ("GSN",), ("IGFBP2",), ("MEGF10",),
    ("GSN", "IGFBP2"), ("GSN", "MEGF10"), ("IGFBP2", "MEGF10"),
    ("GSN", "IGFBP2", "MEGF10"),
)


# ============================================================================
# Data loading
# ============================================================================
RAW_COLS = {
    "event":   "event",
    "time":    "Survival_from_enrollment",       # months from blood sampling
    "age":     "Age at Collection (years)",
    "sex":     "Sex female",
    "slope":   "ALSFRS slope",                   # ALSFRS-R progression slope
    "delay":   "Sampling_delay_from_onset",      # signed months; abs() applied
}
CLINICAL_COLS = ["age", "sex", "slope", "delay"]


def load_tel(path: str = RAW_TEL) -> pd.DataFrame:
    """Load the Tel Aviv table and return a cleaned analysis frame.

    Returned columns:
        time, event, age, sex, slope, delay, GSN, IGFBP2, MEGF10
    Continuous protein columns are z-scored across the cohort.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"TEL raw data not found at {path}. "
            "Set ALS_RAW_TEL to point at Tel_aviv_full_data.csv."
        )
    raw = pd.read_csv(path)
    missing = [v for v in RAW_COLS.values() if v not in raw.columns] \
              + [p for p in PROTEINS if p not in raw.columns]
    if missing:
        raise KeyError(f"Required columns missing from TEL CSV: {missing}")

    df = pd.DataFrame({
        "time":  raw[RAW_COLS["time"]],
        "event": raw[RAW_COLS["event"]].astype(int),
        "age":   raw[RAW_COLS["age"]],
        "sex":   raw[RAW_COLS["sex"]].astype(int),
        "slope": raw[RAW_COLS["slope"]],
        "delay": raw[RAW_COLS["delay"]].abs(),
    })
    for p in PROTEINS:
        df[p] = raw[p]
    df = df.dropna().reset_index(drop=True)
    df = df[df["time"] > 0].reset_index(drop=True)
    # NOTE: protein columns are left at their raw scale; standardisation is
    # performed per fold inside `_fit_fold` (training-set mean / SD applied
    # to both training and held-out folds) to avoid information leakage.
    return df


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

    The model attribute `baseline_estimation_method = "spline"` is set
    OUTSIDE this constructor (in `_make_model`) to mirror the original
    construction order — setting it after `super().__init__` finishes lets
    lifelines initialise its default state first.
    """

    def __init__(self, scale: str = "odds",
                 n_baseline_knots: int = N_BASELINE_KNOTS, **kw):
        super().__init__(**kw)
        self.scale            = scale
        self.n_baseline_knots = n_baseline_knots

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
    m = SplineCox(scale="odds", n_baseline_knots=N_BASELINE_KNOTS,
                  penalizer=PENALIZER)
    m.baseline_estimation_method = "spline"
    return m


def _spline_linear_predictor(model: SplineCox, X_z: pd.DataFrame) -> np.ndarray:
    """Return the model's linear predictor on (already z-scored) covariates."""
    beta = model._model.params_["beta_"].drop("Intercept", errors="ignore")
    cols = [c for c in beta.index if c in X_z.columns]
    return X_z[cols].values @ beta[cols].values


# ============================================================================
# Fold-level helpers
# ============================================================================
def feature_columns(config: str) -> List[str]:
    feats = list(CLINICAL_COLS)
    if config == "clinical":
        return feats
    for token in config.split("+"):
        if token not in PROTEINS:
            raise ValueError(f"Unknown token '{token}' in config '{config}'.")
        feats.append(token)
    return feats


def _fit_fold(df_train: pd.DataFrame, df_test: pd.DataFrame,
              feats: Iterable[str]) -> np.ndarray:
    """Fit one CV fold and return the test-set linear predictors (or NaN)."""
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
        return _spline_linear_predictor(m, Xte_z)
    except Exception:
        return np.full(len(df_test), np.nan)


def _auc_at(time_all: np.ndarray, event_all: np.ndarray,
            time_te: np.ndarray, event_te: np.ndarray,
            lp_te: np.ndarray, horizon: float) -> float:
    """Time-dependent AUC at one horizon, computed on (time_te, event_te)."""
    e_bool = event_te.astype(bool)
    if e_bool.sum() < 2:
        return np.nan
    t_events_max = time_te[e_bool].max()
    h = min(horizon, t_events_max - 1e-3)
    if h <= time_te[e_bool].min():
        return np.nan
    surv_tr = Surv.from_arrays(event=event_all.astype(bool), time=time_all)
    surv_te = Surv.from_arrays(event=e_bool, time=time_te)
    try:
        auc, _ = cumulative_dynamic_auc(surv_tr, surv_te, lp_te, [h])
        return float(auc[0])
    except Exception:
        return np.nan


def _safe_mean(x: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=float)
    x = x[~np.isnan(x)]
    return float(np.mean(x)) if len(x) else np.nan


# ============================================================================
# Real-config CV (no shuffling) — one task per fold per configuration
# ============================================================================
def _run_real_fold(df: pd.DataFrame, train_idx: np.ndarray,
                   test_idx: np.ndarray, config: str
                   ) -> Tuple[str, np.ndarray, np.ndarray]:
    d_tr = df.iloc[train_idx].reset_index(drop=True)
    d_te = df.iloc[test_idx].reset_index(drop=True)
    lp_te = _fit_fold(d_tr, d_te, feature_columns(config))
    return config, np.asarray(test_idx), lp_te


# ============================================================================
# Per-protein train-fold shuffle (one task per fold × shuffle index × protein)
# ============================================================================
def _run_shuffle_fold(df: pd.DataFrame, train_idx: np.ndarray,
                      test_idx: np.ndarray, config: str, protein: str,
                      shuffle_idx: int
                      ) -> Tuple[str, str, int, np.ndarray, np.ndarray]:
    d_tr = df.iloc[train_idx].reset_index(drop=True)
    d_te = df.iloc[test_idx].reset_index(drop=True)
    rng = np.random.default_rng(SEED * 1000 + shuffle_idx * 31 + abs(hash(protein)) % 997)
    d_tr[protein] = rng.permutation(d_tr[protein].values)
    lp_te = _fit_fold(d_tr, d_te, feature_columns(config))
    return config, protein, shuffle_idx, np.asarray(test_idx), lp_te


# ============================================================================
# Cross-validation pipeline
# ============================================================================
def run_cv_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full CV pipeline and return per-seed, indep-table and OOF frames."""
    splits = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                random_state=SEED).split(df))
    n_folds_per_seed = N_SPLITS

    # ------------------------------------------------------------------ Real
    print(f"[real] {len(CONFIGS)} configs x {len(splits)} folds = "
          f"{len(CONFIGS) * len(splits)} fits", flush=True)
    t0 = time.time()
    real_results = Parallel(n_jobs=N_JOBS, batch_size=8)(
        delayed(_run_real_fold)(df, tr, te, cfg)
        for (tr, te) in splits for cfg in CONFIGS
    )
    print(f"[real] done in {time.time() - t0:.1f}s", flush=True)

    # Index real_results by (fold_id, config) for fast lookup.
    real_lp: dict = {}
    for fold_id, (tr, te) in enumerate(splits):
        for cfg_idx, cfg in enumerate(CONFIGS):
            entry = real_results[fold_id * len(CONFIGS) + cfg_idx]
            real_lp[(fold_id, cfg)] = entry  # (config, te_idx, lp_te)

    t_all = df["time"].values
    e_all = df["event"].values.astype(int)

    # ---------------------------- per-seed C-index + multi-horizon AUC ----
    per_seed_rows = []
    for seed_id in range(N_REPEATS):
        folds = range(seed_id * n_folds_per_seed, (seed_id + 1) * n_folds_per_seed)
        for cfg in CONFIGS:
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
    # Pool fold predictions across all seeds and average per patient (one
    # OOF prediction per (patient, model)), for clinical and combined.
    def _pool_oof(cfg: str) -> np.ndarray:
        sums = np.zeros(len(df))
        cnts = np.zeros(len(df))
        for fold_id in range(len(splits)):
            _, te_idx, lp_te = real_lp[(fold_id, cfg)]
            valid = ~np.isnan(lp_te)
            sums[te_idx[valid]] += lp_te[valid]
            cnts[te_idx[valid]] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnts > 0, sums / cnts, np.nan)

    combo_cfg = "+".join(COMBO_FOR_OOF)
    oof_df = pd.DataFrame({
        "time_months":   t_all,
        "event":         e_all,
        "clin_risk_oof": _pool_oof("clinical"),
        "comb_risk_oof": _pool_oof(combo_cfg),
    })

    # ---------------------------- Real-row entries for indep table -------
    table_rows = []
    for cfg in CONFIGS:
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
    shuffle_configs = [c for c in CONFIGS if c != "clinical"]
    shuffle_tasks = []
    for cfg in shuffle_configs:
        for protein in [p for p in cfg.split("+") if p in PROTEINS]:
            for shuffle_idx in range(1, N_SHUFFLES + 1):
                for (tr, te) in splits:
                    shuffle_tasks.append((df, tr, te, cfg, protein, shuffle_idx))
    print(f"[shuffles] {len(shuffle_tasks)} tasks", flush=True)
    t0 = time.time()
    shuf_results = Parallel(n_jobs=N_JOBS, batch_size=16)(
        delayed(_run_shuffle_fold)(*a) for a in shuffle_tasks
    )
    print(f"[shuffles] done in {time.time() - t0:.1f}s", flush=True)

    # Group fold-level shuffle predictions by (cfg, protein, shuffle_idx).
    grouped: dict = {}
    for cfg, protein, shuffle_idx, te_idx, lp_te in shuf_results:
        grouped.setdefault((cfg, protein, shuffle_idx), []).append((te_idx, lp_te))

    for (cfg, protein, shuffle_idx), folds in grouped.items():
        cs, aucs = [], {h: [] for h in AUC_HORIZONS}
        # Aggregate across seeds (each seed contributes n_folds_per_seed folds).
        seed_to_folds: dict = {}
        for fold_id, _ in enumerate(splits):
            seed_to_folds.setdefault(fold_id // n_folds_per_seed, []).append(fold_id)
        for seed_id in range(N_REPEATS):
            lp_oof = np.full(len(df), np.nan)
            for fold_id in seed_to_folds[seed_id]:
                # Match fold-level prediction by te_idx (fold order preserved).
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
def _fit_continuous_cox(df: pd.DataFrame, covs: List[str]) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=PENALIZER)
    cph.fit(df[["time", "event"] + covs], duration_col="time",
            event_col="event", show_progress=False)
    return cph


def _lr_statistic(df: pd.DataFrame, reduced: List[str],
                  added: List[str]) -> float:
    full   = _fit_continuous_cox(df, reduced + added)
    base   = _fit_continuous_cox(df, reduced)
    return 2.0 * (full.log_likelihood_ - base.log_likelihood_)


def run_lrt_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Continuous Cox LRT + permutation null for every comparison.

    Protein columns are z-scored globally (across the cohort) for the LRT
    analysis, mirroring the practice in the original LRT script. This is
    independent of the per-fold standardisation used inside the CV pipeline.
    """
    df = df.copy()
    for p in PROTEINS:
        df[p] = (df[p] - df[p].mean()) / df[p].std(ddof=1)
    rng_master = np.random.default_rng(SEED)
    rows = []
    for added in LRT_ADDED_SETS:
        added = list(added)
        sub = df[["time", "event"] + CLINICAL_COLS + added].dropna().reset_index(drop=True)
        observed = _lr_statistic(sub, CLINICAL_COLS, added)
        # Joint permutation: shuffle the rows of the added-protein block.
        added_block = sub[added].values.copy()
        n = len(sub)
        nulls = []
        for _ in range(N_LRT_PERM):
            perm = sub.copy()
            perm[added] = added_block[rng_master.permutation(n)]
            try:
                nulls.append(_lr_statistic(perm, CLINICAL_COLS, added))
            except Exception:
                pass
        nulls = np.asarray(nulls)
        p_perm = float((np.sum(nulls >= observed) + 1) / (len(nulls) + 1)) \
                 if len(nulls) else np.nan
        rows.append({
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
    print(f"[load] TEL data: {RAW_TEL}", flush=True)
    df = load_tel()
    print(f"  N = {len(df)}, events = {int(df['event'].sum())}", flush=True)

    print("[cv ] running cross-validation + shuffles ...", flush=True)
    per_seed, indep, oof = run_cv_pipeline(df)

    print("[lrt] running likelihood-ratio tests + permutation null ...", flush=True)
    lrt = run_lrt_analysis(df)

    out_per_seed = os.path.join(DATA_DIR, "tel_per_seed_sampling.csv")
    out_indep    = os.path.join(DATA_DIR, "tel_indep_table_sampling.csv")
    out_oof      = os.path.join(DATA_DIR, "tel_oof_scores_sampling.csv")
    out_lrt      = os.path.join(DATA_DIR, "tel_lrt.csv")
    per_seed.to_csv(out_per_seed, index=False)
    indep.to_csv(out_indep, index=False)
    oof.to_csv(out_oof, index=False)
    lrt.to_csv(out_lrt, index=False)
    print(f"saved: {out_per_seed}")
    print(f"saved: {out_indep}")
    print(f"saved: {out_oof}")
    print(f"saved: {out_lrt}")


if __name__ == "__main__":
    main()
