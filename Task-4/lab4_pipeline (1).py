#!/usr/bin/env python
"""Lab 4 - Data pipeline and quality (UCI Adult).

ONE COMMAND:   python lab4_pipeline.py   (only needs: pip install numpy pandas scikit-learn joblib pyyaml)

Stages: ingest -> validate -> split -> clean (train-fit only) -> derived features
        -> baseline comparison -> label-quality audit -> hash + config + data card
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin

log = logging.getLogger("lab4")


# ================= CONFIG (edit here; a copy is saved to results/config.yaml) =================
CONFIG_YAML = '''
# Lab 4 configuration - every tunable value lives here (pipeline is deterministic).
seed: 42
results_dir: results

openml:
  name: adult
  version: 2

expected:            # task 2: logged and verified at ingestion
  rows: 48842
  cols: 15

split:
  test_size: 0.30    # 70/30, stratified on income, done BEFORE cleaning

cleaning:
  lower_quantile: 0.01
  upper_quantile: 0.99

model:
  max_iter: 1000

label_audit:
  n_sample: 50

schema:
  missing_token: "?"
  target: income
  types:
    age: int
    workclass: category
    fnlwgt: int
    education: category
    education-num: int
    marital-status: category
    occupation: category
    relationship: category
    race: category
    sex: category
    capital-gain: int
    capital-loss: int
    hours-per-week: int
    native-country: category
    income: category
  ranges:
    age: [17, 90]
    hours-per-week: [1, 99]
    capital-gain: [0, 99999]
    capital-loss: [0, 4356]
  allowed_categories:      # '?' is always allowed in addition to these
    sex: [Male, Female]
    race: [White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other]
    income: ["<=50K", ">50K"]
'''

PROBLEM_DEFINITION = '''
# Problem definition (Task 1)

**Goal.** Build a reproducible, leakage-free data pipeline for the UCI Adult census data and verify that
it produces a trustworthy table for a supervised learning task.

**Dataset.** UCI Adult, loaded with `fetch_openml(name='adult', version=2, as_frame=True)`:
48,842 rows and 15 columns (14 features plus the target).

**Inputs.** The 14 features: age, workclass, fnlwgt, education, education-num, marital-status, occupation,
relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country. They are a mix of
6 numeric and 8 categorical variables, and three categorical columns contain missing values (`?`).

**Output.** Binary label `income`: `>50K` (positive class, about 24% of rows) versus `<=50K`.

**Success metrics.** F1 of the positive class (`>50K`) and accuracy on a held-out, untouched test set.
F1 is reported next to accuracy because the classes are imbalanced, so accuracy alone can look good
while the minority class is handled poorly. The model is only a LogisticRegression baseline used to
check that the pipeline and derived features behave sensibly; the lab is graded on data quality, not on model score.

**Constraints.**
1. *No leakage:* the 70/30 stratified split happens before any cleaning; imputation values, clipping
   percentiles and duplicate removal use the training split only; the test set is never altered.
2. *Reproducibility:* fixed seed, all parameters in `config.yaml`, fitted transformer saved with joblib,
   SHA256 of the processed dataset stored with the configuration, whole pipeline runs with one command.
3. *Transparency:* validation reports violations rather than silently dropping rows.

**Risks.** Missing values coded as `?`, exact duplicates, conflicting labels for identical feature rows,
outliers in capital-gain/loss, and demographic bias in 1994 census data (documented in `data_card.md`).
'''


# ================= COMPONENTS =================
MISSING = "?"
DERIVED_FEATURES = ["capital_net", "hours_per_age"]

# Data dictionary: education <-> education-num (used for the label-quality audit)
EDUCATION_NUM = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
    "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10,
    "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14,
    "Prof-school": 15, "Doctorate": 16,
}


class CleaningTransformer(BaseEstimator, TransformerMixin):
    """Imputation + percentile clipping. Every statistic is learned in fit()
    (call fit on TRAIN only); transform() merely applies stored values.

    * '?' (and NaN) -> missing
    * categorical: most frequent value (mode) of train
    * numeric: median of train
    * numeric: clipped to the train 1st / 99th percentile
    """

    def __init__(self, numeric_cols, categorical_cols, lower_q=0.01, upper_q=0.99):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.lower_q = lower_q
        self.upper_q = upper_q

    def _normalise(self, X):
        X = X.copy()
        for c in self.categorical_cols:
            col = X[c].astype(object)
            X[c] = col.where(col != MISSING, np.nan)
        for c in self.numeric_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        return X

    def _impute(self, X):
        for c in self.numeric_cols:
            X[c] = X[c].fillna(self.medians_[c])
        for c in self.categorical_cols:
            X[c] = X[c].fillna(self.modes_[c])
        return X

    def fit(self, X, y=None):
        X = self._normalise(X)
        self.medians_ = {c: float(X[c].median()) for c in self.numeric_cols}
        self.modes_ = {c: X[c].mode(dropna=True).iloc[0] for c in self.categorical_cols}
        X = self._impute(X)
        self.lower_ = {c: float(X[c].quantile(self.lower_q)) for c in self.numeric_cols}
        self.upper_ = {c: float(X[c].quantile(self.upper_q)) for c in self.numeric_cols}
        self.n_fit_rows_ = int(len(X))
        return self

    def transform(self, X):
        X = self._impute(self._normalise(X))
        for c in self.numeric_cols:
            X[c] = X[c].clip(lower=self.lower_[c], upper=self.upper_[c])
        return X


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise derived features (no statistics learned -> no leakage possible)."""
    out = df.copy()
    out["capital_net"] = out["capital-gain"] - out["capital-loss"]
    denom = (out["age"] - 16).astype(float)
    out["hours_per_age"] = out["hours-per-week"] / denom.where(denom > 0)  # age<=16 -> NaN
    return out



# ----------------------------------------------------------------- helpers
def setup_logging(results_dir: Path) -> None:
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(results_dir / "pipeline.log", mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)


def fingerprint(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame (used to prove ingestion/validation never modify raw data)."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ------------------------------------------------- Task 2: ingestion
def ingest(cfg: dict) -> pd.DataFrame:
    """Read raw data. No cleaning, no renaming, no dtype changes."""
    ds = fetch_openml(name=cfg["openml"]["name"], version=cfg["openml"]["version"], as_frame=True)
    raw = ds.frame
    exp = cfg["expected"]
    log.info("INGEST  row count = %d (expected %d) | column count = %d (expected %d)",
             raw.shape[0], exp["rows"], raw.shape[1], exp["cols"])
    return raw


def to_typed(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Typed COPY of the raw data (raw itself stays untouched):
    class -> income, int columns -> int64, categoricals -> str with NaN -> '?'."""
    sch = cfg["schema"]
    df = raw.rename(columns={"class": sch["target"]}).copy()
    for col, t in sch["types"].items():
        if t == "int":
            df[col] = df[col].astype("int64")
        else:
            s = df[col].astype(object)
            df[col] = s.where(df[col].notna(), MISSING).astype(str).str.strip()
    return df[list(sch["types"])]


# ------------------------------------------------- Task 3: validation
def impossible_value_flags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Row-level rule checks against the data dictionary (True = violation)."""
    sch = cfg["schema"]
    f = {}
    for col, (lo, hi) in sch["ranges"].items():
        f[f"{col}_out_of_range"] = ~df[col].between(lo, hi)
    f["fnlwgt_not_positive"] = df["fnlwgt"] <= 0
    f["education_vs_education_num_mismatch"] = df["education"].map(EDUCATION_NUM) != df["education-num"]
    f["husband_but_not_male"] = (df["relationship"] == "Husband") & (df["sex"] != "Male")
    f["wife_but_not_female"] = (df["relationship"] == "Wife") & (df["sex"] != "Female")
    for col, allowed in sch["allowed_categories"].items():
        f[f"{col}_invalid_category"] = ~df[col].isin(list(allowed) + [MISSING])
    return pd.DataFrame(f, index=df.index)


def validate(raw: pd.DataFrame, cfg: dict, out_dir: Path) -> dict:
    """Schema / range / missing / duplicate checks. Reports violations, never drops rows."""
    sch = cfg["schema"]
    exp = cfg["expected"]
    view = raw.rename(columns={"class": sch["target"]})  # rename only, data untouched
    checks = []

    def add(name, st, n, detail=""):
        checks.append({"check": name, "status": st, "violations": int(n), "detail": detail})
        log.info("VALIDATE %-34s %-4s violations=%-6d %s", name, st, n, detail)

    add("row_count", status(len(view) == exp["rows"]), int(len(view) != exp["rows"]),
        f"{len(view)} rows (expected {exp['rows']})")
    add("column_count", status(view.shape[1] == exp["cols"]), int(view.shape[1] != exp["cols"]),
        f"{view.shape[1]} columns (expected {exp['cols']})")
    absent = [c for c in sch["types"] if c not in view.columns]
    add("expected_columns_present", status(not absent), len(absent), f"missing: {absent}" if absent else "all 15 present")

    # types
    for col, t in sch["types"].items():
        if col in absent:
            continue
        s = view[col]
        if t == "int":
            ok = (pd.api.types.is_numeric_dtype(s) and bool(s.notna().all())
                  and bool((s % 1 == 0).all()))
            detail = f"raw dtype={s.dtype}, integer-valued & no nulls required"
        else:
            ok = isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_string_dtype(s)
            detail = f"raw dtype={s.dtype}"
        add(f"type[{col}]={t}", status(ok), 0 if ok else int((~ok) * len(s)), detail)

    # ranges
    for col, (lo, hi) in sch["ranges"].items():
        bad = ~view[col].between(lo, hi)
        add(f"range[{col}] {lo}-{hi}", status(not bad.any()), bad.sum(),
            f"observed min={view[col].min()}, max={view[col].max()}")

    # allowed categories ('?' always allowed)
    for col, allowed in sch["allowed_categories"].items():
        vals = view[col].dropna().astype(str)
        bad = ~vals.isin(list(allowed) + [MISSING])
        add(f"categories[{col}]", status(not bad.any()), bad.sum(),
            f"observed: {sorted(vals.unique().tolist())}")

    # missing values: '?' or NaN both mean missing (OpenML v2 stores them as NaN)
    miss = {}
    for col, t in sch["types"].items():
        if t == "category":
            s = view[col]
            n = int(s.isna().sum() + (s.astype(object) == MISSING).sum())
            if n:
                miss[col] = {"n_missing": n, "pct": round(100 * n / len(view), 2)}
    rows_any = int((view[[c for c in miss]].isna().any(axis=1)
                    | (view[[c for c in miss]].astype(object) == MISSING).any(axis=1)).sum()) if miss else 0
    add("missing_values('?' / NaN)", "INFO", 0,
        f"{sum(v['n_missing'] for v in miss.values())} cells in {list(miss)}; {rows_any} rows affected (allowed, imputed later)")

    # numeric nulls
    num_null = {c: int(view[c].isna().sum()) for c, t in sch["types"].items() if t == "int"}
    add("numeric_nulls", status(sum(num_null.values()) == 0), sum(num_null.values()), str(num_null))

    # exact duplicates (info only - handled in the cleaning step, train only)
    dup = int(view.duplicated().sum())
    add("exact_duplicate_rows", "INFO", 0, f"{dup} rows duplicate an earlier row (dropped from TRAIN only later)")

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    report = {"n_rows": int(len(view)), "n_cols": int(view.shape[1]),
              "n_failed_checks": n_fail, "checks": checks, "missing_by_column": miss,
              "note": "No rows were dropped or modified during validation."}
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = ["# Validation report", "",
          f"Rows: **{report['n_rows']}** | Columns: **{report['n_cols']}** | Failed checks: **{n_fail}**", "",
          "| Check | Status | Violations | Detail |", "|---|---|---|---|"]
    md += [f"| {c['check']} | {c['status']} | {c['violations']} | {c['detail']} |" for c in checks]
    md += ["", "No rows were dropped or modified during validation.", ""]
    (out_dir / "validation_report.md").write_text("\n".join(md), encoding="utf-8")
    return report


# ------------------------------------------------- Task 4/5: model helper
def make_model(num_cols, cat_cols, cfg) -> Pipeline:
    enc = ColumnTransformer([("num", StandardScaler(), num_cols),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    return Pipeline([("encode", enc),
                     ("clf", LogisticRegression(max_iter=cfg["model"]["max_iter"], random_state=cfg["seed"]))])


def evaluate_baseline(train_c, test_c, y_tr, y_te, num_cols, cat_cols, cfg) -> dict:
    res = {}
    for name, derived in (("without_derived_features", False), ("with_derived_features", True)):
        Xtr = add_derived_features(train_c) if derived else train_c
        Xte = add_derived_features(test_c) if derived else test_c
        ncols = num_cols + (DERIVED_FEATURES if derived else [])
        feats = ncols + cat_cols
        model = make_model(ncols, cat_cols, cfg).fit(Xtr[feats], y_tr)
        pred = model.predict(Xte[feats])
        res[name] = {"n_features": len(feats),
                     "accuracy": round(float(accuracy_score(y_te, pred)), 4),
                     "f1_>50K": round(float(f1_score(y_te, pred, pos_label=1)), 4)}
        log.info("BASELINE %-26s accuracy=%.4f  F1=%.4f", name, res[name]["accuracy"], res[name]["f1_>50K"])
    return res


# ------------------------------------------------- Task 6: label quality
def label_audit(typed: pd.DataFrame, cfg: dict, out_dir: Path) -> dict:
    target = cfg["schema"]["target"]
    feats = [c for c in typed.columns if c != target]
    n = cfg["label_audit"]["n_sample"]

    # (a) duplicate-conflict audit: identical features, different income label
    g = typed.groupby(feats, dropna=False)[target].agg(["nunique", "size"])
    conflict = g[g["nunique"] > 1]
    n_groups, n_rows = int(len(conflict)), int(conflict["size"].sum())
    log.info("LABEL-AUDIT duplicate-conflict feature groups = %d (%d rows involved)", n_groups, n_rows)

    # (b) 50 random rows re-checked against the data dictionary
    sample = typed.sample(n=n, random_state=cfg["seed"])
    flags = impossible_value_flags(sample, cfg)
    sample_out = sample.copy()
    sample_out["automated_violations"] = flags.apply(lambda r: ";".join(r.index[r]), axis=1)
    sample_out["manual_check_ok"] = ""   # <- fill in by hand after eyeballing the row
    sample_out.to_csv(out_dir / "label_audit_sample_50.csv", index_label="orig_index")
    n_bad_sample = int(flags.any(axis=1).sum())
    log.info("LABEL-AUDIT %d sampled rows: %d with impossible values", n, n_bad_sample)

    # (c) same rules on the whole dataset
    all_flags = impossible_value_flags(typed, cfg)
    per_rule = {k: int(v) for k, v in all_flags.sum().items() if v}
    log.info("LABEL-AUDIT full-dataset impossible-value rows=%d %s", int(all_flags.any(axis=1).sum()), per_rule)

    audit = {"duplicate_conflict_groups": n_groups, "duplicate_conflict_rows": n_rows,
             "sample_size": n, "sample_rows_with_impossible_values": n_bad_sample,
             "full_dataset_rows_with_impossible_values": int(all_flags.any(axis=1).sum()),
             "full_dataset_violations_per_rule": per_rule}
    (out_dir / "label_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


# ------------------------------------------------- Task 7: data card
def write_data_card(path: Path, typed, proc, sha, audit, report, dups_removed, cfg):
    target = cfg["schema"]["target"]
    pos = (typed[target] == ">50K")
    share = lambda s: ", ".join(f"{k} {100*v:.1f}%" for k, v in s.items())
    sex_share = typed["sex"].value_counts(normalize=True)
    race_share = typed["race"].value_counts(normalize=True)
    pos_sex = pos.groupby(typed["sex"]).mean()
    pos_race = pos.groupby(typed["race"]).mean()
    miss = report["missing_by_column"]
    text = f"""# Data Card - UCI Adult (OpenML `adult`, version {cfg['openml']['version']})

## Source
- Census-Income ("Adult") data extracted by Barry Becker from the 1994 U.S. Census Bureau database;
  donated to the UCI Machine Learning Repository (Becker & Kohavi, 1996).
- Loaded with `fetch_openml(name='adult', version=2, as_frame=True)` (OpenML dataset 1590).
- Task: predict whether annual income is `>50K` from 14 demographic / employment features.

## License
UCI lists the Adult dataset under **CC BY 4.0** (attribution required). Re-check the licence on the
UCI / OpenML dataset page before redistributing the data.

## Size
- Raw: {report['n_rows']} rows x {report['n_cols']} columns (14 features + `income`).
- Class balance: {100*pos.mean():.1f}% `>50K` vs {100*(1-pos.mean()):.1f}% `<=50K`.
- Processed dataset (`results/processed_dataset.csv`): {len(proc)} rows
  ({int((proc['split']=='train').sum())} train after removing {dups_removed} exact duplicates, {int((proc['split']=='test').sum())} test untouched),
  columns = 14 cleaned features + 2 derived (`capital_net`, `hours_per_age`) + `income` + `split`.
- **SHA256 of processed dataset:** `{sha}`

## Known biases and quality issues
- **Age of data:** collected in 1994; the USD 50K threshold is not inflation-adjusted and occupations/wages have changed.
- **Representation:** sex = {share(sex_share)}; race = {share(race_share)}. Minority groups are small, so metrics for them are noisy.
- **Outcome disparities in the labels:** share with `>50K` by sex: {share(pos_sex)}; by race: {share(pos_race)}.
  A model trained on these labels will reproduce these historical disparities.
- **Missing data:** `?`/NaN in {', '.join(f'{k} ({v["pct"]}%)' for k, v in miss.items())}; missingness is probably not random (e.g. occupation/workclass).
  Mode imputation was used as required by the lab and may distort these groups.
- **Label noise:** {audit['duplicate_conflict_groups']} groups of identical feature rows ({audit['duplicate_conflict_rows']} rows) carry conflicting income labels
  (any such group puts a ceiling on achievable accuracy). {audit['full_dataset_rows_with_impossible_values']} rows violate dictionary rules (see `label_audit.json`).
- **Sampling:** `fnlwgt` is a census sampling weight, not a personal attribute; it is kept as a feature but is of doubtful predictive meaning.

## Privacy concerns
- Public, anonymised census extract with no names or addresses, but it contains quasi-identifiers
  (age, sex, race, marital status, native country, occupation) that could enable re-identification when linked with other datasets.
- Contains sensitive attributes (race, sex, native country); use them only for fairness auditing, not for individual decisions.

## Processing summary (see `run_pipeline.py`, `config.yaml`)
1. Ingest raw data unmodified -> 2. validate schema/ranges/missing/duplicates (no rows dropped) ->
3. stratified 70/30 split (seed {cfg['seed']}) **before** any cleaning ->
4. drop exact duplicates from train only; fit imputer (mode/median) and 1st/99th-percentile clipping on train only; apply to test ->
5. derived features -> 6. LogisticRegression baseline with/without derived features -> 7. hash + config.

## Intended use / limitations
Educational use for pipeline-quality exercises. Not suitable for real-world income, credit or hiring decisions.
"""
    path.write_text(text, encoding="utf-8")


# ------------------------------------------------- main
def main() -> None:
    
    script = Path(__file__).resolve()
    base = script.parent / "lab4_submission"      # everything is created in this one folder
    base.mkdir(parents=True, exist_ok=True)
    if script.parent != base:
        shutil.copy2(script, base / script.name)   # copy of the script goes into the folder too
    cfg = yaml.safe_load(CONFIG_YAML)
    out = base / cfg["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    (out / "config.yaml").write_text(CONFIG_YAML.strip() + "\n", encoding="utf-8")
    (base / "problem_definition.md").write_text(PROBLEM_DEFINITION.strip() + "\n", encoding="utf-8")
    np.random.seed(cfg["seed"])
    sch = cfg["schema"]
    target = sch["target"]

    # 2. ingestion (raw untouched)
    raw = ingest(cfg)
    fp_raw = fingerprint(raw)

    # 3. validation (report only)
    report = validate(raw, cfg, out)
    if report["n_failed_checks"]:
        log.warning("%d validation check(s) FAILED - see validation_report.md (no rows dropped)", report["n_failed_checks"])
    typed = to_typed(raw, cfg)
    assert fingerprint(raw) == fp_raw, "raw data was modified!"
    log.info("INGEST/VALIDATE raw data fingerprint unchanged: %s", fp_raw[:16])

    num_cols = [c for c, t in sch["types"].items() if t == "int"]
    cat_cols = [c for c, t in sch["types"].items() if t == "category" and c != target]

    # 4. split BEFORE cleaning (70/30, stratified)
    train, test = train_test_split(typed, test_size=cfg["split"]["test_size"],
                                   stratify=typed[target], random_state=cfg["seed"])
    log.info("SPLIT   train=%d test=%d | >50K rate train=%.4f test=%.4f", len(train), len(test),
             (train[target] == ">50K").mean(), (test[target] == ">50K").mean())

    # duplicates: exact duplicates dropped from TRAIN only
    test_dups = int(test.duplicated().sum())
    dups_removed = int(train.duplicated().sum())
    train = train.drop_duplicates()
    log.info("CLEAN   exact duplicates removed from train=%d (test untouched; it holds %d)", dups_removed, test_dups)

    feats = num_cols + cat_cols
    y_tr = (train[target] == ">50K").astype(int)
    y_te = (test[target] == ">50K").astype(int)

    # imputer + clipping fitted on train only
    pre = CleaningTransformer(num_cols, cat_cols,
                              cfg["cleaning"]["lower_quantile"], cfg["cleaning"]["upper_quantile"])
    pre.fit(train[feats])
    train_c, test_c = pre.transform(train[feats]), pre.transform(test[feats])
    joblib.dump(pre, out / "preprocessor.joblib")

    # leakage checks
    assert set(train.index).isdisjoint(test.index), "train/test overlap"
    assert pre.n_fit_rows_ == len(train), "transformer saw rows other than train"
    reloaded = joblib.load(out / "preprocessor.joblib")
    pd.testing.assert_frame_equal(reloaded.transform(test[feats]), test_c)
    log.info("LEAKAGE checks passed: disjoint indices, fit rows == train rows (%d), joblib round-trip identical", len(train))
    log.info("CLEAN   train medians=%s", {k: round(v, 1) for k, v in pre.medians_.items()})
    log.info("CLEAN   train modes=%s", pre.modes_)
    log.info("CLEAN   clip bounds (1%%,99%%)=%s", {c: (round(pre.lower_[c], 1), round(pre.upper_[c], 1)) for c in num_cols})

    # 5. derived features + baseline with/without
    metrics = evaluate_baseline(train_c, test_c, y_tr, y_te, num_cols, cat_cols, cfg)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # 6. label quality
    audit = label_audit(typed, cfg, out)

    # 7. processed dataset, hash, config, data card
    proc = pd.concat([add_derived_features(train_c).assign(**{target: train[target].values, "split": "train"}),
                      add_derived_features(test_c).assign(**{target: test[target].values, "split": "test"})])
    proc.index.name = "orig_index"
    proc_path = out / "processed_dataset.csv"
    proc.to_csv(proc_path, float_format="%.8g")
    sha = sha256_file(proc_path)
    (out / "processed_dataset.sha256").write_text(f"{sha}  processed_dataset.csv\n", encoding="utf-8")
    log.info("HASH    SHA256(processed_dataset.csv) = %s", sha)

    run_cfg = {"config": cfg,
               "runtime": {"python": platform.python_version(), "pandas": pd.__version__,
                           "numpy": np.__version__, "scikit-learn": sklearn.__version__},
               "outputs": {"raw_rows": int(len(raw)), "raw_cols": int(raw.shape[1]),
                           "train_rows_after_dedup": int(len(train)), "test_rows": int(len(test)),
                           "duplicates_removed_from_train": dups_removed,
                           "processed_dataset": "processed_dataset.csv",
                           "processed_dataset_sha256": sha,
                           "metrics": metrics}}
    (out / "config_used.yaml").write_text(yaml.safe_dump(run_cfg, sort_keys=False), encoding="utf-8")

    write_data_card(base / "data_card.md", typed, proc, sha, audit, report, dups_removed, cfg)
    log.info("DONE    outputs in %s | data_card.md written", out)


if __name__ == "__main__":
    main()
