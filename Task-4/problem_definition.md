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
