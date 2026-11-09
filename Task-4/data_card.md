# Data Card - UCI Adult (OpenML `adult`, version 2)

## Source
- Census-Income ("Adult") data extracted by Barry Becker from the 1994 U.S. Census Bureau database;
  donated to the UCI Machine Learning Repository (Becker & Kohavi, 1996).
- Loaded with `fetch_openml(name='adult', version=2, as_frame=True)` (OpenML dataset 1590).
- Task: predict whether annual income is `>50K` from 14 demographic / employment features.

## License
UCI lists the Adult dataset under **CC BY 4.0** (attribution required). Re-check the licence on the
UCI / OpenML dataset page before redistributing the data.

## Size
- Raw: 48842 rows x 15 columns (14 features + `income`).
- Class balance: 23.9% `>50K` vs 76.1% `<=50K`.
- Processed dataset (`results/processed_dataset.csv`): 48818 rows
  (34165 train after removing 24 exact duplicates, 14653 test untouched),
  columns = 14 cleaned features + 2 derived (`capital_net`, `hours_per_age`) + `income` + `split`.
- **SHA256 of processed dataset:** `58db587862b6004ea7b9c585258d84c3e528219ebd02ada2fcea9fcbe88feaa4`

## Known biases and quality issues
- **Age of data:** collected in 1994; the USD 50K threshold is not inflation-adjusted and occupations/wages have changed.
- **Representation:** sex = Male 66.8%, Female 33.2%; race = White 85.5%, Black 9.6%, Asian-Pac-Islander 3.1%, Amer-Indian-Eskimo 1.0%, Other 0.8%. Minority groups are small, so metrics for them are noisy.
- **Outcome disparities in the labels:** share with `>50K` by sex: Female 10.9%, Male 30.4%; by race: Amer-Indian-Eskimo 11.7%, Asian-Pac-Islander 26.9%, Black 12.1%, Other 12.3%, White 25.4%.
  A model trained on these labels will reproduce these historical disparities.
- **Missing data:** `?`/NaN in workclass (5.73%), occupation (5.75%), native-country (1.75%); missingness is probably not random (e.g. occupation/workclass).
  Mode imputation was used as required by the lab and may distort these groups.
- **Label noise:** 5 groups of identical feature rows (10 rows) carry conflicting income labels
  (any such group puts a ceiling on achievable accuracy). 4 rows violate dictionary rules (see `label_audit.json`).
- **Sampling:** `fnlwgt` is a census sampling weight, not a personal attribute; it is kept as a feature but is of doubtful predictive meaning.

## Privacy concerns
- Public, anonymised census extract with no names or addresses, but it contains quasi-identifiers
  (age, sex, race, marital status, native country, occupation) that could enable re-identification when linked with other datasets.
- Contains sensitive attributes (race, sex, native country); use them only for fairness auditing, not for individual decisions.

## Processing summary (see `run_pipeline.py`, `config.yaml`)
1. Ingest raw data unmodified -> 2. validate schema/ranges/missing/duplicates (no rows dropped) ->
3. stratified 70/30 split (seed 42) **before** any cleaning ->
4. drop exact duplicates from train only; fit imputer (mode/median) and 1st/99th-percentile clipping on train only; apply to test ->
5. derived features -> 6. LogisticRegression baseline with/without derived features -> 7. hash + config.

## Intended use / limitations
Educational use for pipeline-quality exercises. Not suitable for real-world income, credit or hiring decisions.
