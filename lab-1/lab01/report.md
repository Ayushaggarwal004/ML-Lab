# Lab 1 - Environment and first system measurements

> Template: fill in the `<...>` parts, delete these hints. No code listings here - code lives in `src/`.

## 1. Goal

Set up a reproducible, pinned Python 3.11.8 environment, train a linear and a tree-ensemble baseline on the Breast Cancer Wisconsin dataset, measure their system cost (training time, single-sample latency, model size, peak memory) and decide which deployment tiers - Cloud, Edge, Mobile, TinyML - each model fits.

## 2. Method

- **Environment:** venv with `requirements.txt`; `src/print_versions.py` output is in `results/versions.txt`. <Mention any deviation: pyarrow 15.0.2 instead of 16.1.0, because mlflow 2.14.1 requires pyarrow<16.>
- **Data and models:** `load_breast_cancer(return_X_y=True)`, 70/30 stratified split, `random_state=42`; `LogisticRegression(max_iter=1000)` and `RandomForestClassifier(n_estimators=100)`; seeds fixed to 42.
- **Timing:** training = median of 5 runs after 1 warm-up; single-sample inference = median of 100 repeats after 1 warm-up (`time.perf_counter`).
- **Model size:** `joblib.dump` to `results/model_<name>.joblib`; 1 KB = 1024 bytes.
- **Peak memory:** RSS of the process sampled every ~1 ms with `psutil`, each phase in a fresh process, median of 5. Reported both as total process RSS and as growth over the pre-phase baseline.
- **Machine:** <CPU, cores, RAM, OS - copy from `results/versions.txt`>

## 3. Results

<Paste the three tables from `results/summary.md`: accuracy, system cost, deployment fit.>

<Optional: one plot, e.g. latency or size per model against the budget lines, saved in `results/`.>

### Deployment-fit argument

<For each model, one short paragraph: which tiers it fits and which budget (memory / latency / size) is the deciding one, quoting your measured numbers. Address the difference between total process RSS and model-only growth.>

## 4. Conclusions

1. <Your own words.>
2. <Your own words.>
3. <Your own words.>

<Questions to think about, not answers: Does the extra accuracy of one model justify its cost? Which resource ruled out the most tiers? Which number would change if you deployed without Python/scikit-learn, and what does that say about the TinyML tier?>
