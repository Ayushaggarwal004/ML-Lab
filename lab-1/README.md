# ML Systems labs

One repository, one folder per lab (`lab01/` ... `lab11/`, `project/`).
Each lab folder has `report.md`, `src/` (code) and `results/` (evidence).

## Setup (Python 3.11.8)

macOS / Linux:

    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Windows (PowerShell):

    py -3.11 -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

In VS Code: `Ctrl+Shift+P` -> **Python: Select Interpreter** -> pick `.venv`.

## Lab 1 - run from the repo root, in this order

    python lab01/src/print_versions.py     # -> lab01/results/versions.txt
    python lab01/src/baselines.py          # -> lab01/results/baseline_accuracy.csv
    python lab01/src/measure_costs.py      # -> system_costs.csv, budget_fit.csv, summary.md, model_*.joblib

Then paste the tables from `lab01/results/summary.md` into `lab01/report.md`.
