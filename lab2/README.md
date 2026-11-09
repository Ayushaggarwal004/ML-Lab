# Lab 2 — A Neural Network from Scratch

Implementation of the Lab 2 specification using NumPy and scikit-learn's `load_digits` dataset.

## Requirements
- Python 3.10+
- NumPy
- Matplotlib
- scikit-learn

## Setup (VS Code terminal)

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python main.py
```

The program checks tensor shapes and softmax row sums, reports the initial loss, performs numerical gradient checking, trains all six combinations of hidden width (`16`, `64`) and learning rate (`0.001`, `0.01`, `0.1`), saves loss curves to `results/loss_curves.png`, writes experiment metrics to `results/metrics.csv`, and times forward/backward passes separately.

## Network
`64 -> W -> ReLU -> W -> ReLU -> 10 -> Softmax`

Weights are sampled from `N(0, 0.01)` and biases start at zero. Inputs are divided by 16. The split is stratified with `test_size=0.30` and `random_state=42`.

## GitHub
```bash
git init
git add .
git commit -m "Implement neural network from scratch for Lab 2"
```

Do not commit `.venv/` or Python cache files; `.gitignore` excludes them.
