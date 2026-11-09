# Lab 5 - Frameworks and automatic differentiation

## Run in VS Code (one command)
1. File -> Open Folder -> select this `lab5` folder.
2. Terminal (Ctrl+`):
```bash
pip install -r requirements.txt
python lab5_autograd.py
```
Takes about 2-5 minutes (the pure-Python engine trains 5 times, torch.compile may compile).

## Output (folder `results/`)
- `report.md`  <- main file: Task 4 table, Task 5 table, Task 6 tables, short note on when compilation helps
- `graph_inspection.txt`  <- PyTorch autograd graph, tensor shape / dtype / contiguity / strides, traced graph
- `results.json`, `run.log`  <- raw numbers and log

## What a correct run shows
- Task 4: max abs difference vs PyTorch and vs numerical gradients < 1e-6 (PASS)
- Task 5: engine and PyTorch loss histories agree (tiny difference), PyTorch much faster
- Task 6: eager vs jit.trace (and torch.compile if your machine has a C++ compiler) timings, non-contiguous slow-downs

## Push to GitHub
git init; git add .; git commit -m "Lab 5"; git branch -M main;
git remote add origin <repo-url>; git push -u origin main
