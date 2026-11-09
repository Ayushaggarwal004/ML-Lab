#!/usr/bin/env python
"""Lab 5 - Frameworks and automatic differentiation.

ONE COMMAND:   python lab5_autograd.py
(needs: pip install numpy scikit-learn torch)

Tasks 1-3  scalar autograd engine (class Value: graph, local derivative rules, topological backward)
Task  4    check vs PyTorch autograd and numerical gradients on ((a*b + c)**2).relu().sum()
Task  5    train 2 -> 8 -> 1 (ReLU hidden, sigmoid output, BCE) on make_moons, own engine vs PyTorch
Task  6    inspect the PyTorch graph; eager vs torch.compile / torch.jit.trace; non-contiguous views

Everything is written to ./results (report.md is the file to read / hand in).
"""
from __future__ import annotations

import json
import logging
import math
import platform
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

warnings.filterwarnings("ignore")
log = logging.getLogger("lab5")

# ------------------------------------------------------------------ settings
SEED = 0
N_SAMPLES, NOISE = 200, 0.2          # make_moons
HIDDEN, LR, EPOCHS = 8, 0.5, 100     # 2 -> 8 -> 1, full-batch gradient descent
N_RUNS = 5                           # median of 5 runs (task 5)
TOL = 1e-6                           # task 4 tolerance


# =========================================================== TASKS 1-3: ENGINE
def _noop() -> None:
    pass


class Value:
    """Scalar node of the computation graph.

    data      : forward value
    grad      : d(output)/d(this node), accumulated by backward()
    _prev     : parent nodes (the inputs this node was computed from)
    _backward : closure = the LOCAL DERIVATIVE RULE of the operation that created this node
    _op       : operation name (for inspection)
    """
    __slots__ = ("data", "grad", "_prev", "_backward", "_op")

    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._backward = _noop
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op='{self._op}')"

    # ---- task 2: add, mul, pow, relu (each builds the graph during the forward pass)
    def __add__(self, other):
        if isinstance(other, Value):
            out = Value(self.data + other.data, (self, other), "+")

            def _backward():                      # d(a+b)/da = 1, d/db = 1
                self.grad += out.grad             # += : accumulate, never overwrite
                other.grad += out.grad
        else:
            out = Value(self.data + other, (self,), "+c")

            def _backward():
                self.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        if isinstance(other, Value):
            out = Value(self.data * other.data, (self, other), "*")

            def _backward():                      # d(a*b)/da = b, d/db = a
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
        else:
            out = Value(self.data * other, (self,), "*c")

            def _backward():
                self.grad += other * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents are supported"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():                          # d(a^k)/da = k * a^(k-1)
            self.grad += k * (self.data ** (k - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():                          # 1 if output > 0 else 0
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # ---- extra ops needed for the moons model (sigmoid + BCE) and for convenience
    def log(self, eps=1e-12):
        x = max(self.data, eps)
        out = Value(math.log(x), (self,), "log")

        def _backward():
            self.grad += (1.0 / x) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        s = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
        out = Value(s, (self,), "sigmoid")

        def _backward():                          # s * (1 - s)
            self.grad += s * (1.0 - s) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (1.0 / other)

    # ---- task 3: backward pass
    def backward(self):
        # 1) topological sort (iterative post-order DFS, parents before children)
        topo, visited, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._prev:
                if id(p) not in visited:
                    stack.append((p, False))
        # 2) output gradient = 1
        self.grad = 1.0
        # 3) visit nodes in reverse topological order; each local rule ACCUMULATES (+=)
        for node in reversed(topo):
            node._backward()
        return len(topo)


def vsum(values):
    """n-ary sum node (one node, many parents): each parent receives out.grad."""
    values = list(values)
    out = Value(sum(v.data for v in values), values, "sum")

    def _backward():
        g = out.grad
        for v in values:
            v.grad += g
    out._backward = _backward
    return out


# ============================================================ TASK 4: VERIFY
def task4_verify() -> dict:
    rng = np.random.default_rng(SEED)
    shape = (3, 4)
    a, b, c = (rng.uniform(-2, 2, shape) for _ in range(3))

    def f_np(a, b, c, pre_square_relu=False):
        z = a * b + c
        return (np.maximum(z, 0) ** 2).sum() if pre_square_relu else (np.maximum(z ** 2, 0)).sum()

    def numeric_grads(pre):
        eps, out = 1e-6, []
        for arr_i in range(3):
            g = np.zeros(shape)
            for idx in np.ndindex(shape):
                args_p = [a.copy(), b.copy(), c.copy()]
                args_m = [a.copy(), b.copy(), c.copy()]
                args_p[arr_i][idx] += eps
                args_m[arr_i][idx] -= eps
                g[idx] = (f_np(*args_p, pre) - f_np(*args_m, pre)) / (2 * eps)
            out.append(g)
        return out

    def run(pre):
        # own engine
        va = [[Value(x) for x in row] for row in a]
        vb = [[Value(x) for x in row] for row in b]
        vc = [[Value(x) for x in row] for row in c]
        terms = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                z = va[i][j] * vb[i][j] + vc[i][j]
                terms.append((z.relu() ** 2) if pre else ((z ** 2).relu()))
        y = vsum(terms)
        y.backward()
        mine = [np.array([[v.grad for v in row] for row in m]) for m in (va, vb, vc)]
        # PyTorch autograd (float64)
        ta, tb, tc = (torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (a, b, c))
        z = ta * tb + tc
        ty = ((z.relu() ** 2) if pre else (z ** 2).relu()).sum()
        ty.backward()
        tgrads = [t.grad.numpy() for t in (ta, tb, tc)]
        num = numeric_grads(pre)
        d_torch = max(float(np.abs(m - t).max()) for m, t in zip(mine, tgrads))
        d_num = max(float(np.abs(m - n).max()) for m, n in zip(mine, num))
        return {"forward_value_engine": y.data, "forward_value_torch": float(ty),
                "max_abs_diff_vs_pytorch": d_torch, "max_abs_diff_vs_numerical": d_num,
                "pass": bool(d_torch < TOL and d_num < TOL)}

    # gradient accumulation sanity check: y = a*a + a  ->  dy/da = 2a + 1
    x = Value(3.0)
    (x * x + x).backward()
    acc_ok = abs(x.grad - 7.0) < 1e-12

    res = {"task_expression": run(pre=False),            # ((a*b + c)**2).relu().sum()
           "extra_relu_branch_check": run(pre=True),     # ((a*b + c).relu()**2).sum(), negatives exercise relu
           "accumulation_check_x*x+x_at_3": {"grad": x.grad, "expected": 7.0, "pass": bool(acc_ok)}}
    for k, v in res.items():
        log.info("TASK4 %-32s %s", k, v)
    return res


# ===================================================== TASK 5: TRAIN vs PYTORCH
def make_init(rng) -> dict:
    k1, k2 = 1 / math.sqrt(2), 1 / math.sqrt(HIDDEN)      # same ranges as nn.Linear defaults
    return {"W1": rng.uniform(-k1, k1, (HIDDEN, 2)), "b1": np.full(HIDDEN, 0.1),
            "W2": rng.uniform(-k2, k2, (1, HIDDEN)), "b2": np.zeros(1)}


def np_forward(P, X):
    h = np.maximum(X @ P["W1"].T + P["b1"], 0)
    z = (h @ P["W2"].T + P["b2"]).ravel()
    return 1 / (1 + np.exp(-z))


def metrics(P, X, y):
    p = np.clip(np_forward(P, X), 1e-12, 1 - 1e-12)
    loss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    return loss, float(((p > 0.5) == y).mean())


def train_engine(X, y, P0):
    W1 = [[Value(w) for w in row] for row in P0["W1"]]
    b1 = [Value(b) for b in P0["b1"]]
    W2 = [Value(w) for w in P0["W2"][0]]
    b2 = Value(P0["b2"][0])
    params = [w for row in W1 for w in row] + b1 + W2 + [b2]
    Xl, yl = X.tolist(), y.tolist()
    hist, n_nodes = [], 0
    for _ in range(EPOCHS):
        for p in params:
            p.grad = 0.0                                   # zero parameter grads each step
        losses = []
        for (x0, x1), yi in zip(Xl, yl):
            hs = [(W1[j][0] * x0 + W1[j][1] * x1 + b1[j]).relu() for j in range(HIDDEN)]
            p = (vsum(h * w for h, w in zip(hs, W2)) + b2).sigmoid()
            losses.append(-(p.log()) if yi == 1 else -((1 - p).log()))   # BCE for one sample
        loss = vsum(losses) * (1.0 / len(losses))
        n_nodes = loss.backward()
        for p in params:
            p.data -= LR * p.grad
        hist.append(loss.data)
    P = {"W1": np.array([[w.data for w in row] for row in W1]), "b1": np.array([b.data for b in b1]),
         "W2": np.array([[w.data for w in W2]]), "b2": np.array([b2.data])}
    return hist, P, n_nodes


def train_torch(X, y, P0):
    model = nn.Sequential(nn.Linear(2, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1), nn.Sigmoid()).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(P0["W1"])); model[0].bias.copy_(torch.tensor(P0["b1"]))
        model[2].weight.copy_(torch.tensor(P0["W2"])); model[2].bias.copy_(torch.tensor(P0["b2"]))
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    lossf = nn.BCELoss()
    Xt, yt = torch.tensor(X), torch.tensor(y).double().unsqueeze(1)
    hist = []
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = lossf(model(Xt), yt)
        loss.backward()
        opt.step()
        hist.append(loss.item())
    P = {"W1": model[0].weight.detach().numpy().copy(), "b1": model[0].bias.detach().numpy().copy(),
         "W2": model[2].weight.detach().numpy().copy(), "b2": model[2].bias.detach().numpy().copy()}
    return hist, P


def task5_train() -> dict:
    X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
    y = y.astype(np.float64)
    P0 = make_init(np.random.default_rng(SEED))
    t_eng, t_tor = [], []
    for r in range(N_RUNS):
        t = time.perf_counter(); h_e, P_e, n_nodes = train_engine(X, y, P0); t_eng.append(time.perf_counter() - t)
        t = time.perf_counter(); h_t, P_t = train_torch(X, y, P0); t_tor.append(time.perf_counter() - t)
        log.info("TASK5 run %d/%d  engine=%.3fs  pytorch=%.4fs", r + 1, N_RUNS, t_eng[-1], t_tor[-1])
    loss_e, acc_e = metrics(P_e, X, y)
    loss_t, acc_t = metrics(P_t, X, y)
    res = {"samples": N_SAMPLES, "epochs": EPOCHS, "lr": LR, "architecture": "2 -> 8 (ReLU) -> 1 (sigmoid), BCE loss",
           "graph_nodes_per_backward": n_nodes,
           "engine": {"median_s": statistics.median(t_eng), "runs_s": t_eng, "final_loss": loss_e, "accuracy": acc_e,
                      "loss_first_epoch": h_e[0], "loss_last_epoch": h_e[-1]},
           "pytorch": {"median_s": statistics.median(t_tor), "runs_s": t_tor, "final_loss": loss_t, "accuracy": acc_t,
                       "loss_first_epoch": h_t[0], "loss_last_epoch": h_t[-1]},
           "max_abs_loss_history_diff": float(np.abs(np.array(h_e) - np.array(h_t)).max()),
           "speedup_pytorch_over_engine": statistics.median(t_eng) / statistics.median(t_tor)}
    log.info("TASK5 engine : loss %.4f -> %.4f  acc=%.3f  median=%.3fs", h_e[0], h_e[-1], acc_e, res["engine"]["median_s"])
    log.info("TASK5 pytorch: loss %.4f -> %.4f  acc=%.3f  median=%.4fs", h_t[0], h_t[-1], acc_t, res["pytorch"]["median_s"])
    log.info("TASK5 loss-history max |diff| = %.2e | PyTorch is %.0fx faster", res["max_abs_loss_history_diff"],
             res["speedup_pytorch_over_engine"])
    return res


# =============================================== TASK 6: GRAPH, EAGER vs COMPILED
def tinfo(name, t):
    return (f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, contiguous={t.is_contiguous()}, "
            f"stride={t.stride()}, requires_grad={t.requires_grad}")


def graph_lines(fn, depth=0, lines=None):
    lines = [] if lines is None else lines
    v = getattr(fn, "variable", None)
    label = type(fn).__name__ + (f"  (parameter shape={tuple(v.shape)})" if v is not None else "")
    lines.append("  " * depth + label)
    for nxt, _ in fn.next_functions:
        if nxt is not None:
            graph_lines(nxt, depth + 1, lines)
    return lines


def timeit(fn, inner, warmup=30, repeats=7):
    """median over `repeats` of the mean time of `inner` calls, after `warmup` calls (microseconds)."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        for _ in range(inner):
            fn()
        ts.append((time.perf_counter() - t) / inner)
    return statistics.median(ts) * 1e6


def task6_frameworks() -> dict:
    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Linear(2, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1), nn.Sigmoid())
    x = torch.randn(4, 2)
    out = model(x)
    insp = ["== Tensor inspection ==",
            tinfo("input x", x), tinfo("W1 (Linear 2->8)", model[0].weight), tinfo("W2 (Linear 8->1)", model[2].weight),
            tinfo("output", out), f"output.grad_fn = {out.grad_fn}", "",
            "== Autograd graph (output -> leaves) ==", *graph_lines(out.grad_fn)]
    try:
        traced = torch.jit.trace(model, x)
        insp += ["", "== torch.jit.trace graph ==", str(traced.graph)]
    except Exception as e:                                   # tracing removed / unsupported in this torch build
        traced = None
        insp += ["", f"== torch.jit.trace unavailable: {type(e).__name__} =="]
    Path("results").mkdir(exist_ok=True)
    log.info("TASK6 graph root: %s", type(out.grad_fn).__name__)

    # ---- eager vs traced vs compiled -------------------------------------
    variants = {"eager": model}
    if traced is not None:
        variants["jit.trace"] = traced
    compile_info = {"torch.compile": "not attempted"}
    first_call = {}
    for name, kw in (("torch.compile(inductor)", {}), ("torch.compile(aot_eager)", {"backend": "aot_eager"})):
        try:
            torch._dynamo.reset()
            comp = torch.compile(model, **kw)
            t0 = time.perf_counter()
            with torch.no_grad():
                comp(torch.randn(64, 2))
            comp(torch.randn(64, 2)).sum().backward()
            first_call[name] = time.perf_counter() - t0
            variants[name] = comp
            compile_info["torch.compile"] = f"worked with {name}"
            break
        except Exception as e:                               # missing C++ compiler / unsupported platform
            compile_info[name] = f"failed: {type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''}"
            log.warning("TASK6 %s failed (%s)", name, type(e).__name__)
    if traced is not None:
        t0 = time.perf_counter(); traced(torch.randn(64, 2)); first_call["jit.trace"] = time.perf_counter() - t0
    log.info("TASK6 compile status: %s", compile_info)

    rows = []
    for bs, inner in ((64, 300), (65536, 30)):
        xb = torch.randn(bs, 2)
        for mode, m in variants.items():
            def fwd(m=m, xb=xb):
                with torch.no_grad():
                    m(xb)

            def fwd_bwd(m=m, xb=xb):
                model.zero_grad(set_to_none=True)
                m(xb).sum().backward()
            for kind, fn in (("forward (no_grad)", fwd), ("forward+backward", fwd_bwd)):
                try:
                    us = timeit(fn, inner)
                except Exception as e:
                    us = float("nan"); log.warning("TASK6 %s %s failed: %s", mode, kind, type(e).__name__)
                rows.append({"mode": mode, "batch": bs, "pass": kind, "median_us": us})
        base = {(r["batch"], r["pass"]): r["median_us"] for r in rows if r["mode"] == "eager"}
        for r in rows:
            r["speedup_vs_eager"] = base[(r["batch"], r["pass"])] / r["median_us"] if r["median_us"] == r["median_us"] else float("nan")
    for r in rows:
        log.info("TASK6 %-26s batch=%-6d %-18s %10.1f us  x%.2f vs eager", r["mode"], r["batch"], r["pass"], r["median_us"], r["speedup_vs_eager"])

    # ---- non-contiguous views --------------------------------------------
    N = 2048
    a = torch.randn(N, N)
    at = a.t()                                   # non-contiguous VIEW (strides swapped)
    sl = a[:, ::2]                               # non-contiguous VIEW (every 2nd column)
    w = torch.randn(N, 64)
    cases = [("transpose view: t + 1", at, at.contiguous(), lambda t: t + 1),
             ("transpose view: t.relu()", at, at.contiguous(), lambda t: t.relu()),
             ("transpose view: t.sum(dim=1)", at, at.contiguous(), lambda t: t.sum(dim=1)),
             ("transpose view: t @ w", at, at.contiguous(), lambda t: t @ w),
             ("strided slice a[:, ::2]: t * 2", sl, sl.contiguous(), lambda t: t * 2)]
    nc_rows = []
    insp += ["", "== Non-contiguous views ==", tinfo("a", a), tinfo("a.t()", at), tinfo("a[:, ::2]", sl)]
    for name, nc, c, fn in cases:
        t_nc = timeit(lambda: fn(nc), inner=10, warmup=3)
        t_c = timeit(lambda: fn(c), inner=10, warmup=3)
        nc_rows.append({"operation": name, "contiguous_us": t_c, "non_contiguous_us": t_nc, "slowdown": t_nc / t_c})
        log.info("TASK6 %-34s contiguous=%9.0f us  non-contig=%9.0f us  slowdown x%.2f", name, t_c, t_nc, t_nc / t_c)
    t_copy = timeit(lambda: at.contiguous(), inner=10, warmup=3)
    log.info("TASK6 cost of at.contiguous() copy itself: %.0f us", t_copy)

    Xt = torch.randn(2, 65536)
    x_nc, x_c = Xt.t(), Xt.t().contiguous()      # model input: (65536, 2) with swapped strides
    insp += [tinfo("model input, non-contiguous", x_nc), tinfo("model input, contiguous copy", x_c)]

    def mfwd(inp):
        with torch.no_grad():
            model(inp)
    m_nc = timeit(lambda: mfwd(x_nc), inner=30)
    m_c = timeit(lambda: mfwd(x_c), inner=30)
    nc_rows.append({"operation": "model forward on (65536,2) input", "contiguous_us": m_c, "non_contiguous_us": m_nc,
                    "slowdown": m_nc / m_c})
    log.info("TASK6 model forward  contiguous=%.0f us  non-contig=%.0f us  slowdown x%.2f", m_c, m_nc, m_nc / m_c)
    (Path("results") / "graph_inspection.txt").write_text("\n".join(insp), encoding="utf-8")
    return {"compile_status": compile_info, "first_call_s": first_call, "timings": rows,
            "non_contiguous": nc_rows, "contiguous_copy_us": t_copy}


# =============================================================== REPORT
def when_compilation_helps(t6) -> str:
    rows = t6["timings"]
    comp = [m for m in {r["mode"] for r in rows} if m.startswith("torch.compile")]
    lines = []
    for mode in ["jit.trace"] + comp:
        for bs in (64, 65536):
            r = next((r for r in rows if r["mode"] == mode and r["batch"] == bs and r["pass"] == "forward (no_grad)"), None)
            if r and r["speedup_vs_eager"] == r["speedup_vs_eager"]:
                lines.append(f"- `{mode}`, batch {bs}, forward: x{r['speedup_vs_eager']:.2f} vs eager "
                             f"({'faster' if r['speedup_vs_eager'] > 1.05 else 'slower' if r['speedup_vs_eager'] < 0.95 else 'about the same'}).")
    fc = ", ".join(f"{k}: {v:.2f}s" for k, v in t6["first_call_s"].items())
    return "\n".join([
        "**Measured in this run**", *lines,
        f"- First call (compile / trace + first run): {fc}. This one-off cost is not in the timings above.",
        f"- torch.compile status: {t6['compile_status']}", "",
        "**When compilation helps**",
        "- Many small operations: a compiler can fuse them and remove Python and dispatcher overhead on every call "
        "(most visible at small batch sizes, where overhead dominates the math).",
        "- The same function is called many times with fixed shapes and dtypes (training loops, serving), so the "
        "compile cost is amortised after warm-up.",
        "- Larger models and GPUs, where kernel fusion saves memory traffic.",
        "",
        "**When it does not help**",
        "- Few or already-large operations (e.g. one big matmul), where the time is spent in the math kernel, not in overhead.",
        "- Short runs or one-off calls, where compile / trace time exceeds the saving.",
        "- Changing shapes or Python control flow, which cause graph breaks or recompilation.",
        "- Platforms without a working compiler toolchain (torch.compile needs one), where tracing/scripting is the fallback.",
    ])


def write_report(out: Path, t4, t5, t6):
    e, t = t5["engine"], t5["pytorch"]
    ok4 = t4["task_expression"]
    md = [
        "# Lab 5 - Frameworks and automatic differentiation: results", "",
        f"Python {platform.python_version()}, torch {torch.__version__}, numpy {np.__version__}, seed {SEED}.", "",
        "## Tasks 1-3: engine",
        "`Value` stores `data`, `grad`, parent nodes (`_prev`) and its local derivative rule (`_backward`). "
        "`+`, `*`, `**`, `relu` (plus `log`, `sigmoid`) build the graph during the forward pass. `backward()` topologically "
        "sorts the graph, sets the output gradient to 1, visits nodes in reverse order and accumulates (`+=`) gradients.", "",
        "## Task 4: verification on ((a*b + c)**2).relu().sum()", "",
        "| Check | max abs diff vs PyTorch | max abs diff vs numerical | < 1e-6 |", "|---|---|---|---|",
        f"| ((a*b+c)**2).relu().sum() | {ok4['max_abs_diff_vs_pytorch']:.2e} | {ok4['max_abs_diff_vs_numerical']:.2e} | {'PASS' if ok4['pass'] else 'FAIL'} |",
        f"| extra: ((a*b+c).relu()**2).sum() | {t4['extra_relu_branch_check']['max_abs_diff_vs_pytorch']:.2e} | "
        f"{t4['extra_relu_branch_check']['max_abs_diff_vs_numerical']:.2e} | {'PASS' if t4['extra_relu_branch_check']['pass'] else 'FAIL'} |", "",
        f"Gradient accumulation check (y = x*x + x at x=3, dy/dx = 7): grad = {t4['accumulation_check_x*x+x_at_3']['grad']} -> "
        f"{'PASS' if t4['accumulation_check_x*x+x_at_3']['pass'] else 'FAIL'}.", "",
        "## Task 5: make_moons, 2 -> 8 -> 1, sigmoid, BCE",
        f"{t5['samples']} samples, {t5['epochs']} full-batch epochs, lr {t5['lr']}, identical initial weights in both implementations, "
        f"median of {N_RUNS} runs.", "",
        "| Implementation | Median time (s) | First-epoch loss | Last-epoch loss | Accuracy |", "|---|---|---|---|---|",
        f"| Own `Value` engine | {e['median_s']:.3f} | {e['loss_first_epoch']:.4f} | {e['loss_last_epoch']:.4f} | {e['accuracy']:.3f} |",
        f"| PyTorch (float64) | {t['median_s']:.4f} | {t['loss_first_epoch']:.4f} | {t['loss_last_epoch']:.4f} | {t['accuracy']:.3f} |", "",
        f"PyTorch is **{t5['speedup_pytorch_over_engine']:.0f}x faster**. Max |difference| between the two loss histories: "
        f"{t5['max_abs_loss_history_diff']:.2e}. The engine builds a graph of {t5['graph_nodes_per_backward']} Python objects per step; "
        "PyTorch uses a few vectorised kernels.", "",
        "## Task 6: PyTorch graph, eager vs traced / compiled, contiguity",
        "Graph and tensor inspection (shape, dtype, contiguity, strides) is in `graph_inspection.txt`.", "",
        "| Mode | Batch | Pass | Median (us) | Speed-up vs eager |", "|---|---|---|---|---|"]
    md += [f"| {r['mode']} | {r['batch']} | {r['pass']} | {r['median_us']:.1f} | x{r['speedup_vs_eager']:.2f} |" for r in t6["timings"]]
    md += ["", "### Non-contiguous views (2048 x 2048 float32)", "",
           "| Operation | Contiguous (us) | Non-contiguous view (us) | Slow-down |", "|---|---|---|---|"]
    md += [f"| {r['operation']} | {r['contiguous_us']:.0f} | {r['non_contiguous_us']:.0f} | x{r['slowdown']:.2f} |" for r in t6["non_contiguous"]]
    md += ["", f"Making a contiguous copy itself costs about {t6['contiguous_copy_us']:.0f} us, so it only pays off when the "
           "view is used by several operations.", "",
           "## Short note: when does compilation help?", "", when_compilation_helps(t6), ""]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")


def main():
    base = Path(__file__).resolve().parent
    out = base / "results"
    out.mkdir(exist_ok=True)
    import os
    os.chdir(base)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    root = logging.getLogger(); root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log", mode="w", encoding="utf-8")):
        h.setFormatter(fmt); root.addHandler(h)
    np.random.seed(SEED); random_seed = torch.manual_seed(SEED)
    t4 = task4_verify()
    t5 = task5_train()
    t6 = task6_frameworks()
    (out / "results.json").write_text(json.dumps({"task4": t4, "task5": t5, "task6": t6}, indent=2, default=float), encoding="utf-8")
    write_report(out, t4, t5, t6)
    log.info("DONE  -> %s", out / "report.md")


if __name__ == "__main__":
    main()
