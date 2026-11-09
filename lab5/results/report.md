# Lab 5 - Frameworks and automatic differentiation: results

Python 3.13.2, torch 2.15.0+cpu, numpy 2.3.4, seed 0.

## Tasks 1-3: engine
`Value` stores `data`, `grad`, parent nodes (`_prev`) and its local derivative rule (`_backward`). `+`, `*`, `**`, `relu` (plus `log`, `sigmoid`) build the graph during the forward pass. `backward()` topologically sorts the graph, sets the output gradient to 1, visits nodes in reverse order and accumulates (`+=`) gradients.

## Task 4: verification on ((a*b + c)**2).relu().sum()

| Check | max abs diff vs PyTorch | max abs diff vs numerical | < 1e-6 |
|---|---|---|---|
| ((a*b+c)**2).relu().sum() | 0.00e+00 | 3.11e-09 | PASS |
| extra: ((a*b+c).relu()**2).sum() | 0.00e+00 | 3.89e-09 | PASS |

Gradient accumulation check (y = x*x + x at x=3, dy/dx = 7): grad = 7.0 -> PASS.

## Task 5: make_moons, 2 -> 8 -> 1, sigmoid, BCE
200 samples, 100 full-batch epochs, lr 0.5, identical initial weights in both implementations, median of 5 runs.

| Implementation | Median time (s) | First-epoch loss | Last-epoch loss | Accuracy |
|---|---|---|---|---|
| Own `Value` engine | 10.750 | 0.6572 | 0.2929 | 0.855 |
| PyTorch (float64) | 0.0400 | 0.6572 | 0.2929 | 0.855 |

PyTorch is **269x faster**. Max |difference| between the two loss histories: 1.11e-16. The engine builds a graph of 10835 Python objects per step; PyTorch uses a few vectorised kernels.

## Task 6: PyTorch graph, eager vs traced / compiled, contiguity
Graph and tensor inspection (shape, dtype, contiguity, strides) is in `graph_inspection.txt`.

| Mode | Batch | Pass | Median (us) | Speed-up vs eager |
|---|---|---|---|---|
| eager | 64 | forward (no_grad) | 34.9 | x1.00 |
| eager | 64 | forward+backward | 262.0 | x1.00 |
| jit.trace | 64 | forward (no_grad) | 40.0 | x0.87 |
| jit.trace | 64 | forward+backward | 262.9 | x1.00 |
| torch.compile(aot_eager) | 64 | forward (no_grad) | 97.6 | x0.36 |
| torch.compile(aot_eager) | 64 | forward+backward | 494.8 | x0.53 |
| eager | 65536 | forward (no_grad) | 902.8 | x1.00 |
| eager | 65536 | forward+backward | 2242.9 | x1.00 |
| jit.trace | 65536 | forward (no_grad) | 947.0 | x0.95 |
| jit.trace | 65536 | forward+backward | 2214.4 | x1.01 |
| torch.compile(aot_eager) | 65536 | forward (no_grad) | 1033.6 | x0.87 |
| torch.compile(aot_eager) | 65536 | forward+backward | 2631.3 | x0.85 |

### Non-contiguous views (2048 x 2048 float32)

| Operation | Contiguous (us) | Non-contiguous view (us) | Slow-down |
|---|---|---|---|
| transpose view: t + 1 | 1695 | 1912 | x1.13 |
| transpose view: t.relu() | 1743 | 1825 | x1.05 |
| transpose view: t.sum(dim=1) | 203 | 840 | x4.15 |
| transpose view: t @ w | 2330 | 4350 | x1.87 |
| strided slice a[:, ::2]: t * 2 | 225 | 868 | x3.85 |
| model forward on (65536,2) input | 940 | 1026 | x1.09 |

Making a contiguous copy itself costs about 7132 us, so it only pays off when the view is used by several operations.

## Short note: when does compilation help?

**Measured in this run**
- `jit.trace`, batch 64, forward: x0.87 vs eager (slower).
- `jit.trace`, batch 65536, forward: x0.95 vs eager (about the same).
- `torch.compile(aot_eager)`, batch 64, forward: x0.36 vs eager (slower).
- `torch.compile(aot_eager)`, batch 65536, forward: x0.87 vs eager (slower).
- First call (compile / trace + first run): torch.compile(aot_eager): 0.22s, jit.trace: 0.00s. This one-off cost is not in the timings above.
- torch.compile status: {'torch.compile': 'worked with torch.compile(aot_eager)', 'torch.compile(inductor)': 'failed: InductorError: InvalidCxxCompiler: Compiler: cl is not found.'}

**When compilation helps**
- Many small operations: a compiler can fuse them and remove Python and dispatcher overhead on every call (most visible at small batch sizes, where overhead dominates the math).
- The same function is called many times with fixed shapes and dtypes (training loops, serving), so the compile cost is amortised after warm-up.
- Larger models and GPUs, where kernel fusion saves memory traffic.

**When it does not help**
- Few or already-large operations (e.g. one big matmul), where the time is spent in the math kernel, not in overhead.
- Short runs or one-off calls, where compile / trace time exceeds the saving.
- Changing shapes or Python control flow, which cause graph breaks or recompilation.
- Platforms without a working compiler toolchain (torch.compile needs one), where tracing/scripting is the fallback.
