# Lab 3: Architectures and Their Costs

## Experimental setup

- Dataset: Fashion-MNIST; official 60,000-image training set split into 55,000 training and 5,000 validation images; official 10,000-image test set.
- Transform: `ToTensor()` followed by `Normalize((0.2860,), (0.3530,))`.
- Batch size: 128; epochs: 5; optimizer: Adam; learning rate: 0.001.
- Random seed: 42; device: `cpu`; PyTorch version: `2.14.1+cpu`.
- Latency protocol: 10 warm-up forwards for each batch size, then median of 100 timed runs for batch 1 and 20 timed runs for batch 128. CUDA synchronization is used when applicable. Latency is per complete forward pass for the stated batch size.

## Architectures and parameter-count constraint

The layer definitions below follow the lab sheet exactly. There is a specification conflict: the requested target of 100k-150k parameters for all three models is incompatible with the explicitly listed layer dimensions. The MLP is above that range, while the CNN and attention model are below it. The layer sizes were not silently changed.

- **MLP:** Flatten 784 → Linear 256 → ReLU → Linear 128 → ReLU → Linear 10.
- **CNN:** Conv2d(1,16,3,padding=1) → ReLU → MaxPool2 → Conv2d(16,32,3,padding=1) → ReLU → MaxPool2 → Flatten → Linear(32×7×7,10).
- **Attention:** 16 non-overlapping 7×7 patches → Linear 49→64 → two Transformer encoder layers (`d_model=64`, `nhead=4`, `dim_feedforward=128`) → mean pooling → Linear 64→10. No positional embedding is added because it is not specified in the lab sheet.

## Parameters, FLOPs, profiling and measured results

Parameter counts are obtained from `torchinfo.summary` and `model.parameters()`. THOP values are reported in the library's own counting convention; they should not automatically be treated as identical to the hand FLOP convention. THOP may omit operations implemented with functional tensor operations, particularly parts of multi-head attention.

| Model | Parameters | Test accuracy (%) | Hand FLOPs for representative layer / block | THOP full-model reported ops | THOP representative-layer reported ops | Batch 1 median (ms) | Batch 128 median (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 235,146 | 87.95 | 401,408 | 234,752.00 | 200,704.00 | 0.095 | 0.485 |
| CNN | 20,490 | 89.55 | 225,792 | 1,031,744.00 | 112,896.00 | 0.230 | 7.396 |
| Attention | 70,794 | 83.84 | 1,114,112 | 591,488.00 | 270,336.00 | 0.608 | 3.606 |

### Hand calculations

- **MLP representative layer:** `2 x 784 x 256 = 401,408 FLOPs/image`. This counts multiply-add work in the first linear layer only.
- **CNN representative layer:** `2 x 3 x 3 x 1 x 16 x 28 x 28 = 225,792 FLOPs/image`. For padding 1, the first convolution's output is 28×28.
- **Attention representative block:** `QKV 393,216 + scores 32,768 + weighted values 32,768 + output projection 131,072 + FFN 524,288 = 1,114,112 FLOPs/image`. This sums matrix multiplications in one encoder layer at sequence length 16 and `d_model=64`; bias additions, softmax, layer normalization, activations and other elementwise operations are excluded from this estimate.

Attention-block breakdown:

| Operation | Hand FLOPs per image |
|---|---:|
| Q, K and V projections | 393,216 |
| Attention score products QK^T | 32,768 |
| Attention-weighted value products | 32,768 |
| Attention output projection | 131,072 |
| Feed-forward Linear(64, 128) and Linear(128, 64) | 524,288 |
| **Total for one encoder layer** | **1,114,112** |

### Hand calculation vs THOP

For a fair scope comparison, the representative THOP profiles use the first MLP Linear layer, the first CNN convolution, and one complete Transformer encoder layer. The separate full-model THOP counts cover each entire model. A difference is expected because the hand estimates explicitly count multiply and add as two FLOPs, while THOP uses its own operator-counting handlers; unsupported or functionally expressed operations can be omitted. The comparison is indicative, not a claim that differently defined operation units are identical.

| Model | Representative layer | Hand calculation (FLOPs/image) | THOP reported ops | Difference / caveat |
|---|---|---:|---:|---|
| MLP | Linear(784, 256) | 401,408 | 200,704.00 | Differences can come from the THOP convention and whether bias operations are included. |
| CNN | Conv2d(1, 16, kernel_size=3, padding=1) | 225,792 | 112,896.00 | Differences can come from the THOP convention and whether bias operations are included. |
| Attention | One Transformer encoder layer (sequence length 16, d_model 64) | 1,114,112 | 270,336.00 | THOP may not count functional QKV projections and attention matrix products; compare conventions before interpreting the gap. |

## Training history

Final epoch validation results:

| Model | Final training accuracy (%) | Final validation accuracy (%) | Final training loss | Final validation loss |
|---|---:|---:|---:|---:|
| MLP | 90.07 | 88.28 | 0.2684 | 0.3232 |
| CNN | 90.93 | 89.74 | 0.2540 | 0.2827 |
| Attention | 84.52 | 83.72 | 0.4247 | 0.4576 |

## Accuracy, cost and access-pattern interpretation

- Highest measured test accuracy: **CNN** (89.55%).
- Highest test-accuracy-per-parameter ratio (a simple, approximate parameter-efficiency metric): **CNN**.
- Highest accuracy per millisecond of batch-1 latency: **MLP**.
- Highest accuracy per millisecond of batch-128 latency: **MLP**. Batch-128 latency is for the whole batch, not per image.

There is no single architecture that is automatically the most cost-effective for every objective; the measured accuracy, parameter count, and latency should be read together. The MLP uses dense global connections after flattening, which is simple to implement but does not explicitly preserve local image structure. The CNN reuses small convolution kernels across spatial positions, typically improving parameter efficiency and local spatial reuse; pooling reduces later feature-map sizes, although convolutions still read overlapping local windows. The attention model performs projection and feed-forward matrix operations over 16 patch tokens, and self-attention allows each token to interact with the others. Its attention work and intermediate access grow quadratically with sequence length, so patch size and token count matter. Here, the sequence length is only 16, which limits that pairwise attention cost. Actual cost-effectiveness should follow the measured results above rather than parameter count alone.

## Plots

- [accuracy_vs_parameters.png](outputs/accuracy_vs_parameters.png)
- [accuracy_vs_latency_batch1.png](outputs/accuracy_vs_latency_batch1.png)
- [accuracy_vs_latency_batch128.png](outputs/accuracy_vs_latency_batch128.png)

## Reproducibility notes

- The split is reproducible with seed 42. Training shuffles the 55,000-image training subset; validation and test loaders do not shuffle.
- Batch size, epoch count, Adam learning rate, transforms and timing repetition counts are fixed to the lab sheet.
- Timing depends on hardware, device, PyTorch version, and system load. Report the device listed above when comparing results.
- THOP and the hand calculation use different operation-counting conventions in some cases. In particular, inspect attention counts for unsupported operations before interpreting them as exact FLOPs.
