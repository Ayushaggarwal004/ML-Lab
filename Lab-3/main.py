from __future__ import annotations

import csv
import random
import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from thop import profile
from torchinfo import summary

from models import build_models


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_PATH = BASE_DIR / "report.md"
SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3
LATENCY_RUNS = {1: 100, 128: 20}
WARMUP_RUNS = 10


# Representative-layer FLOPs use the lab sheet's convention that one
# multiply-accumulate is counted as two FLOPs. Bias, activation, normalization,
# softmax, and pooling costs are not included in these matrix-multiplication
# estimates.
HAND_FLOPS = {
    "MLP": {
        "layer": "Linear(784, 256)",
        "flops": 2 * 784 * 256,
        "formula": "2 x 784 x 256 = 401,408 FLOPs/image",
    },
    "CNN": {
        "layer": "Conv2d(1, 16, kernel_size=3, padding=1)",
        "flops": 2 * 3 * 3 * 1 * 16 * 28 * 28,
        "formula": "2 x 3 x 3 x 1 x 16 x 28 x 28 = 225,792 FLOPs/image",
    },
    "Attention": {
        "layer": "One Transformer encoder layer (sequence length 16, d_model 64)",
        "flops": (
            2 * 16 * 64 * 3 * 64
            + 2 * 16 * 16 * 64
            + 2 * 16 * 16 * 64
            + 2 * 16 * 64 * 64
            + 2 * 16 * 64 * 128
            + 2 * 16 * 128 * 64
        ),
        "formula": "QKV 393,216 + scores 32,768 + weighted values 32,768 + output projection 131,072 + FFN 524,288 = 1,114,112 FLOPs/image",
    },
}


ATTENTION_FLOP_BREAKDOWN = [
    ("Q, K and V projections", 2 * 16 * 64 * 3 * 64),
    ("Attention score products QK^T", 2 * 16 * 16 * 64),
    ("Attention-weighted value products", 2 * 16 * 16 * 64),
    ("Attention output projection", 2 * 16 * 64 * 64),
    ("Feed-forward Linear(64, 128) and Linear(128, 64)", 2 * 16 * 64 * 128 + 2 * 16 * 128 * 64),
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(device: torch.device) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    full_train_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform,
    )

    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [55_000, 5_000],
        generator=torch.Generator().manual_seed(SEED),
    )

    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    return train_loader, validation_loader, test_loader


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def format_number(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value)


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def thop_profile(module: nn.Module, dummy_input: torch.Tensor) -> dict[str, Any]:
    try:
        ops, parameters = profile(module, inputs=(dummy_input,), verbose=False)
        return {"ops": float(ops), "parameters": int(parameters), "error": None}
    except Exception as exc:
        return {"ops": None, "parameters": None, "error": f"{type(exc).__name__}: {exc}"}


def profile_models(models: dict[str, nn.Module], device: torch.device) -> tuple[dict, dict, str]:
    full_profiles: dict[str, dict[str, Any]] = {}
    layer_profiles: dict[str, dict[str, Any]] = {}
    summaries: list[str] = []

    for name, model in models.items():
        model.eval()
        summary_string = str(
            summary(
                model,
                input_size=(1, 1, 28, 28),
                device=str(device),
                verbose=0,
            )
        )
        summaries.append(f"MODEL: {name}\n{summary_string}\n")
        print(f"\n{name} parameter summary:\n{summary_string}")

        image_input = torch.zeros(1, 1, 28, 28, device=device)
        full_profiles[name] = thop_profile(model, image_input)

        if name == "MLP":
            representative_layer = model.network[1]
            layer_input = torch.zeros(1, 784, device=device)
        elif name == "CNN":
            representative_layer = model.conv1
            layer_input = image_input
        else:
            representative_layer = model.transformer.layers[0]
            layer_input = torch.zeros(1, 16, 64, device=device)
        layer_profiles[name] = thop_profile(representative_layer, layer_input)

    summary_path = OUTPUT_DIR / "model_summaries.txt"
    summary_path.write_text("\n".join(summaries), encoding="utf-8")
    return full_profiles, layer_profiles, str(summary_path)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_count = labels.size(0)
            total_loss += loss.item() * batch_count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_count

    return total_loss / total, 100.0 * correct / total


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_count = labels.size(0)
        total_loss += loss.item() * batch_count
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_count

    return total_loss / total, 100.0 * correct / total


def train_models(
    models: dict[str, nn.Module],
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, list[dict[str, float]]], dict[str, float]]:
    criterion = nn.CrossEntropyLoss()
    history: dict[str, list[dict[str, float]]] = {}
    test_accuracies: dict[str, float] = {}

    for name, model in models.items():
        print(f"\n{'=' * 64}\nTraining {name}: {EPOCHS} epochs, Adam, lr={LEARNING_RATE}, batch={BATCH_SIZE}\n{'=' * 64}")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        history[name] = []

        for epoch in range(1, EPOCHS + 1):
            started = time.perf_counter()
            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            validation_loss, validation_accuracy = evaluate(
                model, validation_loader, criterion, device
            )
            elapsed = time.perf_counter() - started
            history[name].append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy_pct": train_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy_pct": validation_accuracy,
                    "epoch_seconds": elapsed,
                }
            )
            print(
                f"{name} | epoch {epoch}/{EPOCHS} | "
                f"train loss {train_loss:.4f} | train acc {train_accuracy:.2f}% | "
                f"val loss {validation_loss:.4f} | val acc {validation_accuracy:.2f}% | "
                f"{elapsed:.1f}s"
            )

        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        test_accuracies[name] = test_accuracy
        print(f"{name} final official test accuracy: {test_accuracy:.2f}% (loss {test_loss:.4f})")
        torch.save(model.state_dict(), OUTPUT_DIR / f"{name.lower()}_state_dict.pth")

    return history, test_accuracies


def benchmark_latency(model: nn.Module, device: torch.device) -> dict[int, float]:
    model.eval()
    medians_ms: dict[int, float] = {}

    for batch_size, run_count in LATENCY_RUNS.items():
        inputs = torch.randn(batch_size, 1, 28, 28, device=device)
        with torch.inference_mode():
            for _ in range(WARMUP_RUNS):
                model(inputs)
            synchronize_if_cuda(device)

            elapsed_ms: list[float] = []
            for _ in range(run_count):
                synchronize_if_cuda(device)
                start = time.perf_counter()
                model(inputs)
                synchronize_if_cuda(device)
                elapsed_ms.append((time.perf_counter() - start) * 1000.0)

        medians_ms[batch_size] = statistics.median(elapsed_ms)
        print(
            f"{batch_size}-image batch latency: median of {run_count} runs = "
            f"{medians_ms[batch_size]:.3f} ms"
        )

    return medians_ms


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_plots(rows: list[dict[str, Any]]) -> list[str]:
    plot_paths: list[str] = []

    plt.figure(figsize=(8, 5))
    for row in rows:
        plt.scatter(row["parameters"], row["test_accuracy_pct"], label=row["model"])
        plt.annotate(
            row["model"],
            (row["parameters"], row["test_accuracy_pct"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    plt.xlabel("Trainable parameters")
    plt.ylabel("Official test accuracy (%)")
    plt.title("Fashion-MNIST accuracy vs parameter count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUTPUT_DIR / "accuracy_vs_parameters.png"
    plt.savefig(path, dpi=160)
    plt.close()
    plot_paths.append(path.name)

    for batch_size in (1, 128):
        latency_key = f"latency_batch{batch_size}_median_ms"
        plt.figure(figsize=(8, 5))
        for row in rows:
            plt.scatter(row[latency_key], row["test_accuracy_pct"], label=row["model"])
            plt.annotate(
                row["model"],
                (row[latency_key], row["test_accuracy_pct"]),
                xytext=(5, 5),
                textcoords="offset points",
            )
        plt.xlabel(f"Median latency for one batch of {batch_size} (ms)")
        plt.ylabel("Official test accuracy (%)")
        plt.title(f"Fashion-MNIST accuracy vs latency (batch {batch_size})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = OUTPUT_DIR / f"accuracy_vs_latency_batch{batch_size}.png"
        plt.savefig(path, dpi=160)
        plt.close()
        plot_paths.append(path.name)

    return plot_paths


def build_report(
    rows: list[dict[str, Any]],
    history: dict[str, list[dict[str, float]]],
    layer_profiles: dict[str, dict[str, Any]],
    device: torch.device,
    plot_paths: list[str],
) -> str:
    lines: list[str] = [
        "# Lab 3: Architectures and Their Costs",
        "",
        "## Experimental setup",
        "",
        f"- Dataset: Fashion-MNIST; official 60,000-image training set split into 55,000 training and 5,000 validation images; official 10,000-image test set.",
        "- Transform: `ToTensor()` followed by `Normalize((0.2860,), (0.3530,))`.",
        f"- Batch size: {BATCH_SIZE}; epochs: {EPOCHS}; optimizer: Adam; learning rate: {LEARNING_RATE}.",
        f"- Random seed: {SEED}; device: `{device}`; PyTorch version: `{torch.__version__}`.",
        f"- Latency protocol: {WARMUP_RUNS} warm-up forwards for each batch size, then median of 100 timed runs for batch 1 and 20 timed runs for batch 128. CUDA synchronization is used when applicable. Latency is per complete forward pass for the stated batch size.",
        "",
        "## Architectures and parameter-count constraint",
        "",
        "The layer definitions below follow the lab sheet exactly. There is a specification conflict: the requested target of 100k-150k parameters for all three models is incompatible with the explicitly listed layer dimensions. The MLP is above that range, while the CNN and attention model are below it. The layer sizes were not silently changed.",
        "",
        "- **MLP:** Flatten 784 → Linear 256 → ReLU → Linear 128 → ReLU → Linear 10.",
        "- **CNN:** Conv2d(1,16,3,padding=1) → ReLU → MaxPool2 → Conv2d(16,32,3,padding=1) → ReLU → MaxPool2 → Flatten → Linear(32×7×7,10).",
        "- **Attention:** 16 non-overlapping 7×7 patches → Linear 49→64 → two Transformer encoder layers (`d_model=64`, `nhead=4`, `dim_feedforward=128`) → mean pooling → Linear 64→10. No positional embedding is added because it is not specified in the lab sheet.",
        "",
        "## Parameters, FLOPs, profiling and measured results",
        "",
        "Parameter counts are obtained from `torchinfo.summary` and `model.parameters()`. THOP values are reported in the library's own counting convention; they should not automatically be treated as identical to the hand FLOP convention. THOP may omit operations implemented with functional tensor operations, particularly parts of multi-head attention.",
        "",
        "| Model | Parameters | Test accuracy (%) | Hand FLOPs for representative layer / block | THOP full-model reported ops | THOP representative-layer reported ops | Batch 1 median (ms) | Batch 128 median (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['model']} | {format_number(row['parameters'])} | {row['test_accuracy_pct']:.2f} | "
            f"{format_number(row['hand_flops_per_image'])} | {format_number(row['thop_full_model_ops'])} | "
            f"{format_number(row['thop_representative_layer_ops'])} | {row['latency_batch1_median_ms']:.3f} | "
            f"{row['latency_batch128_median_ms']:.3f} |"
        )

    lines.extend(
        [
            "",
            "### Hand calculations",
            "",
            f"- **MLP representative layer:** `{HAND_FLOPS['MLP']['formula']}`. This counts multiply-add work in the first linear layer only.",
            f"- **CNN representative layer:** `{HAND_FLOPS['CNN']['formula']}`. For padding 1, the first convolution's output is 28×28.",
            f"- **Attention representative block:** `{HAND_FLOPS['Attention']['formula']}`. This sums matrix multiplications in one encoder layer at sequence length 16 and `d_model=64`; bias additions, softmax, layer normalization, activations and other elementwise operations are excluded from this estimate.",
            "",
            "Attention-block breakdown:",
            "",
            "| Operation | Hand FLOPs per image |",
            "|---|---:|",
        ]
    )
    for label, value in ATTENTION_FLOP_BREAKDOWN:
        lines.append(f"| {label} | {format_number(value)} |")
    lines.append(f"| **Total for one encoder layer** | **{format_number(HAND_FLOPS['Attention']['flops'])}** |")

    lines.extend(
        [
            "",
            "### Hand calculation vs THOP",
            "",
            "For a fair scope comparison, the representative THOP profiles use the first MLP Linear layer, the first CNN convolution, and one complete Transformer encoder layer. The separate full-model THOP counts cover each entire model. A difference is expected because the hand estimates explicitly count multiply and add as two FLOPs, while THOP uses its own operator-counting handlers; unsupported or functionally expressed operations can be omitted. The comparison is indicative, not a claim that differently defined operation units are identical.",
            "",
            "| Model | Representative layer | Hand calculation (FLOPs/image) | THOP reported ops | Difference / caveat |",
            "|---|---|---:|---:|---|",
        ]
    )
    for model_name in ("MLP", "CNN", "Attention"):
        profile_result = layer_profiles[model_name]
        hand = HAND_FLOPS[model_name]
        if profile_result["ops"] is None:
            thop_text = "Not available"
            caveat = profile_result["error"] or "Profiling failed"
        else:
            thop_text = format_number(profile_result["ops"])
            if model_name == "Attention":
                caveat = "THOP may not count functional QKV projections and attention matrix products; compare conventions before interpreting the gap."
            else:
                caveat = "Differences can come from the THOP convention and whether bias operations are included."
        lines.append(
            f"| {model_name} | {hand['layer']} | {format_number(hand['flops'])} | {thop_text} | {caveat} |"
        )

    lines.extend(["", "## Training history", "", "Final epoch validation results:", "", "| Model | Final training accuracy (%) | Final validation accuracy (%) | Final training loss | Final validation loss |", "|---|---:|---:|---:|---:|"])
    for name in ("MLP", "CNN", "Attention"):
        final_epoch = history[name][-1]
        lines.append(
            f"| {name} | {final_epoch['train_accuracy_pct']:.2f} | "
            f"{final_epoch['validation_accuracy_pct']:.2f} | {final_epoch['train_loss']:.4f} | "
            f"{final_epoch['validation_loss']:.4f} |"
        )

    best_accuracy = max(rows, key=lambda item: item["test_accuracy_pct"])
    best_accuracy_per_parameter = max(
        rows, key=lambda item: item["test_accuracy_pct"] / item["parameters"]
    )
    best_accuracy_per_batch1_ms = max(
        rows, key=lambda item: item["test_accuracy_pct"] / item["latency_batch1_median_ms"]
    )
    best_accuracy_per_batch128_ms = max(
        rows, key=lambda item: item["test_accuracy_pct"] / item["latency_batch128_median_ms"]
    )

    lines.extend(
        [
            "",
            "## Accuracy, cost and access-pattern interpretation",
            "",
            f"- Highest measured test accuracy: **{best_accuracy['model']}** ({best_accuracy['test_accuracy_pct']:.2f}%).",
            f"- Highest test-accuracy-per-parameter ratio (a simple, approximate parameter-efficiency metric): **{best_accuracy_per_parameter['model']}**.",
            f"- Highest accuracy per millisecond of batch-1 latency: **{best_accuracy_per_batch1_ms['model']}**.",
            f"- Highest accuracy per millisecond of batch-128 latency: **{best_accuracy_per_batch128_ms['model']}**. Batch-128 latency is for the whole batch, not per image.",
            "",
            "There is no single architecture that is automatically the most cost-effective for every objective; the measured accuracy, parameter count, and latency should be read together. The MLP uses dense global connections after flattening, which is simple to implement but does not explicitly preserve local image structure. The CNN reuses small convolution kernels across spatial positions, typically improving parameter efficiency and local spatial reuse; pooling reduces later feature-map sizes, although convolutions still read overlapping local windows. The attention model performs projection and feed-forward matrix operations over 16 patch tokens, and self-attention allows each token to interact with the others. Its attention work and intermediate access grow quadratically with sequence length, so patch size and token count matter. Here, the sequence length is only 16, which limits that pairwise attention cost. Actual cost-effectiveness should follow the measured results above rather than parameter count alone.",
            "",
            "## Plots",
            "",
        ]
    )
    for plot in plot_paths:
        lines.append(f"- [{plot}](outputs/{plot})")

    lines.extend(
        [
            "",
            "## Reproducibility notes",
            "",
            "- The split is reproducible with seed 42. Training shuffles the 55,000-image training subset; validation and test loaders do not shuffle.",
            "- Batch size, epoch count, Adam learning rate, transforms and timing repetition counts are fixed to the lab sheet.",
            "- Timing depends on hardware, device, PyTorch version, and system load. Report the device listed above when comparing results.",
            "- THOP and the hand calculation use different operation-counting conventions in some cases. In particular, inspect attention counts for unsupported operations before interpreting them as exact FLOPs.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed()
    device = get_device()
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print("Preparing Fashion-MNIST datasets (first run may download the dataset)...")

    train_loader, validation_loader, test_loader = make_dataloaders(device)
    models = build_models()
    for model in models.values():
        model.to(device)

    full_profiles, layer_profiles, summary_path = profile_models(models, device)
    history, test_accuracies = train_models(
        models,
        train_loader,
        validation_loader,
        test_loader,
        device,
    )

    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        print(f"\nBenchmarking inference latency for {name}...")
        latency = benchmark_latency(model, device)
        full_profile = full_profiles[name]
        representative_profile = layer_profiles[name]
        rows.append(
            {
                "model": name,
                "parameters": count_parameters(model),
                "test_accuracy_pct": test_accuracies[name],
                "hand_flops_per_image": HAND_FLOPS[name]["flops"],
                "thop_full_model_ops": full_profile["ops"],
                "thop_full_model_parameters": full_profile["parameters"],
                "thop_full_model_error": full_profile["error"],
                "thop_representative_layer_ops": representative_profile["ops"],
                "thop_representative_layer_parameters": representative_profile["parameters"],
                "thop_representative_layer_error": representative_profile["error"],
                "latency_batch1_median_ms": latency[1],
                "latency_batch128_median_ms": latency[128],
            }
        )

    results_fields = list(rows[0].keys())
    save_csv(OUTPUT_DIR / "results.csv", rows, results_fields)

    history_rows: list[dict[str, Any]] = []
    for name, model_history in history.items():
        for epoch_result in model_history:
            history_rows.append({"model": name, **epoch_result})
    save_csv(
        OUTPUT_DIR / "training_history.csv",
        history_rows,
        [
            "model",
            "epoch",
            "train_loss",
            "train_accuracy_pct",
            "validation_loss",
            "validation_accuracy_pct",
            "epoch_seconds",
        ],
    )

    plot_paths = create_plots(rows)
    report = build_report(rows, history, layer_profiles, device, plot_paths)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("\nCompleted.")
    print(f"Report: {REPORT_PATH}")
    print(f"CSV metrics: {OUTPUT_DIR / 'results.csv'}")
    print(f"Training history: {OUTPUT_DIR / 'training_history.csv'}")
    print(f"Model summaries: {summary_path}")
    print(f"Plots: {OUTPUT_DIR}")
    print("Review report.md and outputs before committing your work.")


if __name__ == "__main__":
    main()
