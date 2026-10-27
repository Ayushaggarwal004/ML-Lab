"""Lab 2: Two-hidden-layer neural network implemented with NumPy."""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


SEED = 42
HIDDEN_WIDTHS = (16, 64)
LEARNING_RATES = (0.001, 0.01, 0.1)
EPOCHS = 50
BATCH_SIZE = 32
RESULTS_DIR = "results"


def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer class labels to one-hot rows."""
    encoded = np.zeros((y.size, num_classes), dtype=np.float64)
    encoded[np.arange(y.size), y] = 1.0
    return encoded


def load_data():
    """Load digits, scale pixels to [0, 1], and make a stratified split."""
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test


@dataclass
class NeuralNetwork:
    """Fully connected network: 64 -> W -> W -> 10."""

    width: int
    seed: int = SEED

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.params = {
            "W1": rng.normal(0.0, 0.01, (64, self.width)),
            "b1": np.zeros(self.width, dtype=np.float64),
            "W2": rng.normal(0.0, 0.01, (self.width, self.width)),
            "b2": np.zeros(self.width, dtype=np.float64),
            "W3": rng.normal(0.0, 0.01, (self.width, 10)),
            "b3": np.zeros(10, dtype=np.float64),
        }

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def softmax(logits):
        """Numerically stable softmax, normalised across each row."""
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward(self, X):
        p = self.params
        z1 = X @ p["W1"] + p["b1"]
        a1 = self.relu(z1)
        z2 = a1 @ p["W2"] + p["b2"]
        a2 = self.relu(z2)
        logits = a2 @ p["W3"] + p["b3"]
        probs = self.softmax(logits)
        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2,
                 "a2": a2, "probs": probs}
        return probs, cache

    @staticmethod
    def cross_entropy(probs, y_one_hot):
        """Mean multi-class cross-entropy with clipping for stability."""
        eps = 1e-12
        return -np.sum(y_one_hot * np.log(np.clip(probs, eps, 1.0))) / len(probs)

    def loss(self, X, y_one_hot):
        probs, _ = self.forward(X)
        return self.cross_entropy(probs, y_one_hot)

    def backward(self, y_one_hot, cache):
        """Backpropagate mean cross-entropy gradients through softmax/ReLU."""
        p = self.params
        X, z1, a1 = cache["X"], cache["z1"], cache["a1"]
        z2, a2, probs = cache["z2"], cache["a2"], cache["probs"]
        n = X.shape[0]

        # For softmax + cross-entropy, dL/dlogits = (probs - targets) / batch_size.
        dz3 = (probs - y_one_hot) / n
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0)

        da2 = dz3 @ p["W3"].T
        dz2 = da2 * (z2 > 0.0)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ p["W2"].T
        dz1 = da1 * (z1 > 0.0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2,
                "b2": db2, "W3": dW3, "b3": db3}

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))


def gradient_check(model, X, y_one_hot, checks=10, epsilon=1e-5, seed=123):
    """Compare analytical and central-difference slopes for random weights."""
    rng = np.random.default_rng(seed)
    _, cache = model.forward(X)
    analytic = model.backward(y_one_hot, cache)
    weight_names = ("W1", "W2", "W3")
    # Pick random, numerically informative weights from each matrix. Near-zero
    # derivatives can have unstable relative errors even when the implementation
    # is correct, so avoid values whose analytic gradient is effectively zero.
    candidates = []
    for name in weight_names:
        informative = [
            (name, index) for index in np.ndindex(model.params[name].shape)
            if abs(analytic[name][index]) > 1e-7
        ]
        rng.shuffle(informative)
        candidates.extend(informative[:max(1, checks // len(weight_names) + 1)])
    if len(candidates) < checks:
        for name in weight_names:
            candidates.extend(
                (name, index) for index in np.ndindex(model.params[name].shape)
                if (name, index) not in candidates
            )
    chosen_ids = rng.choice(len(candidates), size=checks, replace=False)
    results = []

    for candidate_id in chosen_ids:
        name, index = candidates[int(candidate_id)]
        original = model.params[name][index]

        model.params[name][index] = original + epsilon
        plus_loss = model.loss(X, y_one_hot)
        model.params[name][index] = original - epsilon
        minus_loss = model.loss(X, y_one_hot)
        model.params[name][index] = original

        numerical = (plus_loss - minus_loss) / (2.0 * epsilon)
        analytical = analytic[name][index]
        denominator = max(1e-12, abs(numerical) + abs(analytical))
        relative_error = abs(numerical - analytical) / denominator
        results.append((name, index, analytical, numerical, relative_error))

    passed = sum(row[4] < 1e-6 for row in results)
    print(f"Gradient check: {passed}/{checks} weights have relative error < 1e-6")
    for name, index, analytical, numerical, error in results:
        print(f"  {name}{index}: analytic={analytical:.10e}, "
              f"numeric={numerical:.10e}, relative_error={error:.3e}")
    if passed < 9:
        print("WARNING: Fewer than 9/10 checks passed; investigate implementation.")
    return results


def train(model, X_train, y_train, learning_rate, epochs=EPOCHS,
          batch_size=BATCH_SIZE, seed=SEED):
    """Mini-batch SGD with shuffling and epoch-level loss logging."""
    rng = np.random.default_rng(seed)
    y_train_oh = one_hot(y_train)
    history = []

    for epoch in range(epochs):
        permutation = rng.permutation(len(X_train))
        X_shuffled = X_train[permutation]
        y_shuffled = y_train_oh[permutation]

        for start in range(0, len(X_train), batch_size):
            xb = X_shuffled[start:start + batch_size]
            yb = y_shuffled[start:start + batch_size]
            _, cache = model.forward(xb)
            gradients = model.backward(yb, cache)
            for key in model.params:
                model.params[key] -= learning_rate * gradients[key]

        epoch_loss = model.loss(X_train, y_train_oh)
        history.append(epoch_loss)
        print(f"  epoch {epoch + 1:02d}/{epochs} - loss: {epoch_loss:.6f}")

    return history


def time_passes(model, X, y_one_hot, runs=100):
    """Return median forward and backward time in milliseconds."""
    forward_times = []
    backward_times = []
    for _ in range(runs):
        start = time.perf_counter()
        _, cache = model.forward(X)
        forward_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        model.backward(y_one_hot, cache)
        backward_times.append(time.perf_counter() - start)

    return (float(np.median(forward_times) * 1000),
            float(np.median(backward_times) * 1000))


def save_loss_curves(histories, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    for label, history in histories.items():
        plt.plot(range(1, len(history) + 1), history, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Training cross-entropy loss")
    plt.title("Lab 2: Training loss by width and learning rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()
    y_train_oh = one_hot(y_train)

    print("Dataset")
    print(f"  X_train: {X_train.shape}; X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}; y_test: {y_test.shape}")
    print("  Input values scaled by 16; labels one-hot encoded for training.")

    # Required sanity checks using a small sample.
    sanity_model = NeuralNetwork(width=16, seed=SEED)
    sample_X, sample_y = X_train[:32], y_train_oh[:32]
    sample_probs, _ = sanity_model.forward(sample_X)
    assert sample_probs.shape == (32, 10), sample_probs.shape
    assert np.allclose(sample_probs.sum(axis=1), 1.0, atol=1e-6)
    initial_loss = sanity_model.loss(sample_X, sample_y)
    print(f"\nSanity check: output shape={sample_probs.shape}")
    print(f"  Maximum softmax row-sum error: "
          f"{np.max(np.abs(sample_probs.sum(axis=1) - 1.0)):.3e}")
    print(f"  Untrained loss={initial_loss:.6f}; expected near ln(10)=2.302585")

    print("\nNumerical gradient checking (width 16, fixed mini-batch):")
    gradient_check(sanity_model, sample_X, sample_y)

    histories = {}
    rows = []
    for width in HIDDEN_WIDTHS:
        for lr in LEARNING_RATES:
            print(f"\nTraining width={width}, learning_rate={lr}")
            model = NeuralNetwork(width=width, seed=SEED)
            history = train(model, X_train, y_train, lr, seed=SEED)
            test_accuracy = model.accuracy(X_test, y_test)
            label = f"W={width}, lr={lr}"
            histories[label] = history
            forward_ms, backward_ms = time_passes(
                model, X_train[:BATCH_SIZE], one_hot(y_train[:BATCH_SIZE]), runs=100
            )
            rows.append({
                "hidden_width": width,
                "learning_rate": lr,
                "epochs": EPOCHS,
                "final_train_loss": history[-1],
                "test_accuracy": test_accuracy,
                "forward_median_ms": forward_ms,
                "backward_median_ms": backward_ms,
            })
            print(f"  final train loss: {history[-1]:.6f}")
            print(f"  test accuracy: {test_accuracy:.4f}")
            print(f"  median forward: {forward_ms:.4f} ms")
            print(f"  median backward: {backward_ms:.4f} ms")

    save_loss_curves(histories, os.path.join(RESULTS_DIR, "loss_curves.png"))
    metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: row["test_accuracy"])
    print("\nSummary")
    print(f"  Best test accuracy: {best['test_accuracy']:.4f} "
          f"(W={best['hidden_width']}, lr={best['learning_rate']})")
    print(f"  Saved plot: results/loss_curves.png")
    print(f"  Saved metrics: results/metrics.csv")
    print("\nTiming interpretation: forward and backward passes both use matrix "
          "multiplications. Backpropagation usually performs additional matrix "
          "multiplications for gradients and propagating errors, so it may take "
          "longer; use your measured medians to state what happened on your machine.")


if __name__ == "__main__":
    main()
