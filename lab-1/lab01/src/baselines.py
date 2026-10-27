"""Step 2: train the two baselines and record test accuracy (4 decimals)."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score

from common import MODEL_FACTORIES, RESULTS_DIR, load_split, set_seeds


def main() -> None:
    set_seeds()
    X_train, X_test, y_train, y_test = load_split()
    print(f"train: {X_train.shape}  test: {X_test.shape}")

    rows = []
    for name, make in MODEL_FACTORIES.items():
        model = make().fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"{name:<20} test accuracy = {acc:.4f}")
        rows.append({"model": name, "test_accuracy": f"{acc:.4f}"})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "baseline_accuracy.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
