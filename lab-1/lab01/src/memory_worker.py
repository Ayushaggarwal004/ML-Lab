"""Measure peak RSS of ONE phase (train | infer) of ONE model in a fresh process.

Called by measure_costs.py as a subprocess, so every measurement starts from a clean
interpreter and the Random Forest cannot inflate the Logistic Regression numbers (or
vice versa). Prints one JSON line on stdout.

torch is deliberately NOT imported here (set_seeds(include_torch=False)): importing it
would add hundreds of MB to the baseline and hide the model's own footprint. Lab 1 does
not use torch, so the seed has no effect on these results.
"""
from __future__ import annotations

import gc
import json
import sys
import threading
import time

import joblib
import psutil

from common import MODEL_FACTORIES, N_INFER_RUNS, load_split, model_path, set_seeds


class PeakRSS:
    """Samples the process RSS every ~1 ms in a background thread and keeps the maximum."""

    def __init__(self, interval: float = 0.001) -> None:
        self.proc = psutil.Process()
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        self.peak = max(self.peak, self.proc.memory_info().rss)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            time.sleep(self.interval)

    def __enter__(self) -> "PeakRSS":
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()


def main() -> None:
    name, phase = sys.argv[1], sys.argv[2]
    set_seeds(include_torch=False)
    X_train, X_test, y_train, _ = load_split()
    sample = X_test[:1]

    gc.collect()
    baseline = psutil.Process().memory_info().rss  # interpreter + libraries + data, before the phase

    if phase == "train":
        model = MODEL_FACTORIES[name]()
        with PeakRSS() as monitor:
            model.fit(X_train, y_train)
    elif phase == "infer":
        with PeakRSS() as monitor:
            model = joblib.load(model_path(name))  # loading the model is part of the inference footprint
            for _ in range(N_INFER_RUNS):
                model.predict(sample)
    else:
        raise SystemExit(f"unknown phase: {phase}")

    print(json.dumps({"baseline_bytes": baseline, "peak_bytes": max(monitor.peak, baseline)}))


if __name__ == "__main__":
    main()
