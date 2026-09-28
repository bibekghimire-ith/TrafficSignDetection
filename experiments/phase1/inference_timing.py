#!/usr/bin/env python3
"""Run from the project root: python experiments/inference_timing.py
Batched and single-image inference timing for the E0 model (forward pass only, preprocessed input)."""
import json, sys, time
import numpy as np
sys.path.insert(0, "src")
from dataset import load_dataset, prep_dataset
from ffnn import predict
from model_utils import load_model
np.random.seed(1)
_, _, tx, ty = load_dataset(size_in_per=100); X, _ = prep_dataset(tx, ty)
P = load_model("models/e0_baseline")["parameters"]
predict(X[:, :256], P)  # warm-up
batch = []
for _ in range(7):
    t = time.perf_counter(); predict(X, P); batch.append(time.perf_counter() - t)
single = []
for i in range(1000):
    x = X[:, i:i + 1]; t = time.perf_counter(); predict(x, P); single.append(time.perf_counter() - t)
out = {"batch_repeats": 7, "batch_median_s": float(np.median(batch)), "batch_us_per_image_median": float(np.median(batch) / X.shape[1] * 1e6),
       "batch_us_per_image_min": float(min(batch) / X.shape[1] * 1e6), "batch_us_per_image_max": float(max(batch) / X.shape[1] * 1e6),
       "single_image_n": 1000, "single_image_median_us": float(np.median(single) * 1e6), "single_image_p95_us": float(np.percentile(single, 95) * 1e6)}
json.dump(out, open("results/e0_inference_timing.json", "w"), indent=1); print(out)
