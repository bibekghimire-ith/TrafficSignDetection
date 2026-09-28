#!/usr/bin/env python3
"""Run from the project root: python experiments/error_analysis.py
E0 error analysis: which image properties go with misclassification; top-2 gains per class."""
import csv, json, sys
import numpy as np
sys.path.insert(0, "src")
from dataset import load_dataset, prep_dataset, GTSRB_LABELS
from ffnn import predict
from model_utils import load_model

import os
GT = next(p for p in ["dataset/gtsrb/GT-final_test.csv", "dataset/gtsrb/GTSRB/Final_Test/Images/GT-final_test.csv",
                      "dataset/GT-final_test.csv", "/tmp/gt/GT-final_test.csv"] if os.path.exists(p))
rows = list(csv.DictReader(open(GT), delimiter=";"))
np.random.seed(1)
_, _, te_x, te_y = load_dataset(size_in_per=100)
# reproduce load_dataset's permutations: train permutation first, then test
np.random.seed(1); np.random.permutation(39209); perm = np.random.permutation(12630)
cls = np.array([int(r["ClassId"]) for r in rows])[perm]
assert (cls == te_y.ravel()).all(), "permutation mismatch"
w = np.array([int(r["Roi.X2"]) - int(r["Roi.X1"]) for r in rows])[perm]
h = np.array([int(r["Roi.Y2"]) - int(r["Roi.Y1"]) for r in rows])[perm]
roi = np.sqrt(w * h)

X, _ = prep_dataset(te_x, te_y)
P = load_model("models/e0_baseline")["parameters"]
pr = predict(X, P, second_guess=True)
p1 = pr["First Prediction"][0].ravel(); p2 = pr["Second Prediction"][0].ravel(); y = te_y.ravel()
ok = p1 == y
img = te_x.astype(float)
lap = img[:, 1:-1, 1:-1] * 4 - img[:, :-2, 1:-1] - img[:, 2:, 1:-1] - img[:, 1:-1, :-2] - img[:, 1:-1, 2:]
sharp = lap.reshape(len(y), -1).var(1)

out = {"roi_side_px_median_correct": float(np.median(roi[ok])), "roi_side_px_median_wrong": float(np.median(roi[~ok])),
       "sharpness_median_correct": float(np.median(sharp[ok])), "sharpness_median_wrong": float(np.median(sharp[~ok]))}
# accuracy by ROI size quartile-ish bins
bins = [0, 30, 40, 50, 70, 1000]; tab = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (roi >= lo) & (roi < hi); tab.append({"roi_px": "%d–%s" % (lo, hi if hi < 1000 else "max"), "n": int(m.sum()), "acc": float(ok[m].mean())})
out["acc_by_roi_size"] = tab
q = np.quantile(sharp, [0, .25, .5, .75, 1]); tab = []
for i in range(4):
    m = (sharp >= q[i]) & (sharp <= q[i + 1]); tab.append({"sharpness_quartile": i + 1, "n": int(m.sum()), "acc": float(ok[m].mean())})
out["acc_by_sharpness_quartile"] = tab
top2 = ok | (p2 == y); gains = []
for k in range(43):
    m = y == k; gains.append({"class": k, "name": GTSRB_LABELS[k], "top1": float(ok[m].mean()), "top2": float(top2[m].mean()), "gain": float(top2[m].mean() - ok[m].mean())})
out["top2_gain_by_class"] = sorted(gains, key=lambda g: -g["gain"])[:10]
groups = {"speed_limits_0_8": set(range(0, 9)), "triangular_warning_18_31": set(range(18, 32)), "blue_mandatory_33_40": set(range(33, 41)), "end_of_restriction_6_32_41_42": {6, 32, 41, 42}}
off = ~ok; tot = int(off.sum()); g_out = {}
for g, s in groups.items():
    within = int(sum(1 for a, b in zip(y[off], p1[off]) if a in s and b in s))
    g_out[g] = {"errors_true_in_group": int(sum(1 for a in y[off] if a in s)), "errors_within_group": within}
c1 = pr["First Prediction"][1].ravel(); th = []
for t in [0.5, 0.7, 0.8, 0.9]:
    th.append({"threshold": t, "coverage": float((c1 >= t).mean()), "acc_on_accepted": float(ok[c1 >= t].mean()),
               "errors_rejected": float((c1[~ok] < t).mean()), "correct_rejected": float((c1[ok] < t).mean())})
out["confidence_rejection"] = th
out["group_confusions"] = g_out; out["total_errors"] = tot
json.dump(out, open("results/e0_error_analysis.json", "w"), indent=1)
print(json.dumps(out, indent=1))
