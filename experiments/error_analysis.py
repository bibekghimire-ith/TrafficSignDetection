#!/usr/bin/env python3
"""Run from the project root: python experiments/error_analysis.py  (after experiments/analyze.py)

E0 error analysis pooled over seeds: accuracy by original sign size and by sharpness, top-2 gains, within-group
confusions, and confidence-rejection thresholds. Needs GT-final_test.csv (shipped in GTSRB_Final_Test_GT.zip).
"""
import csv, glob, json, os, sys
import numpy as np
sys.path.insert(0, "src")
from dataset import load_dataset, prep_dataset, GTSRB_LABELS
from ffnn import predict
from model_utils import load_model

GT = next(p for p in ["dataset/gtsrb/GT-final_test.csv", "dataset/gtsrb/GTSRB/Final_Test/Images/GT-final_test.csv",
                      "dataset/GT-final_test.csv", "/tmp/gt/GT-final_test.csv"] if os.path.exists(p))
rows = list(csv.DictReader(open(GT), delimiter=";"))
np.random.seed(1)
_, _, te_x, te_y = load_dataset(size_in_per=100)
np.random.seed(1); np.random.permutation(39209); perm = np.random.permutation(12630)   # same order as load_dataset
cls = np.array([int(r["ClassId"]) for r in rows])[perm]
assert (cls == te_y.ravel()).all(), "test-set order mismatch"
w = np.array([int(r["Roi.X2"]) - int(r["Roi.X1"]) for r in rows])[perm]
h = np.array([int(r["Roi.Y2"]) - int(r["Roi.Y1"]) for r in rows])[perm]
roi = np.sqrt(w * h)
X, _ = prep_dataset(te_x, te_y); y = te_y.ravel()
img = te_x.astype(float)
lap = img[:, 1:-1, 1:-1] * 4 - img[:, :-2, 1:-1] - img[:, 2:, 1:-1] - img[:, 1:-1, :-2] - img[:, 1:-1, 2:]
sharp = lap.reshape(len(y), -1).var(1)

oks, ok2s, confs, p1s = [], [], [], []
for path in sorted(glob.glob("models/e0_baseline_s*")):
    pr = predict(X, load_model(path)["parameters"], second_guess=True)
    p1 = pr["First Prediction"][0].ravel(); p2 = pr["Second Prediction"][0].ravel()
    oks.append(p1 == y); ok2s.append((p1 == y) | (p2 == y)); confs.append(pr["First Prediction"][1].ravel().astype(float)); p1s.append(p1)
ok = np.stack(oks); ok2 = np.stack(ok2s); conf = np.stack(confs); P1 = np.stack(p1s); S = len(oks)

out = {"n_seeds": S, "roi_side_px_median_correct": float(np.median(np.tile(roi, (S, 1))[ok])),
       "roi_side_px_median_wrong": float(np.median(np.tile(roi, (S, 1))[~ok]))}
bins = [0, 30, 40, 50, 70, 1000]; out["acc_by_roi_size"] = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (roi >= lo) & (roi < hi)
    a = ok[:, m].mean(1)
    out["acc_by_roi_size"].append({"roi_px": "%d–%s" % (lo, hi if hi < 1000 else "max"), "n": int(m.sum()), "acc_mean": float(a.mean()), "acc_sd": float(a.std(ddof=1))})
q = np.quantile(sharp, [0, .25, .5, .75, 1]); out["acc_by_sharpness_quartile"] = []
for i in range(4):
    m = (sharp >= q[i]) & (sharp <= q[i + 1]); a = ok[:, m].mean(1)
    out["acc_by_sharpness_quartile"].append({"quartile": i + 1, "n": int(m.sum()), "acc_mean": float(a.mean()), "acc_sd": float(a.std(ddof=1))})
gains = []
for k in range(43):
    m = y == k
    gains.append({"class": k, "name": GTSRB_LABELS[k], "top1": float(ok[:, m].mean()), "top2": float(ok2[:, m].mean()),
                  "gain": float(ok2[:, m].mean() - ok[:, m].mean())})
out["top2_gain_by_class"] = sorted(gains, key=lambda g: -g["gain"])[:10]
groups = {"speed_limits_0_8": set(range(0, 9)), "triangular_warning_18_31": set(range(18, 32)),
          "blue_mandatory_33_40": set(range(33, 41)), "end_of_restriction_6_32_41_42": {6, 32, 41, 42}}
tot = int((~ok).sum()); out["total_errors_all_seeds"] = tot; out["group_confusions"] = {}
Y = np.tile(y, (S, 1))
for g, sset in groups.items():
    t_in = np.isin(Y, list(sset)); p_in = np.isin(P1, list(sset))
    out["group_confusions"][g] = {"errors_true_in_group": int((~ok & t_in).sum()), "errors_within_group": int((~ok & t_in & p_in).sum())}
out["confidence_rejection"] = []
for t in [0.5, 0.7, 0.8, 0.9]:
    acc_ = conf >= t
    out["confidence_rejection"].append({"threshold": t, "coverage": float(acc_.mean()), "acc_on_accepted": float(ok[acc_].mean()),
                                        "errors_rejected": float((~acc_ & ~ok).sum() / (~ok).sum()),
                                        "correct_rejected": float((~acc_ & ok).sum() / ok.sum())})
json.dump(out, open("results/e0_error_analysis.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("roi_side_px_median_correct", "roi_side_px_median_wrong", "total_errors_all_seeds")}))
