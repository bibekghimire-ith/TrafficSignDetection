#!/usr/bin/env python3
"""Phase 2 analysis: collate every models/<id>_s<seed> into results/.

Run from the project root:  python experiments/analyze.py

Writes
  results/summary.json            per run: metrics, per-class metrics, history, timings
  results/aggregate.json/.csv     per configuration: mean and sd over seeds
  results/significance.json/.csv  McNemar tests (each config vs E0, same seed) + Welch t-test over seeds
  results/e0_*.csv                baseline detail pooled over seeds
  results/predictions/<run>.npy   test-set predictions (for re-testing without the models)
  results/figures/*.png           every figure used in the report
"""
import csv, glob, json, os, re, sys, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, "src")
from dataset import load_dataset, prep_dataset, train_dev_split, GTSRB_LABELS, NUM_CLASSES
from ffnn import predict
from model_utils import confusion_matrix, model_metrics, load_model

R, F, PRED = "results", "results/figures", "results/predictions"
for d in (F, PRED): os.makedirs(d, exist_ok=True)
# reference categorical palette (fixed order), light surface
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold"})

np.random.seed(1)
_, _, te_x, te_y, _ = load_dataset(size_in_per=100, return_tracks=True)
Xte, _ = prep_dataset(te_x, te_y)
yte = te_y.ravel()
test_support = np.bincount(yte, minlength=NUM_CLASSES)


def train_support_for(seed):
    np.random.seed(seed)
    tx, ty, _, _, tr = load_dataset(size_in_per=100, return_tracks=True)
    _, trs_y, _, dev_y = train_dev_split(tx, ty, tracks=tr)
    return np.bincount(trs_y.ravel(), minlength=NUM_CLASSES), len(trs_y), len(dev_y)


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0; rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
            rows.append((float(lo), float(hi), int(m.sum()), float(conf[m].mean()), float(correct[m].mean())))
    return float(e), rows


def mcnemar(a_ok, b_ok):
    """Exact McNemar test on paired correctness vectors. Returns (b01, b10, p)."""
    b01 = int(np.sum(a_ok & ~b_ok)); b10 = int(np.sum(~a_ok & b_ok))
    n = b01 + b10
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b01, b10), n, 0.5)))
    return b01, b10, p


# ------------------------------------------------------------------ per run
runs = {}
supports = {}
for path in sorted(glob.glob("models/*_s[0-9]*")):
    run = os.path.basename(path)
    cfg, seed = re.match(r"(.+)_s(\d+)$", run).groups(); seed = int(seed)
    if seed not in supports: supports[seed] = train_support_for(seed)
    model = load_model(path); P = model["parameters"]; h = model["history"]
    pr = predict(Xte, P, second_guess=True)
    p1 = pr["First Prediction"][0].ravel(); c1 = pr["First Prediction"][1].ravel(); p2 = pr["Second Prediction"][0].ravel()
    np.save("%s/%s.npy" % (PRED, run), np.stack([p1, p2]).astype(np.int8))
    cm = confusion_matrix(te_y, pr, num_classes=NUM_CLASSES)
    met, mac, acc = model_metrics(cm)
    ok = p1 == yte
    e, calib = ece(c1.astype(float), ok.astype(float))
    tm = json.load(open("results/logs/%s.time.json" % run))
    log = open("results/logs/%s.train.log" % run, errors="ignore").read()
    best = int(np.argmax(h["val_accuracy"])) + 1
    ts, n_tr, n_dev = supports[seed]
    f1 = np.array(met["F1-Score"])
    off = cm.copy(); np.fill_diagonal(off, 0)
    runs[run] = {
        "run": run, "config": cfg, "seed": seed, "layers_dim": model["layers_dim"], "n_params": int(sum(v.size for v in P.values())),
        "n_train": n_tr, "n_dev": n_dev,
        "test_top1": float(acc), "test_top2": float(np.mean(ok | (p2 == yte))),
        "macro_precision": float(mac["Precision"]), "macro_recall": float(mac["Recall"]), "macro_f1": float(mac["F1-Score"]),
        "best_dev_acc": float(max(h["val_accuracy"])), "best_epoch": best, "epochs_completed": len(h["accuracy"]),
        "train_acc_at_best": float(h["accuracy"][best - 1]), "gap_at_best": float(h["accuracy"][best - 1] - max(h["val_accuracy"])),
        "dev_minus_test": float(max(h["val_accuracy"]) - acc), "early_stopped": "halted" in log,
        "train_wall_s": tm["wall_s"], "peak_rss_mb": tm["peak_rss_mb"],
        "mean_conf_correct": float(c1[ok].mean()), "mean_conf_incorrect": float(c1[~ok].mean()), "ece_15bin": e,
        "errors_total": int(off.sum()),
        "corr_trainsupport_f1_pearson": float(np.corrcoef(ts, f1)[0, 1]),
        "corr_trainsupport_f1_spearman": float(stats.spearmanr(ts, f1)[0]),
        "per_class": [{"class": k, "name": GTSRB_LABELS[k], "train_support": int(ts[k]), "test_support": int(test_support[k]),
                       "precision": float(met["Precision"][k]), "recall": float(met["Recall"][k]), "f1": float(met["F1-Score"][k])}
                      for k in range(NUM_CLASSES)],
        "history": {k: [float(x) for x in v] for k, v in h.items()},
        "calibration_bins": calib,
    }
    np.save("%s/cm_%s.npy" % (PRED, run), cm.astype(np.int32))
    print("%-28s top1 %.4f  macroF1 %.4f  ep %2d  %.0fs" % (run, acc, mac["F1-Score"], len(h["accuracy"]), tm["wall_s"]))
json.dump(runs, open(R + "/summary.json", "w"), indent=1)

# ------------------------------------------------------------------ aggregate over seeds
KEYS = ["test_top1", "test_top2", "macro_f1", "macro_precision", "macro_recall", "best_dev_acc", "dev_minus_test",
        "gap_at_best", "epochs_completed", "best_epoch", "train_wall_s", "peak_rss_mb", "ece_15bin",
        "mean_conf_correct", "mean_conf_incorrect", "n_params"]
cfgs = sorted({r["config"] for r in runs.values()})
agg = {}
for c in cfgs:
    rs = [r for r in runs.values() if r["config"] == c]
    agg[c] = {"n_seeds": len(rs), "seeds": sorted(r["seed"] for r in rs)}
    for k in KEYS:
        v = np.array([r[k] for r in rs], dtype=float)
        agg[c][k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "values": v.tolist()}
    agg[c]["per_class_f1_mean"] = np.mean([[pc["f1"] for pc in r["per_class"]] for r in rs], axis=0).tolist()
json.dump(agg, open(R + "/aggregate.json", "w"), indent=1)
with open(R + "/aggregate.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["config", "n_seeds"] + sum([[k + "_mean", k + "_sd"] for k in KEYS], []))
    for c in cfgs: w.writerow([c, agg[c]["n_seeds"]] + sum([[agg[c][k]["mean"], agg[c][k]["sd"]] for k in KEYS], []))

# ------------------------------------------------------------------ significance vs E0
sig = {}
for c in cfgs:
    if c == "e0_baseline": continue
    per_seed = []
    for s in agg[c]["seeds"]:
        a, b = "e0_baseline_s%d" % s, "%s_s%d" % (c, s)
        if a not in runs: continue
        pa = np.load("%s/%s.npy" % (PRED, a))[0]; pb = np.load("%s/%s.npy" % (PRED, b))[0]
        b01, b10, p = mcnemar(pa == yte, pb == yte)
        per_seed.append({"seed": s, "e0_right_other_wrong": b01, "e0_wrong_other_right": b10, "p": p})
    x = agg["e0_baseline"]["macro_f1"]["values"]; y = agg[c]["macro_f1"]["values"]
    t = stats.ttest_ind(y, x, equal_var=False) if len(x) > 1 and len(y) > 1 else None
    sig[c] = {"mcnemar": per_seed, "seeds_p_below_0.05": int(sum(r["p"] < 0.05 for r in per_seed)),
              "macro_f1_diff_mean": float(np.mean(y) - np.mean(x)),
              "welch_t_macro_f1_p": (float(t.pvalue) if t is not None and np.isfinite(t.pvalue) else None)}
# noise floor: the same configuration (E0) trained with different seeds, compared the same way
e0s = sorted(agg["e0_baseline"]["seeds"]); floor = []
for i in range(len(e0s)):
    for j in range(i + 1, len(e0s)):
        pa = np.load("%s/e0_baseline_s%d.npy" % (PRED, e0s[i]))[0]; pb = np.load("%s/e0_baseline_s%d.npy" % (PRED, e0s[j]))[0]
        b01, b10, p = mcnemar(pa == yte, pb == yte)
        floor.append({"seeds": [e0s[i], e0s[j]], "discordant": b01 + b10, "b01": b01, "b10": b10, "p": p})
sig["_e0_seed_vs_seed"] = floor
json.dump(sig, open(R + "/significance.json", "w"), indent=1)
with open(R + "/significance.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["config", "macro_f1_diff_vs_e0", "welch_p", "mcnemar_p_per_seed", "seeds_p<0.05"])
    for c, v in sig.items():
        if c.startswith("_"): continue
        w.writerow([c, v["macro_f1_diff_mean"], v["welch_t_macro_f1_p"], ";".join("%.2g" % r["p"] for r in v["mcnemar"]), v["seeds_p_below_0.05"]])

# ------------------------------------------------------------------ baseline detail (pooled over seeds)
e0 = [r for r in runs.values() if r["config"] == "e0_baseline"]
pcf = {k: np.array([[r["per_class"][c][k] for c in range(NUM_CLASSES)] for r in e0]) for k in ("precision", "recall", "f1", "train_support")}
with open(R + "/e0_per_class_metrics.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["class", "name", "train_support_mean", "test_support", "precision_mean", "recall_mean", "f1_mean", "f1_sd"])
    for c in range(NUM_CLASSES):
        w.writerow([c, GTSRB_LABELS[c], pcf["train_support"][:, c].mean(), test_support[c], pcf["precision"][:, c].mean(),
                    pcf["recall"][:, c].mean(), pcf["f1"][:, c].mean(), pcf["f1"][:, c].std(ddof=1)])
cm_sum = sum(np.load("%s/cm_%s.npy" % (PRED, r["run"])) for r in e0)
off = cm_sum.copy(); np.fill_diagonal(off, 0)
idx = np.dstack(np.unravel_index(np.argsort(-off.ravel()), off.shape))[0][:10]
with open(R + "/e0_top_confusions.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["count_over_seeds", "mean_per_seed", "true", "true_name", "pred", "pred_name"])
    for i, j in idx: w.writerow([off[i, j], off[i, j] / len(e0), i, GTSRB_LABELS[i], j, GTSRB_LABELS[j]])

# ------------------------------------------------------------------ figures
def save(fig, name): fig.tight_layout(); fig.savefig(F + "/" + name); plt.close(fig)

# E0 learning curves: every seed, train and dev
for key, vkey, name, fn in [("loss", "val_loss", "Loss", "fig_e0_loss.png"), ("accuracy", "val_accuracy", "Accuracy", "fig_e0_accuracy.png")]:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for r in sorted(e0, key=lambda r: r["seed"]):
        h = r["history"]; ep = np.arange(1, len(h[key]) + 1)
        ax.plot(ep, h[key], color=C[0], lw=1.4, alpha=0.9, label="train" if r["seed"] == 1 else None)
        ax.plot(ep, h[vkey], color=C[1], lw=1.4, alpha=0.9, label="dev (track-disjoint)" if r["seed"] == 1 else None)
    ax.set_xlabel("Epoch"); ax.set_ylabel(name); ax.set_title("E0 baseline, %s (3 seeds)" % name.lower()); ax.legend()
    save(fig, fn)

cmn = cm_sum / cm_sum.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(8.5, 7.6))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); fig.colorbar(im, fraction=.046, label="Share of true class")
ax.set_xticks(range(43)); ax.set_yticks(range(43)); ax.tick_params(labelsize=6); ax.grid(False)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title("E0 test confusion matrix, pooled over 3 seeds (row-normalised)")
save(fig, "fig_e0_confusion_matrix.png")

f1m, f1s = pcf["f1"].mean(0), pcf["f1"].std(0, ddof=1)
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.bar(range(43), f1m, color=C[0], width=0.8, yerr=f1s, error_kw={"ecolor": INK2, "lw": 0.8, "capsize": 1.5})
ax.set_xticks(range(43)); ax.tick_params(axis="x", labelsize=7); ax.set_ylim(0, 1.02)
ax.set_xlabel("Class id"); ax.set_ylabel("Test F1"); ax.set_title("E0 per-class test F1 (mean ± sd over 3 seeds)"); ax.grid(axis="x", visible=False)
save(fig, "fig_e0_per_class_f1.png")

sup = pcf["train_support"].mean(0)
r_p = float(np.corrcoef(sup, f1m)[0, 1]); r_s = float(stats.spearmanr(sup, f1m)[0])
fig, ax = plt.subplots(figsize=(5.8, 4))
ax.scatter(sup, f1m, s=22, color=C[0], edgecolor="white", linewidth=0.8)
for c in np.argsort(f1m)[:8]: ax.annotate(str(c), (sup[c], f1m[c]), fontsize=7, color=INK2, xytext=(3, 2), textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("Training images in class (log scale)"); ax.set_ylabel("Test F1 (mean of 3 seeds)")
ax.set_title("E0: class size vs F1 (Pearson r = %.2f, Spearman ρ = %.2f)" % (r_p, r_s))
save(fig, "fig_e0_support_vs_f1.png")
json.dump({"pearson": r_p, "spearman": r_s}, open(R + "/e0_support_f1_correlation.json", "w"))

conf_all = []; ok_all = []
for r in e0:
    m = load_model("models/" + r["run"]); pr = predict(Xte, m["parameters"])
    conf_all.append(pr["First Prediction"][1].ravel()); ok_all.append(pr["First Prediction"][0].ravel() == yte)
conf_all = np.concatenate(conf_all).astype(float); ok_all = np.concatenate(ok_all).astype(float)
e_pool, cb = ece(conf_all, ok_all)
fig, ax = plt.subplots(figsize=(4.2, 4))
ax.plot([0, 1], [0, 1], ls="--", color=INK2, lw=1); ax.plot([r[3] for r in cb], [r[4] for r in cb], "o-", color=C[0], lw=2, ms=5)
ax.set_xlabel("Mean confidence in bin"); ax.set_ylabel("Accuracy in bin"); ax.set_title("E0 reliability, 3 seeds pooled (ECE = %.3f)" % e_pool)
save(fig, "fig_e0_reliability.png")

s1 = [r for r in e0 if r["seed"] == 1][0]
p1 = np.load("%s/%s.npy" % (PRED, s1["run"]))[0]; wrong = np.where(p1 != yte)[0][:24]
fig, axs = plt.subplots(3, 8, figsize=(12, 5.2))
for a, i in zip(axs.ravel(), wrong):
    a.imshow(te_x[i], cmap="gray"); a.set_title("T:%d  P:%d" % (yte[i], p1[i]), fontsize=8); a.axis("off")
fig.suptitle("E0 seed 1: first 24 misclassified test images after preprocessing (T = true, P = predicted)")
save(fig, "fig_e0_misclassified.png")

# ablation bar charts: mean ± sd macro-F1 per group
GROUPS = [("E1 initialization", [("e0_baseline", "He (E0)"), ("e1_random", "Random ×0.01")]),
          ("E2 optimizer", [("e0_baseline", "Adam (E0)"), ("e2_mgd", "MGD lr 0.001"), ("e2_mgd_lr0.05", "MGD lr 0.05"), ("e2_bgd", "BGD")]),
          ("E3 regularization", [("e3_dropout_0.6", "Dropout 0.6"), ("e3_dropout_0.7", "Dropout 0.7"), ("e0_baseline", "Dropout 0.8 (E0)"),
                                 ("e3_dropout_0.9", "Dropout 0.9"), ("e3_l2_1", "L2 λ=1"), ("e3_l2_3", "L2 λ=3"), ("e3_l2_10", "L2 λ=10"),
                                 ("e3_l2_30", "L2 λ=30"), ("e3_none", "None")]),
          ("E4 architecture", [("e4_256", "[256]"), ("e0_baseline", "[512,256] (E0)"), ("e4_1024_512_256", "[1024,512,256]"),
                               ("e4_1024_512_256_128", "[1024,512,256,128]")]),
          ("E5 augmentation", [("e0_baseline", "Original (E0)"), ("e5_augmented", "+5 augmented variants")]),
          ("E6 learning-rate decay", [("e0_baseline", "Constant (E0)"), ("e6_decay10", "Halve every 10 epochs")])]
for title, items in GROUPS:
    items = [(c, l) for c, l in items if c in agg]
    if len(items) < 2: continue
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(items) + 1.1))
    y = np.arange(len(items))[::-1]
    mean = [agg[c]["macro_f1"]["mean"] for c, _ in items]; sd = [agg[c]["macro_f1"]["sd"] for c, _ in items]
    col = [C[1] if c == "e0_baseline" else C[0] for c, _ in items]
    ax.barh(y, mean, xerr=sd, color=col, height=0.62, error_kw={"ecolor": INK, "lw": 1, "capsize": 3})
    for yy, (c, _) in zip(y, items):
        ax.scatter(agg[c]["macro_f1"]["values"], [yy] * len(agg[c]["macro_f1"]["values"]), s=10, color=INK, zorder=3)
        ax.text(agg[c]["macro_f1"]["mean"] + agg[c]["macro_f1"]["sd"] + 0.005, yy, "%.3f" % agg[c]["macro_f1"]["mean"], va="center", fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in items])
    lo = max(0, min(m - s for m, s in zip(mean, sd)) - 0.08)
    ax.set_xlim(lo if lo > 0.5 else 0, 1.0); ax.set_xlabel("Test macro-F1 (bar = mean, whisker = sd, dots = seeds)")
    ax.set_title(title + ": test macro-F1 over 3 seeds"); ax.grid(axis="y", visible=False)
    save(fig, "abl_%s.png" % title.split()[0])

# dev-accuracy curves, seed 1, per group
for title, items in GROUPS:
    items = [(c, l) for c, l in items if c + "_s1" in runs]
    if len(items) < 2: continue
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for k, (c, l) in enumerate(items):
        v = runs[c + "_s1"]["history"]["val_accuracy"]; ax.plot(range(1, len(v) + 1), v, color=C[k % 8], lw=1.6, label=l)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dev accuracy"); ax.set_title(title + ": dev accuracy per epoch (seed 1)"); ax.legend(fontsize=7, ncol=2)
    save(fig, "curves_%s.png" % title.split()[0])
print("analysis done:", len(runs), "runs,", len(cfgs), "configs")
