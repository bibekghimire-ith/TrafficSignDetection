#!/usr/bin/env python3
"""Run from the project root: python experiments/analyze.py
Collate every trained model in models/ into results/: metrics, tables, figures."""
import glob, json, os, sys, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
from dataset import load_dataset, prep_dataset, train_dev_split, GTSRB_LABELS, NUM_CLASSES
from ffnn import predict, forward_prop
from model_utils import confusion_matrix, model_metrics, load_model

R = "results"; F = R + "/figures"; os.makedirs(F, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

# identical data path to evaluate.py / train.py (seed 1)
np.random.seed(1)
tr_x, tr_y, te_x, te_y = load_dataset(size_in_per=100)
_, trs_y, _, dev_y = train_dev_split(tr_x, tr_y)
Xte, _ = prep_dataset(te_x, te_y)
yte = te_y.ravel()
train_support = np.bincount(trs_y.ravel(), minlength=NUM_CLASSES)
test_support = np.bincount(yte, minlength=NUM_CLASSES)

def pearson(a, b): return float(np.corrcoef(a, b)[0, 1])
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return pearson(ra, rb)

def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0; rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
            rows.append((lo, hi, int(m.sum()), float(conf[m].mean()), float(correct[m].mean())))
    return float(e), rows

summary = {}
for path in sorted(glob.glob("models/*")):
    rid = os.path.basename(path)
    model = load_model(path); P = model["parameters"]; h = model["history"]
    t0 = time.time(); pred = predict(Xte, P, second_guess=True); inf_s = time.time() - t0
    p1 = pred["First Prediction"][0].ravel(); c1 = pred["First Prediction"][1].ravel()
    p2 = pred["Second Prediction"][0].ravel()
    cm = confusion_matrix(te_y, pred, num_classes=NUM_CLASSES)
    met, mac, acc = model_metrics(cm)
    correct = (p1 == yte)
    e, calib = ece(c1, correct.astype(float))
    tm = json.load(open("results/logs/%s.time.json" % rid)) if os.path.exists("results/logs/%s.time.json" % rid) else {}
    best_ep = int(np.argmax(h["val_accuracy"])) + 1
    log = open("results/logs/%s.train.log" % rid, errors="ignore").read() if os.path.exists("results/logs/%s.train.log" % rid) else ""
    s = {
        "id": rid, "args": (tm["cmd"][tm["cmd"].index("--out") + 2:] if tm else []),
        "layers_dim": model["layers_dim"], "optimizer": model["optimizer"], "regularizer": model["regularizer"],
        "test_top1": float(acc), "test_top2": float(np.mean(correct | (p2 == yte))),
        "macro_precision": float(mac["Precision"]), "macro_recall": float(mac["Recall"]), "macro_f1": float(mac["F1-Score"]),
        "final_train_acc": float(h["accuracy"][-1]), "final_dev_acc": float(h["val_accuracy"][-1]),
        "best_dev_acc": float(max(h["val_accuracy"])), "best_epoch": best_ep,
        "train_acc_at_best": float(h["accuracy"][best_ep - 1]),
        "gap_at_best": float(h["accuracy"][best_ep - 1] - max(h["val_accuracy"])),
        "final_gap": float(h["accuracy"][-1] - h["val_accuracy"][-1]),
        "epochs_completed": len(h["accuracy"]), "early_stopped": "halted" in log,
        "train_wall_s": tm.get("wall_s"), "peak_rss_mb": tm.get("peak_rss_mb"),
        "mean_conf_correct": float(c1[correct].mean()), "mean_conf_incorrect": float(c1[~correct].mean()),
        "ece_15bin": e, "test_inference_s": inf_s, "test_inference_us_per_img": inf_s / len(yte) * 1e6,
        "n_params": int(sum(v.size for v in P.values())),
        "per_class": [{"class": k, "name": GTSRB_LABELS[k], "train_support": int(train_support[k]),
                       "test_support": int(test_support[k]), "precision": float(met["Precision"][k]),
                       "recall": float(met["Recall"][k]), "f1": float(met["F1-Score"][k])} for k in range(NUM_CLASSES)],
        "history": {k: [float(x) for x in v] for k, v in h.items()},
        "calibration_bins": calib,
    }
    f1 = np.array(met["F1-Score"])
    s["corr_trainsupport_f1_pearson"] = pearson(train_support, f1)
    s["corr_trainsupport_f1_spearman"] = spearman(train_support, f1)
    off = cm.copy(); np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-off.ravel()), off.shape))[0][:10]
    s["top_confusions"] = [{"count": int(off[i, j]), "true": int(i), "true_name": GTSRB_LABELS[i],
                            "pred": int(j), "pred_name": GTSRB_LABELS[j]} for i, j in idx]
    s["errors_total"] = int(off.sum())
    summary[rid] = s
    np.save("%s/cm_%s.npy" % (R, rid), cm)
    print("%-24s top1 %.4f top2 %.4f macroF1 %.4f ep %d" % (rid, s["test_top1"], s["test_top2"], s["macro_f1"], s["epochs_completed"]))

json.dump(summary, open(R + "/summary.json", "w"), indent=1)

# ---- flat CSVs ----
import csv
keys = ["id", "test_top1", "test_top2", "macro_precision", "macro_recall", "macro_f1", "final_train_acc", "final_dev_acc",
        "best_dev_acc", "best_epoch", "gap_at_best", "final_gap", "epochs_completed", "early_stopped", "train_wall_s",
        "peak_rss_mb", "mean_conf_correct", "mean_conf_incorrect", "ece_15bin", "test_inference_us_per_img", "n_params",
        "corr_trainsupport_f1_pearson", "corr_trainsupport_f1_spearman"]
with open(R + "/all_runs_summary.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(keys)
    for s in summary.values(): w.writerow([s[k] for k in keys])
if "e0_baseline" in summary:
    b = summary["e0_baseline"]
    with open(R + "/e0_per_class_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(b["per_class"][0].keys())); w.writeheader(); w.writerows(b["per_class"])
    with open(R + "/e0_top_confusions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(b["top_confusions"][0].keys())); w.writeheader(); w.writerows(b["top_confusions"])

    # ---- figures for E0 ----
    hh = b["history"]; ep = np.arange(1, len(hh["loss"]) + 1)
    for key, vkey, name, fn in [("loss", "val_loss", "Loss", "fig7_1_e0_loss.png"), ("accuracy", "val_accuracy", "Accuracy", "fig7_2_e0_accuracy.png")]:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(ep, hh[key], label="train"); ax.plot(ep, hh[vkey], label="dev")
        ax.axvline(b["best_epoch"], color="gray", ls=":", lw=1, label="best dev (restored)")
        ax.set_xlabel("Epoch"); ax.set_ylabel(name); ax.set_title("E0 baseline — %s" % name.lower()); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(F + "/" + fn); plt.close(fig)
    cm = np.load(R + "/cm_e0_baseline.npy"); cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); fig.colorbar(im, fraction=.046)
    ax.set_xticks(range(43)); ax.set_yticks(range(43)); ax.tick_params(labelsize=6)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title("E0 test confusion matrix (row-normalised)")
    fig.tight_layout(); fig.savefig(F + "/fig7_3_e0_confusion_matrix.png"); plt.close(fig)
    pc = b["per_class"]; f1 = [r["f1"] for r in pc]; sup = [r["train_support"] for r in pc]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(sup, f1, s=18)
    for r in pc:
        if r["f1"] < sorted(f1)[8]: ax.annotate(str(r["class"]), (r["train_support"], r["f1"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("Training images in class (log)"); ax.set_ylabel("Test F1")
    ax.set_title("E0: class support vs F1 (Pearson r=%.2f, Spearman ρ=%.2f)" % (b["corr_trainsupport_f1_pearson"], b["corr_trainsupport_f1_spearman"]))
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(F + "/fig7_4_e0_support_vs_f1.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(range(43), f1); ax.set_xticks(range(43)); ax.tick_params(labelsize=7)
    ax.set_xlabel("Class id"); ax.set_ylabel("Test F1"); ax.set_ylim(0, 1.02); ax.set_title("E0 per-class F1"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(F + "/fig7_5_e0_per_class_f1.png"); plt.close(fig)
    cb = b["calibration_bins"]
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.plot([r[3] for r in cb], [r[4] for r in cb], "o-")
    ax.set_xlabel("Mean confidence"); ax.set_ylabel("Accuracy"); ax.set_title("E0 reliability (ECE=%.3f)" % b["ece_15bin"]); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(F + "/fig7_6_e0_reliability.png"); plt.close(fig)
    # misclassified examples
    model = load_model("models/e0_baseline"); pred = predict(Xte, model["parameters"])
    p1 = pred["First Prediction"][0].ravel(); wrong = np.where(p1 != yte)[0][:24]
    fig, axs = plt.subplots(3, 8, figsize=(12, 5.2))
    for a, i in zip(axs.ravel(), wrong):
        a.imshow(te_x[i], cmap="gray"); a.set_title("T:%d P:%d" % (yte[i], p1[i]), fontsize=8); a.axis("off")
    fig.suptitle("E0: first 24 misclassified test images (T=true, P=predicted, after preprocessing)")
    fig.tight_layout(); fig.savefig(F + "/fig7_7_e0_misclassified.png"); plt.close(fig)

# ---- ablation curves ----
groups = {"E1_initialization": ["e0_baseline", "e1_random"],
          "E2_optimizer": ["e0_baseline", "e2_mgd", "e2_bgd"],
          "E3_regularization": ["e3_dropout_0.6", "e3_dropout_0.7", "e0_baseline", "e3_dropout_0.9", "e3_l2_0.1", "e3_l2_0.4", "e3_l2_0.7", "e3_l2_1.0", "e3_none"],
          "E4_architecture": ["e4_256", "e0_baseline", "e4_1024_512_256", "e4_1024_512_256_128"],
          "E5_augmentation": ["e0_baseline", "e5_augmented"],
          "E6_lr_decay": ["e0_baseline", "e6_decay10"]}
for g, ids in groups.items():
    ids = [i for i in ids if i in summary]
    if len(ids) < 2: continue
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i in ids:
        v = summary[i]["history"]["val_accuracy"]; ax.plot(range(1, len(v) + 1), v, label=i)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dev accuracy"); ax.set_title(g.replace("_", " ") + " — dev accuracy per epoch")
    ax.legend(fontsize=7); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(F + "/abl_%s.png" % g); plt.close(fig)
print("done")
