#!/usr/bin/env python3
"""Run from the project root: python experiments/check_ftz_equivalence.py [--candidate PATH]

Shows that the subnormal flush in update_parameters (defect 10) does not change training: retrains E0 seed 1 with
the current code (or uses --candidate) and compares it with models/e0_baseline_s1, which was trained before the
flush existed. Writes results/ftz_equivalence.json.
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, "src")
from model_utils import load_model
ap = argparse.ArgumentParser(); ap.add_argument("--candidate"); a = ap.parse_args()
cand = a.candidate
if not cand:
    cand = os.path.join(tempfile.mkdtemp(), "e0_ftz_s1")
    subprocess.check_call([sys.executable, "train.py", "--epochs", "40", "--no-plot", "--seed", "1", "--out", cand],
                          stdout=subprocess.DEVNULL, env=dict(os.environ, MPLBACKEND="Agg"))
new, ref = load_model(cand), load_model("models/e0_baseline_s1")
out = {"reference": "models/e0_baseline_s1 (trained before the flush)", "candidate": "E0 seed 1 retrained with the flush",
       "max_abs_param_diff": max(float(np.abs(new["parameters"][k] - ref["parameters"][k]).max()) for k in ref["parameters"]),
       "identical_dev_accuracy_history": new["history"]["val_accuracy"] == ref["history"]["val_accuracy"],
       "epochs": [len(new["history"]["accuracy"]), len(ref["history"]["accuracy"])]}
json.dump(out, open("results/ftz_equivalence.json", "w"), indent=1); print(out)
