#!/usr/bin/env python3
"""Run from the project root: python experiments/step_times.py
Per-epoch ms/step parsed from results/logs/<run>.train.log. Shows the subnormal slow-down (defect 10) in the runs
trained before the flush was added (all Phase 2 runs except E5). Writes results/step_times.json."""
import glob, json, os, re
out = {}
for f in sorted(glob.glob("results/logs/*_s[0-9].train.log")):
    run = os.path.basename(f)[:-len(".train.log")]
    txt = open(f, errors="ignore").read().replace("\r", "\n")
    steps = [int(x) for x in re.findall(r"100%\] - [\d.]+s (\d+)ms/step", txt)]
    if steps: out[run] = steps
summary = {}
for cfg in sorted({r.rsplit("_s", 1)[0] for r in out}):
    runs = [v for r, v in out.items() if r.rsplit("_s", 1)[0] == cfg and len(v) >= 6]
    if runs:
        summary[cfg] = {"first3_ms_mean": sum(sum(v[:3]) / 3 for v in runs) / len(runs),
                        "last3_ms_mean": sum(sum(v[-3:]) / 3 for v in runs) / len(runs),
                        "max_ms": max(max(v) for v in runs)}
json.dump({"per_run": out, "per_config": summary}, open("results/step_times.json", "w"), indent=1)
for k, v in summary.items(): print("%-24s first %.1f  last %.1f  max %d" % (k, v["first3_ms_mean"], v["last3_ms_mean"], v["max_ms"]))
