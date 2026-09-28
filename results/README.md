# Experiment results — Phase 2 (final code, 3 seeds)

These are the outputs of the experiment plan in the report (§5.4): 18 configurations × seeds 1–3, 54 runs in total.
They were produced on 27 Sep 2026 on a 2-core x86-64 Linux container (8 GB RAM) with the code in this repository.
The write-up is in `REPORT_completed.md` and `Traffic_Sign_Recognition_Project_Report_completed.docx`, Sections 7–8.
The earlier single-seed run on the original code is in `phase1_original_code/`.

| config | test accuracy | macro-F1 | epochs |
|---|---|---|---|
| e0_baseline | 87.80 ± 0.49% | 0.839 ± 0.007 | 18.3 |
| e1_random | 87.80 ± 1.12% | 0.844 ± 0.012 | 27.7 |
| e2_bgd | 5.66 ± 1.62% | 0.016 ± 0.004 | 40.0 |
| e2_mgd | 70.32 ± 0.52% | 0.520 ± 0.030 | 40.0 |
| e2_mgd_lr0.05 | 87.87 ± 1.36% | 0.844 ± 0.014 | 24.7 |
| e3_dropout_0.6 | 86.20 ± 1.43% | 0.811 ± 0.035 | 20.0 |
| e3_dropout_0.7 | 87.64 ± 0.97% | 0.841 ± 0.007 | 24.0 |
| e3_dropout_0.9 | 86.23 ± 1.89% | 0.824 ± 0.017 | 15.0 |
| e3_l2_1 | 84.92 ± 0.62% | 0.806 ± 0.010 | 10.7 |
| e3_l2_10 | 85.34 ± 2.00% | 0.813 ± 0.023 | 17.3 |
| e3_l2_3 | 85.84 ± 1.64% | 0.823 ± 0.023 | 16.7 |
| e3_l2_30 | 85.99 ± 1.35% | 0.822 ± 0.010 | 18.3 |
| e3_none | 85.42 ± 0.70% | 0.811 ± 0.009 | 14.7 |
| e4_1024_512_256 | 87.82 ± 0.39% | 0.839 ± 0.006 | 22.0 |
| e4_1024_512_256_128 | 86.14 ± 1.71% | 0.819 ± 0.031 | 15.3 |
| e4_256 | 86.96 ± 0.63% | 0.833 ± 0.010 | 20.7 |
| e5_augmented | 90.88 ± 0.11% | 0.874 ± 0.003 | 26.3 |
| e6_decay10 | 89.68 ± 0.67% | 0.866 ± 0.009 | 29.3 |

## Files

- `summary.json`: one entry per run, with metrics, per-class results, history, calibration and timing.
- `aggregate.json` / `.csv`: the mean and sd over seeds for each configuration.
- `significance.json` / `.csv`: McNemar tests (each run against the E0 run with the same seed) and Welch t-tests on
  macro-F1. The `_e0_seed_vs_seed` entry is the noise floor.
- `e0_per_class_metrics.csv`, `e0_top_confusions.csv`, `e0_support_f1_correlation.json`: detail for the baseline,
  pooled over seeds.
- `e0_error_analysis.json`: accuracy by sign size and sharpness, top-2 gains, confusions within groups, and the
  confidence-rejection table.
- `e0_inference_timing.json`, `speed_benchmark.json`, `step_times.json`, `ftz_equivalence.json`: timing and speed
  measurements, plus the check that the subnormal flush leaves results bit-identical.
- `predictions/<run>.npy`: the first and second test-set predictions for each run, in `load_dataset` seed-1 test
  order. `predictions/cm_<run>.npy` holds the confusion matrices.
- `figures/`: every figure used in the report.
- `logs/`: the raw `train.py` / `evaluate.py` / `augment.py` output for each run, plus `.time.json` (wall time and
  peak RSS).

## Reproduce

Run from the project root:

```bash
experiments/run_all.sh                 # all configs x seeds; skips runs that are already done
python experiments/analyze.py          # summary / aggregate / significance / figures
python experiments/error_analysis.py
python experiments/inference_timing.py
python experiments/step_times.py
experiments/speed_benchmark.sh         # original (git HEAD) vs working tree
python experiments/check_ftz_equivalence.py
```
