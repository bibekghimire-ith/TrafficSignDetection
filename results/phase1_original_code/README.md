# Phase 1 results: original code, one seed (26 Sep 2026)

> **Superseded.** These are the first-round results on the original code: random dev split, L2 divided by the mini-batch size, near-inactive LR decay, float64. They are kept because the report's Appendix D and Section 7.7 cite them. The final three-seed results are in `results/` (see `results/README.md`). The paths below are the Phase 1 layout. The models now live in `models/phase1/`, and the scripts in `experiments/phase1/`.

Outputs of the E0 baseline and ablations E1–E6 described in `REPORT.md` §5.4, one seed (1) each.
They were produced in a 2-core x86-64 Linux container (8 GB RAM) from the code in this repository.
The write-up is `REPORT_completed.md` / `Traffic_Sign_Recognition_Project_Report_completed.docx` (Sections 7–8).

| Run | Test acc. | Macro-F1 | Epochs |
|---|---|---|---|
| e0_baseline | 88.46% | 0.841 | 28 |
| e1_random | 88.40% | 0.845 | 40 |
| e2_bgd | 0.70% | 0.001 | 8 |
| e2_mgd | 70.28% | 0.534 | 40 |
| e3_dropout_0.6 | 88.17% | 0.846 | 31 |
| e3_dropout_0.7 | 89.24% | 0.862 | 33 |
| e3_dropout_0.9 | 88.59% | 0.853 | 21 |
| e3_l2_0.1 | 86.71% | 0.829 | 18 |
| e3_l2_0.4 | 85.17% | 0.816 | 15 |
| e3_l2_0.7 | 84.82% | 0.800 | 11 |
| e3_l2_1.0 | 83.94% | 0.777 | 11 |
| e3_none | 86.98% | 0.841 | 25 |
| e4_1024_512_256 | 87.56% | 0.842 | 17 |
| e4_1024_512_256_128 | 88.96% | 0.857 | 34 |
| e4_256 | 87.52% | 0.837 | 21 |
| e5_augmented | 90.93% | 0.873 | 25 |
| e6_decay10 | 89.21% | 0.856 | 34 |

## Files

- `summary.json`: every metric per run, including per-class metrics, training history, top confusions and calibration bins.
- `all_runs_summary.csv`: one row per run.
- `e0_per_class_metrics.csv`, `e0_top_confusions.csv`: baseline detail.
- `e0_error_analysis.json`: accuracy by sign size and sharpness, top-2 gains, within-group confusions and confidence-rejection thresholds.
- `e0_inference_timing.json`: forward-pass timing for batched and single-image inference.
- `cm_<run>.npy`: 43×43 test confusion matrices.
- `figures/`: every figure used in the report.
- `logs/`: `<run>.train.log` / `.eval.log` hold the raw train.py / evaluate.py output, and `.time.json` holds wall time and peak RSS. `e5_augmented_float64_OOM.*` is the unmodified E5 attempt, which ran out of memory.

## Reproduce

Run from the project root:

```bash
experiments/run_all.sh e0_baseline e1_random ...   # run ids are listed in run_all.sh
python experiments/analyze.py
python experiments/error_analysis.py
python experiments/inference_timing.py
```

Trained models are in `models/<run>` (pickle, loadable with `evaluate.py --model`).
