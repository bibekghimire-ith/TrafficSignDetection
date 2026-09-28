# Directory map

<!-- AUTO:START stamp -->
Generated: 2026-09-27 at commit `b11900d` (working tree had 22 uncommitted/untracked paths).
<!-- AUTO:END stamp -->

| path | purpose | key files |
|---|---|---|
| `/` (root) | Entry-point scripts, container, docs, report | `train.py`, `evaluate.py`, `predict.py`, `augment.py`, `gui.py`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `README.md`, `SETUP.md`, `REPORT.md` |
| `src/` | The library: data pipeline, network, utilities. Flat modules on `sys.path` | `dataset.py`, `ffnn.py`, `model_utils.py`, `data_augmentation.py` |
| `tests/` | stdlib `unittest` suite: gradient check, optimizer, RNG, split, metrics | `test_core.py` |
| `experiments/` | Thesis experiment harness (Phase 2: fixed code, 3 seeds) | `run_all.sh`, `analyze.py`, `error_analysis.py`, `inference_timing.py`, `measure.py` |
| `experiments/phase1/` | Archived Phase 1 scripts (original code, one seed) | `README.md`, `run_phase1.sh`, `analyze.py` |
| `tools/` | Repo-map maintenance | `gen_repo_map.py` |
| `docs/architecture/` | Architecture and data-flow diagrams (Mermaid) | `architecture.md`, `data-flow.md` |
| `docs/map/` | This map's detail files | `directory.md`, `symbols.md`, `dependencies.md`, `metrics.md` |
| `ML_Statistics_Teaching_Notes/` | Stand-alone teaching notes and figures, not used by the code | `ML_Statistics_Teaching_Notes.md` |
| `dataset/` *(ignored)* | Downloaded archives, extracted GTSRB, `gtsrb/gtsrb_32.npz` cache, `augmented_data/` | gitignored except `.gitkeep` |
| `models/` *(ignored)* | Pickled trained models (`<run>_s<seed>`, `phase1/`) | gitignored |
| `results/` *(ignored by the map, untracked)* | Experiment outputs: logs, JSON/CSV, figures, predictions | `summary.json`, `aggregate.csv`, `significance.csv`, `figures/` |
| `temp/` *(ignored)* | Stale early-stopping checkpoint from the original code. Current code uses `tempfile` | — |

<!-- AUTO:START tree -->
```
./  .dockerignore, .gitignore, CLAUDE.md, Dockerfile, LICENSE, Project_Walkthrough.ipynb, README.md, REPORT.md, … (+11)
ML_Statistics_Teaching_Notes/  01_Statistics_and_Evaluation_Metrics.md, 02_ML_Project_Walkthroughs.md, ML_Statistics_Teaching_Notes.md
ML_Statistics_Teaching_Notes/assets/  01_distributions.png, 02_mean_median_outlier.png, 03_variance_spread.png, 04_correlation_causation.png, 05_hypothesis_pvalue.png, 06_confusion_matrix.png, 07_roc_pr_curve.png, 08_regression_residuals.png, … (+6)
ML_Statistics_Teaching_Notes/assets/proj1_regression/  01_eda.png, 02_model_comparison.png, 03_best_model_fit.png, 04_learning_curve.png, 05_feature_importance.png
ML_Statistics_Teaching_Notes/assets/proj2_classification/  01_eda.png, 02_model_comparison.png, 03_confusion_matrices.png, 04_threshold_tradeoff.png, 05_feature_importance.png
ML_Statistics_Teaching_Notes/assets/proj3_mnist/  01_sample_digits.png, 02_pixels_as_features.png, 03_pca_projection.png, 04_training_loss.png, 05_confusion_matrix.png, 06_misclassified.png
docs/architecture/  architecture.md, data-flow.md
docs/map/  dependencies.md, directory.md, metrics.md, symbols.md
experiments/  analyze.py, check_ftz_equivalence.py, error_analysis.py, inference_timing.py, measure.py, run_all.sh, speed_benchmark.sh, step_times.py
experiments/phase1/  README.md, analyze.py, error_analysis.py, inference_timing.py, run_phase1.sh, train_lowmem.py
src/  __init__.py, data_augmentation.py, dataset.py, ffnn.py, model_utils.py
tests/  test_core.py
tools/  gen_repo_map.py
```
Archived (listed, not measured): `experiments/phase1/`.
Ignored (not mapped): `.git/`, `.ipynb_checkpoints/`, `.venv/`, `Claude outputs/`, `__pycache__/`, `build/`, `dataset/`, `dist/`, `models/`, `node_modules/`, `results/`, `temp/`, `venv/`, `*.egg-info`.
<!-- AUTO:END tree -->
