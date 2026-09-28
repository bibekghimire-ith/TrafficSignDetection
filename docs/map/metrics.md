# Code health metrics

<!-- AUTO:START stamp -->
Generated: 2026-09-27 at commit `b11900d` (working tree had 22 uncommitted/untracked paths).
<!-- AUTO:END stamp -->

All metrics are **static**: they come from `ast` parsing and `git log`, and no project code is run.
Complexity is the stdlib-`ast` fallback (1 + if/for/while/except/with/assert/ternary + extra boolean operands +
comprehension ifs). radon was not used. Archived `experiments/phase1/`, `tools/`, tests and empty files are
excluded from the module tables.

**How to read churn:** the repository has only five commits, all from August–September 2026, so every source file
shows one commit. Hotspot scores are therefore driven almost entirely by complexity. Uncommitted working-tree
edits, such as the Phase 2 fixes in `src/`, are not counted.

## Size
<!-- AUTO:START metrics-size -->
| module | SLOC | size | functions | classes | complexity sum | max CC |
|---|---|---|---|---|---|---|
| `augment.py` | 48 | ▮▯▯▯▯▯▯▯▯▯ | 2 | 0 | 3 | 2 |
| `evaluate.py` | 43 | ▮▯▯▯▯▯▯▯▯▯ | 2 | 0 | 4 | 3 |
| `experiments/analyze.py` | 233 | ▮▮▮▯▯▯▯▯▯▯ | 4 | 0 | 7 | 3 |
| `experiments/check_ftz_equivalence.py` | 22 | ▯▯▯▯▯▯▯▯▯▯ | 0 | 0 | 0 | 0 |
| `experiments/error_analysis.py` | 64 | ▮▯▯▯▯▯▯▯▯▯ | 0 | 0 | 0 | 0 |
| `experiments/inference_timing.py` | 24 | ▯▯▯▯▯▯▯▯▯▯ | 0 | 0 | 0 | 0 |
| `experiments/measure.py` | 6 | ▯▯▯▯▯▯▯▯▯▯ | 0 | 0 | 0 | 0 |
| `experiments/step_times.py` | 19 | ▯▯▯▯▯▯▯▯▯▯ | 0 | 0 | 0 | 0 |
| `gui.py` | 119 | ▮▮▯▯▯▯▯▯▯▯ | 6 | 1 | 12 | 4 |
| `predict.py` | 46 | ▮▯▯▯▯▯▯▯▯▯ | 3 | 0 | 7 | 4 |
| `src/data_augmentation.py` | 329 | ▮▮▮▮▮▯▯▯▯▯ | 13 | 0 | 77 | 12 |
| `src/dataset.py` | 458 | ▮▮▮▮▮▮▮▯▯▯ | 19 | 0 | 72 | 13 |
| `src/ffnn.py` | 688 | ▮▮▮▮▮▮▮▮▮▮ | 18 | 0 | 81 | 26 |
| `src/model_utils.py` | 565 | ▮▮▮▮▮▮▮▮▯▯ | 21 | 0 | 45 | 6 |
| `train.py` | 120 | ▮▮▯▯▯▯▯▯▯▯ | 2 | 0 | 11 | 10 |
| **total (15 modules)** | **2784** | | **90** | **1** | **319** | |
<!-- AUTO:END metrics-size -->

## Most complex functions
<!-- AUTO:START metrics-complexity -->
| function | file:line | cyclomatic complexity | max nesting |
|---|---|---|---|
| `train` | `src/ffnn.py:717` | 26 ▮▮▮▮▮▮▮▮▮▮ | 4 |
| `load_dataset` | `src/dataset.py:312` | 13 ▮▮▮▮▮▯▯▯▯▯ | 3 |
| `blur_images` | `src/data_augmentation.py:67` | 12 ▮▮▮▮▮▯▯▯▯▯ | 6 |
| `main` | `train.py:57` | 10 ▮▮▮▮▯▯▯▯▯▯ | 1 |
| `download_dataset` | `src/dataset.py:74` | 9 ▮▮▮▯▯▯▯▯▯▯ | 5 |
| `shift_images` | `src/data_augmentation.py:118` | 8 ▮▮▮▯▯▯▯▯▯▯ | 4 |
| `retrive_dataset` | `src/dataset.py:222` | 8 ▮▮▮▯▯▯▯▯▯▯ | 3 |
| `data_generator` | `src/data_augmentation.py:399` | 8 ▮▮▮▯▯▯▯▯▯▯ | 3 |
| `augment_img` | `src/data_augmentation.py:261` | 8 ▮▮▮▯▯▯▯▯▯▯ | 1 |
| `plot_confusion_matrix` | `src/model_utils.py:251` | 6 ▮▮▯▯▯▯▯▯▯▯ | 3 |
<!-- AUTO:END metrics-complexity -->

## Tests (static mapping)
<!-- AUTO:START metrics-tests -->
| module | test files | test functions | test SLOC : source SLOC |
|---|---|---|---|
| `augment.py` | 0 | 0 | — |
| `evaluate.py` | 0 | 0 | — |
| `experiments/analyze.py` | 0 | 0 | — |
| `experiments/check_ftz_equivalence.py` | 0 | 0 | — |
| `experiments/error_analysis.py` | 0 | 0 | — |
| `experiments/inference_timing.py` | 0 | 0 | — |
| `experiments/measure.py` | 0 | 0 | — |
| `experiments/step_times.py` | 0 | 0 | — |
| `gui.py` | 0 | 0 | — |
| `predict.py` | 0 | 0 | — |
| `src/data_augmentation.py` | 1 | 12 | 0.39 |
| `src/dataset.py` | 1 | 12 | 0.28 |
| `src/ffnn.py` | 1 | 12 | 0.18 |
| `src/model_utils.py` | 1 | 12 | 0.22 |
| `train.py` | 0 | 0 | — |

**Untested modules** (no test file imports them), by SLOC: `experiments/analyze.py` (233), `train.py` (120), `gui.py` (119), `experiments/error_analysis.py` (64), `augment.py` (48), `predict.py` (46), `evaluate.py` (43), `experiments/inference_timing.py` (24), `experiments/check_ftz_equivalence.py` (22), `experiments/step_times.py` (19), `experiments/measure.py` (6)

*Static mapping, not measured line coverage.*
<!-- AUTO:END metrics-tests -->

The unit tests cover the network maths (gradient check), Adam, the LR schedule, mini-batching, the track split,
metrics and the empty-augmentation error. **`ffnn.train()`, the most complex function (CC 26), is not unit-tested.**
It is exercised only end-to-end by `train.py` and the experiment runs.

## Hotspots (complexity × churn)
<!-- AUTO:START metrics-hotspots -->
| file | commits (12 mo) | lines changed | complexity sum | hotspot score | tested |
|---|---|---|---|---|---|
| `src/ffnn.py` | 1 | 914 | 81 | 0.87 ▮▮▮▮▮▮▮▮▮▯ | yes |
| `src/model_utils.py` | 1 | 726 | 45 | 0.75 ▮▮▮▮▮▮▮▯▯▯ | yes |
| `src/dataset.py` | 1 | 479 | 72 | 0.69 ▮▮▮▮▮▮▮▯▯▯ | yes |
| `src/data_augmentation.py` | 1 | 487 | 77 | 0.68 ▮▮▮▮▮▮▮▯▯▯ | yes |
| `train.py` | 1 | 130 | 11 | 0.67 ▮▮▮▮▮▮▮▯▯▯ | **no ⚠** |
| `gui.py` | 1 | 152 | 12 | 0.44 ▮▮▮▮▯▯▯▯▯▯ | **no ⚠** |
| `predict.py` | 1 | 64 | 7 | 0.40 ▮▮▮▮▯▯▯▯▯▯ | **no ⚠** |
| `evaluate.py` | 1 | 59 | 4 | 0.25 ▮▮▯▯▯▯▯▯▯▯ | **no ⚠** |
| `augment.py` | 1 | 61 | 3 | 0.19 ▮▮▯▯▯▯▯▯▯▯ | **no ⚠** |
| `experiments/step_times.py` | 0 | 0 | 0 | 0.13 ▮▯▯▯▯▯▯▯▯▯ | **no ⚠** |

`hotspot = pct_rank(complexity sum) × pct_rank(commits in 12 months)`. Churn counts come from `git log --numstat` (commit counts only, no authors). Uncommitted edits are not counted.
<!-- AUTO:END metrics-hotspots -->

## Coupling (intra-repo imports)
<!-- AUTO:START metrics-coupling -->
| module | fan-in | fan-out | instability | imports |
|---|---|---|---|---|
| `src/dataset.py` | 10 | 0 | 0.00 | — |
| `src/model_utils.py` | 10 | 1 | 0.09 | `dataset` |
| `src/ffnn.py` | 7 | 1 | 0.12 | `model_utils` |
| `src/data_augmentation.py` | 2 | 2 | 0.50 | `dataset`, `model_utils` |
| `augment.py` | 0 | 2 | 1.00 | `data_augmentation`, `dataset` |
| `evaluate.py` | 0 | 3 | 1.00 | `dataset`, `ffnn`, `model_utils` |
| `experiments/analyze.py` | 0 | 3 | 1.00 | `dataset`, `ffnn`, `model_utils` |
| `experiments/check_ftz_equivalence.py` | 0 | 1 | 1.00 | `model_utils` |
| `experiments/error_analysis.py` | 0 | 3 | 1.00 | `dataset`, `ffnn`, `model_utils` |
| `experiments/inference_timing.py` | 0 | 3 | 1.00 | `dataset`, `ffnn`, `model_utils` |
| `experiments/measure.py` | 0 | 0 | 0.00 | — |
| `experiments/step_times.py` | 0 | 0 | 0.00 | — |
| `gui.py` | 0 | 3 | 1.00 | `dataset`, `ffnn`, `model_utils` |
| `predict.py` | 0 | 3 | 1.00 | `dataset`, `ffnn`, `model_utils` |
| `train.py` | 0 | 4 | 1.00 | `data_augmentation`, `dataset`, `ffnn`, `model_utils` |

Import cycles: none
<!-- AUTO:END metrics-coupling -->

Hubs are `src/dataset.py` and `src/model_utils.py`, which have the highest fan-in and are stable. A signature change
there ripples into every script. The most fragile module is `train.py` (instability 0.80): it depends on the whole
library.

## TODO / FIXME / HACK markers
<!-- AUTO:START todo -->
No TODO / FIXME / HACK / XXX markers found.
<!-- AUTO:END todo -->
