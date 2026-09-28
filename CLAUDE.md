# TrafficSignDetection — repo map

GTSRB traffic-sign **classifier** (43 classes) built on a feed-forward neural network written from scratch in NumPy:
forward/backprop, Adam, dropout, L2 and early stopping, with no ML framework. It is an MSc project: CLI scripts,
a Tkinter GUI, a Jupyter walkthrough and a report.

<!-- AUTO:START stamp -->
Generated: 2026-09-27 at commit `b11900d` (working tree had 22 uncommitted/untracked paths).
<!-- AUTO:END stamp -->

**Read this file first. Open only the files it points to.** For design, flow or integration questions, open
`docs/architecture/`. For symbols, directory detail or code health, open `docs/map/`. If
`git log <stamp-hash>..HEAD --stat` (or `git status`) shows changes, re-scan only those directories, then run
`python tools/gen_repo_map.py`. When it says *diagrams may be stale*, re-check the diagram against the code and run
`python tools/gen_repo_map.py --stamp-diagrams`.

## Architecture (summary) → [docs/architecture/architecture.md](docs/architecture/architecture.md)
- There are no services or network servers. Every script is a single-process, single-threaded Python CLI run by
  a person (optionally inside Docker, where the default command is a Jupyter server on port 8888).
- `src/` is a flat library on `sys.path`, not a package import. Scripts do `sys.path.insert(0, "src")` and
  `from dataset import …`.
- The layers depend in one direction: `dataset` ← `model_utils` ← `ffnn`. `data_augmentation` uses
  `dataset` + `model_utils`. There are no import cycles.
- The entry scripts (`train.py`, `evaluate.py`, `predict.py`, `augment.py`, `gui.py`) only wire the library
  together.
- `experiments/` holds the thesis experiment harness: a shell loop over train and evaluate, plus analysis scripts
  that read `models/` and write `results/`.

## Data flow (summary) → [docs/architecture/data-flow.md](docs/architecture/data-flow.md)
- The only network call is `download_dataset` (`src/dataset.py:96`), an HTTPS GET of three GTSRB zip archives,
  run once on the first run.
- Pipeline: zip → extract → decode `.ppm` + ROI CSV → `preprocess_image` → cached `dataset/gtsrb/gtsrb_32.npz`
  → split → `prep_dataset` (float32, 0–1, one-hot) → `train()` → pickled model in `models/`.
- Everything else is local files. The pickle model files are the hand-off between `train.py` and
  evaluate/predict/GUI/analysis.
- Scratch state: an early-stopping checkpoint in a `tempfile.mkdtemp` directory (always removed), and augmented
  batches in `dataset/augmented_data/`, which **`load_augmented_data` deletes after reading them**.

## Entry points
| command | file |
|---|---|
| train a model | `train.py` (`main` at `train.py:57`) |
| score on the test set | `evaluate.py` |
| classify image files | `predict.py` |
| desktop app | `gui.py` (`TrafficSignApp`, needs Tkinter; not in Docker) |
| offline augmentation | `augment.py` |
| all thesis experiments | `experiments/run_all.sh` → `experiments/analyze.py`, `error_analysis.py`, `inference_timing.py` |
| timing / numerics checks | `experiments/speed_benchmark.sh` (old vs new code), `experiments/step_times.py`, `experiments/check_ftz_equivalence.py` |
| walkthrough | `Project_Walkthrough.ipynb` |
| refresh this map | `tools/gen_repo_map.py` |

## Where to look for X
| task | open |
|---|---|
| change network maths (forward, backprop, cost) | `src/ffnn.py` (`forward_prop`, `backward_prop`, `softmax_cross_entropy_cost`) ⚠ hotspot |
| optimizer / Adam / learning-rate schedule | `src/ffnn.py` (`update_parameters`, `learning_rate_schedule`) ⚠ hotspot |
| training loop, early stopping, checkpoints | `src/ffnn.py` (`train`, the most complex function, CC 26) ⚠ hotspot |
| CLI flags and hyper-parameters | `train.py` (`parse_args`) ⚠ untested |
| image preprocessing (ROI, equalize, resize) | `src/dataset.py` (`preprocess_image`) |
| download / extract / decode / cache | `src/dataset.py` (`download_dataset`, `decompress_dataset`, `retrive_dataset`, `load_dataset`) |
| train/dev split (track-disjoint vs random) | `src/dataset.py` (`train_dev_split`, `_track_split`) |
| float32 conversion, one-hot | `src/dataset.py` (`normalize_input`, `one_hot_encoding`, `prep_dataset`) |
| augmentation transforms | `src/data_augmentation.py` ⚠ untested |
| augmented batch files | `src/data_augmentation.py` (`data_generator`, `save_generated_data`, `load_augmented_data`) ⚠ untested |
| metrics, confusion matrix | `src/model_utils.py` (`confusion_matrix`, `model_metrics`) |
| mini-batching, activations | `src/model_utils.py` (`rand_mini_batches`, `relu`, `softmax`) |
| model save/load format | `src/model_utils.py` (`save_model`, `load_model`); dict keys in `train.py:119` |
| class names | `src/dataset.py` (`GTSRB_LABELS`, `label_description`) |
| GUI behaviour | `gui.py` ⚠ untested |
| experiment configs and seeds | `experiments/run_all.sh` (`CFG` table) |
| aggregation, significance tests, figures | `experiments/analyze.py` ⚠ untested |
| why later epochs were slow (subnormals) | `src/ffnn.py` (`update_parameters` flush), `experiments/step_times.py` |
| Phase 1 (original-code) results provenance | `experiments/phase1/README.md`, `results/phase1_original_code/` |
| unit and gradient-check tests | `tests/test_core.py` |
| container setup | `Dockerfile`, `docker-compose.yml` |
| report text | `REPORT.md`, `Traffic_Sign_Recognition_Project_Report.docx` |

## Build / run / test
```bash
pip install -r requirements.txt                       # numpy, Pillow, scipy, matplotlib
python -m unittest discover -s tests -v               # 11 tests incl. numerical gradient check (1 needs scipy)
python train.py --epochs 40 --out models/gtsrb_mlp    # first run downloads ~280 MB
python evaluate.py --model models/gtsrb_mlp --no-plot
python predict.py img.png --model models/gtsrb_mlp ;  python gui.py --model models/gtsrb_mlp
experiments/run_all.sh && python experiments/analyze.py   # full thesis plan, 3 seeds (~3 h on 2 cores)
docker compose up                                     # Jupyter on localhost:8888
```
There is no linter or CI config in the repo.

## Conventions & gotchas
- **Column-major data:** `X` is `(features, m)` and `Y` is `(43, m)`. Mini-batches are column slices.
- **dtype:** inputs and parameters are **float32** (`prep_dataset`, `init_parameters(dtype=)`). `update_parameters`
  must not divide by NumPy float64 scalars, or NumPy ≥2 silently upcasts to float64. Use float64 only in tests.
- **L2:** `lambda` is divided by the **training-set size** by default (`--l2-norm dataset`). `--l2-norm batch`
  reproduces the original behaviour.
- **Dev split:** track-disjoint by default (`--split track`). It needs `train_tracks` in the npz cache. An old
  cache is upgraded from the extracted CSVs, or the run fails with a message.
- **RNG:** `train.py` seeds the global RNG once with `--seed`. Mini-batch shuffles use a private
  `RandomState(seed*1000+epoch)`. Do not re-introduce `np.random.seed` inside library functions.
- `augment.py` and `train.py` must use the same `--seed` and `--split`, or augmented data leaks dev images.
- `load_augmented_data` **deletes** the batch files it reads, so regenerate them before every
  `--use-augmented` run. An empty folder raises `ValueError: No augmented batches`.
- Models are **pickles**: `load_model` (`src/model_utils.py:720`) will execute code from an untrusted file.
- `predict.py` and `gui.py` preprocess **without an ROI crop**, unlike training. Feed them tightly cropped
  sign images.
- The Docker Jupyter server runs with token auth disabled (`Dockerfile`). Keep it local.
- Adam moments below 1e-30 are **flushed to zero** in `update_parameters`. Removing this brings back a float32
  subnormal slow-down: 2× on E0 and up to 14× with L2. Results are bit-identical with or without it
  (`results/ftz_equivalence.json`).
- `plt.show()` calls block in scripts. Use `--no-plot`, or `MPLBACKEND=Agg` when headless.

<!-- AUTO:START health -->
**Code health:** 15 source modules, 2784 SLOC; 27% of modules have a test file importing them (static mapping); top hotspots `src/ffnn.py`, `src/model_utils.py`, `src/dataset.py`; complexity by stdlib `ast` (radon not used). Detail: [docs/map/metrics.md](docs/map/metrics.md).
<!-- AUTO:END health -->

## Map index
- [docs/architecture/architecture.md](docs/architecture/architecture.md): system context, components, deployment
- [docs/architecture/data-flow.md](docs/architecture/data-flow.md): DFD with trust boundaries, sequences, flow inventory, stores
- [docs/map/directory.md](docs/map/directory.md): directory purposes and tree
- [docs/map/symbols.md](docs/map/symbols.md): key symbols with file:line
- [docs/map/dependencies.md](docs/map/dependencies.md): dependency versions and why they matter
- [docs/map/metrics.md](docs/map/metrics.md): size, complexity, static test mapping, churn, hotspots, coupling

## Open questions
- The download host in `GTSRB_URLS` is used with no checksum. Should archive hashes be pinned?
- `predict.py`/`gui.py` omit the ROI crop. Is that intended for real photos, or should users crop first? (unverified intent)
- `Project_Walkthrough.ipynb` was written against the original API. Check that its cells still run with the
  `return_tracks`/float32 changes. (unverified)
