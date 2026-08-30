# Setup and Running Guide

Two ways to run this project. **Docker** is the reproducible one — one command,
identical environment everywhere, nothing installed on your machine. **Native
Python** is the one you want if you also intend to use the Tkinter desktop app,
which does not run in the container.

- [1. Hardware requirements](#1-hardware-requirements)
- [2. The dataset](#2-the-dataset)
- [3. Option A — Docker](#3-option-a--docker)
- [4. Option B — native Python](#4-option-b--native-python)
- [5. Step-by-step first run](#5-step-by-step-first-run)
- [6. Command reference](#6-command-reference)
- [7. Troubleshooting](#7-troubleshooting)

---

## 1. Hardware requirements

**No GPU. None of this code can use one** — it is NumPy on the CPU throughout,
and there is no CUDA path to enable. A GPU in your machine will sit idle.

### Measured figures

The memory numbers below are exact — they follow from the array sizes and hold
on any machine. The timings were measured on a **2-core x86-64 container**, so
treat them as a conservative ceiling: a modern laptop with 8 performance cores
will be several times faster.

| Workload | Peak RAM | Time |
|---|---|---|
| Decode GTSRB (one-time, then cached) | ~500 MB | 30 s – 3 min, disk-bound |
| Train, default `[1024, 512, 256, 43]`, 40 epochs | **~820 MB** | ~7 s/epoch → **~5 min** |
| Train, deeper `[1024, 1024, 512, 256, 43]`, 40 epochs | ~920 MB | ~16 s/epoch → ~11 min |
| Evaluate on the test set | ~450 MB | seconds |
| Predict a single image | ~150 MB | milliseconds |
| **Train with `--use-augmented` (6× data)** | **~2.5 GB** | ~6× the epoch time |

### Where the memory goes

The training matrix dominates everything else. GTSRB has 39,209 training
images; after the 90/10 dev split, 35,288 remain:

```
35,288 examples × 1,024 features × 8 bytes (float64) = 289 MB
```

That single array is most of the footprint. The model itself is negligible by
comparison — 667,179 parameters at the default architecture is about 5 MB.

This is also why augmentation is the memory hazard: six variants per image
takes the training matrix to roughly **1.7 GB**, and with the originals still
resident the peak lands near 2.5 GB.

### Recommended specs

| | Minimum | Comfortable |
|---|---|---|
| CPU | any x86-64 or ARM64, 2 cores | 4+ cores |
| RAM | **4 GB** | **8 GB** (16 GB if you use `--use-augmented`) |
| Disk | **2 GB** free | 4 GB free |
| OS | macOS, Linux or Windows | — |
| Python | 3.7+ | 3.10 or 3.11 |

**Docker Desktop users:** the default VM memory allocation is often 2 GB, which
is not enough for a full training run. Raise it to at least 4 GB in
*Settings → Resources*.

### Disk budget

| | Size |
|---|---|
| Downloaded `.zip` archives | ~280 MB |
| Extracted `.ppm` image tree | ~400 MB |
| Decoded `.npz` cache | ~50 MB |
| One trained model | ~5 MB |
| Docker image, if you use it | ~700 MB |

The archives and the extracted tree can both be deleted once the `.npz` cache
exists — see [Troubleshooting](#7-troubleshooting).

---

## 2. The dataset

**GTSRB — German Traffic Sign Recognition Benchmark**, published by the
Institut für Neuroinformatik, Ruhr-Universität Bochum. 39,209 training images
and 12,630 test images across 43 classes, licensed CC BY 4.0.

`train.py` downloads it automatically on first run, so **you normally do not
need to do anything here**. This section is the fallback for when the automatic
download is blocked by a firewall or proxy.

### Primary source — what the code uses

Official project page: <https://benchmark.ini.rub.de/gtsrb_dataset.html>

The archives live in the benchmark's public archive, and these are the exact
three URLs in `GTSRB_URLS` at the top of `src/dataset.py`:

| File | Contents | Approx. size |
|---|---|---|
| [`GTSRB_Final_Training_Images.zip`](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip) | 39,209 training images + per-class annotation CSVs | ~190 MB |
| [`GTSRB_Final_Test_Images.zip`](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip) | 12,630 test images | ~90 MB |
| [`GTSRB_Final_Test_GT.zip`](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip) | `GT-final_test.csv`, the test labels | ~100 KB |

Browse the archive: <https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html>

To install them by hand, drop all three `.zip` files into `dataset/gtsrb/` and
rerun `train.py`. It detects them, skips the download, and extracts them
itself. You do not need to unzip anything yourself.

```
TrafficSignDetection/
└── dataset/
    └── gtsrb/
        ├── GTSRB_Final_Training_Images.zip
        ├── GTSRB_Final_Test_Images.zip
        └── GTSRB_Final_Test_GT.zip
```

### Mirrors

If the primary source is unreachable:

- **Zenodo** — <https://zenodo.org/records/13741936> (DOI
  [10.5281/zenodo.13741936](https://doi.org/10.5281/zenodo.13741936)),
  published by Real-Time Computer Vision at RUB. A single `data.zip`, 320.5 MB.
- **Kaggle** — <https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign>

⚠️ **Both mirrors use a different directory layout and file format from the
official archives, so `src/dataset.py` will not read them as-is.** They are
listed as a way to obtain the data, not as drop-in replacements. Using one
means adapting `retrive_dataset()` to its structure. Prefer the primary source.

### What the loader expects after extraction

```
dataset/gtsrb/GTSRB/
├── Final_Training/Images/
│   ├── 00000/  *.ppm + GT-00000.csv
│   ├── 00001/  *.ppm + GT-00001.csv
│   └── ...     through 00042
└── Final_Test/Images/
    ├── *.ppm
    └── GT-final_test.csv
```

The annotation CSVs are semicolon-delimited with columns `Filename; Width;
Height; Roi.X1; Roi.Y1; Roi.X2; Roi.Y2; ClassId`. The ROI columns matter — the
loader crops to them, discarding the ~10% background margin in each raw file.

---

## 3. Option A — Docker

Reproducible, isolated, and installs nothing on your machine beyond Docker
itself. The container runs training, evaluation, prediction and Jupyter.

**It does not run `gui.py`** — the Tkinter desktop app needs a display server,
which means X11 forwarding, which on macOS means XQuartz and a good deal of
fiddling. If you want the desktop app, use [Option B](#4-option-b--native-python).
Nothing is lost: `gui.py` reads the same model files the container writes.

### Prerequisites

[Docker Desktop](https://www.docker.com/products/docker-desktop/) (macOS or
Windows) or Docker Engine (Linux). Check it is running:

```bash
docker --version
docker compose version
```

### Build

```bash
cd ~/Desktop/TrafficSignDetection
docker build -t traffic-sign-detection .
```

First build takes a few minutes, mostly downloading wheels. Later builds reuse
the cached dependency layer and are near-instant unless `requirements.txt`
changed.

### Run

**Jupyter notebook** — the guided walkthrough:

```bash
docker compose up
```

Open <http://localhost:8888> and click `Project_Walkthrough.ipynb`. There is no
token to paste; auth is disabled because the port is bound to your machine
only. `Ctrl-C` to stop.

**One-off commands** — training, evaluation, prediction:

```bash
docker compose run --rm tsd python train.py --epochs 40
docker compose run --rm tsd python evaluate.py --no-plot
docker compose run --rm tsd python predict.py dataset/my_sign.jpg
```

**An interactive shell inside the container:**

```bash
docker compose run --rm tsd bash
```

### How the volumes work

`docker-compose.yml` bind-mounts four paths, and the reason matters:

| Host path | In container | Why |
|---|---|---|
| `./dataset` | `/app/dataset` | The 300 MB download survives rebuilds and `docker compose down` |
| `./models` | `/app/models` | Trained models land on your disk, where native `gui.py` can open them |
| `./src` | `/app/src` | Edit the code on the host, rerun in the container, no rebuild |
| `./Project_Walkthrough.ipynb` | `/app/...` | Notebook edits and outputs persist |

Because `dataset/` and `models/` are bind mounts rather than baked into the
image, `.dockerignore` excludes them from the build context. Without that, every
`docker build` would spend minutes shipping the dataset to the daemon.

### Notes on plots

`MPLBACKEND=Agg` is set in the image, so matplotlib renders without a display.
Plots appear inline in Jupyter as normal. If you call a plotting function from
`docker compose run`, it will render to nothing — pass `--no-plot` to
`evaluate.py` when running it that way.

---

## 4. Option B — native Python

Needed if you want the Tkinter desktop app. Also simply faster, since there is
no VM between the code and your CPU — noticeably so on Apple Silicon, where
NumPy uses the Accelerate framework directly.

### Prerequisites

Python 3.7 or newer; 3.10/3.11 recommended.

```bash
python3 --version
```

### Install

```bash
cd ~/Desktop/TrafficSignDetection

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install notebook                 # only if you want the walkthrough
```

Verify:

```bash
python -c "import numpy, scipy, PIL, matplotlib; print('core dependencies OK')"
python -c "import tkinter; print('tkinter OK -- gui.py will run')"
```

If the `tkinter` check fails, everything except `gui.py` still works. To fix it:

| Platform | Fix |
|---|---|
| macOS (Homebrew Python) | `brew install python-tk` |
| macOS (python.org installer) | already included — reinstall Python from python.org |
| Debian / Ubuntu | `sudo apt install python3-tk` |
| Fedora | `sudo dnf install python3-tkinter` |
| Windows | already included with the official installer |

---

## 5. Step-by-step first run

Assuming the native install above. For Docker, prefix each command with
`docker compose run --rm tsd`.

### Step 1 — Sanity check on a small sample

Before committing to a full run, prove the pipeline works end to end. This
downloads and decodes the dataset (the slow part, done once) and trains briefly
on a fifth of it.

```bash
python train.py --sample 20 --epochs 5 --no-plot --out models/smoke_test
```

What you should see, in order:

```
GTSRB_Final_Training_Images.zip: downloading...
190.0 MB  [190.0 MB done ==========> 100%]
...
Extracting GTSRB_Final_Training_Images.zip...
decoded 42/42
decoded 39209 training and 12630 test images.
Caching the decoded dataset to dataset/gtsrb/gtsrb_32.npz

Set          Images                     Labels
Training     (7057, 32, 32)             (7057, 1)
...
Network architecture: [1024, 512, 256, 43]

Training The Model...
Epoch 1/5
55/55 [====================  100%] - 1.42s | loss: 3.1204 | acc: 0.1893 | val_loss: 2.8871 | val_acc: 0.2551
...
Model saved to models/smoke_test
```

**Loss falling and accuracy rising across epochs is the signal you want.** If
loss goes to `nan` or accuracy stays at ~0.023 (which is 1/43, i.e. random),
stop and see [Troubleshooting](#7-troubleshooting).

### Step 2 — The real training run

The dataset is cached now, so this starts immediately.

```bash
python train.py --epochs 40 --hidden 512 256 --optimizer adam \
                --regularizer dropout --keep-prob 0.8 --patience 5 \
                --out models/gtsrb_mlp
```

Roughly 5 minutes at the measured rate. `--patience 5` stops early if dev
accuracy stalls for five epochs, and restores the best parameters rather than
the last ones. A plot of the learning curves appears at the end; add
`--no-plot` to suppress it.

### Step 3 — Evaluate

```bash
python evaluate.py --model models/gtsrb_mlp
```

Prints a per-class precision/recall/F1 table with macro averages, then draws
the 43×43 confusion matrix and sample grids of correct and incorrect
predictions.

**Read the macro F1, not the accuracy.** GTSRB is imbalanced roughly 10:1, so
accuracy flatters a model that neglects the rare classes.

### Step 4 — Classify your own images

```bash
python predict.py my_sign.jpg --model models/gtsrb_mlp
```

Crop tightly to the sign first. GTSRB images are tight crops with a small
margin, and a wide street scene fed in whole resembles nothing in the training
set.

### Step 5 — The desktop app (native only)

```bash
python gui.py --model models/gtsrb_mlp
```

Opens a window with *Open Image* / *Classify* / *Clear*. It shows your original
photo beside the 32×32 crop the network actually receives — useful for seeing
why a prediction went wrong.

### Step 6 — The walkthrough notebook

```bash
jupyter notebook Project_Walkthrough.ipynb        # native
docker compose up                                 # or in Docker
```

54 cells taking the project apart step by step, from raw `.ppm` files through
the backpropagation maths to evaluation. Set `QUICK_RUN = True` in Step 0 for a
fast pass.

---

## 6. Command reference

| Command | What it does |
|---|---|
| `python train.py` | Train with defaults, save to `models/gtsrb_mlp` |
| `python train.py --sample 20 --epochs 5` | Quick run on a fifth of the data |
| `python train.py --hidden 1024 512 256` | Deeper network |
| `python train.py --regularizer l2 --lambd 0.7` | L2 instead of dropout |
| `python train.py --optimizer mgd` | Plain mini-batch gradient descent |
| `python train.py --step-decay 10` | Decay the learning rate every 10 epochs |
| `python train.py --use-augmented` | Include the batches from `augment.py` |
| `python augment.py --aug-count 1` | Generate augmented training data |
| `python evaluate.py --no-plot` | Metrics table only, no figures |
| `python predict.py a.jpg b.png` | Classify several files at once |
| `python gui.py` | Desktop app |

Every script accepts `--help`.

**Flag values:** `--optimizer` takes `adam`, `mgd`, `bgd`, `sgd`.
`--regularizer` takes `dropout`, `l2`, `none`. `--initialization` takes `he`,
`random`.

---

## 7. Troubleshooting

**Download fails or hangs.**
Install the three archives by hand — see [The dataset](#2-the-dataset). The
loader detects them and skips the download.

**`MemoryError`, or the process is killed during training.**
The training matrix is 289 MB and peak usage is ~820 MB. Use `--sample 50` to
halve the data, or reduce `--batch-size`. On Docker Desktop, raise the VM
memory in *Settings → Resources* — the 2 GB default is not enough. If you hit
this with `--use-augmented`, that path needs ~2.5 GB and is the likely cause.

**Loss becomes `nan`.**
Almost always too high a learning rate. Try `--lr 0.0001`. If it persists,
check the inputs are normalized — `X.max()` should be 1.0, not 255.

**Accuracy stuck near 0.023.**
That is 1/43 — the model is guessing. Either the learning rate is far too low,
or `--initialization random` is stalling the early epochs. Use `--initialization he`.

**`ModuleNotFoundError: No module named 'tkinter'`.**
Only affects `gui.py`. See the table in [Option B](#4-option-b--native-python).

**`ModuleNotFoundError: No module named 'scipy'`.**
Only affects `augment.py` and `data_augmentation.py`. `pip install scipy`.

**Plots do not appear.**
In Docker this is expected — `MPLBACKEND=Agg` renders headlessly. Use Jupyter
for figures, or pass `--no-plot` to suppress them.

**`docker compose up` says port 8888 is already in use.**
Something else is on that port. Change the host side in `docker-compose.yml`:
`"8889:8888"`, then browse to <http://localhost:8889>.

**Reclaiming disk space.**
Once `dataset/gtsrb/gtsrb_32.npz` exists, neither the archives nor the
extracted tree are needed — the cache is all the code reads. Deleting them
frees about 590 MB:

```bash
rm dataset/gtsrb/*.zip
rm -rf dataset/gtsrb/GTSRB
```

Keep the `.npz`. If you delete it too, the next run re-downloads everything.
