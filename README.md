# Traffic Sign Detection

A German Traffic Sign Recognition Benchmark (GTSRB) classifier built on a
**feed-forward neural network written from scratch in NumPy** — no TensorFlow,
no PyTorch, no scikit-learn. Forward propagation, backpropagation, Adam,
dropout, L2, early stopping and learning-rate decay are all implemented
directly.

The network core is ported from a hand-written digit (MNIST) project. What is
new here is the data layer: GTSRB is harder than MNIST — 43 classes instead of
10, colour photographs of real signs at wildly varying size, exposure and
distance instead of clean centred glyphs, and a badly imbalanced class
distribution (some classes have 2,000 training images, others 200).

---

## Contents

| Path | What it is |
|---|---|
| `Project_Walkthrough.ipynb` | **Start here** — the whole project step by step, with the maths opened up |
| `SETUP.md` | Install, run and troubleshoot: Docker and native, hardware specs, dataset links |
| `REPORT.md` | Project report in research-paper form — methodology, experimental design, references |
| `src/dataset.py` | GTSRB download, extraction, decoding, preprocessing, splitting, encoding |
| `src/ffnn.py` | The network: init, forward, cost, backward, optimizers, train, predict |
| `src/model_utils.py` | Activations, mini-batching, metrics, confusion matrix, plots, save/load |
| `src/data_augmentation.py` | Rotate / shift / blur / zoom / crop-and-pad generators |
| `train.py` | Train a model and save it |
| `evaluate.py` | Score a saved model on the test set, print metrics, plot the confusion matrix |
| `predict.py` | Classify image files from the command line |
| `gui.py` | Tkinter app: open a photo, see the preprocessed crop and the top-2 prediction |
| `augment.py` | Generate augmented training batches offline |
| `Dockerfile`, `docker-compose.yml` | Reproducible container: CLI plus Jupyter on `localhost:8888` |

`dataset/` and `models/` hold data and trained weights and are gitignored.

---

## Quick start

**Docker** — reproducible, installs nothing but Docker:

```bash
docker compose up                                    # Jupyter at localhost:8888
docker compose run --rm tsd python train.py --epochs 40
docker compose run --rm tsd python evaluate.py --no-plot
```

**Native Python** — needed for the desktop GUI, and faster:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# guided walkthrough -- recommended first pass
pip install notebook
jupyter notebook Project_Walkthrough.ipynb

# or straight to the scripts
python train.py --epochs 40 --hidden 512 256 --optimizer adam --regularizer dropout
python evaluate.py --model models/gtsrb_mlp
python gui.py --model models/gtsrb_mlp
```

Full instructions, hardware requirements and troubleshooting: **[SETUP.md](SETUP.md)**.

The first `train.py` run downloads ~280 MB of GTSRB archives, extracts them and
decodes 51,839 `.ppm` files. Decoding is disk-bound — well under a minute on an
SSD — and the result is cached in `dataset/gtsrb/gtsrb_32.npz`, so every later
run loads in seconds. Budget about 2 GB of free disk.

A full 40-epoch training run takes roughly 5 minutes and peaks around 820 MB of
RAM. There is no GPU path; this is NumPy on the CPU throughout. See
[SETUP.md](SETUP.md) for measured figures and the full requirements.

If the download is blocked on your network, fetch the three archives by hand
from the [official benchmark page](https://benchmark.ini.rub.de/gtsrb_dataset.html)
and drop them in `dataset/gtsrb/`; the loader detects them and skips the
download. Direct links and mirrors are in
[SETUP.md § The dataset](SETUP.md#2-the-dataset), and the exact URLs the code
uses are in `GTSRB_URLS` at the top of `src/dataset.py`.

---

## How the data is prepared

Every image goes through the same five steps, in `dataset.preprocess_image()`:

1. **Crop to the annotated ROI.** GTSRB ships a bounding box per image; the
   raw file includes ~10% margin around the sign that is pure background.
2. **Grayscale.** A fully-connected net on raw RGB would need 3× the input
   weights for little gain — sign shape and pictogram carry most of the signal.
3. **Histogram equalization.** The single most valuable step for GTSRB. Images
   are shot in tunnels, direct sun and dusk; equalizing makes a night shot and
   a noon shot of the same sign look comparable.
4. **Resize to 32×32.** Fixed input size; the network needs a constant feature
   count.
5. **Flatten and scale to 0–1.** `(32, 32) → (1024, 1)`, divided by 255.

Labels become one-hot columns of height 43. The convention throughout is
**column-major**: `X` is `(features, m)` and `Y` is `(classes, m)`, so a
mini-batch is a contiguous slice of columns.

---

## Code flow

### Training (`train.py`)

```
load_dataset(size_in_per)                       src/dataset.py
   ├─ download_dataset()      .zip archives, skipped if present
   ├─ decompress_dataset()    extract in place
   ├─ retrive_dataset()       walk Final_Training/Images/000NN + GT csv,
   │                          preprocess_image() each file  → cached .npz
   └─ sample_dataset()        shuffle, take size_in_per %
                                     │
train_dev_split()  →  90% train / 10% dev        src/dataset.py
                                     │
prep_dataset()     →  flatten_input → normalize_input → one_hot_encoding
                      X:(1024,m)  Y:(43,m)
                                     │
[optional] load_augmented_data()  ← augment.py output, concatenated on axis 1
                                     │
init_layers(1024, 43, hidden)  →  [1024, 512, 256, 43]      src/ffnn.py
init_hyperParams(alpha, epochs, batch, lambd, keep_probs)
                                     │
train(...)                                                   src/ffnn.py
  init_parameters(layers_dim, "he")     W1..WL, b1..bL
  initialize_adam(parameters)           v, s moment buffers
  for each epoch:
      learning_rate_schedule()          optional step decay
      rand_mini_batches(X, Y, size)     src/model_utils.py
      for each mini-batch:
          forward_prop      → relu × (L-1), softmax at L, dropout masks
          softmax_cross_entropy_cost   (+ L2 term if regularizer="l2")
          backward_prop     → dZ = AL - Y at the output, then chain back
          update_parameters → Adam (or plain gradient descent)
          evaluate()        → running batch acc / loss
      evaluate(dev)                     val_acc, val_loss
      early stopping        best params checkpointed to a temp dir,
                            reloaded when patience runs out
  → history {parameters, accuracy, loss, val_accuracy, val_loss}
                                     │
save_model("models/gtsrb_mlp", {...})            src/model_utils.py (pickle)
visualize_training_results(...)                  loss + accuracy curves
```

### Evaluation (`evaluate.py`)

```
load_model → parameters
load_dataset → test set → prep_dataset
predict(X, parameters, second_guess=True)        src/ffnn.py
   forward_prop (no dropout) → argmax and max over the softmax column
   second guess = argmax after zeroing the winner
confusion_matrix(y, prediction, num_classes=43)  src/model_utils.py
model_metrics(cm) → per-class precision / recall / F1 + macro averages
metric_summary(...)          printed table
plot_confusion_matrix(cm)    43×43 heat map
visualize_prediction(...) / visualize_mislabelled_images(...)
```

### Inference (`predict.py`, `gui.py`)

```
image file → PIL.Image → preprocess_image() → reshape(1024,1) / 255.
          → predict() → {"First Prediction": [label, prob],
                         "Second Prediction": [label, prob]}
          → label_description()[label]   e.g. 14 → "Stop"
```

### Augmentation (`augment.py`)

```
load_dataset → train_dev_split → data_generator()   src/data_augmentation.py
   for each pass, for each chunk of images:
       augment_img(crop_and_pad, rotate, shift, blur, zoom)
       → 5 variants + original, shuffled
   prep_dataset() the chunk, pickle it to dataset/augmented_data/batch_N
```

---

## The network

`layers_dim = [1024, 512, 256, 43]` by default — two hidden layers.

- **Activations:** ReLU on hidden layers, softmax on the output.
- **Loss:** softmax cross-entropy, computed from logits
  (`z - log Σ exp z`) rather than from probabilities, which avoids the
  `log(0)` blow-up when the network becomes confident.
- **Initialization:** He (`√(2/n_prev)`) by default — the right scale for ReLU;
  `random` (×0.01) is available for comparison.
- **Optimizers:** `adam` (default), `mgd` (mini-batch), `bgd` (full batch),
  `sgd` (single example).
- **Regularization:** `dropout` (inverted, applied to hidden layers only, and
  disabled at prediction time) or `l2`. Never both.
- **Early stopping:** `--patience N` keeps the best-validation-accuracy
  parameters and restores them if N epochs pass without improvement.

### What changed from the digit version

| | Digits (MNIST) | Traffic signs (GTSRB) |
|---|---|---|
| Input | 28×28 grayscale, 784 features | 32×32 grayscale, 1024 features |
| Classes | 10 | 43 |
| Source format | IDX binary blobs | per-class folders of `.ppm` + CSV annotations |
| Preprocessing | normalize only | ROI crop → grayscale → equalize → resize |
| Rotation range | ±60° | ±15° |
| Shift range | ±7 px | ±3 px |
| Horizontal flip | used | **never** — it turns "Turn left ahead" into a mislabelled "Turn right ahead" |
| Confusion matrix | 10×10, annotated cells | 43×43 heat map, cell text suppressed |
| GUI input | draw on canvas or upload | upload only (you cannot hand-draw a sign) |

Three latent bugs in the original code were fixed while porting:

- `precision`/`recall`/`f1_score` divided by zero and returned `nan` for any
  class the model never predicted, which poisoned the macro average. With 43
  classes and a long tail, that is the normal case, not an edge case. They now
  score 0.
- `blur_images(filter_mode="random")` — the documented default — fell through
  to a `ValueError` instead of picking a random filter.
- Early stopping wrote checkpoints to a `temp/` directory in the working
  directory and left it behind on an interrupted run. It now uses a private
  scratch directory that is always cleaned up.

---

## Command reference

```bash
# train
python train.py --epochs 40 --hidden 512 256 --lr 0.001 --batch-size 128 \
                --optimizer adam --initialization he \
                --regularizer dropout --keep-prob 0.8 \
                --patience 5 --out models/gtsrb_mlp

# train on 20% of the data, for a quick sanity run
python train.py --sample 20 --epochs 5 --no-plot

# offline augmentation, then train on it
python augment.py --aug-count 1
python train.py --use-augmented --epochs 30

# evaluate
python evaluate.py --model models/gtsrb_mlp
python evaluate.py --model models/gtsrb_mlp --no-plot   # metrics table only

# classify files
python predict.py path/to/sign.jpg another.png --model models/gtsrb_mlp

# desktop app
python gui.py --model models/gtsrb_mlp
```

`--regularizer` accepts `dropout`, `l2` or `none`. `--optimizer` accepts
`adam`, `mgd`, `bgd` or `sgd`.

---

## Expectations and limits

A fully-connected network on flattened pixels has no translation invariance and
no notion of local structure — it learns each pixel position independently. On
GTSRB it lands well below what a small CNN reaches on the same input, and the
gap is the point: this project is about implementing the mechanics, not about
topping the benchmark. If you want the accuracy, a convolutional model is the
next step, and `src/dataset.py` will feed one unchanged if you skip
`flatten_input`.

Two other honest caveats:

- **Recognition, not detection.** The model classifies an image that already
  contains one cropped sign. It does not find signs in a street scene; that
  needs a detector (sliding window, region proposals or a single-shot model).
- **Class imbalance is not corrected.** Rare classes will show visibly lower
  recall in the metrics table. Class weighting or targeted oversampling of the
  rare classes is the obvious improvement.

---

## Requirements

Python 3.7+, and:

```
numpy       the network itself
Pillow      image decoding and preprocessing
scipy       ndimage transforms for augmentation
matplotlib  training curves, confusion matrix, sample grids
```

`gui.py` also needs Tkinter, which ships with most Python installs (on Debian
and Ubuntu: `sudo apt install python3-tk`; on Homebrew Python: `brew install
python-tk`). `Project_Walkthrough.ipynb` needs `notebook`.

Or skip all of it and use the container — see [SETUP.md](SETUP.md).

---

## Credits

The from-scratch network, metric and augmentation code is adapted from the
MNIST hand-written digit recognition project by **befrenz**
([kattelsameer/HandWritten-Digit-Recognition-using-Deep-Learning](https://github.com/kattelsameer/HandWritten-Digit-Recognition-using-Deep-Learning)).

GTSRB is published by the Institut für Neuroinformatik, Ruhr-Universität
Bochum: J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, *Man vs. computer:
Benchmarking machine learning algorithms for traffic sign recognition*, Neural
Networks, 2012.
