# Traffic Sign Recognition Using a From-Scratch Feed-Forward Neural Network

### A study of the German Traffic Sign Recognition Benchmark using a multi-layer perceptron implemented without machine learning frameworks

**Author:** _[your name]_
**Institution:** _[your institution]_
**Supervisor:** _[supervisor]_
**Date:** _[submission date]_

---

> **STATUS OF THIS DOCUMENT**
>
> All sections are complete. Section 7 reports results from training runs performed on 26 September 2026: the E0 baseline and every ablation, E1–E6, 17 runs in total, each with a single seed (1). The trained models, logs, metric files and figures behind every number are in the `results/` folder of the repository.
>
> Results are single-seed, so differences of about one percentage point between configurations are within noise (Section 9.5). E5 was run through a float32 input wrapper because the unmodified code ran out of memory (Section 7.4).

---

## Abstract

Traffic sign recognition is a core perception task in driver assistance and
autonomous driving, and one where the cost of a misclassification is
asymmetric: mistaking a speed limit sign for another speed limit sign is a
different kind of failure from missing a stop sign entirely. This report
presents a traffic sign classifier for the German Traffic Sign Recognition
Benchmark (GTSRB) — 51,839 images across 43 classes — built on a fully
connected feed-forward neural network implemented from first principles in
NumPy. No machine learning framework is used: forward propagation,
backpropagation, the Adam optimizer, inverted dropout, L2 regularization, He
initialization and early stopping are each derived and implemented directly.

The contribution is not benchmark performance. A fully connected network
operating on flattened pixels lacks both translation invariance and any notion
of spatial locality, and is therefore structurally disadvantaged against the
convolutional architectures that define the state of the art on this dataset.
The contribution is instead a complete, inspectable and reproducible
implementation of the underlying mechanics, together with an experimental
design that isolates the contribution of each component — initialization
scheme, optimizer, regularizer, network depth and data augmentation — and an
honest characterisation of where the architecture's ceiling lies and why.

A methodological finding is reported independently of any training run: three
transformation choices inherited from the digit-recognition domain are
demonstrably invalid for traffic signs. Horizontal mirroring, standard in
digit and general-object augmentation, systematically produces mislabelled
examples for the eight directional sign classes in GTSRB, since a mirrored
"Turn left ahead" is a valid image of "Turn right ahead" carrying the wrong
label.

**Keywords:** traffic sign recognition, GTSRB, multi-layer perceptron,
backpropagation, Adam optimization, dropout, data augmentation, class
imbalance

---

## Table of contents

1. [Introduction](#1-introduction)
2. [Related work](#2-related-work)
3. [Dataset](#3-dataset)
4. [Methodology](#4-methodology)
5. [Experimental setup](#5-experimental-setup)
6. [Evaluation metrics](#6-evaluation-metrics)
7. [Results](#7-results)
8. [Discussion](#8-discussion)
9. [Limitations and threats to validity](#9-limitations-and-threats-to-validity)
10. [Conclusion and future work](#10-conclusion-and-future-work)
- [References](#references)
- [Appendix A — Reproducibility](#appendix-a--reproducibility)
- [Appendix B — Class index](#appendix-b--class-index)
- [Appendix C — Notation](#appendix-c--notation)

---

## 1. Introduction

### 1.1 Background

Automatic traffic sign recognition sits in the perception layer of advanced
driver assistance systems and autonomous vehicles. The task is superficially
easy — signs are standardised, high-contrast, and designed by regulation to be
legible at distance — and in practice hard, because the images a vehicle
actually captures are degraded in ways the designers never intended: motion
blur, low sun, night, rain, partial occlusion by foliage, faded or vandalised
faces, and severe perspective distortion when a sign is approached off-axis.

Stallkamp et al. [1] established the German Traffic Sign Recognition Benchmark
in 2011 to make these methods comparable. Their headline result is worth
restating, because it sets the bar: human classification accuracy on GTSRB is
98.84%, and the best submitted machine learning system exceeded it at 99.46%,
using a committee of convolutional neural networks [2]. Traffic sign
recognition is thus one of the earliest vision tasks on which machines
demonstrably surpassed human performance.

### 1.2 Problem statement

Given a cropped image containing exactly one German traffic sign, assign it to
one of 43 classes. This is a **closed-set, single-label classification**
problem. It is explicitly *not* detection: locating signs within a full street
scene is a distinct problem requiring a different class of model, and is out of
scope (see Section 9.2).

### 1.3 Motivation for the approach

Modern frameworks make it possible to train an accurate classifier without
understanding what the training loop does. That convenience has a cost in
educational settings: the gradient computation, the optimizer's state, the
behavioural difference between dropout at training and at inference time — all
of it is behind an API.

This project takes the opposite approach deliberately. Every component is
implemented in NumPy and can be inspected, unit-checked, and reasoned about
line by line. The accompanying notebook (`Project_Walkthrough.ipynb`) executes
each stage of a training step in isolation so that the intermediate tensors are
directly observable.

The cost of this choice is accepted openly: the architecture cannot compete
with a convolutional model on this dataset. Section 8.4 quantifies why, and
Section 10.2 sets out the path to closing the gap.

### 1.4 Objectives

1. Implement a complete multi-layer perceptron in NumPy — forward propagation,
   backpropagation, Adam, dropout, L2, He initialization, early stopping and
   learning rate scheduling — with no machine learning framework.
2. Build a preprocessing pipeline that adapts GTSRB's variable-size, variable-
   exposure colour photographs into fixed-length feature vectors.
3. Design and specify a controlled experimental protocol isolating the effect
   of each architectural and training choice.
4. Evaluate with metrics appropriate to an imbalanced multi-class problem,
   rather than accuracy alone.
5. Establish which augmentation transformations are valid in the traffic sign
   domain, as opposed to inherited unexamined from other image domains.
6. Deliver the whole thing reproducibly: pinned dependencies, a container
   definition, and a documented single-command path from clean checkout to
   trained model.

### 1.5 Scope

**In scope:** classification of pre-cropped single signs; grayscale
preprocessing; fully connected architectures; the training and regularization
methods listed above; offline data augmentation; per-class evaluation.

**Out of scope:** sign detection or localisation; video and temporal
tracking; convolutional, recurrent or attention-based architectures; real-time
embedded deployment; colour-channel modelling (see Section 9.1).

### 1.6 Contributions

- **C1.** A complete framework-free implementation of a trainable MLP,
  documented to the level of individual equations (Section 4).
- **C2.** A GTSRB preprocessing pipeline with each stage justified against a
  specific property of the data (Section 4.2).
- **C3.** A domain analysis of augmentation validity, identifying horizontal
  mirroring as label-corrupting for this dataset and correcting two
  transformation ranges inherited from the digit domain (Section 4.9).
- **C4.** A controlled experimental protocol of six ablations (Section 5.4).
- **C5.** Three defects identified and corrected in the antecedent
  digit-recognition codebase, two of which are silently wrong rather than
  loudly wrong (Section 4.10).
- **C6.** Measured resource characterisation of the training procedure
  (Section 5.1).

---

## 2. Related work

### 2.1 The benchmark and its results

GTSRB was introduced as a competition at IJCNN 2011 [1]. The final standings
are instructive about what matters on this task:

| Rank | Method | Accuracy | Note |
|---|---|---|---|
| 1 | Committee of CNNs [2] | 99.46% | Ensemble of 25 networks |
| 2 | **Human performance** [1] | **98.84%** | Measured over 32 test subjects |
| 3 | Multi-scale CNN [3] | 98.31% | Features from multiple layers to the classifier |
| 4 | Random forests | 96.14% | On HOG features |
| 5 | LDA on HOG | 95.68% | Classical baseline |

Two things follow. First, convolutional architectures dominate the top of the
table decisively. Second, even the classical HOG-based methods clear 95%,
which sets a meaningful reference point: a method that cannot beat LDA on HOG
features is not exploiting the structure of the problem.

### 2.2 Classical approaches

Before CNNs, GTSRB systems were pipelines of hand-designed features and a
conventional classifier. Histogram of Oriented Gradients (HOG) [4] was the
dominant descriptor, capturing local edge orientation statistics — well matched
to signs, whose identity is largely carried by shape and pictogram outline.
Colour segmentation in HSV space was often used as a first-stage filter,
exploiting the regulatory colour coding (red for prohibitory, blue for
mandatory, yellow for temporary).

The relevance to this work is that these methods succeed by *engineering
translation robustness into the features*. A fully connected network on raw
pixels has neither engineered features nor learned translation invariance,
which is precisely the deficit analysed in Section 8.4.

### 2.3 Convolutional approaches

Ciresan et al. [2] won the benchmark with a committee of deep CNNs trained on
augmented data. Sermanet and LeCun [3] introduced a multi-scale architecture
feeding both early and late convolutional stages to the classifier, on the
argument that traffic sign identity depends on both fine detail (the digits
inside a speed limit sign) and coarse shape (the sign's outline). Their result
also established that grayscale input costs surprisingly little accuracy on
GTSRB relative to full colour — a finding this project's preprocessing relies
on (Section 4.2).

### 2.4 Multi-layer perceptrons on image data

The theoretical position of the MLP is well established. The universal
approximation theorem [5] guarantees that a single hidden layer of sufficient
width can approximate any continuous function on a compact domain — so
representational capacity is not the binding constraint here. The binding
constraint is **sample efficiency**. A fully connected layer treats input
position 100 and position 101 as unrelated coordinates. It must therefore learn
the appearance of a sign separately at every position it can occupy, and every
such position requires its own training examples. LeCun et al. [6] identified
this as the motivation for weight sharing in convolutional networks.

The methods this implementation uses are individually standard: He
initialization [7], scaled for the ReLU nonlinearity; the Adam optimizer [8],
combining momentum with per-parameter adaptive step sizes; and inverted dropout
[9] as a stochastic regularizer.

### 2.5 Positioning of this work

This project does not aim to advance the state of the art on GTSRB, and it
would be dishonest to present it as doing so. It occupies the pedagogical
position: a complete, transparent implementation of the mechanics that
frameworks abstract, applied to a real dataset with real difficulties, together
with an experimental design that makes the resulting limitations legible rather
than mysterious.

---

## 3. Dataset

### 3.1 Description

The German Traffic Sign Recognition Benchmark [1] is published by the Institut
für Neuroinformatik, Ruhr-Universität Bochum, under CC BY 4.0.

| Property | Value |
|---|---|
| Classes | 43 |
| Training images | 39,209 |
| Test images | 12,630 |
| Total | 51,839 |
| Format | PPM (uncompressed RGB) |
| Image dimensions | Variable, 15×15 to 250×250 px |
| Annotation | Per-image ROI bounding box and class id, semicolon-delimited CSV |
| Colour | 24-bit RGB |
| Licence | CC BY 4.0 |

### 3.2 Provenance and structure

Images were extracted from roughly 10 hours of video recorded on German roads.
Critically, **each physical sign contributes a track of 30 consecutive frames**
as the vehicle approaches it. Frames within a track are highly correlated: they
differ mainly in scale and slightly in viewing angle.

This has a direct methodological consequence. Randomly splitting the training
set into training and validation partitions places frames of the *same physical
sign* on both sides of the split, so validation accuracy is optimistically
biased. The official train/test partition is track-disjoint and does not suffer
from this; the internal dev split used here does. This is recorded honestly as
a threat to validity in Section 9.3 rather than glossed over.

### 3.3 Class distribution

The dataset is materially imbalanced. Training images per class range from 210
(class 0, "Speed limit 20") to 2,250 (class 2, "Speed limit 50") — a ratio of
roughly 10.7:1. The distribution reflects real-world sign frequency rather than
sampling error, which is arguably the right choice for a benchmark but makes
accuracy a poor headline metric (Section 6.1).

*Figure 3.1 — class distribution histogram. Generated by
`visualize_data_distribution(train_y, "training")`.*

### 3.4 Sources of difficulty

| Source | Description | Mitigation in this work |
|---|---|---|
| Illumination | Tunnels, direct sun, dusk, headlights | Histogram equalization (§4.2) |
| Scale | 15×15 to 250×250 px | Resize to fixed 32×32 (§4.2) |
| Motion blur | Vehicle in motion | Blur augmentation (§4.9) |
| Perspective | Off-axis approach | Partially, via rotation and shift (§4.9) |
| Occlusion | Foliage, stickers, poles | **Not addressed** |
| Inter-class similarity | Speed limits differ only in digits | **Architectural limitation** (§8.4) |
| Class imbalance | 10.7:1 | Reported per class (§6); **not corrected** |
| Physical degradation | Fading, vandalism, damage | **Not addressed** |

The three "not addressed" entries are stated deliberately. They are known
weaknesses of this system, not oversights.

---

## 4. Methodology

### 4.1 System overview

```
    Raw .ppm                Preprocessing              Network              Output
   ┌──────────┐   ┌───────────────────────────┐   ┌─────────────┐   ┌──────────────┐
   │ variable │   │ ROI crop                  │   │  1024       │   │  43 softmax  │
   │ size RGB │──▶│ grayscale                 │──▶│  ↓ 512 ReLU │──▶│ probabilities│
   │ + ROI    │   │ histogram equalize        │   │  ↓ 256 ReLU │   │  → argmax    │
   │ + label  │   │ resize 32×32              │   │  ↓  43      │   │  → 2nd guess │
   └──────────┘   │ flatten → 1024, ÷255      │   └─────────────┘   └──────────────┘
                  └───────────────────────────┘
```

The implementation is four modules with a strict acyclic dependency order:

```
dataset.py ──▶ data_augmentation.py ──▶ model_utils.py ──▶ ffnn.py
```

`dataset.py` has no project dependencies and owns everything from acquisition
to encoding. `ffnn.py` contains only the network and depends on nothing but
`model_utils.py`.

**Data representation.** All matrices are **column-major**: the design matrix
$X \in \mathbb{R}^{n \times m}$ has features down the rows and examples across
the columns, and labels $Y \in \{0,1\}^{43 \times m}$ likewise. A mini-batch is
therefore a contiguous column slice, and the forward pass is a single matrix
product per layer with no transposition.

### 4.2 Preprocessing

Each image undergoes five deterministic transformations. Each is justified
against a specific property of GTSRB rather than adopted by convention.

**1. Crop to the annotated ROI.** GTSRB supplies a bounding box per image; the
raw file carries approximately 10% background margin beyond it. Since a fully
connected network assigns an independent weight to every input coordinate, that
margin is 20–30% of the input budget spent on pixels containing, by
construction, no signal. The crop reclaims it.

**2. Convert to grayscale.** Retaining RGB triples the first layer's parameter
count — from 524,288 to 1,572,864 weights at the default width — for
information that is largely redundant with shape. Sermanet and LeCun [3] report
that grayscale costs little accuracy on GTSRB. This is a considered trade-off,
not a free choice, and Section 9.1 records what it gives up.

**3. Histogram equalization.** The highest-value step in the pipeline for this
dataset. GTSRB's illumination range is extreme, and equalization redistributes
each image's intensity histogram towards uniform, so that a tunnel exposure and
a noon exposure of the same sign present comparable inputs. Without it the
network must devote capacity to learning illumination invariance from data.

**4. Resize to 32×32 bilinear.** A fixed input dimension is required. 32×32 is
chosen as the smallest size retaining the pictogram detail that separates
visually adjacent classes, at 1,024 features. 28×28 (the MNIST convention)
degrades speed limit digits; 48×48 raises the first layer to 1.18M weights for
marginal detail gain.

**5. Flatten and scale.** $(32,32) \rightarrow (1024,1)$, divided by 255 to
place every feature in $[0,1]$. Unnormalised inputs in $[0,255]$ produce first-
layer gradients two orders of magnitude larger than later layers', which forces
a learning rate too small for the rest of the network.

Labels are one-hot encoded to $\{0,1\}^{43}$.

### 4.3 Network architecture

An $L$-layer fully connected network with widths
$[1024, h_1, \ldots, h_{L-1}, 43]$; the default is $[1024, 512, 256, 43]$.

| Layer | Shape of $W$ | Shape of $b$ | Parameters | Activation |
|---|---|---|---|---|
| 1 | (512, 1024) | (512, 1) | 524,800 | ReLU |
| 2 | (256, 512) | (256, 1) | 131,328 | ReLU |
| 3 | (43, 256) | (43, 1) | 11,051 | Softmax |
| | | **Total** | **667,179** | |

ReLU is used on hidden layers for its non-saturating gradient; softmax on the
output to produce a normalised distribution over the mutually exclusive
classes.

### 4.4 Weight initialization

He initialization [7]:

$$W^{[l]} \sim \mathcal{N}\!\left(0, \frac{2}{n^{[l-1]}}\right), \qquad b^{[l]} = 0$$

The factor $2/n^{[l-1]}$ preserves activation variance across a ReLU layer.
Since ReLU zeroes half its input in expectation, halving the variance, the
factor 2 compensates; without it, activation magnitude decays geometrically
with depth and deep networks train slowly or not at all. Biases start at zero
because the weights already break symmetry.

The alternative implemented for comparison is $W \sim 0.01 \cdot
\mathcal{N}(0,1)$, a depth-independent scale. Experiment E1 (Section 5.4)
quantifies the difference.

### 4.5 Forward propagation

For $l = 1 \ldots L-1$:

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}, \qquad A^{[l]} = \max(0, Z^{[l]})$$

with $A^{[0]} = X$, and at the output layer:

$$Z^{[L]} = W^{[L]} A^{[L-1]} + b^{[L]}, \qquad
\hat{Y}_{k,i} = \frac{e^{Z^{[L]}_{k,i} - \max_j Z^{[L]}_{j,i}}}
{\sum_{c} e^{Z^{[L]}_{c,i} - \max_j Z^{[L]}_{j,i}}}$$

The subtraction of the column maximum is a numerical necessity, not a
refinement: raw logits of even modest magnitude overflow the exponential in
double precision. Subtracting the maximum leaves the softmax mathematically
unchanged while bounding every exponent at zero.

**Caching.** Each layer stores $(A^{[l-1]}, W^{[l]}, b^{[l]})$ and $Z^{[l]}$ on
the forward pass. Backpropagation requires exactly these; recomputing them
would double the cost of a training step.

### 4.6 Loss function

Categorical cross-entropy over $m$ examples:

$$J = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{43} Y_{k,i} \log \hat{Y}_{k,i}$$

Since $Y$ is one-hot, only the true class contributes: the loss is the negative
log-probability assigned to the correct answer.

**Computation from logits.** Evaluating $\log(\text{softmax}(z))$ in two steps
risks $\log(0)$ once the network becomes confident and a probability underflows
to zero. The implementation instead uses the algebraically equivalent but
numerically stable form

$$\log \hat{y}_k = z_k - \max_j z_j - \log\sum_c e^{z_c - \max_j z_j}$$

which cannot underflow. This is why the caches retain $Z$ rather than only $A$.

With L2 regularization the objective gains

$$J_{L2} = J + \frac{\lambda}{2m}\sum_{l=1}^{L}\|W^{[l]}\|_F^2$$

### 4.7 Backpropagation

At the output layer, the softmax and cross-entropy derivatives compose to

$$dZ^{[L]} = \hat{Y} - Y$$

which is why the softmax branch of the backward activation is an identity pass-
through. Then for $l = L \ldots 1$:

$$dW^{[l]} = \frac{1}{m} dZ^{[l]} A^{[l-1]T} + \frac{\lambda}{m}W^{[l]}, \qquad
db^{[l]} = \frac{1}{m}\sum_{i} dZ^{[l]}_{:,i}, \qquad
dA^{[l-1]} = W^{[l]T} dZ^{[l]}$$

and through the ReLU,

$$dZ^{[l]} = dA^{[l]} \odot \mathbb{1}[Z^{[l]} > 0]$$

**Shape assertions.** Every gradient must match the shape of its parameter, and
this is asserted at the point of computation. NumPy broadcasting will silently
produce a wrongly-shaped but valid array from a transposition error, and the
resulting failure surfaces many layers later as poor convergence rather than as
an exception. Asserting at the source converts a silent bug into a loud one.

### 4.8 Optimization

**Adam** [8], with per-parameter first and second moment estimates:

$$v_t = \beta_1 v_{t-1} + (1-\beta_1) dW_t, \qquad
s_t = \beta_2 s_{t-1} + (1-\beta_2) dW_t^2$$

Both are initialised at zero and are therefore biased towards zero early in
training; the bias-corrected estimates are

$$\hat{v}_t = \frac{v_t}{1-\beta_1^t}, \qquad \hat{s}_t = \frac{s_t}{1-\beta_2^t}$$

giving the update

$$W_t = W_{t-1} - \alpha\frac{\hat{v}_t}{\sqrt{\hat{s}_t}+\epsilon}$$

Defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. The counter
$t$ increments **per mini-batch, not per epoch**, and is never reset — the bias
correction is defined in terms of the number of updates applied.

Three alternatives are implemented for comparison: batch gradient descent
(mini-batch = $m$), stochastic gradient descent (mini-batch = 1), and
mini-batch gradient descent without adaptivity.

**Learning rate scheduling** is available as an optional decay applied every
$k$ epochs, floored at $10^{-4}$.

### 4.9 Regularization

**Inverted dropout** [9]. During training each hidden activation is multiplied
by a Bernoulli mask and divided by the keep probability:

$$A^{[l]} \leftarrow \frac{A^{[l]} \odot D^{[l]}}{p}, \qquad D^{[l]}_{ij} \sim \text{Bernoulli}(p)$$

The division is what makes the scheme *inverted*: it preserves
$\mathbb{E}[A^{[l]}]$, so inference requires no compensating rescaling — dropout
is simply not applied. The same masks are reused in the backward pass, so a
dropped unit receives no gradient. Dropout is applied to hidden layers only,
never to the output.

**L2 regularization** as given in Section 4.6, penalising weight magnitude. The
two are treated as alternatives, not combined.

**Early stopping.** Dev accuracy is evaluated each epoch; parameters are
checkpointed whenever it improves, and after $P$ epochs without improvement
training halts and the **best** parameters are restored rather than the last.

### 4.10 Data augmentation

Five transformations are implemented: random rotation, random shift, blur
(Gaussian, maximum, minimum, median or uniform), zoom, and crop-and-pad. Each
pass produces five variants per source image.

**Domain validity of transformations (contribution C3).** The transformation
set was inherited from a hand-written digit recognition system, and three
choices proved invalid when re-examined for traffic signs:

| Transformation | Digit setting | Traffic sign setting | Reason |
|---|---|---|---|
| Horizontal flip | Used | **Removed** | Signs are not chirality-invariant. A mirrored "Turn left ahead" (class 34) is a valid image of "Turn right ahead" (class 33) but retains the label 34. This *manufactures mislabelled training data* for eight directional classes. |
| Rotation | ±60° | **±15°** | A digit at 40° is still that digit. A sign at 40° is a photograph that does not occur; capacity spent on it is wasted. |
| Shift | ±7 px on 28×28 (25%) | **±3 px on 32×32 (9%)** | Post-ROI-crop the sign fills the frame; a 25% shift translates a substantial fraction of it out of view. |

The horizontal flip case is the substantive one. It is not a tuning preference
but a label-correctness error, and it is silent: training proceeds normally and
the damage appears only as depressed accuracy on the affected classes. The
function is retained in the codebase with a documented warning and is never
invoked by the generator.

**Defects corrected in the antecedent codebase (contribution C5).** Three
further issues were found and fixed:

1. `precision()` and `recall()` divided by zero for any class the model never
   predicted, returning `nan` and propagating it into the macro average. With
   43 classes and a long tail this is the expected case, not an edge case.
   Corrected to score 0.
2. `blur_images(filter_mode="random")` — the documented default — fell through
   its dispatch chain to a `ValueError`. The random branch was reachable only
   via a second, separate flag.
3. Early stopping wrote checkpoints to a `temp/` directory in the working
   directory and left it behind if a run was interrupted. Moved to a managed
   scratch directory.

Defects 1 and 3 are silent; only 2 announces itself.

---

## 5. Experimental setup

### 5.1 Hardware and resource characterisation

Resource measurements below were taken on a **2-core x86-64 Linux container**.
Memory figures follow from array sizes and are machine-independent; timings are
a conservative upper bound, since the reference machine is deliberately modest.

| Quantity | Measured |
|---|---|
| Training design matrix, $(1024 \times 35289)$ float64 | 289 MB |
| Peak RSS, $[1024,512,256,43]$ | 816 MB |
| Peak RSS, $[1024,1024,512,256,43]$ | 919 MB |
| Peak RSS, augmented ($6\times$ data) | ~2.5 GB |
| Time per epoch, $[1024,512,256,43]$ | 7.0 s |
| Time per epoch, $[1024,1024,512,256,43]$ | 16.1 s |
| Preprocessing throughput | 7,553 images/s |
| Model size on disk | ~5 MB |

Memory is dominated by the design matrix, not the model: 289 MB of data against
5 MB of parameters. A 40-epoch run at the default architecture therefore
completes in roughly 5 minutes on two cores and requires under 1 GB of RAM.

**Measured during the Section 7 runs** (2-core x86-64 container, 8 GB RAM). A baseline epoch took 4–10 s, depending on load. E0 early-stopped after 28 epochs in 2 min 50 s, with peak RSS 1141 MB. The deepest E4 network took 10 min 58 s. The augmented run's memory was much higher than the ~2.5 GB estimated above. `augment.py` peaked at 4.4 GB, and the unmodified `train.py --use-augmented` was OOM-killed at 5.8 GB. Holding the inputs as float32 brought it to 3.5 GB.

**No GPU is used or usable.** The implementation is NumPy throughout and has no
accelerator path.

### 5.2 Software environment

| Component | Version |
|---|---|
| Python | 3.7+ (3.11 in the container image) |
| NumPy | ≥1.19 |
| Pillow | ≥8.0 |
| SciPy | ≥1.5 (augmentation only) |
| Matplotlib | ≥3.3 (visualisation only) |

A `Dockerfile` and `docker-compose.yml` pin the full environment. No machine
learning framework is present, by design.

### 5.3 Protocol

**Partitioning.** The official GTSRB train/test split is preserved. A 10%
development partition is drawn at random from the training set, giving 35,289
training / 3,920 dev / 12,630 test.

**Test set discipline.** The test partition is used exactly once, for final
evaluation. All model selection, hyperparameter choice and early stopping use
the dev partition only. Repeatedly consulting the test set and adjusting in
response leaks information and inflates the reported figure.

**Seeding.** `np.random.seed(1)` is set at entry to every script. Given
identical data and flags, runs are reproducible.

**Baseline configuration.**

| Hyperparameter | Value |
|---|---|
| Architecture | [1024, 512, 256, 43] |
| Initialization | He |
| Optimizer | Adam ($\beta_1{=}0.9$, $\beta_2{=}0.999$, $\epsilon{=}10^{-8}$) |
| Learning rate | 0.001 |
| Mini-batch size | 128 |
| Epochs | 40 |
| Regularization | Dropout, $p = 0.8$ |
| Early stopping patience | 5 epochs |

### 5.4 Planned experiments

Each varies one factor from the baseline; all other settings and the seed are
held fixed.

| ID | Question | Conditions | Command |
|---|---|---|---|
| **E0** | Baseline | As above | `python train.py --epochs 40` |
| **E1** | Does He initialization help? | `he` vs `random` | `--initialization {he,random}` |
| **E2** | Does adaptivity help? | `adam` vs `mgd` vs `bgd` | `--optimizer {adam,mgd,bgd}` |
| **E3** | Which regularizer? | `dropout` ($p{\in}\{0.6,0.7,0.8,0.9\}$) vs `l2` ($\lambda{\in}\{0.1,0.4,0.7,1.0\}$) vs none | `--regularizer ...` |
| **E4** | Does depth or width help? | $[256]$, $[512,256]$, $[1024,512,256]$, $[1024,512,256,128]$ | `--hidden ...` |
| **E5** | Does augmentation help? | With and without $6\times$ augmented data | `--use-augmented` |
| **E6** | Does LR decay help? | None vs decay every 10 epochs | `--step-decay {0,10}` |

E1 and E2 are expected to show large effects in early epochs and converge
later. E3 and E4 trade bias against variance. E5 is the test of the augmentation
analysis in Section 4.9. Each configuration should be run with at least three
seeds and reported as mean ± standard deviation; single-run differences on this
dataset are not reliably distinguishable from seed noise.

---

## 6. Evaluation metrics

### 6.1 Why accuracy is insufficient

With a 10.7:1 imbalance, a model that performs well on the frequent classes and
poorly on the rare ones posts a respectable accuracy. Accuracy is reported for
comparability with the literature but is **not** the headline metric here.

### 6.2 Per-class metrics

For class $k$ with confusion matrix $C$ (rows true, columns predicted):

$$\text{Precision}_k = \frac{C_{kk}}{\sum_i C_{ik}}, \qquad
\text{Recall}_k = \frac{C_{kk}}{\sum_j C_{kj}}, \qquad
F1_k = \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$

Precision answers "when it says class $k$, is it right?"; recall answers "of the
true class-$k$ images, how many did it find?". Both denominators can be zero —
precision's when the class is never predicted, recall's when it is absent from
the split — and both cases are handled explicitly (Section 4.10).

For traffic signs the asymmetry is real: low recall on "Stop" means missing stop
signs, which is materially worse than low precision on the same class.

### 6.3 Macro averaging

$$\text{Macro-}F1 = \frac{1}{43}\sum_{k=1}^{43} F1_k$$

The unweighted mean gives a rare class exactly the weight of a common one.
**Macro-F1 is the headline metric for this work.**

### 6.4 Top-2 accuracy

The classifier reports a second prediction, obtained by taking the argmax after
suppressing the winner. Top-2 accuracy is informative here because several
class pairs are near-identical at 32×32: a model frequently right in second
place is confusing a specific pair, which is a different and more tractable
failure than not perceiving the sign at all.

### 6.5 Confidence calibration

Mean softmax confidence is reported separately over correct and incorrect
predictions. A well-calibrated model is markedly less confident when wrong,
which makes the confidence score usable as a rejection threshold in
deployment — a property with direct safety relevance.

### 6.6 Reported artefacts

1. Overall accuracy, macro-precision, macro-recall, macro-F1
2. Per-class precision, recall, F1 (43 rows)
3. 43×43 confusion matrix
4. Top-1 and top-2 accuracy
5. Mean confidence, correct vs incorrect
6. Training and dev loss/accuracy curves
7. Sample grid of misclassified images

---

## 7. Results

All figures in this section come from runs performed on 26 September 2026 in a 2-core x86-64 Linux container (8 GB RAM, Python 3.11, NumPy with OpenBLAS), with the code in this repository unmodified. Every run uses seed 1 and the same 35,289 / 3,920 / 12,630 train / dev / test partition. **Each configuration was run once**, so differences smaller than about one percentage point should be read as within seed noise (Section 9.5). The raw outputs — logs, trained models, per-run metrics and every figure — are in the `results/` folder.

### 7.0 How each item was generated

**Table 7.0 — Commands**

| Item | Command |
|---|---|
| All runs (E0–E4, E6) | `experiments/run_all.sh <run-id> …` — calls `train.py --epochs 40 --no-plot` then `evaluate.py --no-plot` per run |
| E5 (augmentation) | `python augment.py --aug-count 1`, then `python experiments/train_lowmem.py --use-augmented` (see Section 7.4) |
| Tables 7.1–7.10, Figures 7.1–7.8 | `python experiments/analyze.py` → `results/summary.json`, `results/*.csv`, `results/figures/` |
| Error analysis (Section 7.3) | `python experiments/error_analysis.py` → `results/e0_error_analysis.json` |

### 7.1 Baseline performance

**Table 7.1 — Baseline (E0) results**

| Metric | Value |
|---|---|
| Test accuracy (top-1) | 88.46% |
| Test accuracy (top-2) | 92.76% |
| Macro-precision | 0.849 |
| Macro-recall | 0.843 |
| **Macro-F1** | **0.841** |
| Training accuracy at restored epoch (23) | 96.71% |
| Best dev accuracy (restored parameters) | 95.08% |
| Train–dev gap at restored epoch | +1.63 pp |
| Final-epoch training / dev accuracy | 97.04% / 94.92% |
| Epochs completed (of 40) | 28 |
| Early stopping triggered | Yes — no dev improvement after epoch 23; epoch-23 parameters restored |
| Wall-clock training time | 2 min 50 s |
| Peak memory (RSS), training | 1141 MB |
| Mean confidence when correct | 0.950 |
| Mean confidence when incorrect | 0.558 |
| Expected calibration error (15 bins) | 0.021 |
| Test errors | 1458 of 12,630 |

![Figure 7.1](results/figures/fig7_1_e0_loss.png)

*Figure 7.1 — E0 training and dev loss per epoch. The dotted line marks the restored epoch.*

![Figure 7.2](results/figures/fig7_2_e0_accuracy.png)

*Figure 7.2 — E0 training and dev accuracy per epoch.*

### 7.2 Per-class performance

**Table 7.2 — E0 per-class test metrics**

| Class | Name | Test support | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 0 | Speed limit 20 | 60 | 0.973 | 0.600 | 0.742 |
| 1 | Speed limit 30 | 720 | 0.901 | 0.887 | 0.894 |
| 2 | Speed limit 50 | 750 | 0.819 | 0.917 | 0.865 |
| 3 | Speed limit 60 | 450 | 0.957 | 0.840 | 0.895 |
| 4 | Speed limit 70 | 660 | 0.925 | 0.856 | 0.889 |
| 5 | Speed limit 80 | 630 | 0.718 | 0.871 | 0.787 |
| 6 | End speed limit 80 | 150 | 0.958 | 0.607 | 0.743 |
| 7 | Speed limit 100 | 450 | 0.887 | 0.851 | 0.868 |
| 8 | Speed limit 120 | 450 | 0.907 | 0.802 | 0.851 |
| 9 | No passing | 480 | 0.950 | 0.992 | 0.970 |
| 10 | No passing >3.5t | 660 | 0.966 | 0.948 | 0.957 |
| 11 | Right-of-way at intersection | 420 | 0.888 | 0.948 | 0.917 |
| 12 | Priority road | 690 | 0.979 | 0.938 | 0.958 |
| 13 | Yield | 720 | 0.966 | 0.994 | 0.980 |
| 14 | Stop | 270 | 0.977 | 0.933 | 0.955 |
| 15 | No vehicles | 210 | 0.935 | 0.895 | 0.915 |
| 16 | No vehicles >3.5t | 150 | 0.980 | 0.973 | 0.977 |
| 17 | No entry | 360 | 0.969 | 0.964 | 0.967 |
| 18 | General caution | 390 | 0.937 | 0.651 | 0.769 |
| 19 | Dangerous curve left | 60 | 0.600 | 0.700 | 0.646 |
| 20 | Dangerous curve right | 90 | 0.738 | 0.689 | 0.713 |
| 21 | Double curve | 90 | 0.497 | 0.856 | 0.629 |
| 22 | Bumpy road | 120 | 0.882 | 0.875 | 0.879 |
| 23 | Slippery road | 150 | 0.813 | 0.813 | 0.813 |
| 24 | Road narrows on right | 90 | 0.782 | 0.756 | 0.768 |
| 25 | Road work | 480 | 0.952 | 0.863 | 0.905 |
| 26 | Traffic signals | 180 | 0.622 | 0.722 | 0.668 |
| 27 | Pedestrians | 60 | 0.568 | 0.417 | 0.481 |
| 28 | Children crossing | 150 | 0.763 | 0.860 | 0.809 |
| 29 | Bicycles crossing | 90 | 0.776 | 0.922 | 0.843 |
| 30 | Beware of ice/snow | 150 | 0.660 | 0.633 | 0.646 |
| 31 | Wild animals crossing | 270 | 0.665 | 0.830 | 0.738 |
| 32 | End of all limits | 60 | 0.841 | 0.967 | 0.899 |
| 33 | Turn right ahead | 210 | 0.912 | 0.986 | 0.947 |
| 34 | Turn left ahead | 120 | 0.868 | 0.983 | 0.922 |
| 35 | Ahead only | 390 | 0.984 | 0.974 | 0.979 |
| 36 | Go straight or right | 120 | 0.983 | 0.950 | 0.966 |
| 37 | Go straight or left | 60 | 0.980 | 0.800 | 0.881 |
| 38 | Keep right | 690 | 0.984 | 0.955 | 0.969 |
| 39 | Keep left | 90 | 0.935 | 0.956 | 0.945 |
| 40 | Roundabout mandatory | 90 | 0.924 | 0.944 | 0.934 |
| 41 | End of no passing | 60 | 0.534 | 0.650 | 0.586 |
| 42 | End of no passing >3.5t | 90 | 0.674 | 0.689 | 0.681 |
|  | **Macro average** | 12630 | **0.849** | **0.843** | **0.841** |

**Table 7.3 — Ten weakest classes by F1 (E0)**

| Rank | Class | Name | F1 | Train images | Test support |
|---|---|---|---|---|---|
| 1 | 27 | Pedestrians | 0.481 | 219 | 60 |
| 2 | 41 | End of no passing | 0.586 | 221 | 60 |
| 3 | 21 | Double curve | 0.629 | 296 | 90 |
| 4 | 19 | Dangerous curve left | 0.646 | 187 | 60 |
| 5 | 30 | Beware of ice/snow | 0.646 | 413 | 150 |
| 6 | 26 | Traffic signals | 0.668 | 532 | 180 |
| 7 | 42 | End of no passing >3.5t | 0.681 | 211 | 90 |
| 8 | 20 | Dangerous curve right | 0.713 | 325 | 90 |
| 9 | 31 | Wild animals crossing | 0.738 | 693 | 270 |
| 10 | 0 | Speed limit 20 | 0.742 | 180 | 60 |

![Figure 7.3](results/figures/fig7_3_e0_confusion_matrix.png)

*Figure 7.3 — E0 43×43 test confusion matrix, each row normalised by the class's test support.*

![Figure 7.4](results/figures/fig7_5_e0_per_class_f1.png)

*Figure 7.4 — E0 per-class test F1.*

**Correlation between class support and F1:** Pearson r = 0.46, Spearman ρ = 0.49 (43 classes; support = training images per class). The correlation is positive and moderate. Rare classes are over-represented among the weak ones (eight of the ten weakest classes have fewer than 420 training images), but support explains only about a fifth of the variance in F1 (r² ≈ 0.21). Class 0 (Speed limit 20, 180 training images) is weak, yet Class 41 (221 images) is weak too while Class 37 (187 images) scores 0.881. The weak classes are rare *and* visually close to another class. Class weighting alone would not close the gap.

![Figure 7.5](results/figures/fig7_4_e0_support_vs_f1.png)

*Figure 7.5 — Training images per class against test F1 (log x-axis). The eight weakest classes are labelled.*

### 7.3 Most frequent confusions

**Table 7.4 — Ten most frequent confused pairs (E0, test set)**

| Count | True class | Predicted class | Plausible cause |
|---|---|---|---|
| 50 | 3 Speed limit 60 | 5 Speed limit 80 | Speed limits differ only in the digits; '6' and '8' are near-identical at 32×32 |
| 46 | 18 General caution | 26 Traffic signals | Same red triangle; the '!' and the traffic-light pictogram are both vertical strokes |
| 42 | 1 Speed limit 30 | 2 Speed limit 50 | Digit confusion, '3' and '5' |
| 36 | 5 Speed limit 80 | 2 Speed limit 50 | Digit confusion, '8' and '5' |
| 35 | 2 Speed limit 50 | 5 Speed limit 80 | Digit confusion, '5' and '8' |
| 28 | 4 Speed limit 70 | 1 Speed limit 30 | Digit confusion, '7' and '3' |
| 27 | 8 Speed limit 120 | 7 Speed limit 100 | Digit confusion, '120' and '100' — the same number of glyphs |
| 26 | 42 End of no passing >3.5t | 41 End of no passing | Same diagonal-bar design; differs only in the small lorry and car pictogram |
| 26 | 27 Pedestrians | 21 Double curve | Same red triangle; low-resolution central pictogram |
| 25 | 25 Road work | 31 Wild animals crossing | Same red triangle; dark central pictogram of similar mass |

The expected pattern holds. Of the 1458 test errors, 630 (43%) have a true class among the speed limits (0–8), and 493 of those are predicted as another class in the same group. 540 errors (37%) have a true class among the triangular warning signs (18–31), and 468 of those stay within that group. Together the two groups account for 80% of all errors and supply nine of the ten pairs in Table 7.4. The blue circular mandatory signs (33–40) cause only 73 errors in total.

**Table 7.5 — E0 test accuracy by original sign size and by sharpness**

| Slice | Images | Accuracy |
|---|---|---|
| Sign ROI 0–30 px (√(w·h)) | 5576 | 81.29% |
| Sign ROI 30–40 px (√(w·h)) | 2714 | 93.70% |
| Sign ROI 40–50 px (√(w·h)) | 1623 | 95.13% |
| Sign ROI 50–70 px (√(w·h)) | 1621 | 95.99% |
| Sign ROI 70–max px (√(w·h)) | 1096 | 90.88% |
| Sharpness quartile 1 (blurriest) | 3158 | 80.62% |
| Sharpness quartile 2 (middle) | 3157 | 89.36% |
| Sharpness quartile 3 (middle) | 3157 | 89.64% |
| Sharpness quartile 4 (sharpest) | 3158 | 94.21% |

The median sign ROI side is 23 px for misclassified images against 33 px for correctly classified ones. Sharpness is the variance of the Laplacian on the preprocessed 32×32 image. It is a proxy: small signs are upsampled and therefore also blurrier, so the two measures are not independent.

![Figure 7.6](results/figures/fig7_7_e0_misclassified.png)

*Figure 7.6 — The first 24 misclassified test images after preprocessing (T = true class, P = predicted class).*

### 7.4 Ablation results

Each table varies one factor from E0. "Epochs run" counts the epochs before early stopping, with the restored best-dev epoch in brackets. The train–dev gap is measured at the restored epoch. One seed per condition, so the tables report single values, not mean ± sd.

**Table 7.6 — Initialization (E1)**

| Condition | Macro-F1 | Test acc. | Best dev acc. | Epochs run | Train–dev gap | Train time |
|---|---|---|---|---|---|---|
| He (E0) | 0.841 | 88.46% | 95.08% | 28 (best 23) | +1.63 pp | 2 min 50 s |
| Random ×0.01 | 0.845 | 88.40% | 95.00% | 40 (best 35) | +2.53 pp | 5 min 06 s |

![Figure 7.7a](results/figures/abl_E1_initialization.png)

*Figure 7.7a — E1: dev accuracy per epoch.*

**Table 7.7 — Optimizer (E2)**

| Condition | Macro-F1 | Test acc. | Best dev acc. | Epochs run | Train–dev gap | Train time |
|---|---|---|---|---|---|---|
| Adam (E0) | 0.841 | 88.46% | 95.08% | 28 (best 23) | +1.63 pp | 2 min 50 s |
| Mini-batch GD | 0.534 | 70.28% | 73.29% | 40 (best 40) | +1.24 pp | 2 min 20 s |
| Batch GD | 0.001 | 0.70% | 1.02% | 8 (best 3) | -0.30 pp | 25 s |

![Figure 7.7b](results/figures/abl_E2_optimizer.png)

*Figure 7.7b — E2: dev accuracy per epoch.*

**Table 7.8 — Regularization (E3)**

| Condition | Macro-F1 | Test acc. | Best dev acc. | Epochs run | Train–dev gap | Train time |
|---|---|---|---|---|---|---|
| Dropout p=0.6 | 0.846 | 88.17% | 93.29% | 31 (best 26) | +0.81 pp | 3 min 32 s |
| Dropout p=0.7 | 0.862 | 89.24% | 94.97% | 33 (best 28) | +1.08 pp | 3 min 51 s |
| Dropout p=0.8 (E0) | 0.841 | 88.46% | 95.08% | 28 (best 23) | +1.63 pp | 2 min 50 s |
| Dropout p=0.9 | 0.853 | 88.59% | 94.74% | 21 (best 16) | +1.88 pp | 1 min 55 s |
| L2 λ=0.1 | 0.829 | 86.71% | 93.37% | 18 (best 13) | +1.26 pp | 1 min 38 s |
| L2 λ=0.4 | 0.816 | 85.17% | 91.30% | 15 (best 10) | -0.13 pp | 1 min 22 s |
| L2 λ=0.7 | 0.800 | 84.82% | 89.46% | 11 (best 6) | -0.96 pp | 1 min 00 s |
| L2 λ=1.0 | 0.777 | 83.94% | 88.11% | 11 (best 6) | -1.41 pp | 1 min 00 s |
| None | 0.841 | 86.98% | 94.80% | 25 (best 20) | +2.72 pp | 2 min 10 s |

![Figure 7.7c](results/figures/abl_E3_regularization.png)

*Figure 7.7c — E3: dev accuracy per epoch.*

**Table 7.9 — Architecture (E4)**

| Condition | Macro-F1 | Test acc. | Best dev acc. | Epochs run | Train–dev gap | Train time | Params |
|---|---|---|---|---|---|---|---|
| [256] | 0.837 | 87.52% | 93.60% | 21 (best 16) | +1.44 pp | 51 s | 273,451 |
| [512, 256] (E0) | 0.841 | 88.46% | 95.08% | 28 (best 23) | +1.63 pp | 2 min 50 s | 667,179 |
| [1024, 512, 256] | 0.842 | 87.56% | 93.70% | 17 (best 12) | +1.13 pp | 3 min 54 s | 1,716,779 |
| [1024, 512, 256, 128] | 0.857 | 88.96% | 94.67% | 34 (best 29) | +1.53 pp | 10 min 58 s | 1,744,171 |

![Figure 7.7d](results/figures/abl_E4_architecture.png)

*Figure 7.7d — E4: dev accuracy per epoch.*

**Table 7.10 — Augmentation (E5)**

| Condition | Macro-F1 | Test acc. | Best dev acc. | Epochs run | Train–dev gap | Train time |
|---|---|---|---|---|---|---|
| Original data (E0) | 0.841 | 88.46% | 95.08% | 28 (best 23) | +1.63 pp | 2 min 50 s |
| + 5 augmented variants (6×) | 0.873 | 90.93% | 96.76% | 25 (best 20) | -2.03 pp | 25 min 36 s |

![Figure 7.7e](results/figures/abl_E5_augmentation.png)

*Figure 7.7e — E5: dev accuracy per epoch.*

**E5 run note.** `augment.py` wrote 176,445 augmented examples (5 per training image) in 41 s, peaking at 4.4 GB of memory. The unmodified `train.py --use-augmented` was then killed by the out-of-memory killer after 37 s, at 5.8 GB RSS on the 8 GB machine. The float64 augmented matrix (1.45 GB), its concatenation with the original data (1.73 GB), and the per-epoch shuffled copy are all alive at the same time. E5 was therefore run through `experiments/train_lowmem.py`, a wrapper that holds the input matrices as float32. Weights and all arithmetic stay float64, and nothing in `src/` changes. Peak RSS was 3.5 GB.

Mean per-class F1 change from E0 to E5: +0.037 for the 19 rare classes (<420 training images), +0.029 for the 24 others.

**Table 7.11 — Learning-rate decay (E6)**

| Condition | Macro-F1 | Test acc. | Best dev acc. | Epochs run | Train–dev gap | Train time |
|---|---|---|---|---|---|---|
| Constant lr (E0) | 0.841 | 88.46% | 95.08% | 28 (best 23) | +1.63 pp | 2 min 50 s |
| Step decay every 10 epochs | 0.856 | 89.21% | 96.07% | 34 (best 29) | +1.43 pp | 3 min 59 s |

![Figure 7.7f](results/figures/abl_E6_lr_decay.png)

*Figure 7.7f — E6: dev accuracy per epoch.*

### 7.5 Calibration, inference cost and resources

![Figure 7.8](results/figures/fig7_6_e0_reliability.png)

*Figure 7.8 — E0 reliability diagram on the test set (15 equal-width confidence bins).*

**Table 7.12 — Rejecting low-confidence predictions (E0, test set)**

| Confidence threshold | Images accepted | Accuracy on accepted | Errors rejected | Correct predictions rejected |
|---|---|---|---|---|
| 0.500 | 92.5% | 93.17% | 45.3% | 2.5% |
| 0.700 | 86.1% | 96.25% | 72.0% | 6.3% |
| 0.800 | 82.5% | 97.36% | 81.1% | 9.2% |
| 0.900 | 77.7% | 98.42% | 89.4% | 13.5% |

Inference for E0 (667,179 parameters) was timed separately on the idle machine with `experiments/inference_timing.py`. The timings cover the forward pass on already-preprocessed input, and exclude image decoding and preprocessing. A batched pass over all 12,630 test images took a median of 0.18 s over 7 repeats, or 14.3 µs per image. Classifying one image at a time took a median of 154 µs (95th percentile 195 µs, 1,000 images), because per-call overhead dominates a network this small. Either figure is far below a camera frame interval. Training memory peaked at 1141 MB for E0, higher than the 816 MB estimated in Section 5.1. The difference is interpreter, NumPy and cached-dataset overhead on top of the design matrix.

---

## 8. Discussion

Sections 8.1–8.3 interpret the results in Section 7. Section 8.4 is an architectural analysis that holds regardless of the runs.

### 8.1 Baseline interpretation

The baseline reaches 88.46% top-1 test accuracy and a macro-F1 of 0.841. The 4.4-point gap between the two means per-class performance is uneven: common classes such as Priority road, Yield, Keep right and No entry score F1 above 0.95, and the classes in Table 7.3 score between 0.48 and 0.74. Accuracy is weighted by the common classes and hides this, which is why macro-F1 is the headline metric.

Measured on the dev split, the model looks well fitted: at the restored epoch the training–dev gap is only 1.63 points, and dev loss flattens from about epoch 12 while training loss keeps falling. The dev split flatters the model, though. Best dev accuracy is 95.08% but test accuracy is 88.46% — a drop of 6.62 points that no dev-side measure predicts. This is the bias anticipated in Section 9.3: random 90/10 splitting puts frames of the same physical sign in both the training and dev partitions, so dev accuracy partly measures recognition of near-duplicates. On the track-disjoint test set the model **generalises less well than the dev curve suggests**. The limiting factor is generalisation to new physical signs, not optimisation.

Early stopping triggered at epoch 28, restoring epoch 23, so the 40-epoch budget was not binding for Adam. Only E1 (random initialisation) and E2-MGD used the full budget, and only E2-MGD was still improving at the end.

Calibration is good. Mean confidence is 0.950 on correct predictions and 0.558 on incorrect ones, and the 15-bin expected calibration error is 0.021. The score is usable as a rejection signal (Section 6.5): Table 7.12 shows that rejecting predictions below 0.7 confidence would catch 72% of the errors, at the cost of 6.3% of the correct predictions, and raise accuracy on the accepted 86% of images to 96.2%.

### 8.2 Ablation interpretation

**E1 — initialization.** He initialisation made no difference to the final result: test accuracy was 88.46% for He and 88.40% for random, and macro-F1 was 0.841 against 0.845. It did speed up training. He reached its best dev accuracy at epoch 23, while random initialisation kept improving until epoch 35 and used the whole budget. Figure 7.7a shows the random run starting lower and catching up. With only two hidden layers, Adam's per-parameter step sizes largely compensate for the poor scale of a ×0.01 initialisation. He initialisation matters more in deeper networks and without an adaptive optimiser.

**E2 — optimizer.** Adam's advantage is large, and within the 40-epoch budget it did not go away. Plain mini-batch gradient descent at the same learning rate reached only 70.28% test accuracy and a macro-F1 of 0.534 after all 40 epochs, and its dev accuracy was still rising. It is under-trained rather than converged to a worse optimum, and a larger learning rate would narrow the gap. Batch gradient descent failed completely (0.70%): one update per epoch meant only 8 gradient steps before early stopping, against 7,728 in E0's 28 epochs. Its loss was falling steadily (4.26 → 4.10 in the log). Under an epoch budget, BGD is handicapped by the number of updates it gets, not by the quality of its gradients. Early stopping on dev accuracy with a patience of 5 is also inappropriate for it. BGD also used the most memory of any E0–E4 run (3.0 GB), because it forward-propagates all 35k examples at once.

**E3 — regularization.** Dropout beat L2 at every setting tried. Dropout with p = 0.7 was the best single configuration among the E3 settings and second only to augmentation overall (89.24% test accuracy, macro-F1 0.862). The dropout results form a shallow optimum: p = 0.6 over-regularises slightly, and p = 0.8 and 0.9 regularise less. The spread (0.841–0.862 macro-F1) is close to seed noise. L2 got worse as λ increased, from macro-F1 0.829 at λ = 0.1 to 0.777 at λ = 1.0. At λ ≥ 0.4 the training–dev gap turns negative and early stopping fires by epoch 15, the signature of underfitting. The implementation divides λ by the *mini-batch* size (128), not the training-set size, so the λ grid carried over from the MNIST project applies much stronger decay than its values suggest, and should be moved down by one to two orders of magnitude. With no regularization the network overfits most (a gap of 2.72 points) and reaches 86.98%. Overall the network has more than enough capacity for this data, and the most effective regulariser was dropout, which also averages over sub-networks.

**E4 — depth and width.** Performance hardly changes with size. The four architectures span 0.837–0.857 macro-F1 while the parameter count grows from 273k to 1.74M. The deepest network, [1024, 512, 256, 128], scored highest (0.857). The three-hidden-layer [1024, 512, 256] was cut off by early stopping at epoch 17 and scored about the same as the baseline and the smallest network, [256]. There is no clear turnover point. Any real trend is smaller than single-seed noise, which is consistent with Section 8.4: extra parameters do not give a fully connected network the translation invariance it lacks. The deepest network cost nearly four times the baseline's training time for half a point of accuracy (1.6 points of macro-F1).

**E5 — augmentation.** Training on the original data plus five augmented variants per image moved test accuracy from 88.46% to 90.93% and macro-F1 from 0.841 to 0.873 (+3.25 points). The mean per-class F1 change was +0.037 for the rare classes and +0.029 for the others: slightly larger for the rare classes, in the direction predicted, but not sharply concentrated there. Shift, rotation, zoom, blur and crop-and-pad give the network the positional variation it cannot infer on its own (Section 8.4), which is the most direct compensation for its missing translation invariance available within this architecture. Augmentation was the most expensive intervention: 25 min 36 s of training against 2 min 50 s for E0, and 3.5 GB of memory even with float32 inputs.

**E6 — learning-rate decay.** E6 scored 0.856 macro-F1 against 0.841 for the baseline, but the difference cannot be credited to decay. The schedule barely decays at all. The run log shows the learning rate going from 0.001000 to 0.000965, 0.000931 and 0.000898 at epochs 11, 21 and 31, a total cut of about 10%. In `train()` the decay rate is set to `lr / ((i+1)/N)`, which is tiny when lr is 0.001, and `learning_rate_schedule` then applies `lr / (1 + decay_rate·epoch)`. Because both runs share seed 1, E0 and E6 are identical for the first 10 epochs and only diverge after the first 3.5% cut. So the difference is the effect of a small perturbation on the training trajectory: in practice a different random draw, at the scale of seed noise, not evidence that decay helps. A meaningful E6 needs a schedule that actually decays, for example halving every 10 epochs.

### 8.3 Error analysis

The confusions are concentrated where Section 7.3 predicted. Speed limits and triangular warning signs account for 80% of test errors, and most of those errors stay within the group: the network recognises the sign family and misreads the pictogram. Among the speed limits, the digit pairs 6/8, 3/5, 5/8, 7/3 and 120/100 dominate. At 32×32 grayscale these digits are separated by a handful of pixels, and a fully connected layer has no local-feature detectors to pick those pixels out. Among the triangles, the model confuses pictograms that are thin vertical strokes (General caution → Traffic signals, 46 errors) or dark blobs of similar mass (Road work → Wild animals, Pedestrians → Double curve). Traffic signals versus General caution is also a case where colour would help: the red, amber and green lamps are lost in grayscale (Section 9.1).

The misclassified images share one clear property: they are small. Accuracy is 81.3% for signs under 30 px across, which is 44% of the test set, against 93.7%–96.0% for signs of 30–70 px. The median misclassified sign is 23 px against 33 px for correct ones. These signs are upsampled to 32×32 and lose detail, and the blurriest quartile by Laplacian variance has 80.6% accuracy against 94.2% for the sharpest. Accuracy falls again for the largest signs (90.9% above 70 px). These are close-range frames, probably with stronger perspective distortion, although this was not measured. Figure 7.6 shows the recurring visual failure modes: sign interiors washed out by histogram equalisation until the digits nearly vanish, heavy blur, and signs that are off-centre or cut off by the ROI crop.

Top-2 accuracy is 92.76% against 88.46% for top-1, so the correct class is the runner-up for 4.31% of all test images, about 37% of the errors. The largest gains are in exactly the confusable classes: End of no passing >3.5t (69% → 89%), Speed limit 20 (60% → 73%), Wild animals crossing (83% → 95%), General caution (65% → 76%), Traffic signals (72% → 82%). For these classes the network has narrowed the answer to two look-alikes but ranks them the wrong way round. That is a sign the discriminating detail is present but weakly encoded, so higher resolution or colour input would help them most.

### 8.4 The architectural ceiling *(analysis, independent of results)*

The dominant limitation is structural, and worth stating precisely because it
is the whole reason this architecture underperforms the state of the art.

**No translation invariance.** A fully connected layer assigns an independent
weight to each input coordinate. Input pixel 100 and input pixel 101 are, to
the network, unrelated variables — the fact that they are horizontally adjacent
is information the architecture cannot represent. A sign shifted three pixels
right is a substantially different input vector, and the network must learn its
appearance separately at every position it can occupy. Convolutional networks
solve this by weight sharing: one learned filter is applied at every position,
so a feature learned at one location transfers to all others for free.

**No spatial locality.** Every hidden unit in layer 1 sees all 1,024 inputs at
once. There is no architectural pressure towards local features — edges,
corners, strokes — that compose into larger structures. The hierarchical
composition that makes CNNs sample-efficient on images is unavailable.

**The consequence is sample efficiency, not capacity.** By the universal
approximation theorem [5], the network *could* represent a good classifier. It
would need vastly more data to *find* one, because it must learn from examples
what a convolutional architecture is given by construction. Section 2.1 shows
the empirical size of the gap: 99.46% for a CNN committee against 95.68% for
classical LDA on HOG features, with an MLP on raw pixels expected below the
latter.

This is a well-understood limitation and the reason the field moved to
convolutional architectures for vision. Documenting it precisely is more useful
than obscuring it.

---

## 9. Limitations and threats to validity

### 9.1 Grayscale conversion discards regulatory colour

German traffic signs encode meaning in colour by regulation: red borders for
prohibitory, blue for mandatory, yellow for temporary. Grayscale conversion
discards this. The trade-off was taken to keep the input dimension at 1,024
(Section 4.2), and Sermanet and LeCun [3] report the cost on GTSRB is modest —
but it is a cost, and it is likely concentrated exactly where colour is the
main discriminant between otherwise similar shapes.

### 9.2 Recognition, not detection

The system classifies images that already contain one cropped, centred sign,
using ROI annotations supplied by the benchmark. A deployed system must first
*find* signs in a full scene. This is a distinct problem requiring a detector,
and no part of it is addressed here. Reported accuracy therefore describes only
the classification stage of a hypothetical pipeline.

### 9.3 Dev split is not track-disjoint

As noted in Section 3.2, GTSRB contributes 30 consecutive frames per physical
sign. The random 90/10 dev split places frames of the same physical sign on both
sides, so **dev accuracy is optimistically biased** — the model has seen
near-duplicates of its validation images. This affects early stopping, which may
halt later than a clean split would justify. The official test partition is
track-disjoint and unaffected, so reported test figures remain sound. A
track-aware split is the correct fix and is listed in Section 10.2.

**Measured effect.** In the Section 7 runs this bias is large. Best dev accuracy exceeded test accuracy by 6.6 points for E0, and by 4.2–7.8 points across every configuration that trained properly (Section 8.1). Early stopping and every dev-based comparison inherit this bias. The test figures do not.

### 9.4 Class imbalance is uncorrected

No class weighting, oversampling or loss reweighting is applied. The 10.7:1
imbalance is passed through to training as-is. This is why macro-F1 rather than
accuracy is the headline metric — the metric reveals the problem rather than
solving it.

### 9.5 Single-seed results if not repeated

Differences of a percentage point or so between configurations are not reliably
distinguishable from initialization noise. The protocol calls for ≥3 seeds with
mean and standard deviation; conclusions drawn from single runs should be
treated as provisional.

**Applies to this report.** Every figure in Section 7 comes from a single run with seed 1, so the ablation conclusions in Section 8.2 are provisional. The exceptions are E2 and L2-regularised E3, whose effects are far larger than seed noise.

### 9.6 Fixed preprocessing is not ablated

The five preprocessing steps are held constant across all experiments and
justified by argument rather than measurement. The contribution of histogram
equalization in particular — argued in Section 4.2 to be the highest-value step
— is untested. An ablation over preprocessing would strengthen this.

### 9.7 Occlusion and physical degradation unaddressed

Neither the preprocessing nor the augmentation models partial occlusion,
fading, or vandalism, all of which occur in the test set. No augmentation
transformation simulates them, and performance on affected images is
uncharacterised.

### 9.8 Inference latency measured for the forward pass only

Inference was measured for E0 (Section 7.5). A forward pass took 14.3 µs per image in a batch and 154 µs for a single image (median; 95th percentile 195 µs) on two cores. This is still not an end-to-end figure: decoding, the ROI crop and the preprocessing of Section 4.2 are excluded, and in a deployed system the sign would first have to be detected (Section 9.2).

---

## 10. Conclusion and future work

### 10.1 Conclusion

This report presented a traffic sign classifier for GTSRB built on a
feed-forward neural network implemented entirely from first principles in
NumPy. Forward propagation, backpropagation, Adam, inverted dropout, L2
regularization, He initialization and early stopping were each derived and
implemented directly, with no machine learning framework.

The methodological contributions stand independently of the training results.
A preprocessing pipeline was designed with each stage justified against a
specific property of the data. Augmentation transformations inherited from the
digit domain were re-examined and found partly invalid, with horizontal
mirroring identified as label-corrupting for eight directional sign classes.
Three defects in the antecedent codebase were found and corrected, two of them
silent. A controlled protocol of six ablations was specified, and the resource
profile of the training procedure was measured.

Measured on the GTSRB test set, the baseline network reaches 88.46% top-1 accuracy (92.76% top-2) and a macro-F1 of 0.841. The best configuration tested, the augmented training set (E5), reaches 90.93% and macro-F1 0.873. Both are well below the reference points in Section 2.1. The baseline is about 7 points below LDA on HOG features (95.68%) and 11 points below the winning CNN committee (99.46%), and the best configuration narrows the first gap only to about 5 points. This is the position the architectural analysis predicted. The errors have the structure Section 8.4 predicts for a network without spatial priors: they concentrate within sign families that differ only in a small interior pictogram, and in small, low-resolution images.

Three ablations produced effects clearly larger than seed noise. For the optimizer (E2), Adam against plain gradient descent at the same learning rate is the difference between a working and a non-working model within 40 epochs. Augmentation (E5) gave the largest gain, +3.2 points of macro-F1 and +2.5 points of accuracy. L2 regularization at the planned strengths hurt clearly, because the implementation scales λ by the mini-batch size. He initialization sped up convergence without changing the final result. Dropout rate, depth, width and the (nearly inactive) learning-rate schedule changed macro-F1 by at most about two points, which one seed cannot separate from noise. The study also exposed a methodological point: in every configuration that trained properly, best dev accuracy overstated test accuracy by 4.2–7.8 points (6.6 for E0), which quantifies the track-leakage bias in Section 9.3 and makes a track-aware dev split the most important protocol fix.

### 10.2 Future work

**Immediate, within this architecture**

1. **Track-aware dev split**, removing the bias identified in Section 9.3.
2. **Class-imbalance correction** — weighted loss or targeted oversampling of
   the rare classes — with macro-F1 as the measure of success.
3. **Preprocessing ablation**, in particular quantifying histogram equalization.
4. **Colour input**: retain RGB at 3,072 features and measure whether the
   accuracy gain justifies tripling the first layer.
5. **End-to-end inference latency benchmark**: forward-pass latency is now measured (Section 7.5), but the full path from camera frame to label, including preprocessing, is not (Section 9.8).

**Implementation fixes found by the ablations.** Two defects surfaced in Section 8.2 and should be fixed before the ablations are repeated. First, the L2 term divides λ by the mini-batch size, so the planned λ grid over-regularises heavily. Second, the step-decay schedule set up in `train()` cuts the learning rate by only about 10% in total, which makes E6 almost a no-op. The augmented-data path should also keep its design matrix in float32 (Section 7.4). Finally, every ablation should be repeated with at least three seeds, as the protocol requires.

**Architectural**

6. **Convolutional layers.** The principled fix for both deficits in Section
   8.4. `src/dataset.py` feeds a CNN unchanged — omit `flatten_input`. This is
   the single highest-value change available.
7. **Batch normalisation**, to stabilise training in deeper configurations.
8. **Ensembling**, following the approach that won the original benchmark [2].

**Beyond classification**

9. **Detection**, converting recognition into a system usable on street scenes.
10. **Temporal aggregation** over the 30-frame tracks, since a deployed system
    sees a sequence and can accumulate evidence rather than deciding per frame.

---

## References

[1] J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel, "Man vs. computer:
Benchmarking machine learning algorithms for traffic sign recognition,"
*Neural Networks*, vol. 32, pp. 323–332, 2012.

[2] D. Cireşan, U. Meier, J. Masci, and J. Schmidhuber, "Multi-column deep
neural network for traffic sign classification," *Neural Networks*, vol. 32,
pp. 333–338, 2012.

[3] P. Sermanet and Y. LeCun, "Traffic sign recognition with multi-scale
convolutional networks," in *Proc. IJCNN*, 2011, pp. 2809–2813.

[4] N. Dalal and B. Triggs, "Histograms of oriented gradients for human
detection," in *Proc. CVPR*, 2005, pp. 886–893.

[5] K. Hornik, M. Stinchcombe, and H. White, "Multilayer feedforward networks
are universal approximators," *Neural Networks*, vol. 2, no. 5, pp. 359–366,
1989.

[6] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning
applied to document recognition," *Proc. IEEE*, vol. 86, no. 11, pp.
2278–2324, 1998.

[7] K. He, X. Zhang, S. Ren, and J. Sun, "Delving deep into rectifiers:
Surpassing human-level performance on ImageNet classification," in *Proc.
ICCV*, 2015, pp. 1026–1034.

[8] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in
*Proc. ICLR*, 2015.

[9] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R.
Salakhutdinov, "Dropout: A simple way to prevent neural networks from
overfitting," *JMLR*, vol. 15, no. 1, pp. 1929–1958, 2014.

[10] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press,
2016.

**Dataset citation.** J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel,
"The German Traffic Sign Recognition Benchmark: A multi-class classification
competition," in *Proc. IJCNN*, 2011, pp. 1453–1460. Available:
<https://benchmark.ini.rub.de/gtsrb_dataset.html>. Licensed CC BY 4.0.

---

## Appendix A — Reproducibility

**Environment.** `Dockerfile` and `docker-compose.yml` pin the complete
environment. See `SETUP.md` for both the container and native paths.

**Reproducing the baseline from a clean checkout:**

```bash
docker build -t traffic-sign-detection .
docker compose run --rm tsd python train.py --epochs 40 --out models/e0_baseline
docker compose run --rm tsd python evaluate.py --model models/e0_baseline --no-plot
```

**Seeding.** `np.random.seed(1)` at entry to every script; `--seed` overrides.
Given identical data and flags, runs reproduce.

**Data provenance.** GTSRB is downloaded from the URLs in `GTSRB_URLS`
(`src/dataset.py`), which are the archives linked from the official benchmark
page. Decoded arrays are cached as `dataset/gtsrb/gtsrb_32.npz`.

**Runs reported in Section 7.** `experiments/run_all.sh <run-id> …` trains and evaluates each configuration and writes logs to `results/logs/`. `experiments/analyze.py` builds `results/summary.json`, `results/all_runs_summary.csv`, the E0 per-class and confusion CSVs, and `results/figures/`. `experiments/error_analysis.py` writes `results/e0_error_analysis.json`. `experiments/train_lowmem.py` is the float32 wrapper used for E5, and `experiments/inference_timing.py` writes `results/e0_inference_timing.json`. Trained models are in `models/`. The preprocessed dataset cache (`dataset/gtsrb/gtsrb_32.npz`, SHA-256 `bdbf50a8…a2523c`) was built from the three official archives.

**Code manifest**

| File | Role |
|---|---|
| `src/dataset.py` | Acquisition, decoding, preprocessing, splitting, encoding |
| `src/ffnn.py` | Network: initialization, forward, cost, backward, optimizers, training, prediction |
| `src/model_utils.py` | Activations, mini-batching, metrics, confusion matrix, plots, persistence |
| `src/data_augmentation.py` | Augmentation transformations and generator |
| `train.py` / `evaluate.py` / `predict.py` / `augment.py` / `gui.py` | Entry points |
| `Project_Walkthrough.ipynb` | Step-by-step executable walkthrough |
| `experiments/` | Experiment runner, analysis and error-analysis scripts used for Section 7 |
| `results/` | Logs, metrics, CSVs and figures from the Section 7 runs |
| `README.md` / `SETUP.md` | Overview and setup guide |

---

## Appendix B — Class index

| id | Class | id | Class | id | Class |
|---|---|---|---|---|---|
| 0 | Speed limit 20 | 15 | No vehicles | 30 | Beware of ice/snow |
| 1 | Speed limit 30 | 16 | No vehicles >3.5t | 31 | Wild animals crossing |
| 2 | Speed limit 50 | 17 | No entry | 32 | End of all limits |
| 3 | Speed limit 60 | 18 | General caution | 33 | Turn right ahead |
| 4 | Speed limit 70 | 19 | Dangerous curve left | 34 | Turn left ahead |
| 5 | Speed limit 80 | 20 | Dangerous curve right | 35 | Ahead only |
| 6 | End speed limit 80 | 21 | Double curve | 36 | Go straight or right |
| 7 | Speed limit 100 | 22 | Bumpy road | 37 | Go straight or left |
| 8 | Speed limit 120 | 23 | Slippery road | 38 | Keep right |
| 9 | No passing | 24 | Road narrows on right | 39 | Keep left |
| 10 | No passing >3.5t | 25 | Road work | 40 | Roundabout mandatory |
| 11 | Right-of-way at intersection | 26 | Traffic signals | 41 | End of no passing |
| 12 | Priority road | 27 | Pedestrians | 42 | End of no passing >3.5t |
| 13 | Yield | 28 | Children crossing | | |
| 14 | Stop | 29 | Bicycles crossing | | |

The eight directional classes affected by the mirroring analysis in Section 4.9
are 19, 20, 33, 34, 36, 37, 38 and 39.

---

## Appendix C — Notation

| Symbol | Meaning |
|---|---|
| $m$ | Number of examples in the current batch |
| $L$ | Number of layers (hidden + output) |
| $n^{[l]}$ | Units in layer $l$; $n^{[0]} = 1024$, $n^{[L]} = 43$ |
| $X$ | Input matrix, $(1024 \times m)$ |
| $Y$ | One-hot labels, $(43 \times m)$ |
| $\hat{Y}$ | Predicted probabilities, $(43 \times m)$ |
| $W^{[l]}, b^{[l]}$ | Weights and biases of layer $l$ |
| $Z^{[l]}, A^{[l]}$ | Pre-activation and activation of layer $l$ |
| $D^{[l]}$ | Dropout mask for layer $l$ |
| $\alpha, \lambda, p$ | Learning rate, L2 strength, dropout keep probability |
| $\beta_1, \beta_2, \epsilon$ | Adam hyperparameters |
| $v_t, s_t$ | Adam first and second moment estimates at step $t$ |
| $J$ | Cost |
| $\odot$ | Element-wise (Hadamard) product |
