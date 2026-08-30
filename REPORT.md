# Traffic Sign Recognition Using a From-Scratch Feed-Forward Neural Network

### A study of the German Traffic Sign Recognition Benchmark using a multi-layer perceptron implemented without machine learning frameworks

**Author:** _[your name]_
**Institution:** _[your institution]_
**Supervisor:** _[supervisor]_
**Date:** _[submission date]_

---

> **STATUS OF THIS DOCUMENT**
>
> Sections 1–6 and 8–10 are complete. **Section 7 (Results) is a structured
> template**: every table and figure it calls for is produced by the code in
> this repository, but no training run has been performed yet, so the cells are
> unfilled. Section 7.0 lists the exact commands that generate each item.
>
> Every quantitative claim made outside Section 7 is either a property of the
> dataset as published, a property of the code, or a resource measurement taken
> on the hardware stated in Section 5.1. No accuracy figure is asserted
> anywhere in this document.

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
| Training design matrix, $(1024 \times 35288)$ float64 | 289 MB |
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
development partition is drawn at random from the training set, giving 35,288
training / 3,921 dev / 12,630 test.

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

> ⚠️ **THIS SECTION IS A TEMPLATE.** No training run has been performed. Every
> table below is produced by the commands given in Section 7.0. Fill them in
> from your own runs; do not cite any figure from this section until you have.

### 7.0 How to generate each item

| Item | Command |
|---|---|
| Table 7.1, Figures 7.1–7.2 | `python train.py --epochs 40 --out models/e0_baseline` |
| Tables 7.2–7.3, Figure 7.3 | `python evaluate.py --model models/e0_baseline` |
| Table 7.4 (E1) | `python train.py --initialization random --out models/e1_random` |
| Table 7.5 (E2) | `python train.py --optimizer mgd --out models/e2_mgd` |
| Table 7.6 (E3) | Vary `--regularizer` / `--keep-prob` / `--lambd` |
| Table 7.7 (E4) | Vary `--hidden` |
| Table 7.8 (E5) | `python augment.py` then `python train.py --use-augmented` |

### 7.1 Baseline performance

**Table 7.1 — Baseline (E0) results**

| Metric | Value |
|---|---|
| Test accuracy (top-1) | _____ |
| Test accuracy (top-2) | _____ |
| Macro-precision | _____ |
| Macro-recall | _____ |
| **Macro-F1** | _____ |
| Final training accuracy | _____ |
| Final dev accuracy | _____ |
| Train–dev gap | _____ |
| Epochs completed (of 40) | _____ |
| Early stopping triggered | Yes / No |
| Wall-clock training time | _____ |
| Mean confidence when correct | _____ |
| Mean confidence when incorrect | _____ |

*Figure 7.1 — Training and dev loss.*
*Figure 7.2 — Training and dev accuracy.*

### 7.2 Per-class performance

**Table 7.2 — Per-class metrics (43 rows)**

| Class | Name | Support | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 0 | Speed limit 20 | | | | |
| … | … | | | | |
| 42 | End of no passing >3.5t | | | | |
| | **Macro average** | | | | |

**Table 7.3 — Ten weakest classes by F1**

| Rank | Class | Name | F1 | Support |
|---|---|---|---|---|
| 1 | | | | |
| … | | | | |

*Figure 7.3 — 43×43 confusion matrix.*

**Correlation between class support and F1:** _____
_(A strong positive correlation indicates the weak classes are weak because
they are rare, which points at class weighting rather than architecture.)_

### 7.3 Most frequent confusions

**Table 7.4 — Top ten confused pairs**

| Count | True class | Predicted class | Plausible cause |
|---|---|---|---|
| | | | |

_Expected pattern: clustering among the speed limits (0–8), which differ only
in their digits, and among the triangular warning signs (18–31), which share an
outline._

### 7.4 Ablation results

**Table 7.5 — Initialization (E1)** · **Table 7.6 — Optimizer (E2)** ·
**Table 7.7 — Regularization (E3)** · **Table 7.8 — Architecture (E4)** ·
**Table 7.9 — Augmentation (E5)** · **Table 7.10 — LR decay (E6)**

Each: condition | macro-F1 | test accuracy | epochs to converge | train–dev gap.
Report mean ± sd over ≥3 seeds.

---

## 8. Discussion

> Sections 8.1–8.3 are to be written against your results. The questions to
> answer are stated. Section 8.4 is an architectural analysis that holds
> independently of any run.

### 8.1 Baseline interpretation *(to be completed)*

Address: How does macro-F1 compare to accuracy, and what does the gap say about
per-class behaviour? Does the train–dev gap indicate over- or underfitting? Did
early stopping trigger, and what does that imply about the epoch budget? Is the
model well calibrated by the Section 6.5 measure?

### 8.2 Ablation interpretation *(to be completed)*

Address each experiment: did He initialization help, and in final performance or
only convergence speed? Did Adam's advantage persist to convergence or only
accelerate early epochs? Which regularizer won, and does that indicate the
network is over- or under-parameterised for this data? How did performance
scale with depth and width, and where did it turn over? Did augmentation help,
and was the gain concentrated in the rare classes as predicted?

### 8.3 Error analysis *(to be completed)*

Address: Are confusions concentrated in the visually similar groups predicted in
Section 7.3? Do the misclassified images share identifiable properties — blur,
angle, low contrast, small original size? Does top-2 accuracy substantially
exceed top-1, and for which classes?

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

### 9.8 No inference latency measurement

Training cost is measured (Section 5.1); inference latency, the operationally
relevant figure for a real-time system, is not. A forward pass through 667k
parameters is inexpensive, but "inexpensive" is not a measurement.

---

## 10. Conclusion and future work

### 10.1 Conclusion *(to be completed against results)*

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

_[Complete with: the achieved macro-F1 and how it compares to the reference
points in Section 2.1; which ablations produced significant effects; whether
the observed error structure matched the architectural prediction of Section
8.4.]_

### 10.2 Future work

**Immediate, within this architecture**

1. **Track-aware dev split**, removing the bias identified in Section 9.3.
2. **Class-imbalance correction** — weighted loss or targeted oversampling of
   the rare classes — with macro-F1 as the measure of success.
3. **Preprocessing ablation**, in particular quantifying histogram equalization.
4. **Colour input**: retain RGB at 3,072 features and measure whether the
   accuracy gain justifies tripling the first layer.
5. **Inference latency benchmark** (Section 9.8).

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

**Code manifest**

| File | Role |
|---|---|
| `src/dataset.py` | Acquisition, decoding, preprocessing, splitting, encoding |
| `src/ffnn.py` | Network: initialization, forward, cost, backward, optimizers, training, prediction |
| `src/model_utils.py` | Activations, mini-batching, metrics, confusion matrix, plots, persistence |
| `src/data_augmentation.py` | Augmentation transformations and generator |
| `train.py` / `evaluate.py` / `predict.py` / `augment.py` / `gui.py` | Entry points |
| `Project_Walkthrough.ipynb` | Step-by-step executable walkthrough |
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
