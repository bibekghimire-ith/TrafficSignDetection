#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GTSRB dataset module.

Downloads, decompresses, decodes and preprocesses the German Traffic Sign
Recognition Benchmark (GTSRB) into the flat, normalized NumPy arrays that the
from-scratch network in ``ffnn.py`` expects.

This is the traffic-sign counterpart of the MNIST loader in the hand-written
digit project. The public surface is deliberately identical -- ``load_dataset``,
``train_dev_split``, ``prep_dataset``, ``label_description`` -- so the model,
metric and augmentation code carries over unchanged.

Pipeline for one image::

    .ppm file -> crop to annotated ROI -> grayscale -> histogram equalize
              -> resize to IMG_SIZE x IMG_SIZE -> uint8 array
              -> flatten to (IMG_SIZE*IMG_SIZE, 1) -> divide by 255
"""
import csv
import os
import os.path
import zipfile
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# ==================================( Dataset configuration )==================================
#: Side length of the square crop fed to the network.
IMG_SIZE = 32
#: Number of flattened input features per example.
NUM_FEATURES = IMG_SIZE * IMG_SIZE
#: GTSRB has 43 sign classes.
NUM_CLASSES = 43

#: Root directory holding the raw and cached data.
DATA_ROOT = "dataset/gtsrb/"

#: Official GTSRB archives, mirrored by the benchmark organisers (Institut fuer
#: Neuroinformatik, Ruhr-Universitaet Bochum).
GTSRB_URLS = {
    "GTSRB_Final_Training_Images.zip":
        "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip",
    "GTSRB_Final_Test_Images.zip":
        "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip",
    "GTSRB_Final_Test_GT.zip":
        "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip",
}

#: Human readable name for every class id.
GTSRB_LABELS = {
    0: "Speed limit 20", 1: "Speed limit 30", 2: "Speed limit 50",
    3: "Speed limit 60", 4: "Speed limit 70", 5: "Speed limit 80",
    6: "End speed limit 80", 7: "Speed limit 100", 8: "Speed limit 120",
    9: "No passing", 10: "No passing >3.5t", 11: "Right-of-way at intersection",
    12: "Priority road", 13: "Yield", 14: "Stop",
    15: "No vehicles", 16: "No vehicles >3.5t", 17: "No entry",
    18: "General caution", 19: "Dangerous curve left", 20: "Dangerous curve right",
    21: "Double curve", 22: "Bumpy road", 23: "Slippery road",
    24: "Road narrows on right", 25: "Road work", 26: "Traffic signals",
    27: "Pedestrians", 28: "Children crossing", 29: "Bicycles crossing",
    30: "Beware of ice/snow", 31: "Wild animals crossing", 32: "End of all limits",
    33: "Turn right ahead", 34: "Turn left ahead", 35: "Ahead only",
    36: "Go straight or right", 37: "Go straight or left", 38: "Keep right",
    39: "Keep left", 40: "Roundabout mandatory", 41: "End of no passing",
    42: "End of no passing >3.5t",
}


# ====================================( Downloading )==========================================
def download_dataset(to_path=DATA_ROOT):
    """Download the GTSRB archives, skipping any that are already present.

        Arguments:
            to_path (str): directory the ``.zip`` archives are written to.

        Example:
            >>> download_dataset("dataset/gtsrb/")
    """
    if not os.path.exists(to_path):
        print("Destination directory does not exist: creating '%s'.\n" % to_path)
        os.makedirs(to_path)

    status = "succeeded"
    for filename, url in GTSRB_URLS.items():
        target = os.path.join(to_path, filename)
        if os.path.exists(target):
            print("%s: already exists." % filename)
            continue

        print("\n%s: downloading..." % filename)
        try:
            remote = urlopen(url)
            total = int(remote.info()["Content-Length"])
            total_mb = total / (1024 * 1024)
            got = 0
            block = 1024 * 64
            with open(target, "wb") as local:
                while True:
                    buffer = remote.read(block)
                    if not buffer:
                        break
                    got += len(buffer)
                    local.write(buffer)
                    pct = (got / total) * 100.
                    inc = int(pct) // 10
                    print("%.1f MB  [%.1f MB done %s>%s %.0f%%]"
                          % (total_mb, got / (1024 * 1024), '=' * inc, '.' * (10 - inc), pct),
                          end="\r")
        except (HTTPError, URLError) as error_message:
            print("Download failed: ", error_message)
            status = "failed"

    print("\n\nDataset download %s...\n" % status)
    if status == "failed":
        raise RuntimeError(
            "Could not download GTSRB. Fetch the three archives listed in "
            "GTSRB_URLS by hand and drop them into '%s'." % to_path)


# ===================================( Decompressing )=========================================
def decompress_dataset(path=DATA_ROOT, keep_original=True):
    """Extract every GTSRB archive found in ``path`` next to itself.

        Arguments:
            path (str): directory holding the ``.zip`` archives.
            keep_original (bool): keep the ``.zip`` files after extraction.
    """
    for filename in GTSRB_URLS:
        archive = os.path.join(path, filename)
        if not os.path.exists(archive):
            continue
        print("Extracting %s..." % filename)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(path)
        if not keep_original:
            os.remove(archive)


def get_files(path, file_type="all"):
    """List the files directly inside ``path``.

        Arguments:
            path (str): directory to list.
            file_type (str): "all", or an extension such as "ppm" or "zip".

        Returns:
            list: - **files** - file names, sorted.
    """
    if not os.path.exists(path):
        return []
    files = [f for f in sorted(os.listdir(path))
             if os.path.isfile(os.path.join(path, f))]
    if file_type != "all":
        files = [f for f in files if f.lower().endswith("." + file_type.lower().lstrip("."))]
    return files


# =====================================( Decoding )============================================
def _read_annotations(csv_path):
    """Read one GTSRB ``GT-*.csv`` annotation file.

        Returns:
            list: one dict per row with ``Filename`` and integer ROI/ClassId fields.
    """
    rows = []
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            rows.append({
                "Filename": row["Filename"],
                "Roi.X1": int(row["Roi.X1"]), "Roi.Y1": int(row["Roi.Y1"]),
                "Roi.X2": int(row["Roi.X2"]), "Roi.Y2": int(row["Roi.Y2"]),
                "ClassId": int(row["ClassId"]),
            })
    return rows


def preprocess_image(image, roi=None):
    """Turn one PIL image into the canonical IMG_SIZE x IMG_SIZE uint8 array.

        Crops to the annotated region of interest, converts to grayscale,
        equalizes the histogram (GTSRB images vary wildly in exposure) and
        resizes.

        Arguments:
            image (PIL.Image): source image.
            roi (tuple, optional): ``(x1, y1, x2, y2)`` bounding box.

        Returns:
            numpy.ndarray: - **array** - uint8 array of shape (IMG_SIZE, IMG_SIZE).
    """
    if roi is not None:
        image = image.crop(roi)
    image = image.convert("L")
    image = ImageOps.equalize(image)
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def retrive_dataset(path=DATA_ROOT):
    """Decode the extracted GTSRB images into NumPy arrays.

        Walks ``Final_Training/Images/000NN`` for the training set and
        ``Final_Test/Images`` plus ``GT-final_test.csv`` for the test set.

        Arguments:
            path (str): directory the archives were extracted into.

        Returns:
            tuple: Following values
                - **train_images** (numpy.ndarray): (m_train, IMG_SIZE, IMG_SIZE) uint8.
                - **train_labels** (numpy.ndarray): (m_train, 1) int.
                - **test_images** (numpy.ndarray): (m_test, IMG_SIZE, IMG_SIZE) uint8.
                - **test_labels** (numpy.ndarray): (m_test, 1) int.
    """
    train_root = os.path.join(path, "GTSRB", "Final_Training", "Images")
    test_root = os.path.join(path, "GTSRB", "Final_Test", "Images")

    if not os.path.exists(train_root):
        raise FileNotFoundError(
            "Expected the extracted training images at '%s'. Run "
            "decompress_dataset() first." % train_root)

    # ---- training set: one folder and one annotation csv per class ----
    train_images, train_labels = [], []
    for class_id in range(NUM_CLASSES):
        class_dir = os.path.join(train_root, "%05d" % class_id)
        annotations = os.path.join(class_dir, "GT-%05d.csv" % class_id)
        for row in _read_annotations(annotations):
            with Image.open(os.path.join(class_dir, row["Filename"])) as img:
                roi = (row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"])
                train_images.append(preprocess_image(img, roi))
            train_labels.append(row["ClassId"])
        print("decoded class %02d/%d" % (class_id, NUM_CLASSES - 1), end="\r")

    # ---- test set: a flat folder plus a single ground-truth csv ----
    gt_path = os.path.join(test_root, "GT-final_test.csv")
    if not os.path.exists(gt_path):  # the GT archive extracts to the dataset root
        gt_path = os.path.join(path, "GT-final_test.csv")

    test_images, test_labels = [], []
    for row in _read_annotations(gt_path):
        with Image.open(os.path.join(test_root, row["Filename"])) as img:
            roi = (row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"])
            test_images.append(preprocess_image(img, roi))
        test_labels.append(row["ClassId"])

    print("\ndecoded %d training and %d test images." % (len(train_images), len(test_images)))

    return (np.asarray(train_images, dtype=np.uint8),
            np.asarray(train_labels, dtype=np.int64).reshape(-1, 1),
            np.asarray(test_images, dtype=np.uint8),
            np.asarray(test_labels, dtype=np.int64).reshape(-1, 1))


# ==============================( Sampling from the dataset )==================================
def sample_dataset(x, y, size_in_per):
    """Shuffle and return ``size_in_per`` percent of the dataset.

        Arguments:
            x (numpy.ndarray): images of shape (m, IMG_SIZE, IMG_SIZE).
            y (numpy.ndarray): labels of shape (m, 1).
            size_in_per (int): sample volume as a percentage.

        Returns:
            tuple: - **x_sample**, **y_sample**.

        Example:
            >>> train_x_s, train_y_s = sample_dataset(train_x, train_y, size_in_per = 25)
    """
    m = y.shape[0]
    sample_m = int(np.multiply(m, np.divide(size_in_per, 100)))

    shuffled = np.random.permutation(m)
    x_shuffled = x[shuffled, :, :]
    y_shuffled = y[shuffled, :]

    x_sample = x_shuffled[0:sample_m, :, :]
    y_sample = y_shuffled[0:sample_m, :]

    assert (x_sample.shape == (sample_m, IMG_SIZE, IMG_SIZE))
    assert (y_sample.shape == (sample_m, 1))

    return x_sample, y_sample


# ===================================( Loading the dataset )===================================
def load_dataset(dataset="gtsrb", size_in_per=100, path=DATA_ROOT, use_cache=True):
    """Load GTSRB, downloading, extracting and decoding it if needed.

        Decoding ~52,000 ``.ppm`` files takes a couple of minutes, so the
        decoded arrays are cached in ``gtsrb_<IMG_SIZE>.npz`` and reused.

        Arguments:
            dataset (str): only "gtsrb" is supported; kept for signature parity
                with the MNIST version of this module.
            size_in_per (int, optional): percentage of the data to return.
            path (str, optional): directory the data lives in.
            use_cache (bool, optional): read/write the decoded ``.npz`` cache.

        Returns:
            tuple: - **train_x_orig**, **train_y_orig**, **test_x_orig**, **test_y_orig**.

        Example:
            >>> train_x, train_y, test_x, test_y = load_dataset(size_in_per = 100)
    """
    if dataset != "gtsrb":
        raise ValueError("Only the 'gtsrb' dataset is supported by this module")

    cache = os.path.join(path, "gtsrb_%d.npz" % IMG_SIZE)

    if use_cache and os.path.exists(cache):
        print("Loading the decoded dataset from cache: %s" % cache)
        with np.load(cache) as data:
            train_x, train_y = data["train_x"], data["train_y"]
            test_x, test_y = data["test_x"], data["test_y"]
    else:
        if not os.path.exists(os.path.join(path, "GTSRB", "Final_Training", "Images")):
            download_dataset(path)
            decompress_dataset(path)
        train_x, train_y, test_x, test_y = retrive_dataset(path)
        if use_cache:
            print("Caching the decoded dataset to %s" % cache)
            np.savez_compressed(cache, train_x=train_x, train_y=train_y,
                                test_x=test_x, test_y=test_y)

    train_x_orig, train_y_orig = sample_dataset(train_x, train_y, size_in_per)
    test_x_orig, test_y_orig = sample_dataset(test_x, test_y, size_in_per)

    return train_x_orig, train_y_orig, test_x_orig, test_y_orig


# ========================( Splitting training into train and dev )============================
def train_dev_split(train_x, train_y, dev_fraction=0.1):
    """Randomly split the training set into a training and a development set.

        Arguments:
            train_x (numpy.ndarray): images of shape (m, IMG_SIZE, IMG_SIZE).
            train_y (numpy.ndarray): labels of shape (m, 1).
            dev_fraction (float, optional): share held out for the dev set.

        Returns:
            tuple: - **train_x**, **train_y**, **dev_x**, **dev_y**.

        Example:
            >>> tr_x, tr_y, dev_x, dev_y = train_dev_split(train_x, train_y)
    """
    m = train_y.shape[0]
    dev_m = int(m * dev_fraction)

    shuffled = np.random.permutation(m)
    x_shuffled = train_x[shuffled, :, :]
    y_shuffled = train_y[shuffled, :]

    dev_x, dev_y = x_shuffled[0:dev_m], y_shuffled[0:dev_m]
    new_train_x, new_train_y = x_shuffled[dev_m:], y_shuffled[dev_m:]

    return new_train_x, new_train_y, dev_x, dev_y


# ==================================( Label descriptions )=====================================
def label_description(dataset="gtsrb"):
    """Return the ``{class_id: name}`` mapping for the dataset.

        Example:
            >>> label_desc = label_description()
            >>> label_desc[14]
            'Stop'
    """
    if dataset != "gtsrb":
        raise ValueError("Dataset must be 'gtsrb'")
    return GTSRB_LABELS


# ====================================( Visualization )========================================
def visualize_data_distribution(y_orig, dataset_type):
    """Plot how many examples each class has -- GTSRB is markedly imbalanced.

        Arguments:
            y_orig (numpy.ndarray): labels of shape (m, 1).
            dataset_type (str): "training", "dev" or "test".
    """
    if not len(dataset_type):
        raise ValueError("Dataset type must be training, dev or test")

    labels, counts = np.unique(y_orig, return_counts=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(labels, counts)
    ax.set_title("Class distribution of the %s set (%d examples)"
                 % (dataset_type.capitalize(), y_orig.shape[0]))
    ax.set_xlabel("Class id")
    ax.set_ylabel("Number of examples")
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    plt.show()


def visualize_dataset(x_orig, y_orig, dataset="gtsrb", dataset_type="training"):
    """Plot 10 random sample images with their labels.

        Arguments:
            x_orig (numpy.ndarray): images of shape (m, IMG_SIZE, IMG_SIZE).
            y_orig (numpy.ndarray): labels of shape (m, 1).
            dataset (str): dataset name.
            dataset_type (str): "training", "dev" or "test".
    """
    label_desc = label_description(dataset)
    index = np.random.randint(0, x_orig.shape[0], 10)

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 7))
    fig.subplots_adjust(hspace=0.6)
    fig.suptitle("Sample %s set images" % dataset_type.capitalize())

    for ax, i in zip(axes.flatten(), index):
        ax.imshow(x_orig[i].squeeze(), interpolation="nearest", cmap="gray")
        ax.set(title="%d | %s" % (y_orig[i, 0], label_desc[int(y_orig[i, 0])]))
        ax.axis("off")
    plt.show()


# =================================( Preparing the dataset )===================================
def flatten_input(x_orig):
    """Flatten (m, IMG_SIZE, IMG_SIZE) images into a (NUM_FEATURES, m) matrix.

        Example:
            >>> train_x_flatten = flatten_input(train_x)
    """
    m = x_orig.shape[0]
    x_flatten = x_orig.reshape(x_orig.shape[0], -1).T
    assert (x_flatten.shape == (NUM_FEATURES, m))
    return x_flatten


def normalize_input(x_flatten):
    """Scale pixel values from 0-255 into the 0-1 range.

        Example:
            >>> train_x_norm = normalize_input(train_x_flatten)
    """
    m = x_flatten.shape[1]
    x_norm = np.divide(x_flatten, 255.)
    assert (x_norm.shape == (NUM_FEATURES, m))
    return x_norm


def one_hot_encoding(y_orig, num_class=NUM_CLASSES):
    """One-hot encode labels of shape (1, m) into (num_class, m).

        Example:
            >>> train_y_encoded = one_hot_encoding(train_y.T, num_class = 43)
    """
    m = y_orig.shape[1]
    y_encoded = np.eye(num_class)[y_orig.reshape(-1)].T
    assert (y_encoded.shape == (num_class, m))
    return y_encoded


def prep_dataset(x_orig, y_orig, num_class=NUM_CLASSES):
    """Flatten and normalize images and one-hot encode labels.

        Arguments:
            x_orig (numpy.ndarray): images of shape (m, IMG_SIZE, IMG_SIZE).
            y_orig (numpy.ndarray): labels of shape (m, 1).
            num_class (int, optional): number of classes.

        Returns:
            tuple: - **x_norm** (NUM_FEATURES, m), **y_encoded** (num_class, m).

        Example:
            >>> train_x_norm, train_y_encoded = prep_dataset(train_x, train_y)
    """
    x_flatten = flatten_input(x_orig)
    x_norm = normalize_input(x_flatten)
    y_encoded = one_hot_encoding(y_orig.T, num_class)
    return x_norm, y_encoded
