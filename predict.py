#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classify one or more traffic sign images from the command line.

Example:
    python predict.py samples/stop.jpg samples/yield.png --model models/gtsrb_mlp
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from dataset import label_description, preprocess_image
from ffnn import predict as predict_fn
from model_utils import load_model


def load_image(path):
    """Read an image file and turn it into a normalized (NUM_FEATURES, 1) column."""
    with Image.open(path) as img:
        array = preprocess_image(img.copy())
    return array, (array.reshape(-1, 1) / 255.)


def parse_args():
    p = argparse.ArgumentParser(description="Classify traffic sign images")
    p.add_argument("images", nargs="+", help="image files to classify")
    p.add_argument("--model", default="models/gtsrb_mlp")
    return p.parse_args()


def main():
    args = parse_args()

    model = load_model(args.model)
    if model is None:
        raise SystemExit("Could not read a model from '%s'. Train one first." % args.model)
    parameters = model["parameters"]
    labels = label_description()

    for path in args.images:
        if not os.path.exists(path):
            print("%s: not found" % path)
            continue

        _, column = load_image(path)
        prediction = predict_fn(column, parameters, second_guess=True)

        first_lbl, first_prob = prediction["First Prediction"]
        sec_lbl, sec_prob = prediction["Second Prediction"]

        print("\n%s" % path)
        print("  1st: %-28s (class %2d)  %.2f%%"
              % (labels[int(first_lbl)], int(first_lbl), float(first_prob) * 100))
        print("  2nd: %-28s (class %2d)  %.2f%%"
              % (labels[int(sec_lbl)], int(sec_lbl), float(sec_prob) * 100))


if __name__ == "__main__":
    main()
