#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a trained model on the GTSRB test set.

Example:
    python evaluate.py --model models/gtsrb_mlp
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from dataset import NUM_CLASSES, load_dataset, prep_dataset
from ffnn import predict
from model_utils import (confusion_matrix, load_model, metric_summary,
                         model_metrics, plot_confusion_matrix,
                         visualize_mislabelled_images, visualize_prediction)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a GTSRB traffic sign classifier")
    p.add_argument("--model", default="models/gtsrb_mlp")
    p.add_argument("--sample", type=int, default=100,
                   help="percentage of the test set to evaluate on")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    model = load_model(args.model)
    if model is None:
        raise SystemExit("Could not read a model from '%s'. Train one first." % args.model)
    parameters = model["parameters"]

    _, _, test_x_orig, test_y_orig = load_dataset(size_in_per=args.sample)
    test_x_norm, _ = prep_dataset(test_x_orig, test_y_orig)

    prediction = predict(test_x_norm, parameters, second_guess=True)

    cm = confusion_matrix(test_y_orig, prediction, num_classes=NUM_CLASSES)
    metrics, macro_metrics, acc = model_metrics(cm)
    metric_summary(metrics, macro_metrics, acc)

    if not args.no_plot:
        plot_confusion_matrix(cm, dataset_type="test")
        visualize_prediction(test_x_orig, test_y_orig.T, prediction, dataset_type="test")
        visualize_mislabelled_images(test_x_orig, test_y_orig.T, prediction,
                                     dataset_type="test")


if __name__ == "__main__":
    main()
