#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the from-scratch MLP on GTSRB.

Example:
    python train.py --epochs 40 --hidden 512 256 --optimizer adam --regularizer dropout
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from dataset import (IMG_SIZE, NUM_CLASSES, NUM_FEATURES, load_dataset,
                     prep_dataset, train_dev_split)
from ffnn import init_hyperParams, init_layers, train
from model_utils import save_model, visualize_training_results


def parse_args():
    p = argparse.ArgumentParser(description="Train a traffic sign classifier on GTSRB")
    p.add_argument("--sample", type=int, default=100,
                   help="percentage of GTSRB to use (default: 100)")
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 256],
                   help="hidden layer sizes (default: 512 256)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--optimizer", choices=["sgd", "bgd", "mgd", "adam"], default="adam")
    p.add_argument("--initialization", choices=["random", "he"], default="he")
    p.add_argument("--regularizer", choices=["none", "l2", "dropout"], default="dropout")
    p.add_argument("--lambd", type=float, default=0.7,
                   help="L2 strength, used when --regularizer l2")
    p.add_argument("--keep-prob", type=float, default=0.8,
                   help="dropout keep probability per hidden layer")
    p.add_argument("--patience", type=int, default=5,
                   help="early stopping patience in epochs; 0 disables it")
    p.add_argument("--step-decay", type=int, default=0,
                   help="multiply the learning rate by --decay-gamma every N epochs; 0 disables it")
    p.add_argument("--decay-gamma", type=float, default=0.5,
                   help="step-decay factor (default 0.5: halve the learning rate)")
    p.add_argument("--l2-norm", choices=["dataset", "batch"], default="dataset",
                   help="divide lambda by the training-set size (default) or the mini-batch size (original behaviour)")
    p.add_argument("--split", choices=["track", "random"], default="track",
                   help="dev split: hold out whole physical-sign tracks (default) or random frames (original)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="models/gtsrb_mlp",
                   help="path the trained model is written to")
    p.add_argument("--use-augmented", action="store_true",
                   help="also train on the batches written by augment.py")
    p.add_argument("--no-plot", action="store_true", help="skip the training curves")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # ---- 1. load and split ----------------------------------------------------
    train_x_orig, train_y_orig, test_x_orig, test_y_orig, tracks = load_dataset(
        size_in_per=args.sample, return_tracks=True)
    train_x, train_y, dev_x, dev_y = train_dev_split(
        train_x_orig, train_y_orig, tracks=tracks if args.split == "track" else None)

    print("\nSet\t\tImages\t\t\tLabels")
    print("=" * 60)
    print("Training\t%s\t%s" % (train_x.shape, train_y.shape))
    print("Dev\t\t%s\t%s" % (dev_x.shape, dev_y.shape))
    print("Test\t\t%s\t%s" % (test_x_orig.shape, test_y_orig.shape))
    print("=" * 60)

    # ---- 2. flatten, normalize, one-hot encode --------------------------------
    train_x_norm, train_y_enc = prep_dataset(train_x, train_y)
    dev_x_norm, dev_y_enc = prep_dataset(dev_x, dev_y)

    if args.use_augmented:
        from data_augmentation import load_augmented_data
        aug_x, aug_y = load_augmented_data()
        print("Adding %d augmented examples to the training set." % aug_x.shape[1])
        train_x_norm = np.concatenate((train_x_norm, aug_x), axis=1)
        train_y_enc = np.concatenate((train_y_enc, aug_y), axis=1)

    # ---- 3. build the network -------------------------------------------------
    layers_dim = init_layers(NUM_FEATURES, NUM_CLASSES, hidden_layers=args.hidden)
    print("\nNetwork architecture: %s\n" % layers_dim)

    regularizer = None if args.regularizer == "none" else args.regularizer
    keep_probs = [args.keep_prob] * len(args.hidden) if regularizer == "dropout" else []

    hyper_params = init_hyperParams(
        alpha=args.lr,
        num_epoch=args.epochs,
        minibatch_size=args.batch_size,
        lambd=args.lambd if regularizer == "l2" else 0,
        keep_probs=keep_probs,
    )

    # ---- 4. train -------------------------------------------------------------
    history = train(
        training_data=(train_x_norm, train_y_enc),
        validation_data=(dev_x_norm, dev_y_enc),
        layers_dim=layers_dim,
        hyperParams=hyper_params,
        initialization=args.initialization,
        optimizer=args.optimizer,
        regularizer=regularizer,
        verbose=3,
        patience=args.patience or None,
        step_decay=args.step_decay or None,
        seed=args.seed,
        decay_gamma=args.decay_gamma,
        l2_norm=args.l2_norm,
    )

    # ---- 5. persist -----------------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model = {
        "parameters": history["parameters"],
        "layers_dim": layers_dim,
        "img_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "hyper_params": hyper_params,
        "optimizer": args.optimizer,
        "regularizer": regularizer,
        "seed": args.seed,
        "split": args.split,
        "l2_norm": args.l2_norm,
        "step_decay": args.step_decay,
        "decay_gamma": args.decay_gamma,
        "history": {k: history[k] for k in
                    ("accuracy", "loss", "val_accuracy", "val_loss")},
    }
    save_model(args.out, model)
    print("\nModel saved to %s" % args.out)
    print("Final train acc: %.4f | dev acc: %.4f"
          % (history["accuracy"][-1], history["val_accuracy"][-1]))

    if not args.no_plot:
        visualize_training_results(history["accuracy"], history["val_accuracy"],
                                   history["loss"], history["val_loss"])


if __name__ == "__main__":
    main()
