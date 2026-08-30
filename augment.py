#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline data augmentation for the GTSRB training set.

Writes preprocessed, augmented batches into ``dataset/augmented_data/`` which
``train.py --use-augmented`` then folds into the training set. Each pass
produces 5 extra variants per image (crop-and-pad, rotate, shift, blur, zoom)
plus the original, so one pass over ~35k training images yields ~210k examples.

Horizontal flip is deliberately excluded: it relabels directional signs.

Example:
    python augment.py --aug-count 1 --batch-size 512
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_augmentation import data_generator
from dataset import load_dataset, train_dev_split


def parse_args():
    p = argparse.ArgumentParser(description="Generate augmented GTSRB training data")
    p.add_argument("--sample", type=int, default=100,
                   help="percentage of GTSRB to augment (default: 100)")
    p.add_argument("--aug-count", type=int, default=1,
                   help="number of augmentation passes; each writes one batch file")
    p.add_argument("--batch-size", type=int, default=512,
                   help="images processed per chunk, to bound memory use")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    train_x_orig, train_y_orig, _, _ = load_dataset(size_in_per=args.sample)
    # augment only the training split; the dev set must stay untouched
    train_x, train_y, _, _ = train_dev_split(train_x_orig, train_y_orig)

    print("Augmenting %d training images (%d pass(es))..."
          % (train_x.shape[0], args.aug_count))

    data_generator(train_x, train_y,
                   batch_size=args.batch_size,
                   aug_count=args.aug_count,
                   verbose=2,
                   pre_process_data=True)

    print("\nAugmented batches written to dataset/augmented_data/")
    print("Train on them with: python train.py --use-augmented")


if __name__ == "__main__":
    main()
