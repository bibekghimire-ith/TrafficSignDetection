#!/usr/bin/env python3
"""Run train.py with the design matrices held as float32 instead of float64.

Used only for E5: with 6x augmented data the unmodified train.py needs ~6 GB
(float64 augmented matrix + concatenated copy + per-epoch shuffled copy) and was
OOM-killed on the 8 GB sandbox. Network parameters and arithmetic stay float64
(float32 inputs are upcast on the first matmul); nothing in src/ is changed.
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
import data_augmentation, train
_orig_prep, _orig_aug = train.prep_dataset, data_augmentation.load_augmented_data
def prep32(x, y):
    X, Y = _orig_prep(x, y); return X.astype(np.float32), Y.astype(np.float32)
def aug32():
    X, Y = _orig_aug(); return X.astype(np.float32, copy=False), Y.astype(np.float32, copy=False)
train.prep_dataset = prep32
data_augmentation.load_augmented_data = aug32
if __name__ == "__main__":
    train.main()
