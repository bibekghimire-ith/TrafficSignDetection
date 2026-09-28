#!/usr/bin/env python3
"""Unit tests for the from-scratch network. Standard library only (unittest) plus NumPy.

Run from the project root:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ffnn import (backward_prop, forward_prop, init_parameters, initialize_adam,  # noqa: E402
                  learning_rate_schedule, softmax_cross_entropy_cost, update_parameters)
from model_utils import confusion_matrix, model_metrics, rand_mini_batches  # noqa: E402
from dataset import train_dev_split  # noqa: E402


def _cost(X, Y, P, lambd=0.0, reg=None, l2_m=None):
    AL, caches, _ = forward_prop(X, P)
    return softmax_cross_entropy_cost(AL, Y, caches, lambd=lambd, regularizer=reg, from_logits=True, l2_m=l2_m)


def _grad_check(reg=None, lambd=0.0, l2_m=None, n_checks=60, eps=1e-6):
    """Max relative error between backprop and central-difference gradients, float64."""
    rng = np.random.RandomState(0)
    np.random.seed(0)
    X = rng.randn(12, 7)
    Y = np.eye(4)[rng.randint(0, 4, 7)].T
    P = init_parameters([12, 8, 6, 4], "he", dtype=np.float64)
    AL, caches, _ = forward_prop(X, P)
    grads = backward_prop(AL, Y, caches, lambd=lambd, regularizer=reg, l2_m=l2_m)
    worst = 0.0
    for key in P:
        for _ in range(n_checks // len(P) + 1):
            idx = tuple(rng.randint(0, d) for d in P[key].shape)
            old = P[key][idx]
            P[key][idx] = old + eps; jp = _cost(X, Y, P, lambd, reg, l2_m)
            P[key][idx] = old - eps; jm = _cost(X, Y, P, lambd, reg, l2_m)
            P[key][idx] = old
            num = (jp - jm) / (2 * eps)
            ana = grads["d" + key][idx]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    return worst


class GradientCheck(unittest.TestCase):
    def test_backprop_matches_numeric_gradient(self):
        self.assertLess(_grad_check(), 1e-6)

    def test_backprop_matches_numeric_gradient_l2_batch_normalised(self):
        self.assertLess(_grad_check(reg="l2", lambd=0.7), 1e-6)

    def test_backprop_matches_numeric_gradient_l2_dataset_normalised(self):
        self.assertLess(_grad_check(reg="l2", lambd=0.7, l2_m=1000), 1e-6)


class Optimiser(unittest.TestCase):
    def test_adam_keeps_float32(self):
        np.random.seed(1)
        P = init_parameters([10, 5, 3], "he", dtype=np.float32)
        v, s = initialize_adam(P)
        g = {"d" + k: np.ones_like(val) for k, val in P.items()}
        for t in range(1, 4):
            update_parameters(P, g, 0.001, optimizer="adam", beta1=0.9, beta2=0.999, epsilon=1e-8, v=v, s=s, t=t)
        self.assertTrue(all(val.dtype == np.float32 for val in P.values()))

    def test_adam_first_step_moves_each_weight_by_lr(self):
        np.random.seed(1)
        P = init_parameters([4, 3], "he", dtype=np.float64)
        W0 = P["W1"].copy()
        v, s = initialize_adam(P)
        g = {"dW1": np.full_like(P["W1"], 0.5), "db1": np.full_like(P["b1"], -2.0)}
        update_parameters(P, g, 0.01, optimizer="adam", beta1=0.9, beta2=0.999, epsilon=1e-8, v=v, s=s, t=1)
        np.testing.assert_allclose(W0 - P["W1"], 0.01, rtol=1e-5)  # bias-corrected first step = lr*sign(g)

    def test_adam_flushes_subnormal_moments(self):
        # zero gradient for many steps: moments must reach exactly 0, never linger as float32 subnormals
        np.random.seed(1)
        P = init_parameters([6, 4], "he", dtype=np.float32)
        v, s = initialize_adam(P)
        g1 = {"d" + k: np.full_like(val, 1e-3) for k, val in P.items()}
        g0 = {"d" + k: np.zeros_like(val) for k, val in P.items()}
        update_parameters(P, g1, 0.001, optimizer="adam", beta1=0.9, beta2=0.999, epsilon=1e-8, v=v, s=s, t=1)
        for t in range(2, 870):   # m = 1e-4 * 0.9**868 ~ 1e-44 would be subnormal without the flush
            update_parameters(P, g0, 0.001, optimizer="adam", beta1=0.9, beta2=0.999, epsilon=1e-8, v=v, s=s, t=t)
        tiny = np.finfo(np.float32).tiny
        for d in (v, s):
            for arr in d.values():
                self.assertFalse(np.any((arr != 0) & (np.abs(arr) < tiny)))

    def test_step_decay_schedule(self):
        self.assertEqual([learning_rate_schedule(0.001, e, step=10) for e in (0, 9, 10, 25)],
                         [0.001, 0.001, 0.0005, 0.00025])


class Data(unittest.TestCase):
    def test_minibatches_do_not_reseed_global_rng(self):
        np.random.seed(123); a = np.random.rand()
        np.random.seed(123); rand_mini_batches(np.zeros((2, 10)), np.zeros((3, 10)), 4, seed=5); b = np.random.rand()
        self.assertEqual(a, b)

    def test_minibatches_cover_every_example_once(self):
        X = np.arange(20).reshape(1, 20).astype(float)
        mbs = rand_mini_batches(X, X.copy(), 6, seed=2)
        self.assertEqual(sorted(np.concatenate([m[0].ravel() for m in mbs]).tolist()), list(range(20)))

    def test_track_split_is_disjoint_and_stratified(self):
        np.random.seed(0)
        y = np.repeat(np.arange(3), 300).reshape(-1, 1)             # 3 classes x 10 tracks x 30 frames
        tracks = (y.ravel() * 100000 + np.repeat(np.arange(30), 30)).reshape(-1, 1)
        x = np.arange(900 * 4).reshape(900, 2, 2)
        trx, try_, dvx, dvy = train_dev_split(x, y, 0.1, tracks=tracks)
        tr_ids = set(tracks.ravel()[np.isin(x[:, 0, 0], trx[:, 0, 0])])
        dv_ids = set(tracks.ravel()[np.isin(x[:, 0, 0], dvx[:, 0, 0])])
        self.assertFalse(tr_ids & dv_ids)
        self.assertEqual(len(trx) + len(dvx), 900)
        self.assertEqual(sorted(np.unique(dvy)), [0, 1, 2])


try:
    import scipy  # noqa: F401  (data_augmentation needs scipy.ndimage)
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


@unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
class AugmentedBatches(unittest.TestCase):
    def test_empty_folder_raises_clear_error(self):
        import tempfile
        from data_augmentation import load_augmented_data
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "dataset", "augmented_data"))
            os.chdir(d)
            try:
                with self.assertRaisesRegex(ValueError, "No augmented batches"):
                    load_augmented_data()
            finally:
                os.chdir(cwd)


class Metrics(unittest.TestCase):
    def test_unpredicted_class_scores_zero_not_nan(self):
        y = np.array([[0], [1], [2], [2]])
        pred = {"First Prediction": [np.array([[0, 1, 1, 1]]), None]}
        cm = confusion_matrix(y, pred, num_classes=3)
        metrics, macro, acc = model_metrics(cm)
        self.assertEqual(metrics["Precision"][2], 0)
        self.assertFalse(np.isnan(macro["F1-Score"]))
        self.assertAlmostEqual(acc, 0.5)


if __name__ == "__main__":
    unittest.main()
