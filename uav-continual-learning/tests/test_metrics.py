"""Pure-numpy unit tests for the continual-learning metrics (no torch needed).

Hand-computed on a 3-task accuracy matrix R[i, j] = acc on task j after task i.
"""
import numpy as np

from uavcl.metrics import average_accuracy, average_forgetting, backward_transfer

R = np.array([
    [0.90, 0.00, 0.00],
    [0.80, 0.85, 0.00],
    [0.70, 0.80, 0.95],
])


def test_average_accuracy():
    assert abs(average_accuracy(R) - (0.70 + 0.80 + 0.95) / 3) < 1e-9


def test_backward_transfer():
    # mean[(0.70-0.90), (0.80-0.85)] = -0.125
    assert abs(backward_transfer(R) - (-0.125)) < 1e-9


def test_average_forgetting():
    # task0: max(0.90,0.80)-0.70=0.20 ; task1: 0.85-0.80=0.05 ; mean=0.125
    assert abs(average_forgetting(R) - 0.125) < 1e-9
