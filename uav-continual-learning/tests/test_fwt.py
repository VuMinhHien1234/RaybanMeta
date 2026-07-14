"""Test metric Forward Transfer — thuần numpy."""
import numpy as np

from uavcl.metrics import forward_transfer


def test_fwt_reads_upper_diagonal():
    # R[j-1, j]: R[0,1]=0.4, R[1,2]=0.6 -> FWT = mean(0.4, 0.6) = 0.5
    R = np.array([
        [0.9, 0.4, 0.0],
        [0.8, 0.9, 0.6],
        [0.7, 0.8, 0.9],
    ])
    assert abs(forward_transfer(R) - 0.5) < 1e-9
    # trừ mức đoán mò
    assert abs(forward_transfer(R, chance=0.5) - 0.0) < 1e-9


def test_fwt_single_task_zero():
    assert forward_transfer(np.array([[1.0]])) == 0.0
