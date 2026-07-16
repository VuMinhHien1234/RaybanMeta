"""Thao tác trên "state" của bộ nhớ Titans (G2).

State do titans-pytorch trả về là cấu trúc lồng nhau (namedtuple/dict/tuple)
chứa tensor. Các hàm ở đây đi đệ quy qua cấu trúc đó, áp dụng phép biến đổi
lên TỪNG tensor, giữ nguyên hình dạng cấu trúc — nhờ vậy code không phụ
thuộc vào version cụ thể của thư viện.

- detach_state : cắt gradient (truncated BPTT) nhưng GIỮ giá trị ký ức.
- clone_state  : bản sao độc lập — dùng khi eval ("chấm thi không ghi trí nhớ").
- state_to_cpu : chuyển về CPU để torch.save (checkpoint S10).
- state_norm   : độ lớn tổng của state — log mỗi task để phát hiện "phình" (C2).
- count_floats : đếm phần tử — báo cáo chi phí bộ nhớ.
"""
from __future__ import annotations

import math

import torch


def _tree_map(fn, obj):
    """Áp fn lên mọi tensor trong cấu trúc lồng nhau, giữ nguyên khung."""
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
        return type(obj)(*(_tree_map(fn, o) for o in obj))
    if isinstance(obj, tuple):
        return tuple(_tree_map(fn, o) for o in obj)
    if isinstance(obj, list):
        return [_tree_map(fn, o) for o in obj]
    if isinstance(obj, dict):
        return {k: _tree_map(fn, v) for k, v in obj.items()}
    return obj  # int/float/None/str... giữ nguyên


def _tree_tensors(obj):
    """Liệt kê mọi tensor trong cấu trúc (generator)."""
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            yield from _tree_tensors(o)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _tree_tensors(v)


def detach_state(state):
    return _tree_map(lambda t: t.detach(), state)


def clone_state(state):
    return _tree_map(lambda t: t.detach().clone(), state)


def state_to_cpu(state):
    return _tree_map(lambda t: t.detach().cpu(), state)


def state_norm(state) -> float:
    """sqrt(tổng bình phương norm các tensor float) — 0.0 nếu state rỗng/None."""
    if state is None:
        return 0.0
    total = 0.0
    for t in _tree_tensors(state):
        if t.is_floating_point() and t.numel() > 0:
            total += float(t.detach().float().norm()) ** 2
    return math.sqrt(total)


def count_floats(state) -> int:
    if state is None:
        return 0
    return sum(t.numel() for t in _tree_tensors(state))
