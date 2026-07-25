"""Dataset sources for G1+ experiments.

The synthetic source is self-contained and used for local smoke tests. Real
datasets are imported lazily so a broken or missing torchvision does not block
the CPU-only synthetic pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import random


@dataclass
class DatasetSplit:
    dataset: object
    labels: list[int]


@dataclass
class DataSource:
    name: str
    splits: dict[str, DatasetSplit]
    num_classes: int
    class_names: list[str]


class SyntheticScenes:
    """Small deterministic color-pattern dataset for smoke tests."""

    def __init__(self, labels: list[int], image_size: int = 32, seed: int = 0):
        self.labels = [int(y) for y in labels]
        self.image_size = int(image_size)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        import torch

        y = int(self.labels[idx])
        g = torch.Generator().manual_seed(self.seed + idx * 9973 + y * 1000003)
        h = w = self.image_size
        yy = torch.linspace(0, 1, h).view(1, h, 1).expand(1, h, w)
        xx = torch.linspace(0, 1, w).view(1, 1, w).expand(1, h, w)
        base = torch.zeros(3, h, w)
        base[y % 3].fill_(0.82)
        base[(y + 1) % 3] += (0.10 + 0.08 * (y % 4)) * xx.squeeze(0)
        base[(y + 2) % 3] += (0.08 + 0.06 * (y % 5)) * yy.squeeze(0)
        noise = 0.035 * torch.randn((3, h, w), generator=g)
        return (base + noise).clamp(0.0, 1.0).float(), y


def _labels(num_classes: int, per_class: int) -> list[int]:
    return [c for c in range(num_classes) for _ in range(per_class)]


def _synthetic(cfg: dict) -> DataSource:
    num_classes = int(cfg.get("num_classes", 6))
    image_size = int(cfg.get("image_size", 32))
    seed = int(cfg.get("split_seed", 0))
    counts = {
        "train": int(cfg.get("train_per_class", 24)),
        "val": int(cfg.get("val_per_class", 8)),
        "test": int(cfg.get("test_per_class", 8)),
    }
    splits = {
        name: DatasetSplit(
            SyntheticScenes(_labels(num_classes, n), image_size=image_size, seed=seed + k * 10000),
            _labels(num_classes, n),
        )
        for k, (name, n) in enumerate(counts.items())
    }
    return DataSource(
        name="synthetic",
        splits=splits,
        num_classes=num_classes,
        class_names=[f"class_{i}" for i in range(num_classes)],
    )


def _transform(image_size: int, train: bool) -> Callable:
    from torchvision import transforms

    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend([transforms.ToTensor()])
    return transforms.Compose(ops)


def _split_dataset(dataset, labels: list[int], val_frac: float, test_frac: float, seed: int):
    from torch.utils.data import Subset

    idx = list(range(len(labels)))
    by_class: dict[int, list[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)
    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for cls in sorted(by_class):
        vals = list(by_class[cls])
        rng.shuffle(vals)
        n_test = int(round(len(vals) * test_frac))
        n_val = int(round(len(vals) * val_frac))
        test_idx.extend(vals[:n_test])
        val_idx.extend(vals[n_test:n_test + n_val])
        train_idx.extend(vals[n_test + n_val:])
    return {
        "train": DatasetSplit(Subset(dataset, sorted(train_idx)), [labels[i] for i in sorted(train_idx)]),
        "val": DatasetSplit(Subset(dataset, sorted(val_idx)), [labels[i] for i in sorted(val_idx)]),
        "test": DatasetSplit(Subset(dataset, sorted(test_idx)), [labels[i] for i in sorted(test_idx)]),
    }


def _eurosat(cfg: dict) -> DataSource:
    from torch.utils.data import Subset
    from torchvision.datasets import EuroSAT

    root = Path(cfg.get("root", "./data"))
    image_size = int(cfg.get("image_size", 64))
    seed = int(cfg.get("split_seed", 0))
    base = EuroSAT(root=str(root), download=True, transform=None)
    labels = [int(y) for y in base.targets]
    splits = _split_dataset(base, labels, val_frac=0.1, test_frac=0.1, seed=seed)
    for name, split in splits.items():
        indices = list(split.dataset.indices)
        dataset = EuroSAT(
            root=str(root),
            download=False,
            transform=_transform(image_size, train=name == "train"),
        )
        split.dataset = Subset(dataset, indices)
    class_names = list(getattr(base, "classes", [f"class_{i}" for i in range(10)]))
    return DataSource("eurosat", splits, len(class_names), class_names)


def _resisc45(cfg: dict) -> DataSource:
    from collections import Counter

    from datasets import concatenate_datasets, load_dataset
    from torch.utils.data import Dataset
    from torchvision import transforms

    image_size = int(cfg.get("image_size", 224))
    repo = str(cfg.get("hf_repo", "timm/resisc45"))
    root = Path(cfg.get("root", "./data"))
    hf = load_dataset(repo, cache_dir=str(root / "hf_cache"))
    # The Hugging Face repository provides train/validation/test already, but
    # this project uses one reproducible 80/10/10 split for every source.
    # Combine all source splits first so no RESISC45 samples are discarded.
    split_names = [name for name in ("train", "validation", "test") if name in hf]
    split_names.extend(name for name in hf if name not in split_names)
    datasets = [hf[name] for name in split_names]
    base = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    names = base.features["label"].names
    labels = [int(y) for y in base["label"]]
    expected_total = int(cfg.get("expected_total_samples", 31500))
    expected_per_class = int(cfg.get("expected_samples_per_class", 700))
    counts = Counter(labels)
    bad_counts = {int(cls): int(counts.get(cls, 0)) for cls in range(len(names))
                  if counts.get(cls, 0) != expected_per_class}
    if len(base) != expected_total or bad_counts:
        raise RuntimeError(
            "RESISC45 cache/split không đầy đủ: "
            f"total={len(base)} (cần {expected_total}), "
            f"class_counts_sai={bad_counts}. Không chạy benchmark trên hai nguồn dữ liệu khác nhau."
        )
    tfm = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

    class HFDataset(Dataset):
        def __len__(self):
            return len(base)

        def __getitem__(self, idx):
            row = base[int(idx)]
            return tfm(row["image"].convert("RGB")), int(row["label"])

    splits = _split_dataset(HFDataset(), labels, val_frac=0.1, test_frac=0.1, seed=int(cfg.get("split_seed", 0)))
    return DataSource("resisc45", splits, len(names), list(names))


def get_source(cfg: dict) -> DataSource:
    name = str(cfg.get("name", "synthetic")).lower()
    if name == "synthetic":
        return _synthetic(cfg)
    if name == "eurosat":
        return _eurosat(cfg)
    if name in {"resisc45", "nwpu-resisc45", "nwpu_resisc45"}:
        return _resisc45(cfg)
    raise KeyError(f"Unknown data source '{name}'. Available: synthetic, eurosat, resisc45")
