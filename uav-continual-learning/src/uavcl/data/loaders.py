"""Build PyTorch DataLoaders for a class-incremental stream."""
from __future__ import annotations

from torch.utils.data import DataLoader, Subset

from .sources import DataSource
from .stream import TaskSpec


def build_task_loaders(source: DataSource, stream: list[TaskSpec], data_cfg: dict) -> list[dict]:
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))
    loaders = []
    for spec in stream:
        loaders.append(
            {
                "train": DataLoader(
                    Subset(source.splits["train"].dataset, spec.train_idx),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                ),
                "val": DataLoader(
                    Subset(source.splits["val"].dataset, spec.val_idx),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                ),
                "test": DataLoader(
                    Subset(source.splits["test"].dataset, spec.test_idx),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                ),
            }
        )
    return loaders
