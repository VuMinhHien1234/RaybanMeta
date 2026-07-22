"""Data sources and class-incremental task streams."""
from .sources import DataSource, DatasetSplit, get_source
from .stream import TaskSpec, build_stream, describe_stream

__all__ = [
    "DataSource",
    "DatasetSplit",
    "TaskSpec",
    "build_stream",
    "describe_stream",
    "get_source",
]
