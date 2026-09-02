# Copyright (c) 2026 PaddlePaddle Authors.
"""Provenance-bound fixed-token dataset adapter for formal GLM52 runs."""
from __future__ import annotations

from typing import Any, Dict, Iterator


class FixedTokenSFTDataset:
    """Read the formal indexed fixed-token derivative without retokenization."""

    def __init__(self, **dataset_config: Dict[str, Any]):
        from paddleformers.datasets.loader import create_indexed_dataset

        path = dataset_config.get("train_dataset_path") or dataset_config.get("task_group")
        if not path:
            raise ValueError("fixed_token dataset requires train_dataset_path")
        self.path = path
        self.dataset = create_indexed_dataset(path, skip_warmup=True, warmup_only_rank0=True)
        if len(self.dataset) != 1:
            raise ValueError(f"fixed_token dataset must contain exactly one sample: {len(self.dataset)}")
        sample = self.dataset[0][0]
        self.readback = {
            "path": path,
            "token_ids": list(sample.token_ids),
            "labels": list(sample.labels),
            "position_ids": list(sample.position_ids),
            "num_examples": int(sample.num_examples),
            "loss_mask": [int(label != -100) for label in sample.labels],
        }

    @property
    def column_names(self):
        return None

    def map(self, function, **kwargs):
        """Formal trainer hook: authoritative fixed tokens bypass tokenization."""
        return self

    def __getitem__(self, index: int) -> Any:
        return [self.dataset[index][0]]

    def __iter__(self) -> Iterator[Any]:
        for index in range(len(self.dataset)):
            yield self[index]

    def __len__(self) -> int:
        return len(self.dataset)
