from typing import Tuple, NamedTuple, Iterator
import torch
from torch.utils.data import Dataset
import numpy as np

Batch = NamedTuple("batch", [("inputs", torch.Tensor), ("targets", torch.Tensor)])


class DataIterator:
    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BatchIterator(DataIterator):
    """
    Returns an iterator.
    """

    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> Iterator[Batch]:
        start_idx = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(start_idx)

        for indx in start_idx:
            end = indx + self.batch_size
            batch_inputs = inputs[indx:end]
            batch_targets = targets[indx:end]
            yield Batch(batch_inputs, batch_targets)


class CustomDataLoader(Dataset):
    def __init__(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self.X = data[0]
        self.Y = data[1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx, :]
