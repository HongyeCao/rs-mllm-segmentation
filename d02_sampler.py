"""Deterministic, resumable epoch sampling for D02 training."""

from __future__ import annotations

from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


def epoch_seed(seed: int, epoch: int) -> int:
    """Mix a run seed and zero-based epoch into a stable Torch seed."""

    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    # Keep the value in the positive int64 range accepted by manual_seed.
    return (int(seed) + 0x9E3779B97F4A7C15 * (int(epoch) + 1)) & 0x7FFFFFFFFFFFFFFF


def epoch_permutation(length: int, seed: int, epoch: int) -> list[int]:
    """Return the epoch permutation without touching the process Torch RNG."""

    if length < 0:
        raise ValueError("length must be non-negative")
    generator = torch.Generator()
    generator.manual_seed(epoch_seed(seed, epoch))
    return torch.randperm(length, generator=generator).tolist()


class ResumableEpochSampler(Sampler[int]):
    """A fixed epoch permutation sliced from an absolute consumed position."""

    def __init__(
        self,
        data_source: Sized,
        *,
        seed: int,
        epoch: int,
        start_position: int = 0,
    ) -> None:
        self.length = len(data_source)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.start_position = int(start_position)
        if not 0 <= self.start_position <= self.length:
            raise ValueError(
                f"sampler_position {self.start_position} is outside [0, {self.length}]"
            )
        self._permutation = epoch_permutation(self.length, self.seed, self.epoch)

    def __iter__(self) -> Iterator[int]:
        return iter(self._permutation[self.start_position :])

    def __len__(self) -> int:
        return self.length - self.start_position

    @property
    def permutation(self) -> tuple[int, ...]:
        return tuple(self._permutation)
