"""Deterministic fixed-ratio sampling for concatenated LeRobot datasets."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import torch
from torch.utils.data import Sampler


def mixture_offsets(dataset_lengths: Sequence[int]) -> list[int]:
    """Return the starting global index of each concatenated dataset."""
    offsets: list[int] = []
    current = 0
    for length in dataset_lengths:
        offsets.append(current)
        current += length
    return offsets


def mixture_bucket_counts(indices: Sequence[int], dataset_lengths: Sequence[int]) -> list[int]:
    """Count concatenated indices by source dataset; useful for verification."""
    offsets = mixture_offsets(dataset_lengths)
    total_length = sum(dataset_lengths)
    counts = [0] * len(dataset_lengths)
    for index in indices:
        if index < 0 or index >= total_length:
            raise ValueError(f"Index {index} is outside the concatenated dataset.")
        for bucket in range(len(dataset_lengths) - 1, -1, -1):
            if index >= offsets[bucket]:
                counts[bucket] += 1
                break
        else:
            raise ValueError(f"Index {index} is outside the concatenated dataset.")
    return counts


def _validate_inputs(dataset_lengths: Sequence[int], weights: Sequence[float], block_size: int) -> None:
    if len(dataset_lengths) == 0:
        raise ValueError("At least one dataset is required.")
    if len(dataset_lengths) != len(weights):
        raise ValueError("dataset_lengths and weights must have the same length.")
    if any(length <= 0 for length in dataset_lengths):
        raise ValueError("Every mixed dataset must contain a positive number of frames.")
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Mixture weights must be non-negative and have a positive sum.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")


def _largest_remainder_quotas(weights: Sequence[float], count: int) -> list[int]:
    total_weight = sum(weights)
    raw = [count * weight / total_weight for weight in weights]
    quotas = [int(value) for value in raw]
    remaining = count - sum(quotas)
    order = sorted(range(len(weights)), key=lambda idx: (-(raw[idx] - quotas[idx]), idx))
    for index in order[:remaining]:
        quotas[index] += 1
    return quotas


UNIFIED_GOAL_BUCKET_QUOTAS: dict[str, int] = {
    "complete/red": 63,
    "complete/green": 63,
    "complete/yellow": 62,
    "complete/full": 62,
    "grasp_transition/red": 75,
    "grasp_transition/green": 75,
    "grasp_transition/yellow": 75,
    "grasp_transition/full": 75,
    "grasp/red": 25,
    "grasp/green": 25,
    "grasp/yellow": 25,
    "grasp/full": 25,
    "release/red": 75,
    "release/green": 75,
    "release/yellow": 75,
    "release/full": 75,
    "place/red": 13,
    "place/green": 13,
    "place/yellow": 12,
    "place/full": 12,
}

_UNIFIED_GOAL_PHASES = ("complete", "grasp_transition", "grasp", "release", "place", "atomic")
_UNIFIED_GOAL_COLORS = ("red", "green", "yellow")


@dataclass(frozen=True)
class FrameIndexPool(Sequence[int]):
    """Memory-efficient sequence of global frame indices stored as contiguous ranges."""

    ranges: tuple[range, ...]

    def __post_init__(self) -> None:
        if not self.ranges or any(index_range.step != 1 or len(index_range) == 0 for index_range in self.ranges):
            raise ValueError("FrameIndexPool requires non-empty, unit-step ranges.")
        previous_stop = -1
        for index_range in self.ranges:
            if index_range.start < previous_stop:
                raise ValueError("FrameIndexPool ranges must be ordered and non-overlapping.")
            previous_stop = index_range.stop

    @cached_property
    def _ends(self) -> tuple[int, ...]:
        total = 0
        ends = []
        for index_range in self.ranges:
            total += len(index_range)
            ends.append(total)
        return tuple(ends)

    @cached_property
    def _starts(self) -> tuple[int, ...]:
        return tuple(index_range.start for index_range in self.ranges)

    def __len__(self) -> int:
        return self._ends[-1]

    def __getitem__(self, position: int | slice) -> int | list[int]:
        if isinstance(position, slice):
            return [self[index] for index in range(*position.indices(len(self)))]
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError(position)
        range_index = bisect_right(self._ends, position)
        previous_end = self._ends[range_index - 1] if range_index else 0
        return self.ranges[range_index].start + position - previous_end

    def contains(self, index: int) -> bool:
        range_index = bisect_right(self._starts, index) - 1
        return range_index >= 0 and index < self.ranges[range_index].stop


def classify_unified_goal_task(task: str, phase: str) -> str:
    """Return the deterministic sampling bucket for one unified-goal task string."""
    if phase not in _UNIFIED_GOAL_PHASES:
        raise ValueError(f"Unknown unified-goal phase: {phase!r}")

    lowered = task.lower()
    if phase != "atomic" and all(f"put the {color} block" in lowered for color in _UNIFIED_GOAL_COLORS):
        return f"{phase}/full"

    matches = [color for color in _UNIFIED_GOAL_COLORS if f"the {color} block" in lowered]
    if len(matches) != 1:
        raise ValueError(f"Could not classify task for phase {phase!r}: {task!r}")
    return f"{phase}/{matches[0]}"


def build_unified_goal_bucket_indices(datasets: Sequence[Any]) -> dict[str, FrameIndexPool]:
    """Partition the six converted unified-goal datasets into global frame-index buckets."""
    if len(datasets) != len(_UNIFIED_GOAL_PHASES):
        raise ValueError(
            "Unified-goal sampling requires complete, grasp_transition, grasp, release, place, and atomic datasets."
        )

    bucket_ranges: dict[str, list[range]] = {bucket: [] for bucket in UNIFIED_GOAL_BUCKET_QUOTAS}
    offset = 0
    for phase, dataset in zip(_UNIFIED_GOAL_PHASES, datasets, strict=True):
        if not any(bucket.startswith(f"{phase}/") for bucket in UNIFIED_GOAL_BUCKET_QUOTAS):
            offset += dataset.num_frames
            continue
        for episode in dataset.meta.episodes:
            tasks = episode["tasks"]
            if len(tasks) != 1:
                raise ValueError(f"Unified-goal episode has {len(tasks)} tasks instead of one: {tasks!r}")
            task = str(tasks[0])
            bucket = classify_unified_goal_task(task, phase)
            if bucket not in bucket_ranges:
                raise ValueError(f"Task {task!r} was assigned to unsupported bucket {bucket!r}.")
            bucket_ranges[bucket].append(
                range(offset + int(episode["dataset_from_index"]), offset + int(episode["dataset_to_index"]))
            )
        offset += dataset.num_frames

    buckets = {bucket: FrameIndexPool(tuple(ranges)) for bucket, ranges in bucket_ranges.items() if ranges}
    missing = [bucket for bucket in UNIFIED_GOAL_BUCKET_QUOTAS if bucket not in buckets]
    if missing:
        raise ValueError(f"Unified-goal dataset is missing required buckets: {', '.join(missing)}")
    return buckets


def _bucket_contains(candidates: Sequence[int], index: int) -> bool:
    if isinstance(candidates, FrameIndexPool):
        return candidates.contains(index)
    return index in candidates


def unified_goal_bucket_counts(
    indices: Sequence[int], bucket_indices: Mapping[str, Sequence[int]]
) -> dict[str, int]:
    """Count sampled global indices by unified-goal bucket."""
    counts = {bucket: 0 for bucket in bucket_indices}
    for index in indices:
        matching = [
            bucket
            for bucket, candidates in bucket_indices.items()
            if _bucket_contains(candidates, index)
        ]
        if len(matching) != 1:
            raise ValueError(f"Global index {index} belongs to {len(matching)} unified-goal buckets.")
        counts[matching[0]] += 1
    return counts


def scaled_bucket_quotas(quotas: Mapping[str, int], count: int) -> dict[str, int]:
    """Scale named bucket quotas to a partial block with largest-remainder rounding."""
    buckets = list(quotas)
    values = _largest_remainder_quotas([quotas[bucket] for bucket in buckets], count)
    return dict(zip(buckets, values, strict=True))


class FixedRatioMixtureSampler(Sampler[int]):
    """Sample concatenated datasets with exact source quotas per block.

    The source order within a block and each selected frame index are randomized
    with a seeded generator, while the count from each source is fixed.  This
    prevents longer videos from silently changing the intended training mix.
    """

    def __init__(
        self,
        dataset_lengths: Sequence[int],
        weights: Sequence[float],
        num_samples: int,
        block_size: int = 1_000,
        seed: int = 0,
    ) -> None:
        _validate_inputs(dataset_lengths, weights, block_size)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        self.dataset_lengths = list(dataset_lengths)
        self.weights = list(weights)
        self.num_samples = num_samples
        self.block_size = block_size
        self.seed = seed
        self.offsets = mixture_offsets(dataset_lengths)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        produced = 0

        while produced < self.num_samples:
            block_count = min(self.block_size, self.num_samples - produced)
            sources: list[int] = []
            for source_index, quota in enumerate(_largest_remainder_quotas(self.weights, block_count)):
                sources.extend([source_index] * quota)

            order = torch.randperm(len(sources), generator=generator).tolist()
            for position in order:
                source_index = sources[position]
                local_index = torch.randint(
                    self.dataset_lengths[source_index], (1,), generator=generator
                ).item()
                yield self.offsets[source_index] + local_index
            produced += block_count

    def __len__(self) -> int:
        return self.num_samples


class FixedQuotaBucketSampler(Sampler[int]):
    """Sample global frame indices with exact named-bucket quotas per block."""

    def __init__(
        self,
        bucket_indices: Mapping[str, Sequence[int]],
        quotas: Mapping[str, int],
        num_samples: int,
        block_size: int = 1_000,
        seed: int = 0,
    ) -> None:
        if not quotas:
            raise ValueError("At least one bucket quota is required.")
        if set(bucket_indices) != set(quotas):
            missing = sorted(set(quotas).difference(bucket_indices))
            extra = sorted(set(bucket_indices).difference(quotas))
            details = []
            if missing:
                details.append(f"missing buckets: {', '.join(missing)}")
            if extra:
                details.append(f"unexpected buckets: {', '.join(extra)}")
            raise ValueError("Bucket indices and quotas must have identical names (" + "; ".join(details) + ").")
        if any(quota <= 0 for quota in quotas.values()):
            raise ValueError("Every bucket quota must be positive.")
        if any(len(indices) == 0 for indices in bucket_indices.values()):
            empty = [bucket for bucket, indices in bucket_indices.items() if not indices]
            raise ValueError(f"Every bucket must contain at least one frame: {', '.join(empty)}")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if block_size <= 0:
            raise ValueError("block_size must be positive.")

        self.bucket_indices = {bucket: list(indices) for bucket, indices in bucket_indices.items()}
        self.quotas = dict(quotas)
        self.num_samples = num_samples
        self.block_size = block_size
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        buckets = list(self.quotas)
        produced = 0

        while produced < self.num_samples:
            block_count = min(self.block_size, self.num_samples - produced)
            sources: list[str] = []
            for bucket, quota in zip(
                buckets,
                scaled_bucket_quotas(self.quotas, block_count).values(),
                strict=True,
            ):
                sources.extend([bucket] * quota)

            order = torch.randperm(len(sources), generator=generator).tolist()
            for position in order:
                candidates = self.bucket_indices[sources[position]]
                yield candidates[torch.randint(len(candidates), (1,), generator=generator).item()]
            produced += block_count

    def __len__(self) -> int:
        return self.num_samples
