from __future__ import annotations

import unittest


class TestFixedRatioMixtureSampler(unittest.TestCase):
    def test_each_full_block_uses_the_configured_source_quotas(self) -> None:
        from lerobot.datasets.weighted_mixture import FixedRatioMixtureSampler, mixture_bucket_counts

        lengths = [10, 20, 30, 40]
        sampler = FixedRatioMixtureSampler(
            dataset_lengths=lengths,
            weights=[0.4, 0.25, 0.25, 0.1],
            num_samples=2_000,
            block_size=1_000,
            seed=7,
        )

        indices = list(sampler)

        self.assertEqual(len(indices), 2_000)
        self.assertEqual(mixture_bucket_counts(indices[:1_000], lengths), [400, 250, 250, 100])
        self.assertEqual(mixture_bucket_counts(indices[1_000:], lengths), [400, 250, 250, 100])

    def test_partial_block_uses_largest_remainder_quotas(self) -> None:
        from lerobot.datasets.weighted_mixture import FixedRatioMixtureSampler, mixture_bucket_counts

        lengths = [3, 5, 7, 11]
        sampler = FixedRatioMixtureSampler(
            dataset_lengths=lengths,
            weights=[0.4, 0.25, 0.25, 0.1],
            num_samples=13,
            block_size=1_000,
            seed=7,
        )

        self.assertEqual(mixture_bucket_counts(list(sampler), lengths), [5, 3, 3, 2])

    def test_rejects_empty_source_dataset(self) -> None:
        from lerobot.datasets.weighted_mixture import FixedRatioMixtureSampler

        with self.assertRaisesRegex(ValueError, "positive"):
            FixedRatioMixtureSampler(
                dataset_lengths=[10, 0, 30, 40],
                weights=[0.4, 0.25, 0.25, 0.1],
                num_samples=100,
                block_size=100,
                seed=7,
            )

    def test_bucket_counter_rejects_index_after_last_dataset(self) -> None:
        from lerobot.datasets.weighted_mixture import mixture_bucket_counts

        with self.assertRaisesRegex(ValueError, "outside"):
            mixture_bucket_counts([30], [10, 20])


class TestUnifiedGoalMixtureSampler(unittest.TestCase):
    def test_each_full_block_uses_phase_and_target_quotas(self) -> None:
        from lerobot.datasets.weighted_mixture import (
            FixedQuotaBucketSampler,
            UNIFIED_GOAL_BUCKET_QUOTAS,
            unified_goal_bucket_counts,
        )

        bucket_indices = {
            bucket: list(range(index * 10, index * 10 + 10))
            for index, bucket in enumerate(UNIFIED_GOAL_BUCKET_QUOTAS)
        }
        sampler = FixedQuotaBucketSampler(
            bucket_indices=bucket_indices,
            quotas=UNIFIED_GOAL_BUCKET_QUOTAS,
            num_samples=1_000,
            block_size=1_000,
            seed=7,
        )

        self.assertEqual(
            unified_goal_bucket_counts(list(sampler), bucket_indices),
            UNIFIED_GOAL_BUCKET_QUOTAS,
        )

    def test_grasp_focus_quotas_prioritize_transition_and_release(self) -> None:
        from collections import Counter

        from lerobot.datasets.weighted_mixture import UNIFIED_GOAL_BUCKET_QUOTAS

        per_phase = Counter()
        for bucket, quota in UNIFIED_GOAL_BUCKET_QUOTAS.items():
            phase, _ = bucket.split("/", 1)
            per_phase[phase] += quota

        self.assertEqual(
            dict(per_phase),
            {
                "complete": 250,
                "grasp_transition": 300,
                "grasp": 100,
                "release": 300,
                "place": 50,
            },
        )
        self.assertEqual(sum(UNIFIED_GOAL_BUCKET_QUOTAS.values()), 1_000)
        self.assertEqual(UNIFIED_GOAL_BUCKET_QUOTAS["complete/red"], 63)
        self.assertEqual(UNIFIED_GOAL_BUCKET_QUOTAS["grasp_transition/full"], 75)
        self.assertEqual(UNIFIED_GOAL_BUCKET_QUOTAS["release/yellow"], 75)
        self.assertEqual(UNIFIED_GOAL_BUCKET_QUOTAS["place/yellow"], 12)
        self.assertFalse(any(bucket.startswith("atomic/") for bucket in UNIFIED_GOAL_BUCKET_QUOTAS))

    def test_classifies_full_goal_before_individual_color(self) -> None:
        from lerobot.datasets.weighted_mixture import classify_unified_goal_task

        full_task = (
            "put the red block into the black frame, then put the green block into the black frame, "
            "then put the yellow block into the black frame"
        )

        self.assertEqual(classify_unified_goal_task(full_task, "complete"), "complete/full")
        self.assertEqual(
            classify_unified_goal_task("put the yellow block into the black frame", "place"),
            "place/yellow",
        )
        self.assertEqual(
            classify_unified_goal_task("put the green block into the black frame", "grasp_transition"),
            "grasp_transition/green",
        )

    def test_rejects_missing_required_bucket(self) -> None:
        from lerobot.datasets.weighted_mixture import FixedQuotaBucketSampler, UNIFIED_GOAL_BUCKET_QUOTAS

        bucket_indices = {
            bucket: [index]
            for index, bucket in enumerate(UNIFIED_GOAL_BUCKET_QUOTAS)
            if bucket != "release/yellow"
        }

        with self.assertRaisesRegex(ValueError, "release/yellow"):
            FixedQuotaBucketSampler(
                bucket_indices=bucket_indices,
                quotas=UNIFIED_GOAL_BUCKET_QUOTAS,
                num_samples=1_000,
            )


class TestDatasetMixtureConfig(unittest.TestCase):
    def test_validates_six_dataset_weights(self) -> None:
        from lerobot.configs.train import DatasetMixtureConfig

        config = DatasetMixtureConfig(
            repo_ids=[
                "local/complete",
                "local/grasp",
                "local/place",
                "local/grasp_lift",
                "local/release",
                "local/atomic",
            ],
            root="/tmp/lerobot_data",
            weights=[0.4, 0.15, 0.15, 0.1, 0.1, 0.1],
        )

        config.validate()
        self.assertEqual(config.normalized_weights, [0.4, 0.15, 0.15, 0.1, 0.1, 0.1])

    def test_rejects_wrong_number_of_datasets(self) -> None:
        from lerobot.configs.train import DatasetMixtureConfig

        config = DatasetMixtureConfig(
            repo_ids=["local/complete", "local/grasp"],
            root="/tmp/lerobot_data",
            weights=[0.5, 0.5],
        )

        with self.assertRaisesRegex(ValueError, "exactly 6"):
            config.validate()


class TestGradientAccumulationConfig(unittest.TestCase):
    def test_rejects_non_positive_gradient_accumulation_steps(self) -> None:
        from lerobot.configs.train import validate_gradient_accumulation_steps

        with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
            validate_gradient_accumulation_steps(0)


if __name__ == "__main__":
    unittest.main()
