"""Metric definitions, adaptive protocol, and formal experiment guardrails."""

import copy
import unittest

import numpy as np

from pims_net.evaluation.experiment import MATCHED_CONFIG, validate_experiment
from pims_net.evaluation.metrics import boundary_iou, correct_region_damage, mask_boundary, mask_iou
from pims_net.evaluation.protocol import EvaluationClicker, EvaluationProtocol, aggregate_results, evaluate_session


class SequenceSession:
    def __init__(self, predictions):
        self.predictions = predictions
        self.clicks = []

    def add_click(self, *click):
        self.clicks.append(click)
        return self.predictions[min(len(self.clicks), len(self.predictions)) - 1].astype(float)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.target = np.zeros((40, 40), bool)
        self.target[10:30, 10:30] = True

    def test_iou_and_boundary_at_image_edges(self):
        empty = np.zeros((5, 5), bool)
        full = ~empty
        self.assertEqual(mask_iou(empty, empty), 1)
        self.assertEqual(mask_iou(full, empty), 0)
        expected = full.copy()
        expected[1:-1, 1:-1] = False
        np.testing.assert_array_equal(mask_boundary(full), expected)
        self.assertEqual(boundary_iou(full, full), 1)
        self.assertEqual(boundary_iou(full, empty), 0)
        shifted = self.target.copy()
        shifted[10:30, 10] = False
        self.assertLess(boundary_iou(shifted, self.target), mask_iou(shifted, self.target))

    def test_damage_counts_both_foreground_and_background(self):
        target = np.array([[1, 1, 0, 0]], bool)
        old = np.array([[1, 0, 0, 0]], bool)
        new = np.array([[0, 1, 1, 0]], bool)
        result = correct_region_damage(old, new, target)
        self.assertEqual(result, {"damaged_pixels": 2, "previous_correct_pixels": 3, "damage_rate": 2 / 3})
        self.assertEqual(correct_region_damage(~target, target, target)["damage_rate"], 0)

    def test_deterministic_clicks_and_changing_errors(self):
        clicker = EvaluationClicker(self.target)
        first = clicker.next_click(np.zeros_like(self.target))
        self.assertEqual(first, (19, 19, 1))
        self.assertNotEqual(clicker.next_click(np.zeros_like(self.target)), first)
        x, y, label = clicker.next_click(np.ones_like(self.target))
        self.assertEqual(label, 0)
        self.assertFalse(self.target[y, x])
        self.assertIsNone(clicker.next_click(self.target))
        # Equal FN/FP distances select negative, then row-major coordinates.
        target = np.array([[1, 0]], bool)
        self.assertEqual(EvaluationClicker(target).next_click(~target), (1, 0, 0))

    def test_perfect_early_stop_forward_fills_and_zero_damage(self):
        session = SequenceSession([self.target])
        result = evaluate_session(session, self.target)
        self.assertEqual(result["clicks_used"], 1)
        for key in ("IoU@1", "IoU@3", "IoU@5", "BoundaryIoU@5", "NoC@90"):
            self.assertEqual(result[key], 1)
        self.assertEqual(result["correct_region_damage_rate@2:5"], 0)

    def test_success_still_measures_five_clicks_and_failure_is_capped(self):
        nearly = self.target.copy()
        nearly[10, 10:30] = False  # IoU=.95 but more than five error pixels.
        result = evaluate_session(SequenceSession([nearly]), self.target)
        self.assertEqual(result["NoC@90"], 1)
        self.assertEqual(result["clicks_used"], 5)
        failed = evaluate_session(SequenceSession([np.zeros_like(self.target)]), self.target, EvaluationProtocol(max_clicks=7))
        self.assertEqual(failed["NoC@90"], 7)
        self.assertEqual(failed["clicks_used"], 7)
        self.assertFalse(failed["reached_90"])
        mean = aggregate_results([result, failed])
        self.assertEqual(mean["failure_rate@90"], .5)
        self.assertEqual(mean["IoU@1"], .475)

    def test_success_after_five_and_damage_excludes_late_transitions(self):
        empty = np.zeros_like(self.target)
        result = evaluate_session(SequenceSession([empty] * 5 + [self.target]), self.target, EvaluationProtocol(max_clicks=8))
        self.assertEqual(result["NoC@90"], 6)
        self.assertEqual(result["IoU@5"], 0)
        self.assertEqual(result["correct_region_damage_rate@2:5"], 0)
        predictions = [empty, np.ones_like(empty), empty, np.ones_like(empty), empty]
        result = evaluate_session(SequenceSession(predictions), self.target, EvaluationProtocol(max_clicks=5))
        self.assertEqual(result["correct_region_damage_rate@2:5"], 1)

    def test_invalid_outputs_and_protocol(self):
        for kwargs in ({"max_clicks": 4}, {"target_iou": .8}, {"boundary_ratio": float("nan")}):
            with self.assertRaises(ValueError):
                EvaluationProtocol(**kwargs)
        with self.assertRaises(ValueError):
            evaluate_session(SequenceSession([np.full_like(self.target, np.nan, dtype=float)]), self.target)

    def test_independent_training_provenance_and_matched_budget(self):
        config = {key: 1 for key in MATCHED_CONFIG}
        config.update(baseline_sha256="baseline", sam_frozen=True, optimizer_updates=2)
        checkpoints = {name: {"variant": name, "format_version": 2, "epoch": 1, "global_step": 2,
                              "config": {**config, "variant": name}} for name in ("a1", "a2", "a3")}
        validate_experiment(checkpoints, "baseline")
        for field, value in (("variant", "a3"), ("optimizer_updates", 0), ("max_clicks", 3), ("baseline_sha256", "changed")):
            changed = copy.deepcopy(checkpoints)
            changed["a1"]["config"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_experiment(changed, "baseline")
        changed = copy.deepcopy(checkpoints)
        changed["a2"]["global_step"] += 1
        with self.assertRaises(ValueError):
            validate_experiment(changed, "baseline")


if __name__ == "__main__":
    unittest.main()
