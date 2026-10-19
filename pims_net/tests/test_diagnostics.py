"""Diagnostic metrics, probability-domain heatmaps, and observer noninterference."""

import json
from types import SimpleNamespace
import unittest
import uuid

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from pims_net.evaluation.diagnostics import DiagnosticWriter, intent_statistics, round_summary
from pims_net.evaluation.diagnostic_report import probability_statistics, transition_analysis, write_report
from pims_net.evaluation.methods import PIMSSession
from pims_net.evaluation.metrics import segmentation_statistics
from pims_net.evaluation.protocol import EvaluationProtocol, evaluate_session
from pims_net.paths import OUTPUT_ROOT


class DiagnosticTests(unittest.TestCase):
    def test_confusion_precision_recall_and_foreground_area(self):
        truth = np.array([[1, 1, 0, 0]], bool)
        predicted = np.array([[1, 0, 1, 1]], bool)
        result = segmentation_statistics(predicted, truth)
        self.assertEqual([result[k] for k in ("tp", "fp", "fn", "tn")], [1, 2, 1, 0])
        self.assertEqual(result["precision"], 1 / 3)
        self.assertEqual(result["recall"], .5)
        self.assertEqual(result["foreground_ratio"], .75)
        self.assertEqual(result["gt_foreground_ratio"], .5)
        self.assertEqual(result["pred_to_gt_area_ratio"], 1.5)
        empty = segmentation_statistics(np.zeros_like(truth), truth)
        self.assertEqual(empty["precision"], 0)
        self.assertEqual(empty["recall"], 0)
        confidence = probability_statistics(np.array([[[.5, 1., .5, .2]]]), truth,
                                            np.array([[1, 1, 0, 1]], bool))
        self.assertEqual(confidence["foreground"]["near_05_fraction"], .5)
        self.assertEqual(confidence["background"]["near_05_fraction"], 0)
        self.assertEqual(confidence["background"]["pixels"], 1)

    def test_intent_conditioning_effective_changes_and_excluded_padding(self):
        old = np.array([[[.8, .1, .1, .9, .8]]], np.float32)
        gt = np.array([[1, 0, 1, 0, 1]], bool)
        valid = np.array([[1, 1, 1, 1, 0]], bool)
        intents = np.array([[[0, 1, 1, 0, 0]], [[1, 0, 0, 1, 1]], [[0, 0, 0, 0, 0]]], np.float32)
        current = intents[2:3] * old + intents[:1]
        result = intent_statistics(old, current, intents, gt, valid)
        self.assertLess(result["formula_max_error"], 1e-6)
        self.assertEqual(result["groups"]["tp"]["pixels"], 1)
        self.assertEqual(result["groups"]["tp"]["REMOVE"], 1)
        self.assertAlmostEqual(result["groups"]["tp"]["effective_remove"], .8)
        self.assertEqual(result["groups"]["tp"]["background_crossings"], 1)
        self.assertEqual(result["groups"]["tn"]["foreground_crossings"], 1)
        self.assertEqual(result["groups"]["fn"]["ADD"], 1)
        self.assertEqual(result["groups"]["fp"]["REMOVE"], 1)

    def test_intent_snapshot_resizes_probabilities_without_sigmoid(self):
        intents = torch.zeros(1, 3, 2, 2)
        intents[:, 0] = 1
        initializer = SimpleNamespace(postprocess=lambda x, sizes: [F.interpolate(x, (3, 5), mode="bilinear", align_corners=False)])
        session = PIMSSession(SimpleNamespace(sam_initializer=initializer), None, [(3, 5)])
        session.state = SimpleNamespace(edit_probabilities=intents, logits=torch.zeros(1, 1, 2, 2), probabilities=torch.ones(1, 1, 2, 2))
        snapshot = session.diagnostic_snapshot()
        np.testing.assert_allclose(snapshot["original_intents"][0], 1)
        np.testing.assert_array_equal(snapshot["original_intents"][1:], 0)
        self.assertEqual(snapshot["original_intents"].shape, (3, 3, 5))

    def test_round_deltas_transitions_and_observer_equivalence(self):
        truth = np.zeros((16, 16), bool)
        truth[4:12, 4:12] = True
        first = truth.copy()
        first[4:8, 4:8] = False
        second = truth.copy()
        second[:2, :2] = True
        class Session:
            def __init__(self): self.n = 0
            def add_click(self, *click):
                self.n += 1
                return [first, second, truth][min(self.n, 3) - 1].astype(float)
        captured = []
        plain = evaluate_session(Session(), truth)
        observed = evaluate_session(Session(), truth, on_step=lambda **kw: captured.append(kw["item"].copy()))
        self.assertEqual(plain, observed)
        self.assertIsNone(captured[0]["iou_delta"])
        self.assertEqual(captured[1]["tn_lost"], 4)
        self.assertEqual(captured[1]["fn_fixed"], 16)
        self.assertEqual(captured[2]["fp_fixed"], 4)
        self.assertAlmostEqual(captured[1]["iou_delta"], 64 / 68 - .75)
        summary = round_summary({"a3": [observed]}, 5)
        self.assertEqual(summary[-1]["iou"], 1)
        self.assertEqual(summary[-1]["iou_delta"], 0)
        self.assertEqual(summary[-1]["objects_updated"], 0)
        analysis = transition_analysis({"a2": [observed], "a3": [observed]})
        self.assertEqual(analysis["a2"]["transitions"]["fn_fixed"]["rate"], 1)
        self.assertEqual(analysis["a2"]["transitions"]["fp_fixed"]["rate"], 1)
        self.assertEqual(analysis["a2"]["transitions"]["tp_lost"]["rate"], 0)
        self.assertEqual(analysis["a3_minus_a2"]["tied_objects"], 1)

    def test_complete_artifacts_and_absent_intents(self):
        directory = (OUTPUT_ROOT / f"diagnostic_test_{uuid.uuid4().hex}").resolve()
        directory.mkdir(parents=True)
        try:
            truth = np.zeros((16, 24), bool)
            truth[4:12, 6:18] = True
            native_gt = cv2.resize(truth.astype(np.float32), (256, 171), interpolation=cv2.INTER_NEAREST)
            padded = np.zeros((256, 256), np.float32)
            padded[:171] = native_gt
            class Session:
                def __init__(self, edit): self.n, self.edit = 0, edit
                def add_click(self, *click):
                    self.n += 1
                    return np.zeros_like(truth, float) if self.n == 1 else truth.astype(float)
                def diagnostic_snapshot(self):
                    previous = np.full((1, 256, 256), .1, np.float32)
                    current = previous if self.n == 1 else padded[None]
                    intents = np.stack((padded, 1 - padded, np.zeros_like(padded))) if self.n > 1 and self.edit else None
                    original = np.stack((truth, ~truth, np.zeros_like(truth))).astype(np.float32) if intents is not None else None
                    return {"logits": np.zeros_like(previous), "probabilities": current,
                            "previous_probabilities": previous if self.n > 1 else None,
                            "intents": intents, "original_intents": original}
            writer = DiagnosticWriter(directory)
            writer.begin_object(0, {"id": "test", "image": np.zeros((16, 24, 3), np.uint8), "mask": truth})
            records = {}
            for method in ("a0", "a1", "a2", "a3"):
                records[method] = [evaluate_session(Session(method in ("a2", "a3")), truth,
                                                   EvaluationProtocol(max_clicks=5), writer.callback(method))]
            writer.end_object()
            summary = writer.finalize(records, 5)
            root = directory / "diagnostics/object_0000"
            self.assertEqual(summary["actual_rounds_saved"], 8)
            self.assertFalse((root / "a3/round_01/ADD.png").exists())
            self.assertFalse((root / "a1/round_02/ADD.png").exists())
            self.assertTrue((root / "a3/round_02/ADD.png").exists())
            with Image.open(root / "a3/round_02/mask.png") as mask:
                np.testing.assert_array_equal(np.array(mask) > 0, truth)
            with np.load(root / "a3/round_02/native.npz") as data:
                np.testing.assert_array_equal(data["intents"].sum(0), 1)
            metrics = json.loads((root / "a3/round_02/metrics.json").read_text())
            self.assertEqual(metrics["iou"], 1)
            self.assertEqual(metrics["iou_delta"], 1)
            self.assertEqual(len(json.loads((root / "a3/round_02/clicks.json").read_text())), 2)
            with (directory / "per_object.jsonl").open("w", encoding="utf-8") as stream:
                for method, objects in records.items():
                    stream.write(json.dumps({"method": method, **objects[0]}) + "\n")
            analysis = write_report(directory)
            self.assertEqual(analysis["a3_minus_a2"]["tied_objects"], 1)
            self.assertTrue((directory / "diagnostics/trends.png").is_file())
            self.assertTrue((directory / "diagnostics/analysis.html").is_file())
        finally:
            self.assertTrue(directory.is_relative_to(OUTPUT_ROOT.resolve()))
            for path in sorted(directory.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                self.assertTrue(path.resolve().is_relative_to(directory))
                path.unlink() if path.is_file() else path.rmdir()
            directory.rmdir()


if __name__ == "__main__":
    unittest.main(verbosity=2)
