"""Click sampling, closed-loop stopping, loss semantics, data and checkpoint tests."""

import json
import uuid
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.paths import OUTPUT_ROOT
from pims_net.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from pims_net.training.click_simulator import RITMClickSimulator
from pims_net.training.interactive_losses import LossWeights, interactive_losses, prepare_target
from pims_net.training.interactive_rollout import interactive_rollout


class ClickSimulatorTests(unittest.TestCase):
    def test_positive_negative_perfect_and_no_duplicates(self):
        target = np.zeros((32, 48), dtype=bool)
        target[4:28, 10:38] = True
        simulator = RITMClickSimulator(target, np.random.default_rng(4))
        clicks = [simulator.next_click(np.zeros_like(target)) for _ in range(5)]
        self.assertEqual(len(set(clicks)), 5)
        for x, y, label in clicks:
            self.assertEqual(label, 1)
            self.assertTrue(target[y, x])
        x, y, label = simulator.next_click(np.ones_like(target))
        self.assertEqual(label, 0)
        self.assertFalse(target[y, x])
        self.assertIsNone(simulator.next_click(target))

    def test_r_seed_and_distance_choice(self):
        target = np.zeros((32, 32), dtype=bool)
        target[8:24, 8:24] = True
        prediction = np.zeros_like(target)
        prediction[:2, :2] = True  # FP interior is much smaller than FN interior.
        first = RITMClickSimulator(target, np.random.default_rng(5)).next_click(prediction)
        second = RITMClickSimulator(target, np.random.default_rng(5)).next_click(prediction)
        self.assertEqual(first, second)
        x, y, label = first
        self.assertEqual(label, 1)
        self.assertTrue(12 <= x < 20 and 12 <= y < 20)

    def test_closed_loop_stops_after_perfect_initial_mask(self):
        target = np.zeros((256, 256), dtype=bool)
        target[60:160, 80:180] = True
        logits = torch.where(torch.from_numpy(target)[None, None], 10., -10.)
        class PerfectInitialModel:
            sam_initializer = SimpleNamespace(postprocess=lambda values, sizes: [values])
            def initialize(self, image, points, labels):
                self.first_click = points, labels
                return SimpleNamespace(logits=logits, probabilities=logits.sigmoid(), image_sizes=torch.tensor([[256, 256]]))
            def refine(self, *args):
                raise AssertionError("Perfect predictions must not receive a fabricated extra click")
        result = interactive_rollout(PerfectInitialModel(), np.zeros((256, 256, 3), np.uint8), target)
        self.assertEqual(len(result["clicks"]), 1)
        self.assertEqual(result["clicks"][0]["iou"], 1)
        self.assertIsNone(result["loss"])

    def test_next_click_tracks_changing_fn_and_fp_predictions(self):
        target = np.zeros((256, 256), dtype=bool)
        target[60:180, 80:200] = True
        first_prediction = target.copy()
        first_prediction[90:130, 110:150] = False
        second_prediction = target.copy()
        second_prediction[10:35, 15:40] = True

        class ChangingModel:
            sam_initializer = SimpleNamespace(postprocess=lambda values, sizes: [values])
            stage = 0
            def state(self, prediction):
                logits = torch.where(torch.from_numpy(prediction)[None, None], 10., -10.)
                return SimpleNamespace(logits=logits, probabilities=logits.sigmoid(),
                                       image_sizes=torch.tensor([[256, 256]]),
                                       edit_probabilities=torch.full((1, 3, 256, 256), 1 / 3))
            def initialize(self, *args):
                return self.state(first_prediction)
            def refine(self, *args):
                self.stage += 1
                return self.state(second_prediction if self.stage == 1 else target)

        image = np.zeros((256, 256, 3), np.uint8)
        result = interactive_rollout(ChangingModel(), image, target, rng=np.random.default_rng(3))
        self.assertEqual(len(result["clicks"]), 3)
        second, third = result["clicks"][1:]
        self.assertEqual(second["label"], 1)
        self.assertTrue(110 <= second["x"] < 150 and 90 <= second["y"] < 130)
        self.assertEqual(third["label"], 0)
        self.assertTrue(15 <= third["x"] < 40 and 10 <= third["y"] < 35)
        limited = interactive_rollout(ChangingModel(), image, target, max_clicks=2)
        self.assertEqual(len(limited["clicks"]), 2)
        with self.assertRaises(ValueError):
            interactive_rollout(ChangingModel(), image, target, max_clicks=6)


class InteractiveLossTests(unittest.TestCase):
    def test_target_geometry_and_padding(self):
        target, valid = prepare_target(np.ones((128, 256), dtype=np.uint8), "cpu")
        self.assertEqual(tuple(target.shape), (1, 1, 256, 256))
        self.assertEqual(valid.sum().item(), 128 * 256)
        torch.testing.assert_close(target, valid)

    def test_class_balance_and_preserve_definition(self):
        old = torch.tensor([[[[.1, .9, .1, .1]]]], requires_grad=True)
        target = torch.tensor([[[[1., 0., 0., 0.]]]])  # ADD, REMOVE, KEEP, KEEP.
        intents = torch.tensor([[[[.6, .2, .1, .4]], [[.2, .6, .1, .4]], [[.2, .2, .8, .2]]]])
        current = torch.full_like(old, .4, requires_grad=True)
        state = SimpleNamespace(probabilities=current, logits=torch.logit(current), edit_probabilities=intents)
        losses = interactive_losses(state, old, target, torch.ones_like(target))
        expected = (-torch.tensor(.6).log() - torch.tensor(.6).log()
                    - (torch.tensor(.8).log() + torch.tensor(.2).log()) / 2) / 3
        torch.testing.assert_close(losses["edit"], expected)
        torch.testing.assert_close(losses["preserve"], torch.tensor(.3))
        losses["preserve"].backward()
        self.assertIsNone(old.grad)  # Detached teacher probabilities.
        torch.testing.assert_close(current.grad, torch.tensor([[[[0., 0., .5, .5]]]]))

    def test_absent_classes_empty_preserve_and_padding_exclusion(self):
        old = torch.zeros(1, 1, 1, 2)
        target = torch.ones_like(old)
        current = torch.tensor([[[[.4, .9]]]], requires_grad=True)
        intents = torch.tensor([[[[.7, .1]], [[.2, .8]], [[.1, .1]]]], requires_grad=True)
        state = SimpleNamespace(probabilities=current, logits=torch.logit(current), edit_probabilities=intents)
        losses = interactive_losses(state, old, target, torch.tensor([[[[1., 0.]]]]))
        torch.testing.assert_close(losses["edit"], -torch.tensor(.7).log())
        self.assertEqual(losses["preserve"].item(), 0)
        losses["loss"].backward()
        self.assertEqual(current.grad[0, 0, 0, 1].item(), 0)
        self.assertEqual(intents.grad[..., 1].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            LossWeights(edit=-1)


class DataAndCheckpointTests(unittest.TestCase):
    def setUp(self):
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        self.directory = (OUTPUT_ROOT / f"interactive_test_{uuid.uuid4().hex}").resolve()
        self.assertTrue(self.directory.is_relative_to(OUTPUT_ROOT.resolve()))
        self.directory.mkdir()
        self.addCleanup(self.cleanup_files)

    def cleanup_files(self):
        # Only files produced by this test; no recursive removal or restricted
        # Windows tempfile ACLs, which interfere with the managed execution host.
        for path in self.directory.iterdir():
            path.unlink()
        self.directory.rmdir()

    def test_manifest_binary_data_and_reject_instance_ids(self):
        Image.fromarray(np.zeros((12, 20, 3), dtype=np.uint8)).save(self.directory / "image.png")
        mask = np.zeros((12, 20), dtype=np.uint8)
        mask[2:8, 4:12] = 255
        Image.fromarray(mask).save(self.directory / "mask.png")
        manifest = self.directory / "train.jsonl"
        manifest.write_text(json.dumps({"image": "image.png", "mask": "mask.png"}) + "\n", encoding="utf-8")
        dataset = InteractiveDataset(manifest)
        sample = dataset[0]
        self.assertEqual(sample["image"].shape, (12, 20, 3))
        self.assertEqual(sample["mask"].dtype, np.bool_)
        mask[0, 0] = 2
        Image.fromarray(mask).save(self.directory / "mask.png")
        with self.assertRaises(ValueError):
            dataset[0]

    def test_research_weights_and_optimizer_roundtrip(self):
        model = nn.Module()
        model.sam_initializer = nn.Linear(2, 2).requires_grad_(False)
        model.edit_intent_decoder = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.edit_intent_decoder.parameters())
        model.edit_intent_decoder(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        saved = model.edit_intent_decoder.weight.detach().clone()
        baseline = model.sam_initializer.weight.detach().clone()
        path = self.directory / "last.pt"
        save_training_checkpoint(path, model, optimizer, 1, 2, {"seed": 42})
        with torch.no_grad():
            model.edit_intent_decoder.weight.zero_()
        checkpoint = load_training_checkpoint(path, model, optimizer)
        torch.testing.assert_close(model.edit_intent_decoder.weight, saved)
        torch.testing.assert_close(model.sam_initializer.weight, baseline)
        self.assertFalse(any(key.startswith("sam_initializer.") for key in checkpoint["research_state_dict"]))
        self.assertEqual(checkpoint["global_step"], 2)
        self.assertTrue(optimizer.state)
        model.variant = "a2"  # A2/A3 share keys but must not silently exchange weights.
        with self.assertRaisesRegex(ValueError, "variant"):
            load_training_checkpoint(path, model)


if __name__ == "__main__":
    unittest.main()
