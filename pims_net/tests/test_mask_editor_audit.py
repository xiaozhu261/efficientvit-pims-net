"""Mask-editor invariants and explicit reproductions of current numerical issues.

Expected failures document unmet invariants; they are findings, not successful
identity tests. Production model/training code is intentionally unchanged.
"""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from pims_net.models import MaskEditor, PIMSNet
from pims_net.models.pims_net import InteractionState
from pims_net.training.interactive_losses import prepare_target


def oracle_intents(previous, target, threshold=.5):
    """One-hot FN=ADD, FP=REMOVE, correct=KEEP, with explicit threshold."""
    truth, foreground = target.bool(), previous > threshold
    return torch.cat((truth & ~foreground, ~truth & foreground, truth == foreground), 1).to(previous.dtype)


def pure_keep(previous):
    intents = previous.new_zeros((previous.shape[0], 3, *previous.shape[-2:]))
    intents[:, 2] = 1
    return intents


class EditorAuditTests(unittest.TestCase):
    def test_pure_keep_five_rounds_bit_exact_and_identity_gradient(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(device=device, dtype=dtype):
                    old = torch.rand(2, 1, 256, 256, dtype=dtype, device=device)
                    old[..., 0, :4] = torch.tensor([0., 1., .49, .5], dtype=dtype, device=device)
                    old.requires_grad_()
                    current = old
                    for _ in range(5):
                        current = MaskEditor()(current, pure_keep(current))
                        torch.testing.assert_close(current, old, atol=0, rtol=0)
                    current.sum().backward()
                    torch.testing.assert_close(old.grad, torch.ones_like(old), atol=0, rtol=0)

    def test_gt_oracle_fixes_all_binary_errors_and_preserves_correct_probabilities(self):
        old = torch.linspace(0, 1, 2 * 256 * 256).reshape(2, 1, 256, 256)
        old[..., 0, :4] = torch.tensor([0., 1., .495, .5])
        target = (torch.arange(old.numel()).reshape_as(old) % 3 == 0).float()
        intents = oracle_intents(old, target)
        self.assertTrue(all(intents[:, c].any() for c in range(3)))
        current = MaskEditor()(old, intents)
        self.assertTrue(torch.equal(current > .5, target.bool()))
        correct = (old > .5) == target.bool()
        torch.testing.assert_close(current[correct], old[correct], atol=0, rtol=0)
        for _ in range(4):
            current = MaskEditor()(current, oracle_intents(current, target))
            self.assertTrue(torch.equal(current > .5, target.bool()))

    def test_gt_probability_oracle_and_no_double_sigmoid(self):
        old = torch.rand(2, 1, 256, 256)
        target = torch.randint(0, 2, old.shape).float()
        intents = torch.cat((target, 1 - target, torch.zeros_like(target)), 1)
        result = MaskEditor()(old, intents)
        torch.testing.assert_close(result, target, atol=0, rtol=0)
        # Sending raw logits is rejected rather than silently sigmoid-ing again.
        with self.assertRaises(ValueError):
            MaskEditor()(torch.full_like(old, -2), intents)

    @unittest.expectedFailure
    def test_training_threshold_oracle_should_be_perfect_at_eval_threshold(self):
        old = torch.tensor([[[[.495]]]])
        truth = torch.ones_like(old)
        result = MaskEditor()(old, oracle_intents(old, truth, threshold=.49))
        self.assertTrue(torch.equal(result > .5, truth.bool()))

    @unittest.expectedFailure
    def test_predictor_and_evaluator_thresholds_should_be_bit_identical(self):
        logits = torch.tensor([0., 1e-8, -1e-8], dtype=torch.float32)
        self.assertTrue(torch.equal(logits > 0, logits.sigmoid() > .5))

    def test_target_geometry_matches_independent_coordinate_mapping(self):
        for height, width in ((173, 281), (251, 137), (427, 640), (640, 427), (511, 513)):
            with self.subTest(size=(height, width)):
                yy, xx = np.indices((height, width))
                source = ((xx // 13 + 2 * (yy // 17)) % 3 == 0)
                target, valid = prepare_target(source, "cpu")
                scale = 1024 / max(height, width)
                resized_h, resized_w = int(height * scale + .5), int(width * scale + .5)
                y, x = np.indices((256, 256))
                expected_valid = (4 * y < resized_h) & (4 * x < resized_w)
                source_y = np.minimum((4 * y * height / resized_h).astype(int), height - 1)
                source_x = np.minimum((4 * x * width / resized_w).astype(int), width - 1)
                expected = source[source_y, source_x] & expected_valid
                np.testing.assert_array_equal(valid[0, 0].numpy().astype(bool), expected_valid)
                np.testing.assert_array_equal(target[0, 0].numpy().astype(bool), expected)


class PipelineAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = PIMSNet().to(cls.device).eval()

    def make_state(self, saturated=False):
        features = torch.randn(1, 256, 64, 64, device=self.device)
        logits = torch.full((1, 1, 256, 256), -2., device=self.device)
        logits[..., 128:] = 80. if saturated else 2.
        return InteractionState(features, torch.tensor([[1024., 1024.]], device=self.device),
                                torch.tensor([[[100., 100.]]], device=self.device),
                                torch.ones(1, 1, device=self.device, dtype=torch.long), logits,
                                self.model.spatial_memory.reset_state(features), logits.sigmoid())

    def keep_refine(self, state):
        with patch.object(self.model.edit_intent_decoder, "forward", return_value=pure_keep(state.probabilities)):
            return self.model.refine(state, [[[300., 300.]]], [[0]])

    @torch.no_grad()
    def test_keep_pipeline_probability_identity_five_rounds(self):
        initial = self.make_state(saturated=True)
        current = initial
        for _ in range(5):
            current = self.keep_refine(current)
            torch.testing.assert_close(current.probabilities, initial.probabilities, atol=0, rtol=0)
            self.assertTrue(torch.isfinite(current.logits).all())
        self.assertEqual(initial.points.shape[1], 1)
        self.assertEqual(current.points.shape[1], 6)

    @unittest.expectedFailure
    @torch.no_grad()
    def test_keep_pipeline_should_preserve_raw_logits(self):
        initial = self.make_state(saturated=True)
        current = self.keep_refine(initial)
        torch.testing.assert_close(current.logits, initial.logits, atol=1e-5, rtol=0)

    @unittest.expectedFailure
    @torch.no_grad()
    def test_keep_pipeline_should_preserve_original_resolution_binary_mask(self):
        initial = self.make_state(saturated=True)
        current = self.keep_refine(initial)
        before = self.model.sam_initializer.postprocess(initial.logits, initial.image_sizes)[0] > 0
        after = self.model.sam_initializer.postprocess(current.logits, current.image_sizes)[0] > 0
        self.assertTrue(torch.equal(before, after))

    @torch.no_grad()
    def test_keep_unsaturated_control_preserves_original_binary_mask(self):
        initial = self.make_state()
        current = self.keep_refine(initial)
        before = self.model.sam_initializer.postprocess(initial.logits, initial.image_sizes)[0] > 0
        after = self.model.sam_initializer.postprocess(current.logits, current.image_sizes)[0] > 0
        self.assertTrue(torch.equal(before, after))

    @torch.no_grad()
    def test_state_memory_receives_latest_mask_and_hidden_without_mutation(self):
        initial = self.make_state()
        saved_logits, saved_hidden = initial.logits.clone(), initial.hidden.clone()
        with patch.object(self.model.spatial_memory.conv_gru, "forward", wraps=self.model.spatial_memory.conv_gru.forward) as gru:
            current = self.keep_refine(initial)
            inputs, hidden = gru.call_args.args
            expected_mask = F.interpolate(initial.probabilities, (64, 64), mode="bilinear", align_corners=False)
            torch.testing.assert_close(inputs[:, -1:], expected_mask)
            self.assertIs(hidden, initial.hidden)
        with patch.object(self.model.spatial_memory, "forward", wraps=self.model.spatial_memory.forward) as memory:
            self.keep_refine(current)
            self.assertIs(memory.call_args.args[2], current.logits)
            self.assertIs(memory.call_args.args[3], current.hidden)
        torch.testing.assert_close(initial.logits, saved_logits, atol=0, rtol=0)
        torch.testing.assert_close(initial.hidden, saved_hidden, atol=0, rtol=0)
        self.assertGreater(current.hidden.abs().sum().item(), 0)
        self.assertEqual(self.model.spatial_memory.reset_state(initial.image_features).count_nonzero().item(), 0)

    @torch.no_grad()
    def test_gt_oracle_through_full_refine_and_following_keep(self):
        initial = self.make_state()
        target = torch.zeros_like(initial.probabilities)
        target[..., 20:220, 60:180] = 1
        intents = oracle_intents(initial.probabilities, target)
        with patch.object(self.model.edit_intent_decoder, "forward", return_value=intents):
            current = self.model.refine(initial, [[[300., 300.]]], [[1]])
        self.assertTrue(torch.equal(current.probabilities > .5, target.bool()))
        self.assertTrue(torch.equal(current.logits > 0, target.bool()))
        for _ in range(4):
            current = self.keep_refine(current)
            self.assertTrue(torch.equal(current.probabilities > .5, target.bool()))

    @torch.no_grad()
    def test_postprocess_matches_official_resize_and_crop_on_rectangular_images(self):
        from segment_anything.utils.transforms import ResizeLongestSide
        logits = torch.randn(1, 1, 256, 256, device=self.device)
        for shape in ((173, 281), (251, 137), (427, 640), (640, 427)):
            sizes = torch.tensor([shape], dtype=torch.float32, device=self.device)
            actual = self.model.sam_initializer.postprocess(logits, sizes)[0]
            resized = ResizeLongestSide.get_preprocess_shape(*shape, 1024)
            expected = F.interpolate(logits, (1024, 1024), mode="bilinear", align_corners=False)
            expected = F.interpolate(expected[..., :resized[0], :resized[1]], shape, mode="bilinear", align_corners=False)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
