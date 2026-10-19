"""Real-weight first-click parity, memory semantics, and independent trainable heads."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pims_net.evaluation.methods import EVSAMSession, create_sessions
from pims_net.models import DirectMaskDecoder, PIMSNet
from pims_net.training.interactive_losses import interactive_losses, prepare_target


class DirectDecoderTests(unittest.TestCase):
    def test_batched_logits_are_complete_masks_not_residuals(self):
        decoder = DirectMaskDecoder()
        with torch.no_grad():
            decoder.head.weight.zero_()
            decoder.head.bias.fill_(-2)
            output = decoder(torch.randn(2, 256, 64, 64), torch.randn(2, 128, 64, 64),
                             torch.rand(2, 1, 256, 256))
        self.assertEqual(tuple(output.shape), (2, 1, 256, 256))
        torch.testing.assert_close(output, torch.full_like(output, -2), atol=0, rtol=0)


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.models = {variant: PIMSNet(variant=variant).to(cls.device) for variant in ("a1", "a2", "a3")}
        cls.image = np.random.default_rng(42).integers(0, 256, (96, 144, 3), dtype=np.uint8)
        cls.features, cls.sizes = cls.models["a3"].sam_initializer.encode(cls.image)

    def test_all_methods_share_frozen_first_click(self):
        sessions = create_sessions(self.models, self.features, self.sizes)
        results = [session.add_click(50, 40, 1) for session in sessions.values()]
        for result in results[1:]:
            np.testing.assert_allclose(result, results[0], rtol=0, atol=0)
        for model in self.models.values():
            self.assertTrue(all(not p.requires_grad for p in model.sam_initializer.parameters()))

    def test_a0_cumulative_points_and_mask_feedback_match_official_predictor(self):
        from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
        initializer = self.models["a3"].sam_initializer
        session = EVSAMSession(initializer, self.features, self.sizes)
        predictor = EfficientViTSamPredictor(initializer.sam)
        predictor.set_image(self.image)
        points, labels, previous = [], [], None
        for x, y, label in ((50, 40, 1), (15, 20, 0), (80, 70, 1)):
            points.append([x, y])
            labels.append(label)
            actual = session.add_click(x, y, label)
            masks, scores, low = predictor.predict(point_coords=np.array(points), point_labels=np.array(labels),
                                                   mask_input=previous, multimask_output=True, return_logits=True)
            best = scores.argmax()
            previous = low[best:best + 1]
            expected = torch.from_numpy(masks[best]).sigmoid().numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_history_reset_in_both_training_and_inference(self):
        for variant, model in self.models.items():
            for training in (True, False):
                with self.subTest(variant=variant, training=training), torch.no_grad():
                    model.train(training)
                    state = model.initial_state(self.features, self.sizes, [[[50, 40]]], [[1]])
                    state = model.refine(state, [[[15, 20]]], [[0]])
                    self.assertGreater(state.hidden.abs().sum().item(), 0)
                    changed = replace(state, hidden=state.hidden + 1)
                    with patch.object(model.spatial_memory, "forward", wraps=model.spatial_memory.forward) as memory:
                        normal = model.refine(state, [[[80, 70]]], [[1]])
                        history = memory.call_args.args[3]
                        if variant == "a2":
                            self.assertEqual(history.count_nonzero().item(), 0)
                        else:
                            self.assertIs(history, state.hidden)
                    perturbed = model.refine(changed, [[[80, 70]]], [[1]])
                    if variant == "a2":
                        torch.testing.assert_close(normal.logits, perturbed.logits, atol=0, rtol=0)
                    else:
                        self.assertFalse(torch.equal(normal.hidden, perturbed.hidden))

    def test_each_variant_trains_its_own_head_and_memory(self):
        target_mask = np.zeros(self.image.shape[:2], bool)
        target_mask[20:70, 30:110] = True
        target, valid = prepare_target(target_mask, self.device)
        for variant, model in self.models.items():
            with self.subTest(variant=variant):
                model.train()
                model.zero_grad(set_to_none=True)
                initial = model.initial_state(self.features, self.sizes, [[[50, 40]]], [[1]])
                state = model.refine(initial, [[[15, 20]]], [[0]])
                self.assertEqual(tuple(state.logits.shape), (1, 1, 256, 256))
                losses = interactive_losses(state, initial.probabilities, target, valid)
                losses["loss"].backward()
                head = model.direct_mask_decoder if variant == "a1" else model.edit_intent_decoder
                self.assertGreater(head.head.weight.grad.abs().sum().item(), 0)
                self.assertGreater(sum(p.grad.abs().sum().item() for p in model.spatial_memory.parameters()), 0)
                self.assertTrue(all(p.grad is None for p in model.sam_initializer.parameters()))
                if variant == "a1":
                    self.assertIsNone(state.edit_probabilities)
                    self.assertEqual(losses["edit"].item(), 0)
                    torch.testing.assert_close(state.probabilities, state.logits.sigmoid())
                else:
                    self.assertGreater(losses["edit"].item(), 0)


if __name__ == "__main__":
    unittest.main()
