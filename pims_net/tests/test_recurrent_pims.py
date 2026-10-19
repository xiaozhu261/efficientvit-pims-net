"""Recurrent component tests and real-weight first-click/training/inference checks."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from pims_net.inference import PIMSPredictor
from pims_net.models import ConvGRU, EditDecoder, PIMSNet
from pims_net.training.step import training_step


class RecurrentUnitTests(unittest.TestCase):
    def test_gru_shapes_state_and_gradients(self):
        for batch_size in (1, 2):
            gru = ConvGRU(input_channels=7, hidden_channels=8)
            inputs = torch.randn(batch_size, 7, 64, 64, requires_grad=True)
            first = gru(inputs)
            second = gru(inputs, first)
            self.assertEqual(tuple(second.shape), (batch_size, 8, 64, 64))
            self.assertFalse(torch.equal(first, second))
            second.square().mean().backward()
            self.assertTrue(torch.isfinite(inputs.grad).all().item())
            self.assertGreater(gru.gates.weight.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            gru(inputs, torch.zeros(2, 7, 64, 64))

    def test_edit_decoder_is_residual(self):
        decoder = EditDecoder(8)
        hidden = torch.randn(2, 8, 64, 64)
        previous = torch.randn(2, 1, 256, 256)
        self.assertEqual(tuple(decoder(hidden, previous).shape), (2, 1, 256, 256))
        with torch.no_grad():
            decoder.layers[-1].weight.zero_()
            decoder.layers[-1].bias.zero_()
        torch.testing.assert_close(decoder(hidden, previous), previous, rtol=0, atol=0)


class PIMSIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(5)
        cls.model = PIMSNet().to(cls.device)
        rng = np.random.default_rng(5)
        cls.images = [rng.integers(0, 256, size=shape, dtype=np.uint8)
                      for shape in ((173, 281, 3), (251, 137, 3))]
        cls.points = torch.tensor([[[80., 60.], [120., 100.], [30., 40.]],
                                   [[60., 120.], [80., 150.], [20., 80.]]])
        cls.labels = torch.tensor([[1, 0, 1], [1, 1, 0]])

    def setUp(self):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

    def test_first_click_matches_official_predictor(self):
        from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

        predictor = EfficientViTSamPredictor(self.model.sam_initializer.sam)
        with patch.object(self.model.spatial_memory, "forward", wraps=self.model.spatial_memory.forward) as gru:
            state = self.model.initialize(self.images, self.points[:, :1], self.labels[:, :1])
            self.assertEqual(gru.call_count, 0)
        self.assertFalse(state.logits.requires_grad)
        self.assertEqual(torch.count_nonzero(state.hidden).item(), 0)
        for index, image in enumerate(self.images):
            # Compare the same batch size: CUDA kernels can produce different
            # floating-point results for B=1 and B=2, including in official SAM.
            single = self.model.initialize(image, self.points[index:index + 1, :1], self.labels[index:index + 1, :1])
            predictor.set_image(image)
            _, scores, low_res = predictor.predict(
                point_coords=self.points[index, :1].numpy(),
                point_labels=self.labels[index, :1].numpy(), multimask_output=True, return_logits=True
            )
            expected = torch.from_numpy(low_res[scores.argmax()]).to(self.device)
            torch.testing.assert_close(single.logits[0, 0], expected, rtol=1e-4, atol=1e-4)

    def test_batched_first_click_matches_official_forward(self):
        from segment_anything.utils.transforms import ResizeLongestSide

        sam = self.model.sam_initializer.sam
        records = []
        for index, image in enumerate(self.images):
            coords = ResizeLongestSide(1024).apply_coords(self.points[index, :1].numpy(), image.shape[:2])
            records.append({
                "image": sam.transform(image).to(self.device),
                "point_coords": torch.tensor(coords, dtype=torch.float32, device=self.device).unsqueeze(0),
                "point_labels": self.labels[index:index + 1, :1].to(self.device),
            })
        with torch.no_grad():
            masks, scores = sam(records, multimask_output=True)
            state = self.model.initialize(self.images, self.points[:, :1], self.labels[:, :1])
        for index in range(2):
            expected = masks[index, 0, scores[index, 0].argmax()]
            torch.testing.assert_close(state.logits[index, 0], expected, rtol=1e-4, atol=1e-4)

    def test_sequence_uses_sam_only_on_first_click(self):
        sam = self.model.sam_initializer.sam
        with patch.object(sam.image_encoder, "forward", wraps=sam.image_encoder.forward) as image_encoder, \
             patch.object(sam.mask_decoder, "forward", wraps=sam.mask_decoder.forward) as mask_decoder, \
             patch.object(self.model.spatial_memory, "forward", wraps=self.model.spatial_memory.forward) as gru:
            with torch.no_grad():
                result = self.model(self.images, self.points, self.labels)
            self.assertEqual(tuple(result.shape), (2, 3, 1, 256, 256))
            self.assertTrue(torch.isfinite(result).all().item())
            self.assertEqual(image_encoder.call_count, 1)
            self.assertEqual(mask_decoder.call_count, 2)  # One call per image, first click only.
            self.assertEqual(gru.call_count, 2)

    def test_training_step_updates_only_research_modules(self):
        sam = self.model.sam_initializer.sam
        before_buffers = {name: value.clone() for name, value in sam.named_buffers()}
        before_weight = next(sam.parameters()).detach().clone()
        before_edit = self.model.edit_intent_decoder.head.weight.detach().clone()
        optimizer = torch.optim.Adam(self.model.trainable_parameters(), lr=1e-3)
        result = training_step(self.model, self.images, self.points, self.labels, torch.zeros(2, 1, 256, 256))
        self.assertTrue(result["loss"].requires_grad)
        self.assertTrue(torch.isfinite(result["loss"]).item())
        result["loss"].backward()
        self.assertTrue(all(not m.training for m in sam.modules()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in sam.parameters()))
        for component in (self.model.spatial_memory.feature_projection, self.model.spatial_memory.conv_gru, self.model.edit_intent_decoder):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in component.parameters()))
            self.assertGreater(sum(p.grad.abs().sum().item() for p in component.parameters()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before_edit, self.model.edit_intent_decoder.head.weight))
        torch.testing.assert_close(next(sam.parameters()), before_weight, rtol=0, atol=0)
        for name, value in sam.named_buffers():
            torch.testing.assert_close(value, before_buffers[name], rtol=0, atol=0)

    def test_recurrent_graph_reaches_previous_state(self):
        state1 = self.model.initialize(self.images[:1], self.points[:1, :1], self.labels[:1, :1])
        state2 = self.model.refine(state1, self.points[:1, 1:2], self.labels[:1, 1:2])
        state2.hidden.retain_grad()
        state2.logits.retain_grad()
        state3 = self.model.refine(state2, self.points[:1, 2:3], self.labels[:1, 2:3])
        state3.logits.square().mean().backward()
        self.assertGreater(state2.hidden.grad.abs().sum().item(), 0)
        self.assertGreater(state2.logits.grad.abs().sum().item(), 0)
        self.assertEqual(state1.points.shape[1], 1)  # Older state is not mutated.
        self.assertEqual(state3.points.shape[1], 3)

    def test_cached_inference_matches_forward_and_resets(self):
        predictor = PIMSPredictor(self.model)
        with self.assertRaises(RuntimeError):
            predictor.add_click(1, 1)
        sam = self.model.sam_initializer.sam
        with patch.object(sam.image_encoder, "forward", wraps=sam.image_encoder.forward) as encode:
            predictor.set_image(self.images[0])
            outputs = []
            for index in range(3):
                x, y = self.points[0, index].tolist()
                result = predictor.add_click(x, y, self.labels[0, index].item())
                self.assertEqual(result["click_count"], index + 1)
                self.assertEqual(tuple(result["logits"].shape), (1, 1, 173, 281))
                self.assertEqual(result["mask"].dtype, torch.bool)
                self.assertFalse(result["low_res_logits"].requires_grad)
                outputs.append(result["low_res_logits"])
            self.assertEqual(encode.call_count, 1)
        with torch.no_grad():
            expected = self.model(self.images[:1], self.points[:1], self.labels[:1])
        torch.testing.assert_close(torch.stack(outputs, dim=1), expected)
        predictor.set_image(self.images[1])
        self.assertIsNone(predictor.state)
        result = predictor.add_click(60, 120)
        self.assertEqual(result["click_count"], 1)
        predictor.reset_image()
        self.assertIsNone(predictor.features)

    def test_tensor_input_and_bad_sequences(self):
        with torch.no_grad():
            batch = torch.stack([self.model.sam_initializer.sam.transform(image) for image in self.images])
            sizes = [image.shape[:2] for image in self.images]
            actual = self.model(batch, self.points[:, :1], self.labels[:, :1], sizes)
            expected = self.model(self.images, self.points[:, :1], self.labels[:, :1])
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            self.model(batch, self.points[:, :1], self.labels[:, :1])
        with self.assertRaises(ValueError):
            training_step(self.model, self.images, self.points[:, :1], self.labels[:, :1], torch.zeros(2, 1, 256, 256))
        with self.assertRaises(ValueError):
            self.model(self.images, self.points[:, :0], self.labels[:, :0])
        with self.assertRaises(ValueError):
            self.model(self.images, self.points[:, :1], torch.full((2, 1), -1))

    def test_refinement_uses_intent_update_formula(self):
        with torch.no_grad():
            first = self.model.initialize(self.images, self.points[:, :1], self.labels[:, :1])
            second = self.model.refine(first, self.points[:, 1:2], self.labels[:, 1:2])
        self.assertIsNone(first.edit_probabilities)
        self.assertEqual(tuple(second.edit_probabilities.shape), (2, 3, 256, 256))
        add, remove, keep = second.edit_probabilities.split(1, dim=1)
        torch.testing.assert_close(add + remove + keep, torch.ones_like(add))
        torch.testing.assert_close(second.probabilities, keep * first.probabilities + add, rtol=0, atol=0)
        torch.testing.assert_close(second.logits.sigmoid(), second.probabilities.clamp(1e-6, 1 - 1e-6))

    def test_five_interactions_and_new_image_memory_reset(self):
        predictor = PIMSPredictor(self.model)
        predictor.set_image(self.images[0])
        first_image_states = []
        for index, (x, y, label) in enumerate(((80, 60, 1), (120, 100, 0), (30, 40, 1), (90, 70, 0), (50, 30, 1))):
            result = predictor.add_click(x, y, label)
            hidden = predictor.state.hidden
            self.assertEqual(tuple(hidden.shape), (1, 128, 64, 64))
            self.assertTrue(torch.isfinite(hidden).all().item())
            self.assertEqual(result["click_count"], index + 1)
            self.assertEqual(tuple(result["low_res_logits"].shape), (1, 1, 256, 256))
            torch.testing.assert_close(result["low_res_probabilities"], predictor.state.probabilities)
            if index == 0:
                self.assertIsNone(result["edit_probabilities"])
                self.assertEqual(torch.count_nonzero(hidden).item(), 0)
            else:
                self.assertEqual(tuple(result["edit_probabilities"].shape), (1, 3, 256, 256))
                self.assertGreater((hidden - first_image_states[-1]).abs().sum().item(), 0)
            first_image_states.append(hidden.clone())
        predictor.set_image(self.images[1])
        self.assertIsNone(predictor.state)
        first = predictor.add_click(60, 120, 1)
        self.assertEqual(torch.count_nonzero(predictor.state.hidden).item(), 0)
        self.assertEqual(first["click_count"], 1)
        second = predictor.add_click(80, 150, 0)
        fresh = PIMSPredictor(self.model)
        fresh.set_image(self.images[1])
        fresh.add_click(60, 120, 1)
        expected = fresh.add_click(80, 150, 0)
        torch.testing.assert_close(second["low_res_logits"], expected["low_res_logits"])
        torch.testing.assert_close(predictor.state.hidden, fresh.state.hidden)


if __name__ == "__main__":
    unittest.main()
