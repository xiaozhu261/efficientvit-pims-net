"""Integration tests using real local official weights; no mocked encoder."""

import unittest

import numpy as np
import torch
from torch import nn

from pims_net.models.baseline import build_baseline
from pims_net.models.encoder_adapter import EncoderAdapter


class EncoderAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.adapter = EncoderAdapter().to(cls.device)
        cls.reference = build_baseline("efficientvit-sam-l0").to(cls.device).eval()
        rng = np.random.default_rng(42)
        cls.images = [
            rng.integers(0, 256, size=(317, 529, 3), dtype=np.uint8),
            rng.integers(0, 256, size=(713, 281, 3), dtype=np.uint8),
        ]

    def test_retains_complete_pretrained_encoder_only(self):
        self.assertEqual(set(dict(self.adapter.named_children())), {"image_encoder"})
        self.assertEqual(set(dict(self.adapter.image_encoder.named_children())), {"backbone", "neck", "norm"})
        actual = self.adapter.image_encoder.state_dict()
        expected = self.reference.image_encoder.state_dict()
        self.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertTrue(all(key.startswith("image_encoder.") for key in self.adapter.state_dict()))

    def test_official_preprocessing(self):
        # Cover downsize, upsize, and the official already-long-side-512 branch.
        images = self.images + [self.images[0][:97, :83], np.zeros((512, 129, 3), dtype=np.uint8)]
        expected = torch.stack([self.reference.transform(image) for image in images])
        actual = self.adapter.preprocess(images)
        self.assertEqual(tuple(actual.shape), (4, 3, 512, 512))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Padding happens after normalization and therefore remains exactly zero.
        self.assertTrue(torch.all(actual[0, :, -1, :] == 0).item())
        self.assertTrue(torch.all(actual[1, :, :, -1] == 0).item())

    def test_single_and_batched_output_shapes(self):
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                output = self.adapter(self.images[:batch_size])
                self.assertEqual(tuple(output.shape), (batch_size, 256, 64, 64))
                self.assertEqual(output.device, next(self.adapter.parameters()).device)
                self.assertTrue(torch.isfinite(output).all().item())
                self.assertFalse(output.requires_grad)

    def test_matches_official_encoder(self):
        image = self.images[0]
        with torch.no_grad():
            expected = self.reference.image_encoder(self.reference.transform(image).unsqueeze(0).to(self.device))
        torch.testing.assert_close(self.adapter(image), expected, rtol=1e-5, atol=1e-5)

    def test_frozen_under_parent_train_and_head_backward(self):
        parent = nn.Sequential(self.adapter)
        self.assertIs(parent.train()[0], self.adapter)
        self.assertFalse(self.adapter.training)
        self.assertTrue(all(not module.training for module in self.adapter.modules()))
        self.assertTrue(all(not parameter.requires_grad for parameter in self.adapter.parameters()))
        before = {name: value.clone() for name, value in self.adapter.named_buffers()}
        batch = self.adapter.preprocess(self.images[:1]).requires_grad_()
        output = self.adapter(batch)
        self.assertEqual(tuple(output.shape), (1, 256, 64, 64))
        self.assertFalse(output.requires_grad)
        # no_grad features must still be usable by a trainable research head.
        head = nn.Conv2d(256, 1, 1).to(self.device)
        head(output).square().mean().backward()
        self.assertIsNotNone(head.weight.grad)
        self.assertIsNone(batch.grad)
        self.assertTrue(all(parameter.grad is None for parameter in self.adapter.parameters()))
        for name, value in self.adapter.named_buffers():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_rejects_ambiguous_or_invalid_inputs(self):
        cases = [
            ([], ValueError),
            (np.zeros((10, 10, 3), dtype=np.float32), TypeError),
            (np.zeros((10, 10), dtype=np.uint8), ValueError),
            (np.zeros((0, 10, 3), dtype=np.uint8), ValueError),
            (torch.zeros(1, 3, 256, 256), ValueError),
            (torch.zeros(1, 3, 512, 512, dtype=torch.uint8), TypeError),
        ]
        for value, error in cases:
            with self.subTest(error=error, input_type=type(value)):
                with self.assertRaises(error):
                    self.adapter(value)


if __name__ == "__main__":
    unittest.main()
