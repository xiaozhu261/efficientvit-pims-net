"""Independent forward, geometry, and gradient tests; no SAM weights needed."""

import math
import unittest

import torch
from torch import nn

from pims_net.models import ClickEncoder, InitialMaskDecoder


class ClickEncoderTests(unittest.TestCase):
    def test_shapes_and_signed_gaussians(self):
        encoder = ClickEncoder(sigma=2)
        for batch_size in (1, 2):
            points = torch.tensor([[[128., 256.], [384., 64.]]]).repeat(batch_size, 1, 1)
            maps = encoder(points, torch.tensor([[1, 0]]).repeat(batch_size, 1))
            self.assertEqual(tuple(maps.shape), (batch_size, 2, 64, 64))
            self.assertEqual(maps.dtype, torch.float32)
            self.assertEqual(maps[0, 0, 32, 16].item(), 1.0)
            self.assertEqual(maps[0, 1, 8, 48].item(), 1.0)
            self.assertAlmostEqual(maps[0, 0, 32, 18].item(), math.exp(-0.5), places=6)
            self.assertTrue(((maps >= 0) & (maps <= 1)).all().item())

    def test_original_image_resize_and_padding_alignment(self):
        points = torch.tensor([[[128., 128.]], [[128., 128.]]])
        maps = ClickEncoder()(points, torch.ones(2, 1), torch.tensor([[256, 512], [512, 256]]))
        # A rectangular image is not stretched to fill the padded square.
        self.assertEqual(maps[0, 0].argmax().item(), 16 * 64 + 16)
        self.assertEqual(maps[1, 0].argmax().item(), 16 * 64 + 16)
        # Official rounding: (317, 529) -> (307, 512).
        maps = ClickEncoder()(torch.tensor([[[264.5, 158.5]]]), torch.ones(1, 1), (317, 529))
        expected = math.exp(-((19 - 19.1875) ** 2) / 8)
        self.assertAlmostEqual(maps[0, 0, 19, 32].item(), expected, places=6)

    def test_multiple_clicks_max_and_padding(self):
        encoder = ClickEncoder()
        points = torch.tensor([[[80., 80.], [160., 160.], [float('nan'), -1.]]])
        actual = encoder(points, torch.tensor([[1, 1, -1]]))
        first = encoder(points[:, :1], torch.tensor([[1]]))
        second = encoder(points[:, 1:2], torch.tensor([[1]]))
        torch.testing.assert_close(actual, torch.maximum(first, second))
        self.assertEqual(torch.count_nonzero(actual[:, 1]).item(), 0)
        self.assertEqual(torch.count_nonzero(encoder(points, torch.full((1, 3), -1))).item(), 0)
        self.assertEqual(tuple(encoder(torch.empty(2, 0, 2), torch.empty(2, 0)).shape), (2, 2, 64, 64))

    def test_invalid_clicks(self):
        encoder = ClickEncoder()
        with self.assertRaises(ValueError):
            ClickEncoder(sigma=0)
        for point, label in (([-1., 0.], 1), ([512., 0.], 0), ([0., 0.], 2)):
            with self.assertRaises(ValueError):
                encoder(torch.tensor([[point]]), torch.tensor([[label]]))
        with self.assertRaises(ValueError):
            encoder(torch.zeros(1, 1, 2), torch.ones(1, 1), (0, 512))


class InitialMaskDecoderTests(unittest.TestCase):
    def test_independent_forward_shapes(self):
        decoder = InitialMaskDecoder().eval()
        self.assertLess(sum(p.numel() for p in decoder.parameters()), 50000)
        with torch.no_grad():
            for batch_size in (1, 2):
                logits = decoder(torch.randn(batch_size, 256, 64, 64), torch.zeros(batch_size, 2, 64, 64))
                self.assertEqual(tuple(logits.shape), (batch_size, 1, 256, 256))
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_combined_forward_backward_and_click_influence(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            with self.subTest(device=device):
                torch.manual_seed(42)
                encoder = ClickEncoder().to(device)
                decoder = InitialMaskDecoder().to(device)
                features = torch.randn(2, 256, 64, 64, device=device)
                points = torch.tensor([[[128., 256.]], [[256., 128.]]], device=device)
                maps = encoder(points, torch.tensor([[1], [0]], device=device))
                logits = decoder(features, maps)
                self.assertEqual(tuple(logits.shape), (2, 1, 256, 256))
                self.assertEqual(logits.device, features.device)
                with torch.no_grad():
                    no_click_logits = decoder(features, torch.zeros_like(maps))
                self.assertGreater((logits - no_click_logits).abs().max().item(), 1e-6)
                nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
                for parameter in decoder.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(decoder.fusion[0].weight.grad[:, 256:].abs().sum().item(), 0)

    def test_returns_logits_without_sigmoid(self):
        decoder = InitialMaskDecoder()
        with torch.no_grad():
            decoder.decoder[-1].weight.zero_()
            decoder.decoder[-1].bias.fill_(-2)
            logits = decoder(torch.zeros(1, 256, 64, 64), torch.zeros(1, 2, 64, 64))
        torch.testing.assert_close(logits, torch.full_like(logits, -2))

    def test_rejects_shape_mismatch(self):
        decoder = InitialMaskDecoder()
        with self.assertRaises(ValueError):
            decoder(torch.zeros(1, 255, 64, 64), torch.zeros(1, 2, 64, 64))
        with self.assertRaises(ValueError):
            decoder(torch.zeros(1, 256, 64, 64), torch.zeros(2, 2, 64, 64))


if __name__ == "__main__":
    unittest.main()
