"""Intent normalization, exact editing semantics, shapes, and gradient tests."""

import unittest

import torch

from pims_net.models import EditIntentDecoder, MaskEditor


class EditIntentDecoderTests(unittest.TestCase):
    def test_shapes_normalization_and_all_input_gradients(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for batch_size in (1, 2):
                with self.subTest(device=device, batch_size=batch_size):
                    decoder = EditIntentDecoder().to(device)
                    features = torch.randn(batch_size, 256, 64, 64, device=device, requires_grad=True)
                    hidden = torch.randn(batch_size, 128, 64, 64, device=device, requires_grad=True)
                    previous = torch.rand(batch_size, 1, 256, 256, device=device, requires_grad=True)
                    intents = decoder(features, hidden, previous)
                    self.assertEqual(tuple(intents.shape), (batch_size, 3, 256, 256))
                    self.assertTrue(((intents >= 0) & (intents <= 1)).all().item())
                    torch.testing.assert_close(intents.sum(1), torch.ones_like(intents[:, 0]))
                    # Decoder-only loss verifies all three inputs affect intent,
                    # rather than testing only the editor's direct mask path.
                    (-intents[:, 0].log().mean()).backward()
                    for value in (features, hidden, previous):
                        self.assertTrue(torch.isfinite(value.grad).all().item())
                        self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_invalid_mask_and_hidden(self):
        decoder = EditIntentDecoder()
        features = torch.zeros(1, 256, 64, 64)
        with self.assertRaises(ValueError):
            decoder(features, torch.zeros(1, 32, 64, 64), torch.zeros(1, 1, 256, 256))
        with self.assertRaises(ValueError):
            decoder(features, torch.zeros(1, 128, 64, 64), torch.full((1, 1, 256, 256), -2.))


class MaskEditorTests(unittest.TestCase):
    def test_pure_add_remove_keep_including_endpoints(self):
        previous = torch.tensor([[[[0., .2], [.8, 1.]]]])
        for channel, expected in ((0, torch.ones_like(previous)), (1, torch.zeros_like(previous)), (2, previous)):
            intents = torch.zeros(1, 3, 2, 2)
            intents[:, channel] = 1
            actual = MaskEditor()(previous, intents)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_mixed_formula_and_valid_range(self):
        previous = torch.rand(2, 1, 256, 256)
        intents = torch.randn(2, 3, 256, 256).softmax(1)
        actual = MaskEditor()(previous, intents)
        torch.testing.assert_close(actual, intents[:, 2:3] * previous + intents[:, :1], rtol=0, atol=0)
        self.assertTrue(((actual >= 0) & (actual <= 1)).all().item())
        self.assertEqual(tuple(actual.shape), (2, 1, 256, 256))

    def test_gradients_through_all_three_intents_and_previous_mask(self):
        previous = torch.full((1, 1, 2, 2), .25, requires_grad=True)
        intent_logits = torch.zeros(1, 3, 2, 2, requires_grad=True)
        output = MaskEditor()(previous, intent_logits.softmax(1))
        output.sum().backward()
        torch.testing.assert_close(previous.grad, torch.full_like(previous, 1 / 3))
        for channel in range(3):
            self.assertGreater(intent_logits.grad[:, channel].abs().sum().item(), 0)
        self.assertLess(intent_logits.grad[:, 1].max().item(), 0)  # REMOVE decreases foreground.

    def test_rejects_nonprobabilities_and_shape_errors(self):
        editor = MaskEditor()
        previous = torch.full((1, 1, 2, 2), .5)
        for intents in (torch.zeros(1, 3, 2, 2), torch.ones(1, 3, 2, 2),
                        torch.full((1, 3, 2, 2), float("nan")), torch.zeros(1, 2, 2, 2)):
            with self.assertRaises(ValueError):
                editor(previous, intents)
        with self.assertRaises(ValueError):
            editor(torch.full_like(previous, -1), torch.full((1, 3, 2, 2), 1 / 3))
        with self.assertRaises(TypeError):
            editor(previous.to(torch.long), torch.full((1, 3, 2, 2), 1 / 3))


if __name__ == "__main__":
    unittest.main()
