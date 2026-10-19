"""Five-round state, fusion, gradient, and reset tests without SAM dependencies."""

import unittest

import torch

from pims_net.models import SpatialStateMemory


class SpatialStateMemoryTests(unittest.TestCase):
    def test_five_rounds_and_reset_cpu_and_cuda(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for batch_size in (1, 2):
                with self.subTest(device=device, batch_size=batch_size), torch.no_grad():
                    memory = SpatialStateMemory().to(device).eval()
                    features = torch.randn(batch_size, 256, 64, 64, device=device)
                    clicks = torch.rand(batch_size, 2, 64, 64, device=device)
                    logits = torch.randn(batch_size, 1, 256, 256, device=device)
                    state = memory.reset_state(features)
                    self.assertEqual(torch.count_nonzero(state).item(), 0)
                    for _ in range(5):
                        previous = state.clone()
                        next_state = memory(features, clicks, logits, state)
                        self.assertEqual(tuple(next_state.shape), (batch_size, 128, 64, 64))
                        self.assertTrue(torch.isfinite(next_state).all().item())
                        self.assertFalse(torch.equal(next_state, state))
                        torch.testing.assert_close(state, previous, rtol=0, atol=0)
                        state = next_state
                    fresh_state = memory.reset_state(features)
                    self.assertEqual(torch.count_nonzero(fresh_state).item(), 0)
                    torch.testing.assert_close(memory(features, clicks, logits, fresh_state), memory(features, clicks, logits))

    def test_all_inputs_and_early_state_receive_gradients(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        memory = SpatialStateMemory().to(device)
        features = torch.randn(1, 256, 64, 64, device=device, requires_grad=True)
        clicks = torch.rand(1, 2, 64, 64, device=device, requires_grad=True)
        logits = torch.randn(1, 1, 256, 256, device=device, requires_grad=True)
        first = memory(features, clicks, logits)
        first.retain_grad()
        state = first
        for _ in range(4):
            state = memory(features, clicks, logits, state)
        state.square().mean().backward()
        for tensor in (features, clicks, logits, first):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_explicit_states_isolate_interleaved_images(self):
        memory = SpatialStateMemory().eval()
        features_a, features_b = torch.randn(2, 1, 256, 64, 64)
        clicks = torch.rand(1, 2, 64, 64)
        logits = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            state_a = memory(features_a, clicks, logits)
            state_b = memory(features_b, clicks, logits)
            second_a = memory(features_a, clicks, logits, state_a)
            second_b = memory(features_b, clicks, logits, state_b)
            isolated_a = memory(features_a, clicks, logits, memory(features_a, clicks, logits))
            isolated_b = memory(features_b, clicks, logits, memory(features_b, clicks, logits))
        torch.testing.assert_close(second_a, isolated_a)
        torch.testing.assert_close(second_b, isolated_b)

    def test_rejects_wrong_state_or_input_shapes(self):
        memory = SpatialStateMemory()
        features = torch.zeros(1, 256, 64, 64)
        clicks = torch.zeros(1, 2, 64, 64)
        logits = torch.zeros(1, 1, 256, 256)
        with self.assertRaises(ValueError):
            memory(features, clicks, logits, torch.zeros(1, 32, 64, 64))
        with self.assertRaises(ValueError):
            memory(features, clicks, logits, torch.zeros(1, 128, 64, 64, dtype=torch.float64))
        with self.assertRaises(ValueError):
            memory(features, clicks, logits[:, :, :64, :64])
        with self.assertRaises(ValueError):
            memory(features, clicks[:, :1], logits)


if __name__ == "__main__":
    unittest.main()
