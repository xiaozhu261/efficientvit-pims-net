from types import SimpleNamespace
import unittest

import numpy as np
import torch

from pims_net.overfit_directional import binary_for_clicker, describe, distribution, gradient_pair
from pims_net.training.click_simulator import RITMClickSimulator
from pims_net.training.directional_losses import direction_target, directional_losses
from pims_net.training.interactive_losses import interactive_losses


class OverfitThresholdTests(unittest.TestCase):
    def test_half_threshold_labels_and_preserve_consistent(self):
        old = torch.tensor([[[[.495, .495, .5, .5, .5001, .5001]]]])
        gt = torch.tensor([[[[1., 0., 1., 0., 1., 0.]]]])
        wanted = torch.tensor([[[[1., 0., 1., 0., 0., -1.]]]])
        torch.testing.assert_close(direction_target(old, gt, .5), wanted)
        p = torch.tensor([[[[.8, .4, .7, .2, .9, .3]]]])
        state = SimpleNamespace(probabilities=p, logits=torch.logit(p), direction=torch.zeros_like(p), edit_probabilities=None)
        losses = directional_losses(state, old, gt, torch.ones_like(gt), threshold=.5)
        torch.testing.assert_close(losses['preserve'], (p-old).abs()[wanted == 0].mean())

    def test_binary_adapter_enforces_half_before_original_clicker(self):
        p = np.array([[.495, .5, .5001]], np.float32)
        np.testing.assert_array_equal(binary_for_clicker(p), [[False, False, True]])
        clicker = RITMClickSimulator(np.array([[True, False, True]]), np.random.default_rng(42))
        self.assertEqual(clicker.next_click(binary_for_clicker(p)), (0, 0, 1))

    def test_default_directional_mask_terms_unchanged(self):
        old, p, gt = torch.rand(1, 1, 3, 4), torch.rand(1, 1, 3, 4), torch.randint(2, (1, 1, 3, 4)).float()
        state = SimpleNamespace(probabilities=p, logits=torch.logit(p), direction=torch.zeros_like(p), edit_probabilities=None)
        original = interactive_losses(state, old, gt, torch.ones_like(gt))
        actual = directional_losses(state, old, gt, torch.ones_like(gt))
        for key in ('bce', 'dice', 'preserve'):
            torch.testing.assert_close(actual[key], original[key], atol=0, rtol=0)

    def test_distributions_and_gradient_cosine(self):
        old = np.array([[.5, .6, .4, .7]])
        gt = np.array([[True, False, False, True]])
        valid = np.array([[True, True, True, False]])
        result = distribution(np.array([[0., -.5, .5, 1.]]), np.array([[.1, .2, .3, .4]]), old, gt, valid)
        self.assertEqual(result['all']['direction']['count'], 3)
        self.assertEqual(sum(result['all']['direction']['histogram']), 3)
        self.assertEqual(result['FN']['direction']['count'], 1)
        self.assertIsNone(result['TP']['direction'])
        g = torch.tensor([1., 2.])
        self.assertAlmostEqual(gradient_pair([g], [-g])['cosine'], -1.)
        self.assertIsNone(gradient_pair([g], [torch.zeros_like(g)])['cosine'])


if __name__ == '__main__':
    unittest.main()
