from dataclasses import dataclass, replace
import unittest

import numpy as np
import torch

from pims_net.evaluate_eta import band_statistics, refine_with_eta


@dataclass(frozen=True)
class State:
    probabilities: torch.Tensor
    logits: torch.Tensor
    hidden: int = 0


class Refiner:
    def refine(self, state, points, labels):
        self.input = state
        p = .2 * state.probabilities + .4
        self.proposal = replace(state, probabilities=p, logits=torch.logit(p, eps=1e-6), hidden=state.hidden + 1)
        return self.proposal


class EtaTests(unittest.TestCase):
    def test_zero_preserves_extreme_logits_while_history_advances(self):
        logits = torch.tensor([-100., 100.])
        initial = State(logits.sigmoid(), logits)
        state, model = initial, Refiner()
        for _ in range(4):
            state = refine_with_eta(model, state, None, None, 0)
            self.assertTrue(torch.equal(state.logits, initial.logits))
            self.assertTrue(torch.equal(state.probabilities, initial.probabilities))
        self.assertEqual(state.hidden, 4)

    def test_one_returns_exact_original_proposal(self):
        initial = State(torch.tensor([.1, .9]), torch.tensor([-2., 2.]))
        model = Refiner()
        self.assertIs(refine_with_eta(model, initial, None, None, 1), model.proposal)

    def test_intermediate_scales_and_feeds_back_its_own_mask(self):
        initial = State(torch.tensor([.1, .9]), torch.tensor([-2., 2.]))
        for eta in (.1, .25, .5):
            model = Refiner()
            first = refine_with_eta(model, initial, None, None, eta)
            expected = initial.probabilities + eta * (.2 * initial.probabilities + .4 - initial.probabilities)
            torch.testing.assert_close(first.probabilities, expected, rtol=0, atol=0)
            second = refine_with_eta(model, first, None, None, eta)
            self.assertIs(model.input, first)
            torch.testing.assert_close(second.probabilities, first.probabilities + eta * (.2 * first.probabilities + .4 - first.probabilities), rtol=0, atol=0)
            self.assertTrue(((second.probabilities >= 0) & (second.probabilities <= 1)).all())

    def test_invalid_eta_rejected(self):
        for eta in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                refine_with_eta(Refiner(), None, None, None, eta)

    def test_band_excludes_padding_and_separates_foreground(self):
        p = np.array([[.49, .51, .9, .5]], np.float32)
        target = np.array([[1, 1, 1, 0]], bool)
        valid = np.array([[1, 1, 1, 0]], bool)
        result = band_statistics(p, target, valid)
        self.assertEqual(result['band_fg'], 2 / 3)
        self.assertEqual(result['band_all'], 2 / 3)
        self.assertIsNone(result['band_bg'])


if __name__ == '__main__':
    unittest.main()
