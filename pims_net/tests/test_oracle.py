from dataclasses import dataclass, replace
import unittest

import numpy as np
import torch

from pims_net.evaluate_oracle import clicked_error_component, oracle_update


@dataclass(frozen=True)
class State:
    probabilities: torch.Tensor
    logits: torch.Tensor
    edit_probabilities: torch.Tensor | None = None
    hidden: int = 0


class OracleTests(unittest.TestCase):
    def setUp(self):
        logits = torch.tensor([[[[-100., 100., -1., 1.]]]])
        self.previous = State(logits.sigmoid(), logits)
        q = torch.tensor([[[[.3, .1, .4, .2]], [[.2, .5, .1, .1]], [[.5, .4, .5, .7]]]])
        p = q[:, 2:3] * self.previous.probabilities + q[:, :1]
        self.proposal = State(p, torch.logit(p, eps=1e-6), q, hidden=1)
        self.target = torch.tensor([[[[1., 0., 1., 0.]]]])
        self.valid = torch.tensor([[[[1., 1., 1., 0.]]]])

    def test_b1_preserves_total_strength_and_uses_gt_direction(self):
        state = oracle_update(self.previous, self.proposal, 'b1', self.target, self.valid)
        m = self.proposal.edit_probabilities[:, :2].sum(1, keepdim=True)
        expected = self.previous.probabilities + m * (self.target - self.previous.probabilities)
        torch.testing.assert_close(state.probabilities[self.valid.bool()], expected[self.valid.bool()], atol=0, rtol=0)
        self.assertTrue(torch.equal(state.logits[~self.valid.bool()], self.previous.logits[~self.valid.bool()]))
        self.assertTrue(torch.equal(state.probabilities[~self.valid.bool()], self.previous.probabilities[~self.valid.bool()]))
        self.assertTrue(((state.probabilities - self.target).abs() <= (self.previous.probabilities - self.target).abs()).all())
        self.assertEqual(state.hidden, 1)

    def test_b1_zero_strength_preserves_raw_extreme_logits(self):
        q = torch.zeros_like(self.proposal.edit_probabilities)
        q[:, 2] = 1
        proposal = replace(self.proposal, edit_probabilities=q)
        state = oracle_update(self.previous, proposal, 'b1', self.target, torch.ones_like(self.valid))
        self.assertTrue(torch.equal(state.logits, self.previous.logits))
        self.assertTrue(torch.equal(state.probabilities, self.previous.probabilities))

    def test_b2_preserves_prediction_inside_and_previous_outside(self):
        gate = torch.tensor([[[[True, False, True, True]]]])
        actual_gate = gate & self.valid.bool()
        state = oracle_update(self.previous, self.proposal, 'b2', self.target, self.valid, gate)
        for key in ('probabilities', 'logits'):
            self.assertTrue(torch.equal(getattr(state, key)[actual_gate], getattr(self.proposal, key)[actual_gate]))
            self.assertTrue(torch.equal(getattr(state, key)[~actual_gate], getattr(self.previous, key)[~actual_gate]))
        self.assertIs(state.edit_probabilities, self.proposal.edit_probabilities)
        self.assertEqual(state.hidden, 1)

    def test_empty_b2_still_advances_history_and_b3_is_exact(self):
        state = self.previous
        for i in range(4):
            proposal = replace(self.proposal, hidden=state.hidden + 1)
            state = oracle_update(state, proposal, 'b2', self.target, self.valid, torch.zeros_like(self.valid))
        self.assertEqual(state.hidden, 4)
        self.assertTrue(torch.equal(state.logits, self.previous.logits))
        self.assertTrue(torch.equal(state.probabilities, self.previous.probabilities))
        self.assertIs(oracle_update(self.previous, self.proposal, 'b3', self.target, self.valid), self.proposal)

    def test_component_connectivity_sign_and_resolved_click(self):
        old, truth, valid = np.zeros((256, 256), np.float32), np.zeros((256, 256), bool), np.ones((256, 256), bool)
        truth[10, 10] = truth[11, 11] = truth[30, 30] = True
        click = dict(x=20, y=20, label=1)
        gate, info = clicked_error_component(old, truth, valid, click, (512, 512))
        self.assertEqual(info['error_components'], 2)
        self.assertEqual(int(gate.sum()), 2)  # Diagonal neighbors: 8-connected.
        self.assertFalse(gate[30, 30])
        negative, _ = clicked_error_component(old, truth, valid, dict(click, label=0), (512, 512))
        self.assertFalse(negative.any())
        old[10, 10] = old[11, 11] = .6
        gate, info = clicked_error_component(old, truth, valid, click, (512, 512))
        self.assertFalse(gate.any())
        self.assertFalse(info['selected_error'])

    def test_negative_component_and_padding_exclusion(self):
        old, truth, valid = np.zeros((256, 256), np.float32), np.zeros((256, 256), bool), np.zeros((256, 256), bool)
        valid[:128] = True
        old[30:33, 40:43] = .7
        old[140, 40] = .7
        gate, info = clicked_error_component(old, truth, valid, dict(x=80, y=60, label=0), (256, 512))
        self.assertEqual(info['error_components'], 1)
        self.assertEqual(int(gate.sum()), 9)
        self.assertFalse(gate[140, 40])


if __name__ == '__main__':
    unittest.main()
