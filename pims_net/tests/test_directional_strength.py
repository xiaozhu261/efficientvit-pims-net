import unittest
from types import SimpleNamespace
from dataclasses import dataclass

import numpy as np
import torch

from pims_net.evaluate_directional_strength import edit_counts, rates, refine_scaled
from pims_net.models.directional_mask_editor import DirectionalMaskEditor


@dataclass
class FakeState:
    probabilities: torch.Tensor
    logits: torch.Tensor
    direction: torch.Tensor
    strength: torch.Tensor


class StrengthSweepTests(unittest.TestCase):
    def test_scale_clip_sign_and_keep(self):
        p = torch.full((1, 1, 1, 3), .4)
        d = torch.tensor([[[[1., -1., 0.]]]])
        s = torch.full_like(p, .2)
        previous = FakeState(p, torch.logit(p), d, s)
        editor = DirectionalMaskEditor()
        q, logits = editor.update(p, previous.logits, d, s)
        proposal = FakeState(q, logits, d, s)
        model = SimpleNamespace(refine=lambda *args: proposal, directional_editor=editor)
        unchanged, effective = refine_scaled(model, previous, None, None, 1)
        self.assertIs(unchanged, proposal)
        for factor in (2, 4, 8):
            result, effective = refine_scaled(model, previous, None, None, factor)
            strength = min(.2 * factor, 1.)
            torch.testing.assert_close(result.probabilities, torch.tensor([[[[.4 + .6 * strength, .4 * (1-strength), .4]]]]))
            self.assertTrue(torch.equal(result.strength, s))
            self.assertEqual(result.logits[0, 0, 0, 2], previous.logits[0, 0, 0, 2])
            self.assertLessEqual(float(effective.max()), 1)

    def test_repair_damage_denominators(self):
        gt = np.array([1, 1, 0, 0, 1, 0], bool)
        old = np.array([0, 0, 1, 1, 1, 0], bool)
        new = np.array([1, 0, 0, 1, 0, 0], bool)
        counts = edit_counts(old, new, gt)
        self.assertEqual(counts, dict(fn_before=2, fn_repaired=1, fp_before=2, fp_removed=1, correct_before=2, correct_damaged=1))
        self.assertEqual(rates(counts), dict(fn_repair_rate=.5, fp_removal_rate=.5, damage_rate=.5))
        self.assertIsNone(rates(edit_counts(gt, gt, gt))['fn_repair_rate'])


if __name__ == '__main__':
    unittest.main()
