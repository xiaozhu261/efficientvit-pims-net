"""Calibration algebra and same-history contracts; no checkpoint edits."""

from dataclasses import dataclass
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from pims_net.evaluate_gate_calibration import calibrated_update,calibrated_refine,contribution,ratio
from pims_net.models.positive_correction_gate import positive_update


class GateCalibrationTests(unittest.TestCase):
    def test_zero_alpha_all_thresholds_exact(self):
        z=torch.tensor([[[[-100.,-2.,0.,14.,100.]]]])
        for tau in (0.,.3,.5,.7):
            p,l,g=calibrated_update(z.sigmoid(),z,torch.rand_like(z),torch.tensor([1]),0.,tau)
            self.assertTrue(torch.equal(p,z.sigmoid()) and torch.equal(l,z) and not g.any())

    def test_original_setting_matches_G2(self):
        z=torch.randn(2,1,12,12)*15
        raw=torch.randn_like(z)
        labels=torch.tensor([1,0])
        actual=calibrated_update(z.sigmoid(),z,raw.sigmoid(),labels,1.,0.)
        expected=positive_update(z.sigmoid(),z,raw,labels)
        for a,b in zip(actual,expected): self.assertTrue(torch.equal(a,b))

    def test_threshold_before_alpha_and_inclusive(self):
        p=torch.full((1,1,1,4),.2)
        raw=torch.tensor([[[[.2999,.3,.5,.9]]]])
        q,l,g=calibrated_update(p,torch.logit(p),raw,torch.tensor([1]),.1,.3)
        torch.testing.assert_close(g,torch.tensor([[[[0.,.03,.05,.09]]]]))
        self.assertTrue(torch.equal(q,p+g*(1-p)))

    def test_negative_exact_and_monotone_endpoints(self):
        z=torch.randn(2,1,3,3)*20
        for a in (0.,.1,.25,.5,.75,1.):
            for t in (0.,.3,.5,.7):
                q,l,g=calibrated_update(z.sigmoid(),z,torch.rand_like(z),torch.tensor([1,0]),a,t)
                self.assertTrue((q>=z.sigmoid()).all() and (l>=z).all())
                self.assertTrue(torch.equal(q[1],z.sigmoid()[1]) and torch.equal(l[1],z[1]))

    def test_one_shared_S0_transition(self):
        @dataclass
        class Proposal:
            base_probabilities:torch.Tensor
            base_logits:torch.Tensor
            probabilities:torch.Tensor
            logits:torch.Tensor
            gate:torch.Tensor
            labels:torch.Tensor
            hidden:int
        class FakeModel:
            calls=0
            def refine(self,previous,points,labels):
                self.calls+=1
                base=torch.full((1,1,1,1),.2)
                return Proposal(base,torch.logit(base),torch.full_like(base,.6),torch.logit(torch.full_like(base,.6)),
                    torch.full_like(base,.5),torch.tensor(labels),previous.hidden+1)
        model=FakeModel()
        state=calibrated_refine(model,SimpleNamespace(hidden=3),[[[1,2]]],[[1]],.5,0.)
        self.assertEqual(model.calls,1)
        self.assertEqual(state.hidden,4)
        self.assertAlmostEqual(float(state.probabilities),.4)

    def test_counts_ratio_and_delta_identity(self):
        old=np.array([[1,0,0,0]],bool)
        gt=np.array([[1,1,0,0]],bool)
        new=np.array([[1,1,1,0]],bool)
        row=contribution(old,new,gt)
        self.assertEqual((row['K'],row['E']),(1,1))
        self.assertAlmostEqual(row['delta_iou'],1/6)
        self.assertEqual(ratio(1,0)['ratio_status'],'infinite')
        self.assertIsNone(ratio(0,0)['K_over_E'])
        self.assertEqual(ratio(0,0)['ratio_status'],'undefined')

    def test_invalid_calibration_rejected(self):
        p=torch.zeros(1,1,1,1)
        for a,t in ((-1.,0.),(1.1,0.),(1.,float('nan'))):
            with self.assertRaises(ValueError): calibrated_update(p,p,p,torch.tensor([1]),a,t)


if __name__=='__main__': unittest.main()
