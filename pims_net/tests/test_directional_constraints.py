import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from pims_net.evaluate_directional_constraints import intervene, latest_gaussian, transition_counts, transition_rates
from pims_net.models.click_encoder import ClickEncoder
from pims_net.models.directional_mask_editor import DirectionalMaskEditor


@dataclass
class State:
    probabilities: torch.Tensor
    logits: torch.Tensor
    direction: torch.Tensor
    strength: torch.Tensor
    labels: torch.Tensor


class ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.editor = DirectionalMaskEditor()
        self.model = SimpleNamespace(directional_editor=self.editor)
        p = torch.tensor([[[[.2,.8],[.4,.6]]]])
        self.old = State(p,torch.logit(p),None,None,torch.tensor([[1]]))
        self.d = torch.tensor([[[[1.,-1.],[-.5,.5]]]])
        self.s = torch.full_like(p,.5)
        q,logits = self.editor.update(p,self.old.logits,self.d,self.s)
        self.proposal = State(q,logits,self.d,self.s,torch.tensor([[1]]))
        self.g = torch.tensor([[[[1.,.5],[0.,.25]]]])

    def test_C0_identity_and_click_signs(self):
        self.assertIs(intervene(self.model,self.old,self.proposal,'C0',self.g),self.proposal)
        for sign in (0,1):
            self.proposal.labels=torch.tensor([[sign]])
            for branch in ('C1','C3'):
                new=intervene(self.model,self.old,self.proposal,branch,self.g)
                delta=new.probabilities-self.old.probabilities
                self.assertTrue((delta>=0).all() if sign else (delta<=0).all())
                suppressed=self.d<0 if sign else self.d>0
                self.assertTrue(torch.equal(new.logits[suppressed],self.old.logits[suppressed]))

    def test_gaussian_scaling_and_exact_keep(self):
        new=intervene(self.model,self.old,self.proposal,'C2',self.g)
        expected=self.editor(self.old.probabilities,self.d,self.s*self.g)
        self.assertTrue(torch.equal(new.probabilities,expected))
        self.assertTrue(torch.equal(new.logits[self.g==0],self.old.logits[self.g==0]))
        # A tiny nonzero Gaussian at saturated P must not cause logit-only edits.
        self.old.probabilities=torch.ones_like(self.old.probabilities)
        self.old.logits=torch.full_like(self.old.logits,100.)
        new=intervene(self.model,self.old,self.proposal,'C2',torch.full_like(self.g,1e-30))
        self.assertTrue(torch.equal(new.logits,self.old.logits))

    def test_oracle_gt_distance_and_no_damage(self):
        gt=torch.tensor([[[[1.,0.],[0.,1.]]]])
        new=intervene(self.model,self.old,self.proposal,'local_gt_oracle',self.g,gt)
        torch.testing.assert_close(new.probabilities,self.old.probabilities+self.g*(gt-self.old.probabilities))
        self.assertTrue(((new.probabilities-gt).abs()<=(self.old.probabilities-gt).abs()).all())
        correct=(self.old.probabilities>.5)==gt.bool()
        self.assertTrue((((new.probabilities>.5)==gt.bool()) | ~correct).all())

    def test_gaussian_latest_only_and_padding(self):
        model=SimpleNamespace(click_encoder=ClickEncoder())
        state=SimpleNamespace(points=torch.tensor([[[40.,60.],[200.,100.]]]),labels=torch.tensor([[1,0]]),image_sizes=torch.tensor([[256,512]]))
        valid=torch.zeros(1,1,256,256)
        valid[:,:,:128]=1
        g=latest_gaussian(model,state,valid)
        self.assertEqual(g.shape,(1,1,256,256))
        self.assertTrue((g[:,:,128:]==0).all())
        y,x=np.unravel_index(int(g.argmax()),(256,256))
        self.assertLessEqual(abs(x-100),2)
        self.assertLessEqual(abs(y-50),2)
        self.assertEqual(model.click_encoder.sigma,2)

    def test_TP_denominator_excludes_TN(self):
        old=np.array([1,1,0,1,0,0],bool)
        gt=np.array([1,1,1,0,0,0],bool)
        new=np.array([0,1,1,0,1,0],bool)
        counts=transition_counts(old,new,gt)
        self.assertEqual(counts['tp_before'],2)
        self.assertEqual(counts['tp_damaged'],1)
        self.assertEqual(transition_rates(counts),dict(fn_repair_rate=1.,fp_removal_rate=1.,tp_damage_rate=.5,tn_damage_rate=.5))


if __name__=='__main__':
    unittest.main()
