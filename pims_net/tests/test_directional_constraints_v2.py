import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from pims_net.evaluate_directional_constraints_v2 import signed_direction, gaussian_map, intervene
from pims_net.evaluate_oracle import clicked_error_component
from pims_net.models.click_encoder import ClickEncoder
from pims_net.models.directional_mask_editor import DirectionalMaskEditor


@dataclass
class State:
    probabilities: torch.Tensor
    logits: torch.Tensor
    direction: torch.Tensor
    strength: torch.Tensor
    labels: torch.Tensor


class RevisedConstraintTests(unittest.TestCase):
    def test_abs_direction_batch_preserves_magnitude(self):
        d=torch.tensor([[[[-.8,.4,0.]]],[[[-.8,.4,0.]]]])
        labels=torch.tensor([[0,1],[1,0]])
        result=signed_direction(d,labels)
        self.assertTrue(torch.equal(result.abs(),d.abs()))
        self.assertTrue((result[0]>=0).all() and (result[1]<=0).all())
        self.assertEqual(float(result[0,0,0,0]),float(d.abs()[0,0,0,0]))

    def test_C0_C1_C3_updates(self):
        editor=DirectionalMaskEditor()
        model=SimpleNamespace(directional_editor=editor)
        p=torch.full((1,1,2,2),.5)
        old=State(p,torch.logit(p),None,None,torch.tensor([[1]]))
        d=torch.tensor([[[[-.8,.4],[0.,-.2]]]])
        s=torch.full_like(p,.5)
        proposal=State(*editor.update(p,old.logits,d,s),d,s,torch.tensor([[1]]))
        self.assertIs(intervene(model,old,proposal,'C0'),proposal)
        g=torch.tensor([[[[1.,.5],[0.,.25]]]])
        for branch,weight in [('C1',s),('C3',s*g)]:
            result=intervene(model,old,proposal,branch,g)
            torch.testing.assert_close(result.probabilities,editor(p,d.abs(),weight))
            self.assertTrue((result.probabilities>=p).all())

    def test_component_selection_both_signs_and_empty(self):
        valid=np.ones((256,256),bool)
        truth=np.zeros_like(valid)
        truth[10:14,10:14]=1
        truth[30:32,30:32]=1
        old=np.zeros((256,256),np.float32)
        gate,_=clicked_error_component(old,truth,valid,dict(x=11,y=11,label=1),(256,256))
        self.assertEqual(int(gate.sum()),16)
        self.assertFalse(gate[30,30])
        old[10:14,10:14]=.9
        gate,_=clicked_error_component(old,truth,valid,dict(x=11,y=11,label=1),(256,256))
        self.assertFalse(gate.any())
        old[50:53,50:53]=.9
        gate,_=clicked_error_component(old,truth,valid,dict(x=51,y=51,label=0),(256,256))
        self.assertEqual(int(gate.sum()),9)

    def test_oracle_support_empty_and_correct_keep(self):
        p=torch.tensor([[[[.1,.9],[.8,.2]]]])
        old=State(p,torch.logit(p),None,None,torch.tensor([[1]]))
        gt=torch.tensor([[[[1.,0.],[1.,0.]]]])
        g=torch.full_like(p,.8)
        gate=torch.tensor([[[[1,0],[0,0]]]],dtype=torch.bool)
        result=intervene(None,old,old,'Oracle',g,gt,gate)
        torch.testing.assert_close(result.probabilities,p+g*gate*(gt-p))
        self.assertTrue(torch.equal(result.logits[~gate],old.logits[~gate]))
        empty=intervene(None,old,old,'Oracle',g,gt,torch.zeros_like(gate))
        self.assertTrue(torch.equal(empty.probabilities,p) and torch.equal(empty.logits,old.logits))

    def test_sigma_widens_external_gaussian(self):
        state=SimpleNamespace(points=torch.tensor([[[256.,256.]]]),labels=torch.tensor([[1]]),image_sizes=torch.tensor([[512,512]]))
        valid=torch.ones(1,1,256,256)
        maps=[gaussian_map(ClickEncoder(s),state,valid) for s in (2,4,8)]
        self.assertTrue((maps[0]<=maps[1]).all() and (maps[1]<=maps[2]).all())
        self.assertLess(float(maps[0].sum()),float(maps[1].sum()))
        self.assertLess(float(maps[1].sum()),float(maps[2].sum()))


if __name__=='__main__':
    unittest.main()
