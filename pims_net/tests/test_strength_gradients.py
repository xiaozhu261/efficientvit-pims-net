import unittest

import torch
from torch.nn import functional as F

from pims_net.diagnose_strength_gradients import gradients,vector_summary,regional_strength_gradients
from pims_net.models.directional_mask_editor import DirectionalMaskEditor


class StrengthGradientTests(unittest.TestCase):
    def test_small_sigmoid_derivative_does_not_imply_bce_gradient_vanishes(self):
        z=torch.tensor([[[[-12.]]]],requires_grad=True)
        s=z.sigmoid()
        old=torch.zeros_like(s)
        d=torch.ones_like(s)
        p,logits=DirectionalMaskEditor().update(old,torch.full_like(s,-100.),d,s)
        loss=F.binary_cross_entropy_with_logits(logits,torch.ones_like(s))
        gs,gz=gradients(loss,[s,z])
        self.assertLess(float(s*(1-s)),1e-4)
        self.assertLess(float(gz),-.99)
        torch.testing.assert_close(gz,gs*s*(1-s))

    def test_logit_clipping_blocks_bce_but_not_dice_probability_path(self):
        z=torch.tensor([[[[-20.]]]],requires_grad=True)
        s=z.sigmoid()
        p,logits=DirectionalMaskEditor().update(torch.zeros_like(s),torch.full_like(s,-100.),torch.ones_like(s),s)
        bce=F.binary_cross_entropy_with_logits(logits,torch.ones_like(s))
        gs,gz=gradients(bce,[s,z])
        self.assertEqual(float(gs),0.)
        self.assertEqual(float(gz),0.)
        dice=1-(2*p.sum()+1)/(p.sum()+2)
        self.assertLess(float(gradients(dice,[z])[0]),0.)

    def test_direction_no_direct_current_strength_path_but_recurrent_head_gradient(self):
        weight=torch.tensor(-2.,requires_grad=True)
        previous=weight.sigmoid()
        direction=(previous*3).tanh()
        current_s=(weight*2).sigmoid()
        loss=(direction-1).square()+current_s.sum()*0
        gs,gw=gradients(loss,[current_s,weight])
        self.assertEqual(float(gs),0.)
        self.assertNotEqual(float(gw),0.)

    def test_regional_strength_vjp_additivity_and_row_selection(self):
        head=torch.nn.Conv2d(3,2,1)
        features=torch.randn(1,3,2,2)
        raw=head(features)
        a=torch.tensor([[[[True,False],[False,True]]]])
        b=~a
        g=torch.randn(1,1,2,2)
        groups=regional_strength_gradients(raw,list(head.parameters()),g,dict(a=a,b=b,all=torch.ones_like(a)))
        for ga,gb,gall in zip(groups['a'],groups['b'],groups['all']):
            torch.testing.assert_close(ga+gb,gall)
            self.assertTrue(torch.all(gall[0]==0))

    def test_projection_identity_and_undefined_zero_cosine(self):
        a=torch.tensor([1.,2.])
        b=torch.tensor([-2.,1.])
        vectors=dict(a=a,b=b,zero=a*0,total=a+b)
        contrib,pairs=vector_summary(vectors)
        self.assertAlmostEqual(sum(r['projection_on_total'] for r in contrib if r['loss']!='total'),1.)
        self.assertTrue(all(r['cosine'] is None for r in pairs if r['right']=='zero'))


if __name__=='__main__':
    unittest.main()
