from types import SimpleNamespace
import unittest

import torch

from pims_net.training.directional_losses import directional_losses
from pims_net.training.strength_supervised_losses import strength_supervised_losses,harmful_strength_region
from pims_net.models.directional_mask_editor import DirectionalMaskEditor
from pims_net.overfit_strength_supervision import vector_report


class StrengthSupervisionTests(unittest.TestCase):
    def test_gradient_norm_projection_and_zero_norm_handling(self):
        report=vector_report(dict(mask=torch.tensor([1.,2.]),activate=torch.tensor([-2.,1.]),inactive=torch.zeros(2)))
        self.assertAlmostEqual(report['norms']['mask'],5**.5)
        self.assertAlmostEqual(report['pairs']['mask__activate']['cosine'],0.)
        self.assertIsNone(report['pairs']['mask__inactive']['cosine'])
        self.assertAlmostEqual(sum(v for k,v in report['projections'].items() if k!='total'),1.)

    def inputs(self, bias=-10.):
        raw=torch.full((1,2,2,2),bias,requires_grad=True)
        old=torch.tensor([[[[.1,.9],[.7,.2]]]],requires_grad=True)
        d=raw[:,:1].tanh()
        s=raw[:,1:2].sigmoid()
        p,logits=DirectionalMaskEditor().update(old,old.detach().logit(),d,s)
        state=SimpleNamespace(direction=d,strength=s,probabilities=p,logits=logits,edit_probabilities=None)
        gt=torch.tensor([[[[1.,0.],[1.,0.]]]])
        valid=torch.tensor([[[[True,True],[True,False]]]])
        component=torch.tensor([[[[True,False],[False,False]]]])
        return state,old,gt,valid,component,raw

    def test_s0_exact_original_loss_and_gradients(self):
        s,old,gt,v,c,raw=self.inputs()
        a=strength_supervised_losses(s,old,gt,v,c,1,raw,'S0')
        b=directional_losses(s,old,gt,v,threshold=.5)
        self.assertTrue(torch.equal(a['loss'],b['loss']))
        ga=torch.autograd.grad(a['loss'],raw,retain_graph=True)[0]
        gb=torch.autograd.grad(b['loss'],raw)[0]
        self.assertTrue(torch.equal(ga,gb))

    def test_activation_uses_raw_logits_and_keeps_gradient_when_s_is_tiny(self):
        s,old,gt,v,c,raw=self.inputs(-30.)
        loss=strength_supervised_losses(s,old,gt,v,c,1,raw,'S1')
        g=torch.autograd.grad(.05*loss['strength_activate'],raw)[0]
        self.assertAlmostEqual(float(loss['strength_activate']),30.,places=5)
        self.assertAlmostEqual(float(g[:,1:2][c]),-.05,places=6)
        self.assertTrue(torch.all(g[:,:1]==0))
        self.assertTrue(torch.all(g[:,1:2][~c]==0))

    def test_s2_only_harmful_correct_pixels_and_excludes_unrepaired_errors(self):
        s,old,gt,v,c,raw=self.inputs(30.)
        v=torch.ones_like(v)
        s.direction=torch.tensor([[[[1.,-1.],[-1.,1.]]]],requires_grad=True)
        terms=strength_supervised_losses(s,old,gt,v,c,1,raw,'S2')
        gate=harmful_strength_region(old,gt,v,s.direction)
        self.assertEqual(gate.tolist(),[[[[False,False],[True,True]]]])
        self.assertFalse(gate.requires_grad)
        g=torch.autograd.grad(.01*terms['strength_suppress'],raw,retain_graph=True)[0][:,1:2]
        self.assertTrue(torch.all(g[~gate]==0))
        torch.testing.assert_close(g[gate],torch.full_like(g[gate],.005))
        base=directional_losses(s,old,gt,v,threshold=.5)
        torch.testing.assert_close(terms['loss'],base['loss']+.05*terms['strength_activate']+.01*terms['strength_suppress'])

    def test_harmful_filter_excludes_padding_safe_directions_and_zero(self):
        s,old,gt,v,c,raw=self.inputs()
        safe=torch.tensor([[[[-1.,1.],[1.,-1.]]]])
        self.assertFalse(harmful_strength_region(old,gt,v,safe).any())
        harmful=torch.tensor([[[[-1.,1.],[-1.,1.]]]])
        self.assertEqual(int(harmful_strength_region(old,gt,v,harmful).sum()),1)
        self.assertFalse(harmful_strength_region(old,gt,v,torch.zeros_like(harmful)).any())

    def test_negative_click_skips_activation_but_s2_suppresses_non_target(self):
        s,old,gt,v,c,raw=self.inputs()
        for name in ('S1','S2'):
            loss=strength_supervised_losses(s,old,gt,v,c,0,raw,name)
            self.assertEqual(float(loss['strength_activate']),0.)
            self.assertEqual(float(loss['strength_suppress'])>0,name=='S2')

    def test_empty_target_skips_activation_and_only_suppresses_harmful_correct(self):
        s,old,gt,v,c,raw=self.inputs()
        loss=strength_supervised_losses(s,old,gt,v,torch.zeros_like(c),1,raw,'S2')
        self.assertEqual(float(loss['strength_activate']),0.)
        g=torch.autograd.grad(loss['strength_suppress'],raw)[0][:,1:2]
        mask=harmful_strength_region(old,gt,v,s.direction)
        self.assertTrue(torch.all(g[mask]>0))
        self.assertTrue(torch.all(g[~mask]==0))


if __name__=='__main__':
    unittest.main()
