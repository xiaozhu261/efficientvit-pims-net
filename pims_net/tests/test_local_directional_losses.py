from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from pims_net.training.directional_losses import directional_losses
from pims_net.training.local_directional_losses import local_directional_losses
from pims_net.overfit_directional_losses import loss_gradients, paired


class LocalLossTests(unittest.TestCase):
    def data(self):
        logits=torch.tensor([[[[-1., 1.], [2., -2.]]]], requires_grad=True)
        direction=torch.tensor([[[[.2, -.3], [.4, -.5]]]], requires_grad=True)
        state=SimpleNamespace(logits=logits, probabilities=logits.sigmoid(), direction=direction, edit_probabilities=None)
        old=torch.tensor([[[[.1, .9], [.8, .2]]]], requires_grad=True)
        target=torch.tensor([[[[1., 0.], [1., 0.]]]])
        valid=torch.tensor([[[[True, True], [True, False]]]])
        component=torch.tensor([[[[True, False], [False, False]]]])
        return state,old,target,valid,component

    def test_l0_exact_original(self):
        s,p,t,v,c=self.data()
        a=local_directional_losses(s,p,t,v,c,1,'L0')
        b=directional_losses(s,p,t,v,threshold=.5)
        for key in a:
            self.assertTrue(torch.equal(a[key],b[key]))

    def test_mask_local_preserve_outside_and_padding_excluded(self):
        s,p,t,v,c=self.data()
        losses=local_directional_losses(s,p,t,v,c,1,'L1')
        self.assertAlmostEqual(float(losses['bce']),float(F.softplus(torch.tensor(1.))),places=6)
        mask_grad=torch.autograd.grad(losses['bce']+losses['dice'],s.logits,retain_graph=True)[0]
        preserve_grad=torch.autograd.grad(losses['preserve'],s.logits)[0]
        self.assertTrue(torch.all(mask_grad[~c]==0))
        self.assertLess(float(mask_grad[c][0]),0)
        self.assertTrue(torch.all(preserve_grad[c | ~v]==0))
        self.assertTrue(torch.all(preserve_grad[v & ~c]!=0))
        self.assertEqual(float(losses['direction']),0.)

    def test_local_direction_sign_and_support(self):
        for label in (0,1):
            s,p,t,v,c=self.data()
            loss=local_directional_losses(s,p,t,v,c,label,'L2')
            g=torch.autograd.grad(loss['direction'],s.direction)[0]
            self.assertTrue(torch.all(g[~c]==0))
            self.assertEqual(bool(g[c][0]<0),bool(label))
            base=local_directional_losses(s,p,t,v,c,label,'L1')
            torch.testing.assert_close(loss['loss'],base['loss']+loss['direction'])

    def test_empty_component_only_preserve_and_detached_reference(self):
        s,p,t,v,c=self.data()
        loss=local_directional_losses(s,p,t,v,torch.zeros_like(c),1,'L2')
        for key in ('bce','dice','direction'):
            self.assertEqual(float(loss[key]),0.)
        torch.testing.assert_close(loss['loss'],.1*loss['preserve'])
        loss['loss'].backward()
        self.assertIsNone(p.grad)
        self.assertTrue(torch.isfinite(s.logits.grad).all())

    def test_inactive_and_zero_gradient_are_not_conflict_observations(self):
        raw=torch.tensor([1.,2.],requires_grad=True)
        objectives=dict(mask=raw.sum(),direction=raw.sum()*0,preserve=-raw.sum())
        pairs=paired(loss_gradients(objectives,[raw],False))
        self.assertIsNone(pairs['mask_direction'])
        self.assertAlmostEqual(pairs['mask_preserve']['cosine'],-1.)
        zero=paired(loss_gradients(objectives,[raw],True))
        self.assertIsNone(zero['mask_direction']['cosine'])


if __name__=='__main__':
    unittest.main()
