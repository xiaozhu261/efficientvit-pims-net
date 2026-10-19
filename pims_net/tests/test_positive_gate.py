"""Single-round contracts without loading or altering any existing checkpoint."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn

from pims_net.models.positive_correction_gate import PositiveCorrectionGate, positive_update, PositiveCorrectionPIMSNet
from pims_net.models.directional_pims_net import DirectionalState
from pims_net.models.click_encoder import ClickEncoder
from pims_net.training.positive_gate_losses import gate_target, positive_gate_losses


class PositiveGateTests(unittest.TestCase):
    def test_shape_and_gradient(self):
        torch.set_num_threads(2)
        gate = PositiveCorrectionGate()
        raw = gate(torch.randn(2, 256, 64, 64), torch.randn(2, 2, 64, 64),
                   torch.randn(2, 128, 64, 64), torch.rand(2, 1, 256, 256))
        self.assertEqual(raw.shape, (2, 1, 256, 256))
        raw.mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in gate.parameters()))

    def test_positive_formula_negative_exact(self):
        z = torch.randn(2, 1, 4, 4) * 10
        p = z.sigmoid()
        raw = torch.randn_like(z, requires_grad=True)
        new, logits, gate = positive_update(p, z, raw, torch.tensor([1, 0]))
        self.assertTrue(torch.equal(new[0], (p + raw.sigmoid() * (1-p))[0]))
        self.assertTrue((new >= p).all())
        self.assertTrue(torch.equal(new[1], p[1]) and torch.equal(logits[1], z[1]))
        new.sum().backward()
        self.assertEqual(int(torch.count_nonzero(raw.grad[1])), 0)

    def test_disabled_and_zero_gate_exact(self):
        z = torch.tensor([[[[-100., 0., 100.]]]])
        p = z.sigmoid()
        for raw, enabled in ((torch.randn_like(z), False), (torch.full_like(z, -float('inf')), True)):
            q, l, _ = positive_update(p, z, raw, torch.tensor([1]), enabled)
            self.assertTrue(torch.equal(q, p) and torch.equal(l, z))

    def test_saturated_positive_logits_never_decrease(self):
        z=torch.tensor([[[[-100.,-15.,0.,14.,15.,100.]]]])
        p=z.sigmoid()
        q,l,_=positive_update(p,z,torch.zeros_like(z),torch.tensor([1]))
        self.assertTrue((q>=p).all() and (l>=z).all())
        self.assertTrue(torch.isfinite(l).all())

    def test_residual_component_after_base_and_padding(self):
        target = torch.zeros(1, 1, 256, 256)
        target[:, :, 10:20, 10:20] = 1
        target[:, :, 30:40, 30:40] = 1
        valid = torch.ones_like(target)
        base = torch.zeros_like(target)
        selected, _ = gate_target(base, target, valid, dict(x=12,y=12,label=1), (256,256))
        self.assertEqual(int(selected.sum()), 100)
        base[:,:,12,12] = 1
        selected, _ = gate_target(base, target, valid, dict(x=12,y=12,label=1), (256,256))
        self.assertFalse(selected.any())
        selected, _ = gate_target(base, target, valid, dict(x=32,y=32,label=0), (256,256))
        self.assertFalse(selected.any())

    def test_balanced_loss_empty_and_gradient(self):
        raw = torch.zeros(1,1,2,2,requires_grad=True)
        base = torch.full_like(raw,.2)
        p,l,_ = positive_update(base,torch.logit(base),raw,torch.tensor([1]))
        state=SimpleNamespace(gate_logits=raw, probabilities=p, logits=l)
        target=torch.tensor([[[[1.,0.],[0.,0.]]]])
        selected=target.bool()
        valid=torch.ones_like(target)
        terms=positive_gate_losses(state,target,valid,selected,'G1')
        g=torch.autograd.grad(terms['loss'],raw,retain_graph=True)[0]
        self.assertAlmostEqual(float(g[0,0,0,0]),-.25)
        self.assertAlmostEqual(float(g[0,0,0,1]),1/12,places=6)
        empty=positive_gate_losses(state,target,valid,torch.zeros_like(selected),'G2')
        empty['loss'].backward()
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_base_freeze_train_override(self):
        base=nn.Sequential(nn.Linear(2,2),nn.Dropout())
        model=PositiveCorrectionPIMSNet(base).train()
        self.assertFalse(base.training)
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))
        self.assertTrue(model.gate.training)

    def test_frozen_transition_and_corrected_feedback(self):
        class FakeBase(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=nn.Parameter(torch.tensor(.1))
                self.click_encoder=ClickEncoder()
                self.inputs=[]

            def refine(self,state,points,labels):
                assert not torch.is_grad_enabled()
                self.inputs.append(state.probabilities.detach().clone())
                logits=.9*state.logits+self.weight
                return replace(state,points=torch.cat((state.points,torch.tensor(points,dtype=torch.float)),1),
                    labels=torch.cat((state.labels,torch.tensor(labels)),1),probabilities=logits.sigmoid(),
                    logits=logits,hidden=state.hidden+.1)

        base=FakeBase()
        model=PositiveCorrectionPIMSNet(base).train()
        state=DirectionalState(image_features=torch.randn(1,256,64,64),image_sizes=torch.tensor([[256,256]]),
            points=torch.tensor([[[100.,100.]]]),labels=torch.tensor([[1]]),logits=torch.zeros(1,1,256,256),
            hidden=torch.zeros(1,128,64,64),probabilities=torch.full((1,1,256,256),.5))
        model.eval()
        eval_state=model.refine(state,[[[110.,110.]]],[[1]])
        model.train()
        train_state=model.refine(state,[[[110.,110.]]],[[1]])
        self.assertTrue(torch.equal(train_state.probabilities,eval_state.probabilities))
        self.assertTrue(torch.equal(train_state.logits,eval_state.logits))
        self.assertTrue(torch.equal(train_state.hidden,eval_state.hidden))
        base.inputs.clear()
        weight=base.weight.detach().clone()
        outputs=[]
        intermediate=[]
        for label in (1,0,1):
            state=model.refine(state,[[[110.,110.]]],[[label]])
            self.assertFalse(state.hidden.requires_grad or state.base_probabilities.requires_grad)
            outputs.append(state.probabilities.detach().clone())
            state.gate_logits.retain_grad()
            intermediate.append(state.gate_logits)
        self.assertTrue(torch.equal(base.inputs[1],outputs[0]))
        self.assertTrue(torch.equal(base.inputs[2],outputs[1]))
        state.gate_logits.mean().backward()
        self.assertIsNone(intermediate[0].grad)
        self.assertIsNone(intermediate[1].grad)
        self.assertIsNotNone(intermediate[2].grad)
        self.assertIsNone(base.weight.grad)
        self.assertTrue(torch.equal(base.weight,weight))
        self.assertTrue(all(p.grad is not None for p in model.gate.parameters()))


if __name__ == '__main__':
    unittest.main()
