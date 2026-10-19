"""Analytical checks against the production loss; no model/optimizer updates."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from pims_net.evaluation.edit_loss_audit import click_regions, labels_from_old, local_statistics, combine_local, pair_statistics
from pims_net.models.mask_editor import MaskEditor
from pims_net.training.interactive_losses import LossWeights, interactive_losses


def evaluate(z, old, truth, valid=None, weights=LossWeights()):
    q = z.softmax(1)
    p = MaskEditor()(old, q)
    state = SimpleNamespace(probabilities=p, logits=torch.logit(p, eps=1e-6), edit_probabilities=q)
    return interactive_losses(state, old, truth, torch.ones_like(old) if valid is None else valid, weights)


class EditLossAuditTests(unittest.TestCase):
    def test_actual_labels_channels_and_049_boundary(self):
        old = torch.tensor([[[[.1, .9, .9, .1, .49, .495, .495]]]], dtype=torch.float64)
        truth = torch.tensor([[[[1., 0., 1., 0., 1., 1., 0.]]]], dtype=torch.float64)
        expected = torch.tensor([[[0, 1, 2, 2, 0, 2, 1]]])
        z = torch.randn(1, 3, 1, 7, dtype=torch.float64, requires_grad=True)
        loss = evaluate(z, old, truth)['edit']
        nll = F.cross_entropy(z, expected, reduction='none')
        manual = torch.stack([nll[expected == i].mean() for i in range(3)]).mean()
        torch.testing.assert_close(loss, manual)
        np.testing.assert_array_equal(labels_from_old(old[0, 0].numpy(), truth[0, 0].numpy().astype(bool)), expected[0].numpy())
        alternative = labels_from_old(old[0, 0].numpy(), truth[0, 0].numpy().astype(bool), .5)
        self.assertEqual(int((alternative != expected[0].numpy()).sum()), 2)

    def test_class_balancing_exact_logit_gradient_and_padding(self):
        old = torch.tensor([[[[.1, .9, .1, .9, .1]]]], dtype=torch.float64, requires_grad=True)
        truth = torch.tensor([[[[1., 0., 0., 1., 1.]]]], dtype=torch.float64)
        valid = torch.tensor([[[[1., 1., 1., 1., 0.]]]], dtype=torch.float64)
        z = torch.randn(1, 3, 1, 5, dtype=torch.float64, requires_grad=True)
        loss = evaluate(z, old, truth, valid)['edit']
        actual, old_grad = torch.autograd.grad(loss, (z, old), allow_unused=True)
        onehot = F.one_hot(torch.tensor([[[0, 1, 2, 2, 0]]]), 3).permute(0, 3, 1, 2)
        expected = (z.softmax(1) - onehot) * torch.tensor([1/3, 1/3, 1/6, 1/6, 0], dtype=torch.float64)
        torch.testing.assert_close(actual, expected)
        self.assertIsNone(old_grad)  # Edit target is detached, not a soft teacher.

    def test_absent_classes_are_excluded_from_divisor(self):
        old = torch.zeros(1, 1, 1, 3, dtype=torch.float64)
        z = torch.randn(1, 3, 1, 3, dtype=torch.float64, requires_grad=True)
        loss = evaluate(z, old, torch.ones_like(old))['edit']
        torch.testing.assert_close(loss, -z.log_softmax(1)[:, 0].mean())

    def test_weighted_total_gradient_and_finite_difference(self):
        old = torch.tensor([[[[.2, .8, .1, .9]]]], dtype=torch.float64)
        truth = torch.tensor([[[[1., 0., 0., 1.]]]], dtype=torch.float64)
        z = torch.tensor([[[[.2, .4, .5, .3]], [[.4, .1, .3, .8]], [[.1, .9, .7, .2]]]], dtype=torch.float64, requires_grad=True)
        weights = LossWeights(bce=.7, dice=1.3, edit=2., preserve=.2)
        terms = evaluate(z, old, truth, weights=weights)
        total = torch.autograd.grad(terms['loss'], z, retain_graph=True)[0]
        separate = sum(getattr(weights, name) * torch.autograd.grad(terms[name], z, retain_graph=True)[0]
                       for name in ('bce', 'dice', 'edit', 'preserve'))
        torch.testing.assert_close(total, separate)
        self.assertTrue(torch.autograd.gradcheck(lambda raw: evaluate(raw, old, truth, weights=weights)['loss'], (z,)))

    def test_keep_can_conflict_with_mask_confidence_sharpening(self):
        old = torch.tensor([[[[.8]]]], dtype=torch.float64)
        z = torch.tensor([[[[.6]], [[.1]], [[.3]]]], dtype=torch.float64).log().requires_grad_()
        terms = evaluate(z, old, torch.ones_like(old))
        mask = torch.autograd.grad(terms['bce'] + terms['dice'], z, retain_graph=True)[0]
        edit = torch.autograd.grad(terms['edit'], z)[0]
        self.assertLess(float((mask * edit).sum()), 0)

    def test_click_disks_geometry_and_history(self):
        regions = click_regions([dict(x=128, y=64, label=1), dict(x=256, y=128, label=0)], (256, 512))
        self.assertTrue(regions['center_negative'][64, 128])
        self.assertTrue(regions['history_positive'][32, 64])
        self.assertFalse(regions['latest_positive'].any())
        self.assertEqual(int(regions['latest_negative'].sum()), 197)

    def test_confusion_rows_and_gradient_pooled_aggregation(self):
        q = np.array([[[.8, .1, .6]], [[.1, .8, .1]], [[.1, .1, .3]]])
        labels = np.array([[0, 1, 2]])
        grad = np.ones_like(q)
        row = local_statistics(q, labels, labels, np.ones((1, 3)), np.ones((1, 3), bool), np.ones((1, 3), bool), grad, -grad)
        np.testing.assert_array_equal(row['confusion'], [[1, 0, 0], [0, 1, 0], [1, 0, 0]])
        combined = combine_local([row, row])
        self.assertEqual(combined['pixels'], 6)
        self.assertEqual(combined['gradient_conflict_fraction'], 1)
        self.assertAlmostEqual(combined['pooled_gradient_cosine'], -1)
        self.assertAlmostEqual(pair_statistics(grad, -grad)['cosine'], -1)


if __name__ == '__main__':
    unittest.main()
