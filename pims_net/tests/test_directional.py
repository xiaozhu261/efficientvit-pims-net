"""Signed editor, supervision, frozen first click, BPTT, and checkpoint isolation."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from uuid import uuid4

import numpy as np
import torch

from pims_net.models.directional_mask_editor import DirectionalMaskEditor
from pims_net.models.directional_pims_net import DirectionalEditDecoder, DirectionalPIMSNet
from pims_net.models.pims_net import PIMSNet
from pims_net.paths import OUTPUT_ROOT
from pims_net.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from pims_net.training.directional_losses import DirectionalLossWeights, direction_target, directional_losses
from pims_net.training.directional_rollout import directional_rollout
from pims_net.training.interactive_losses import prepare_target


class SignedEditorTests(unittest.TestCase):
    def test_zero_direction_exact_probability_and_extreme_logits_keep(self):
        old_logits = torch.tensor([[[[-100., -20., 0., 20., 100.]]]])
        old = old_logits.sigmoid()
        p, logits = DirectionalMaskEditor().update(old, old_logits, torch.zeros_like(old), torch.rand_like(old))
        self.assertTrue(torch.equal(p, old))
        self.assertTrue(torch.equal(logits, old_logits))

    def test_zero_strength_keep(self):
        old_logits = torch.tensor([[[[-100., 100.]]]])
        old = old_logits.sigmoid()
        p, logits = DirectionalMaskEditor().update(old, old_logits, torch.ones_like(old), torch.zeros_like(old))
        self.assertTrue(torch.equal(p, old))
        self.assertTrue(torch.equal(logits, old_logits))

    def test_batched_shape_sign_bounds_and_full_strength_endpoints(self):
        old, d, s = torch.rand(2, 1, 256, 256), torch.rand(2, 1, 256, 256) * 2 - 1, torch.rand(2, 1, 256, 256)
        editor = DirectionalMaskEditor()
        p = editor(old, d, s)
        self.assertEqual(tuple(p.shape), (2, 1, 256, 256))
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertTrue((p[d > 0] >= old[d > 0]).all())
        self.assertTrue((p[d < 0] <= old[d < 0]).all())
        torch.testing.assert_close(editor(old, torch.ones_like(d), torch.ones_like(s)), torch.ones_like(old))
        torch.testing.assert_close(editor(old, -torch.ones_like(d), torch.ones_like(s)), torch.zeros_like(old))

    def test_update_gradient_finite_difference(self):
        old = torch.tensor([[[[.2, .8]]]], dtype=torch.float64, requires_grad=True)
        d = torch.tensor([[[[.4, -.6]]]], dtype=torch.float64, requires_grad=True)
        s = torch.tensor([[[[.3, .7]]]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(DirectionalMaskEditor(), (old, d, s)))

    def test_invalid_direction_and_shape_rejected(self):
        old = torch.full((1, 1, 2, 2), .5)
        for d in (torch.full_like(old, 1.1), torch.full_like(old, float('nan')), torch.zeros(1, 2, 2, 2)):
            with self.assertRaises(ValueError):
                DirectionalMaskEditor()(old, d, old)


class DirectionSupervisionTests(unittest.TestCase):
    def test_labels_threshold_and_detachment(self):
        old = torch.tensor([[[[.1, .9, .9, .1, .49, .495]]]], requires_grad=True)
        truth = torch.tensor([[[[1., 0., 1., 0., 1., 1.]]]])
        expected = torch.tensor([[[[1., -1., 0., 0., 1., 0.]]]])
        actual = direction_target(old, truth)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(actual.requires_grad)

    def test_balanced_gradient_padding_and_zero_direction_can_learn(self):
        old = torch.tensor([[[[.1, .9, .1, .9, .9]]]], requires_grad=True)
        truth = torch.tensor([[[[1., 0., 0., 1., 0.]]]])
        d = torch.zeros_like(old, requires_grad=True)
        state = SimpleNamespace(probabilities=old, logits=torch.logit(old), direction=d, edit_probabilities=None)
        losses = directional_losses(state, old, truth, torch.tensor([[[[1., 1., 1., 1., 0.]]]]))
        gd, gp = torch.autograd.grad(losses['direction'], (d, old), allow_unused=True)
        torch.testing.assert_close(losses['direction'], torch.tensor(1/3))
        torch.testing.assert_close(gd, torch.tensor([[[[-1/3, 1/3, 0., 0., 0.]]]]))
        self.assertIsNone(gp)

    def test_missing_class_and_weighted_loss(self):
        old = torch.full((1, 1, 1, 2), .1)
        d = torch.zeros_like(old, requires_grad=True)
        truth = torch.ones_like(old)
        state = SimpleNamespace(probabilities=old, logits=torch.logit(old), direction=d, edit_probabilities=None)
        weights = DirectionalLossWeights(0., 0., 2., 0.)
        result = directional_losses(state, old, truth, torch.ones_like(old), weights)
        torch.testing.assert_close(result['direction'], torch.tensor(.5))
        torch.testing.assert_close(result['loss'], torch.tensor(1.))
        with self.assertRaises(ValueError):
            DirectionalLossWeights(direction=-1)

    def test_perfect_first_mask_stops_without_training(self):
        mask = np.zeros((256, 256), bool)
        mask[40:180, 60:210] = True
        logits = torch.where(torch.from_numpy(mask)[None, None], 20., -20.)
        model = SimpleNamespace(sam_initializer=SimpleNamespace(postprocess=lambda x, sizes: [x]),
            initialize=lambda *args: SimpleNamespace(logits=logits, image_sizes=torch.tensor([[256, 256]])))
        result = directional_rollout(model, np.zeros((256, 256, 3), np.uint8), mask)
        self.assertIsNone(result['loss'])
        self.assertEqual(len(result['clicks']), 1)


class DirectionalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(42)
        cls.model = DirectionalPIMSNet().to(cls.device)
        cls.a3 = PIMSNet().to(cls.device).eval()
        cls.image = np.random.default_rng(9).integers(0, 256, (96, 144, 3), dtype=np.uint8)
        cls.features, cls.sizes = cls.model.sam_initializer.encode(cls.image)

    def initial(self):
        return self.model.initial_state(self.features, self.sizes, [[[50, 40]]], [[1]])

    def test_first_click_same_as_a3_and_new_image_resets(self):
        self.model.train()
        initial = self.initial()
        reference = self.a3.initial_state(self.features, self.sizes, [[[50, 40]]], [[1]])
        self.assertTrue(torch.equal(initial.logits, reference.logits))
        with torch.no_grad():
            state = self.model.refine(initial, [[[10, 10]]], [[0]])
        self.assertGreater(int(state.hidden.count_nonzero()), 0)
        self.assertEqual(int(self.initial().hidden.count_nonzero()), 0)
        self.assertFalse(self.model.sam_initializer.training)
        self.assertFalse(any(p.requires_grad for p in self.model.sam_initializer.parameters()))

    def test_decoder_batch_shape_and_activations(self):
        decoder = DirectionalEditDecoder().to(self.device)
        with torch.no_grad():
            decoder.head.weight.zero_()
            decoder.head.bias.copy_(torch.tensor([0., 0.], device=self.device))
            d, s = decoder(self.features.expand(2, -1, -1, -1), torch.zeros(2, 128, 64, 64, device=self.device),
                           torch.rand(2, 1, 256, 256, device=self.device))
        self.assertEqual(tuple(d.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(s.shape), (2, 1, 256, 256))
        self.assertTrue(torch.equal(d, torch.zeros_like(d)))
        self.assertTrue(torch.equal(s, torch.full_like(s, .5)))

    def test_five_rounds_full_bptt_and_both_head_gradients(self):
        model = self.model.train()
        model.zero_grad(set_to_none=True)
        state = self.initial()
        truth = np.zeros((96, 144), bool)
        truth[20:75, 30:115] = True
        target, valid = prepare_target(truth, self.device)
        terms = []
        first_hidden = None
        for i, (x, y, label) in enumerate(((10, 10, 0), (60, 50, 1), (110, 10, 0), (80, 60, 1))):
            old = state.probabilities
            state = model.refine(state, [[[x, y]]], [[label]])
            if first_hidden is None:
                first_hidden = state.hidden
            self.assertEqual(tuple(state.hidden.shape), (1, 128, 64, 64))
            self.assertEqual(tuple(state.probabilities.shape), (1, 1, 256, 256))
            self.assertEqual(state.points.shape[1], i + 2)
            self.assertTrue(torch.isfinite(state.logits).all())
            terms.append(directional_losses(state, old, target, valid)['loss'])
        historical = torch.autograd.grad(state.direction.sum(), first_hidden, retain_graph=True)[0]
        self.assertGreater(float(historical.abs().sum()), 0)
        torch.stack(terms).mean().backward()
        for channel in (0, 1):
            grad = model.directional_decoder.head.weight.grad[channel]
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.spatial_memory.parameters()), 0)
        self.assertTrue(all(p.grad is None for p in model.sam_initializer.parameters()))

    def test_checkpoint_roundtrip_and_a3_cross_load_rejected(self):
        directory = OUTPUT_ROOT / f'directional_test_{uuid4().hex}'
        directory.mkdir()
        path = directory / 'directional.pt'
        try:
            optimizer = torch.optim.AdamW(self.model.trainable_parameters(), lr=1e-4)
            expected = self.model.directional_decoder.head.weight.detach().clone()
            save_training_checkpoint(path, self.model, optimizer, 0, 0, {'variant': 'directional'})
            with torch.no_grad():
                self.model.directional_decoder.head.weight.add_(1)
            load_training_checkpoint(path, self.model)
            self.assertTrue(torch.equal(expected, self.model.directional_decoder.head.weight))
            with self.assertRaisesRegex(ValueError, 'variant'):
                load_training_checkpoint(path, self.a3)
            with self.assertRaisesRegex(ValueError, 'variant'):
                load_training_checkpoint(OUTPUT_ROOT / 'coco_subset_20261017/seed_42/training/a3/last.pt', self.model)
        finally:
            path.unlink(missing_ok=True)
            directory.rmdir()


if __name__ == '__main__':
    unittest.main()
