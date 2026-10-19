import unittest

import numpy as np

from pims_net.evaluation.edit_competition import Accumulator, pixel_fields


class CompetitionTests(unittest.TestCase):
    def evaluate(self, old, q, truth, valid=None):
        old = np.array([[old]], dtype=np.float32)
        q = np.array(q, np.float32)[:, None, :]
        new = q[2:3] * old + q[:1]
        truth = np.array([truth], bool)
        valid = np.ones_like(truth) if valid is None else np.array([valid], bool)
        return pixel_fields(old, new, q, truth, valid)

    def test_keep_is_identity_not_competition(self):
        fields, regions, errors = self.evaluate([.9, .1], [[0, 0], [0, 0], [1, 1]], [1, 0])
        acc = Accumulator()
        acc.add(fields, regions['all'])
        row = acc.result()
        self.assertEqual(row['shrink'], 0)
        self.assertEqual(row['unchanged_distance'], 1)
        self.assertEqual(row['active'], 0)
        self.assertIsNone(row['cancellation_ratio'])
        self.assertIsNone(row['competing_given_active'])

    def test_symmetric_edit_competes_and_contracts(self):
        fields, regions, errors = self.evaluate([.9, .1], [[.4, .4], [.4, .4], [.2, .2]], [1, 0])
        np.testing.assert_allclose(fields['new_probability'], [[.58, .42]], atol=1e-7)
        acc = Accumulator()
        acc.add(fields, regions['all'])
        row = acc.result()
        self.assertEqual(row['competing_given_active'], 1)
        self.assertEqual(row['shrink_given_competing'], 1)
        self.assertAlmostEqual(row['cancellation_ratio'], .2, places=6)
        self.assertAlmostEqual(row['gt_error_delta'], .32, places=6)
        self.assertEqual(row['binary_flip'], 0)
        self.assertLess(errors['centered_identity'], 1e-6)

    def test_oracle_correction_can_expand_from_center_without_competition(self):
        fields, regions, errors = self.evaluate([.1, .9], [[1, 0], [0, 1], [0, 0]], [1, 0])
        acc = Accumulator()
        acc.add(fields, regions['all'])
        row = acc.result()
        self.assertEqual(row['competing_given_active'], 0)
        self.assertEqual(row['expand'], 1)
        self.assertEqual(row['cancellation_ratio'], 0)
        self.assertLess(row['gt_error_delta'], 0)
        self.assertEqual(row['binary_flip'], 1)

    def test_regions_padding_histograms_and_macro_weighting(self):
        fields, regions, errors = self.evaluate([.9, .1, .9, .1, .9], [[.4]*5, [.4]*5, [.2]*5],
                                                [1, 0, 0, 1, 1], [1, 1, 1, 1, 0])
        for group in ('tp', 'tn', 'fp', 'fn'):
            self.assertEqual(regions[group].sum(), 1)
        acc = Accumulator()
        acc.add(fields, regions['all'])
        acc.add(fields, regions['tp'])
        row = acc.result()
        self.assertEqual(row['pixels'], 5)
        self.assertEqual(row['snapshots'], 2)
        self.assertEqual(acc.histograms['q_add'].sum(), 5)
        self.assertEqual(acc.joint.sum(), 5)
        self.assertAlmostEqual(row['delta'], -.064, places=6)
        self.assertAlmostEqual(row['macro']['delta'], -.16, places=6)
        lower, upper = row['quantile_intervals']['gap']['50']
        self.assertLessEqual(lower, 0)
        self.assertGreater(upper, 0)
        self.assertAlmostEqual(upper - lower, .005)

    def test_tolerance_categories_cover_every_pixel(self):
        rng = np.random.default_rng(17)
        old = rng.uniform(0, 1, (1, 32, 32)).astype(np.float32)
        q = np.zeros((3, 32, 32), np.float32)
        q[:2] = rng.uniform(0, 4e-6, (2, 32, 32))
        q[2] = 1 - q[0] - q[1]
        new = q[2:3] * old + q[:1]
        fields, _, _ = pixel_fields(old, new, q, np.zeros((32, 32), bool), np.ones((32, 32), bool))
        count = sum(fields[k].astype(int) for k in ('shrink', 'expand', 'unchanged_distance'))
        np.testing.assert_array_equal(count, np.ones((32, 32), int))


if __name__ == '__main__':
    unittest.main()
