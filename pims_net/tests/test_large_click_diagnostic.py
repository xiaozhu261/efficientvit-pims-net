import unittest
import numpy as np
import torch

from pims_net.diagnose_large_clicks import describe,probability_update,counterfactuals,classify_component
from pims_net.models.directional_mask_editor import DirectionalMaskEditor


class ClickDiagnosticTests(unittest.TestCase):
    def test_disjoint_bottlenecks_positive_and_negative(self):
        p=torch.tensor([.1,.1,.1,.4])
        d=torch.tensor([-.8,.8,.2,.8])
        s=torch.tensor([.9,.1,1.,.9])
        for positive,pold,direction in ((True,p,d),(False,1-p,-d)):
            cf=counterfactuals(pold,direction,s,positive)
            classes=classify_component(direction,cf,positive)
            for name,index in [('wrong_or_zero_direction',0),('strength_limited',1),('direction_magnitude_limited',2),('crossed',3)]:
                self.assertEqual(classes[name].tolist(),[i==index for i in range(4)])
            self.assertTrue(cf['fix_sign'][0])
            self.assertFalse(cf['strength_one'][0])
            self.assertTrue(cf['strength_one'][1])
            self.assertFalse(cf['strength_one'][2])

    def test_threshold_equality_and_zero_direction(self):
        for positive,p,d in ((True,0.,1.),(False,1.,-1.)):
            cf=counterfactuals(torch.tensor([p]),torch.tensor([d]),torch.tensor([.5]),positive)
            self.assertEqual(bool(cf['actual']),not positive)
        cf=counterfactuals(torch.tensor([.5]),torch.tensor([0.]),torch.tensor([1.]),True)
        self.assertTrue(classify_component(torch.tensor([0.]),cf,True)['wrong_or_zero_direction'][0])

    def test_update_matches_model_and_does_not_mutate(self):
        torch.manual_seed(7)
        p=torch.rand(2,1,8,8)
        d=torch.rand_like(p)*2-1
        s=torch.rand_like(p)
        copies=[v.clone() for v in (p,d,s)]
        self.assertTrue(torch.equal(probability_update(p,d,s),DirectionalMaskEditor()(p,d,s)))
        counterfactuals(p,d,s,True)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(copies,(p,d,s))))

    def test_distribution_exact_and_empty(self):
        self.assertIsNone(describe(np.array([])))
        stats=describe(np.array([0.,1.,2.,3.]))
        self.assertEqual(stats['count'],4)
        self.assertEqual(stats['mean'],1.5)
        self.assertEqual(stats['median'],1.5)
        self.assertAlmostEqual(stats['q90'],2.7)


if __name__=='__main__':
    unittest.main()
