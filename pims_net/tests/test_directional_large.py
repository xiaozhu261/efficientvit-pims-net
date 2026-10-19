import unittest
import torch
from pims_net.models.directional_large import LargeDirectionalEditDecoder, LargeDirectionalPIMSNet
from pims_net.models.directional_pims_net import DirectionalPIMSNet
from pims_net.models.directional_wide import DecoderResidualBlock, WideDirectionalEditDecoder


class LargeTests(unittest.TestCase):
    def test_capacity_shapes_gradients_and_initial_head(self):
        torch.set_num_threads(4)
        decoder=LargeDirectionalEditDecoder()
        self.assertEqual(sum(isinstance(m,DecoderResidualBlock) for m in decoder.modules()),4)
        self.assertEqual([decoder.fusion[0].out_channels,decoder.fusion[2].out_channels,decoder.decoder[1].out_channels],[256,128,64])
        torch.testing.assert_close(decoder.head.bias,torch.tensor([0.,-2.]))
        d,s=decoder(torch.randn(2,256,64,64),torch.randn(2,128,64,64),torch.rand(2,1,256,256))
        self.assertEqual(d.shape,(2,1,256,256))
        self.assertEqual(s.shape,d.shape)
        self.assertTrue((d.abs()<=1).all() and ((s>=0)&(s<=1)).all())
        (d.square().mean()+s.mean()).backward()
        for p in decoder.parameters():
            self.assertTrue(torch.isfinite(p.grad).all() and p.grad.abs().sum()>0)
        self.assertGreater(sum(p.numel() for p in decoder.parameters()),sum(p.numel() for p in WideDirectionalEditDecoder().parameters()))

    def test_nondecode_parameters_and_methods_unchanged(self):
        torch.manual_seed(42)
        original=DirectionalPIMSNet()
        torch.manual_seed(42)
        large=LargeDirectionalPIMSNet()
        state=large.state_dict()
        for name,p in original.state_dict().items():
            if not name.startswith('directional_decoder.'):
                self.assertTrue(torch.equal(p,state[name]),name)
        self.assertEqual(large.variant,'directional_large')
        self.assertIs(large.refine.__func__,original.refine.__func__)
        self.assertIs(large.initialize.__func__,original.initialize.__func__)
        self.assertEqual(type(large.directional_editor),type(original.directional_editor))
        self.assertTrue(all(not p.requires_grad for p in large.sam_initializer.parameters()))


if __name__=='__main__':
    unittest.main()
