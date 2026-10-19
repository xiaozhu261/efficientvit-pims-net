import unittest
import torch
from pims_net.models.directional_pims_net import DirectionalEditDecoder, DirectionalPIMSNet
from pims_net.models.directional_wide import DecoderResidualBlock, WideDirectionalEditDecoder, WideDirectionalPIMSNet


class WideTests(unittest.TestCase):
    def test_width_blocks_shapes_and_gradients(self):
        torch.set_num_threads(4)
        decoder = WideDirectionalEditDecoder()
        self.assertEqual(sum(isinstance(m, DecoderResidualBlock) for m in decoder.modules()), 2)
        self.assertEqual([decoder.fusion[0].out_channels, decoder.fusion[2].out_channels, decoder.decoder[1].out_channels], [128, 64, 32])
        d, s = decoder(torch.randn(2,256,64,64), torch.randn(2,128,64,64), torch.rand(2,1,256,256))
        self.assertEqual(d.shape, (2,1,256,256))
        self.assertEqual(s.shape, d.shape)
        self.assertTrue((d.abs()<=1).all() and ((s>=0)&(s<=1)).all())
        (d.square().mean() + s.mean()).backward()
        for p in decoder.parameters():
            self.assertTrue(torch.isfinite(p.grad).all() and p.grad.abs().sum()>0)
        self.assertGreater(sum(p.numel() for p in decoder.parameters()), sum(p.numel() for p in DirectionalEditDecoder().parameters()))

    def test_residual_identity(self):
        block = DecoderResidualBlock(4)
        for p in block.parameters():
            torch.nn.init.zeros_(p)
        x = torch.randn(2,4,8,8)
        self.assertTrue(torch.equal(x, block(x)))

    def test_shared_initialization_and_plumbing(self):
        torch.manual_seed(42)
        original = DirectionalPIMSNet()
        torch.manual_seed(42)
        wide = WideDirectionalPIMSNet()
        self.assertEqual(wide.variant, 'directional_wide')
        for k,v in original.state_dict().items():
            if not k.startswith('directional_decoder.'):
                self.assertTrue(torch.equal(v, wide.state_dict()[k]), k)
        self.assertIs(WideDirectionalPIMSNet.refine, DirectionalPIMSNet.refine)
        self.assertIs(WideDirectionalPIMSNet.initialize, DirectionalPIMSNet.initialize)
        self.assertEqual(type(original.directional_editor), type(wide.directional_editor))
        self.assertTrue(all(not p.requires_grad for p in wide.sam_initializer.parameters()))


if __name__ == '__main__':
    unittest.main()
