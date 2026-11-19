"""CPU contract tests; CUDA and the compiled extension are not required."""
import importlib.util
from pathlib import Path
import unittest

import torch

_spec = importlib.util.spec_from_file_location(
    'mxfp4_weights', Path(__file__).resolve().parents[1] / 'deep_gemm/mega/mxfp4.py')
mxfp4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mxfp4)


class MXFP4WeightContracts(unittest.TestCase):
    def test_ue8m0_endpoints(self):
        powers = torch.exp2(torch.tensor([-127., -126., -1., 0., 127.]))
        expected = torch.tensor([0, 1, 126, 127, 254], dtype=torch.uint8)
        self.assertTrue(torch.equal(mxfp4._normalize_mxfp4_ue8m0(powers), expected))
        for value in (0., -1., 1.5, float('inf'), float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mxfp4._normalize_mxfp4_ue8m0(torch.tensor([value]))

    def test_sign_layout_preserves_all_packed_codes(self):
        raw = torch.arange(256, dtype=torch.int16).to(torch.uint8).reshape(2, 128)
        processed = mxfp4._reorder_mxfp4_sign_bits_for_sm90(raw)
        restored = mxfp4._restore_mxfp4_sign_bits_from_sm90(processed)
        self.assertTrue(torch.equal(restored.view(torch.uint8), raw))

    def test_tiled_stage_byte_set_and_inverse(self):
        raw = torch.arange(2 * 512 * 128, dtype=torch.int64).remainder(251).to(torch.int8).reshape(2, 512, 128)
        tiled = mxfp4._tile_mxfp4_weights_for_sm90(raw)
        self.assertEqual(tuple(tiled.shape), (2, 2, 2, 16384))
        # Compare natural E2M1 codes: sign bits are regrouped within words,
        # so processed byte histograms themselves need not be preserved.
        def codes(v):
            v = mxfp4._restore_mxfp4_sign_bits_from_sm90(v.contiguous().reshape(-1, 16)).view(torch.uint8)
            return torch.cat(((v & 15).flatten(), (v >> 4).flatten())).sort().values
        # Each contiguous stage contains exactly one N256 x K128 region.
        for expert in range(2):
            for n in range(2):
                for k in range(2):
                    expected = raw[expert, n * 256:(n + 1) * 256, k * 64:(k + 1) * 64].flatten()
                    self.assertTrue(torch.equal(codes(tiled[expert, n, k]), codes(expected)))
        self.assertTrue(torch.equal(mxfp4._untile_mxfp4_weights_from_sm90(tiled), raw))

    def test_scale_payload_physical_order(self):
        raw = torch.arange(2 * 32 * 8).reshape(2, 32, 8)
        result = mxfp4._transpose_mxfp4_scales_for_sm90(raw)
        self.assertTrue(torch.equal(result.flatten(), raw.reshape(2, 32, 2, 4).permute(0, 2, 1, 3).flatten()))
        self.assertTrue(torch.equal(mxfp4._restore_mxfp4_scales_from_sm90(result), raw))

    def test_secondary_and_layout_agree(self):
        l1 = (torch.full((1, 512, 256), 0x21, dtype=torch.int8), torch.full((1, 512, 16), 127, dtype=torch.uint8))
        l2 = (torch.full((1, 512, 128), 0x43, dtype=torch.int8), torch.full((1, 512, 8), 127, dtype=torch.uint8))
        row = mxfp4.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(l1, l2)
        tiled = mxfp4.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(l1, l2, torch.tensor([2.]), torch.tensor([4.]), layout='tiled')
        for i, factor in enumerate((2, 4)):
            self.assertTrue(torch.equal(mxfp4._untile_mxfp4_weights_from_sm90(tiled[i][0]), row[i][0]))
            self.assertTrue(torch.equal(tiled[i][1], row[i][1]))
            self.assertTrue(torch.equal(tiled[i][2], row[i][2] * factor))

    def test_invalid_scale_codes_and_dynamic_range(self):
        weight = torch.zeros((1, 16, 32), dtype=torch.int8)
        for codes in ([255, 127], [0, 254]):
            scales = torch.tensor(codes, dtype=torch.uint8).expand(1, 16, 2).contiguous()
            with self.subTest(codes=codes), self.assertRaises(ValueError):
                mxfp4._process_mxfp4_e8m0(weight, scales)


if __name__ == '__main__':
    unittest.main()
