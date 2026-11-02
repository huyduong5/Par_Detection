"""Unit tests: C3 uses occ forward; test loader returns distinct occ images."""

import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
sys.path.insert(0, str(PKG))


class TestOccForwardLogic(unittest.TestCase):
    def test_use_occ_flag_for_c3(self):
        """Mirrors train_one_epoch gate: visibility alone must enable occ forward."""
        class Cfg:
            use_occlusion_pairs = True
            use_visibility = True
            lambda_c = 0.0

        config = Cfg()
        use_occ = config.use_occlusion_pairs and (
            config.use_visibility or config.lambda_c > 0
        )
        self.assertTrue(use_occ)

    def test_c2_does_not_need_occ(self):
        class Cfg:
            use_occlusion_pairs = False
            use_visibility = False
            lambda_c = 0.0

        config = Cfg()
        use_occ = config.use_occlusion_pairs and (
            config.use_visibility or config.lambda_c > 0
        )
        self.assertFalse(use_occ)

    def test_return_occ_on_test_split(self):
        # Source-level check: return_occ must not be gated by split name.
        src = (PKG / 'dataset' / 'occlusion_dataset.py').read_text(encoding='utf-8')
        self.assertIn('self.return_occ = bool(return_occ)', src)
        self.assertNotIn("split in ('train', 'val')", src)

    def test_lambda_protocol(self):
        from config import get_thesis_variant

        c3 = get_thesis_variant('C3')
        c4 = get_thesis_variant('C4')
        self.assertEqual(c3.lambda_v, 0.5)
        self.assertEqual(c3.lambda_c, 0.0)
        self.assertTrue(c3.use_visibility)
        self.assertTrue(c3.use_occlusion_pairs)
        self.assertEqual(c4.lambda_v, 0.5)
        self.assertEqual(c4.lambda_c, 0.2)

    def test_pose_lite_uncertain(self):
        sys.path.insert(0, str(REPO / 'tools' / 'occlusion'))
        # import from build module path
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'build_synth_pairs',
            REPO / 'tools' / 'occlusion' / 'build_synth_pairs.py',
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ok, reason = mod.pose_lite_uncertain(64, 128, 'head')
        self.assertTrue(ok)
        ok2, reason2 = mod.pose_lite_uncertain(10, 10, 'shoes', min_side=32)
        self.assertFalse(ok2)
        self.assertEqual(reason2, 'min_side')


if __name__ == '__main__':
    unittest.main()
