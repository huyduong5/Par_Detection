"""Protocol checks: loss terms, consist mask, unpaired defaults, gap aggregation."""

import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
sys.path.insert(0, str(PKG))

from dataset.region_utils import protocol_consist_mask  # noqa: E402
from metrics.metrics_aq import consistency_gap  # noqa: E402


ATTRS = ['Adult', 'Hat', 'Shirt', 'Trousers', 'Sandals', 'Walking']
REGION_MAP = {
    'Adult': 'global',
    'Hat': 'head',
    'Shirt': 'upper',
    'Trousers': 'lower',
    'Sandals': 'shoes',
    'Walking': 'global',
}


class TestConsistMask(unittest.TestCase):
    def test_only_intact_evidence(self):
        mask = protocol_consist_mask(ATTRS, REGION_MAP, 'head', bbox_ok=True)
        # Adult/Walking global = 0, Hat occluded = 0, other evidence = 1
        self.assertEqual(mask.tolist(), [0, 0, 1, 1, 1, 0])

    def test_uncertain_all_zero(self):
        mask = protocol_consist_mask(ATTRS, REGION_MAP, 'shoes', bbox_ok=False)
        self.assertEqual(float(mask.sum()), 0.0)

    def test_builder_matches_protocol(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'build_synth_pairs', REPO / 'tools' / 'occlusion' / 'build_synth_pairs.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        vis_t, vis_v, affected, consist = mod.vis_targets_for_occlusion(
            ATTRS, REGION_MAP, 'upper', bbox_ok=True)
        self.assertAlmostEqual(float(vis_t[ATTRS.index('Shirt')]), 0.15)
        self.assertAlmostEqual(float(vis_t[ATTRS.index('Hat')]), 0.85)
        self.assertEqual(float(vis_v[ATTRS.index('Adult')]), 0.0)
        self.assertEqual(float(vis_v[ATTRS.index('Walking')]), 0.0)
        self.assertEqual(float(consist[ATTRS.index('Adult')]), 0.0)
        self.assertEqual(float(consist[ATTRS.index('Shirt')]), 0.0)
        self.assertEqual(float(consist[ATTRS.index('Hat')]), 1.0)
        self.assertTrue(affected[ATTRS.index('Shirt')])


class TestGapAggregation(unittest.TestCase):
    def test_global_mean_not_batch_mean(self):
        # Batch A: 1 sample, gap 0. Large batch B: gap 1. Equal batch-mean would be 0.5.
        p_c = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        p_o = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        mask = np.ones_like(p_c)
        gap = consistency_gap(p_c, p_o, mask)['consistency_gap']
        self.assertAlmostEqual(gap, 2.0 / 3.0, places=5)
        batch_mean = np.mean([
            consistency_gap(p_c[:1], p_o[:1], mask[:1])['consistency_gap'],
            consistency_gap(p_c[1:], p_o[1:], mask[1:])['consistency_gap'],
        ])
        self.assertGreater(abs(batch_mean - gap), 0.1)


try:
    import torch
except ImportError:
    torch = None


class TestLossTerms(unittest.TestCase):
    @unittest.skipUnless(torch is not None, 'torch required')
    def test_vis_and_consist(self):
        import torch
        from losses.climpar_loss import CLIMPPARLoss

        class Cfg:
            loss_type = 'bce'
            global_contrastive_weight = 0.0
            finegrained_contrastive_weight = 0.0
            temperature_init = 0.07
            lambda_v = 0.5
            lambda_c = 0.2
            use_visibility = True
            lambda_kd = 0.0

        crit = CLIMPPARLoss(config=Cfg())
        logits = torch.zeros(2, 3)
        labels = torch.zeros(2, 3)
        vis_pred = torch.full((2, 3), 0.5)
        vis_target = torch.full((2, 3), 0.5)
        empty = torch.zeros(2, 3)
        _, parts = crit(
            logits, labels, features={'vis_pred': vis_pred},
            vis_target=vis_target, vis_valid_mask=empty,
            logits_occ=logits, consist_mask=empty,
        )
        self.assertEqual(float(parts['L_vis']), 0.0)
        self.assertEqual(float(parts['L_consist']), 0.0)
        self.assertFalse(torch.isnan(parts['_total']))

        valid = torch.ones(2, 3)
        loss_far, parts_far = crit(
            logits, labels, features={'vis_pred': vis_pred},
            vis_target=vis_target, vis_valid_mask=valid,
            logits_occ=torch.full_like(logits, 4.0), consist_mask=valid,
        )
        loss_near, parts_near = crit(
            logits, labels, features={'vis_pred': vis_pred},
            vis_target=vis_target, vis_valid_mask=valid,
            logits_occ=logits.clone(), consist_mask=valid,
        )
        self.assertLess(float(parts_near['L_consist']), float(parts_far['L_consist']))
        self.assertIn('L_vis', parts_near)
        self.assertFalse(torch.isnan(loss_far))
        self.assertFalse(torch.isnan(loss_near))


class TestUnpairedSample(unittest.TestCase):
    @unittest.skipUnless(torch is not None, 'torch required')
    def test_vis_valid_zero_and_occ_equals_clean(self):
        from types import SimpleNamespace
        from PIL import Image
        from dataset.occlusion_dataset import OcclusionPairDataset

        tmp = tempfile.mkdtemp()
        img_dir = os.path.join(tmp, 'images')
        os.makedirs(img_dir)
        Image.new('RGB', (32, 48), (10, 20, 30)).save(os.path.join(img_dir, 'a.jpg'))
        info = SimpleNamespace(
            image_name=['a.jpg'],
            label=np.zeros((1, len(ATTRS)), dtype=np.float32),
            partition={'test': np.array([0])},
            attributes=ATTRS,
            root=img_dir,
        )
        pkl = os.path.join(tmp, 'data.pkl')
        with open(pkl, 'wb') as f:
            pickle.dump(info, f)
        occ_dir = os.path.join(tmp, 'occ')
        os.makedirs(occ_dir)
        with open(os.path.join(occ_dir, 'body_region_map.json'), 'w', encoding='utf-8') as f:
            import json
            json.dump(REGION_MAP, f)
        occ_pkl = os.path.join(occ_dir, 'occlusion_pairs_test.pkl')
        with open(occ_pkl, 'wb') as f:
            pickle.dump({'by_imgname': {}, 'split': 'test'}, f)

        ds = OcclusionPairDataset(
            pkl_path=pkl, occlusion_pkl_path=occ_pkl, img_dir=img_dir,
            split='test', transform=None, return_occ=True,
        )
        sample = ds[0]
        self.assertEqual(float(sample['vis_valid'].sum()), 0.0)
        self.assertEqual(float(sample['consist_mask'].sum()), 0.0)
        self.assertTrue(torch_equal_img(sample['img_clean'], sample['img_occ']))


def torch_equal_img(a, b):
    import numpy as np
    from PIL import Image
    if isinstance(a, Image.Image):
        return np.array(a).tobytes() == np.array(b).tobytes()
    return True


if __name__ == '__main__':
    unittest.main()
