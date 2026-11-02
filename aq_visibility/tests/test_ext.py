"""Tests for the extension matrix: small-attr gate, sg_bce consistency, multi-scale."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
sys.path.insert(0, str(PKG))

from metrics.metrics_aq import compute_metrics, per_attr_ma_dict  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        'small_attr_gate', REPO / 'experiments' / 'aq_matrix' / 'small_attr_gate.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPerAttrMA(unittest.TestCase):
    def test_mean_equals_mA(self):
        rng = np.random.default_rng(0)
        labels = (rng.random((200, 57)) > 0.7).astype(np.float32)
        preds = rng.random((200, 57)).astype(np.float32)
        attrs = [f'a{i}' for i in range(57)]
        per = per_attr_ma_dict(preds, labels, attrs)
        self.assertEqual(len(per), 57)
        self.assertAlmostEqual(np.mean(list(per.values())),
                               compute_metrics(preds, labels)['mA'], places=6)


class TestSmallAttrGate(unittest.TestCase):
    def test_flags_only_weak_small_attrs(self):
        gate = _load_gate()
        d = {
            'clean': {'mA': 0.80},
            'per_attr_mA': {
                'Glasses': 0.70,      # small, weak
                'Sandals': 0.78,      # small, not weak (margin 0.05)
                'Female': 0.60,       # weak but not small
                'Boots': 0.74,        # small, weak
            },
        }
        weak = gate.weak_small_attrs(d, margin=0.05)
        self.assertEqual([a for a, _ in weak], ['Glasses', 'Boots'])

    def test_missing_per_attr(self):
        gate = _load_gate()
        self.assertEqual(gate.weak_small_attrs({'clean': {'mA': 0.8}}), [])
        md = gate.render_md({'C3': {'clean': {'mA': 0.8}}})
        self.assertIn('No eval JSON with per_attr_mA', md)


def _crit(consistency_type):
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

    Cfg.consistency_type = consistency_type
    return CLIMPPARLoss(config=Cfg())


@unittest.skipUnless(torch is not None, 'torch required')
class TestConsistencySG(unittest.TestCase):
    def test_zero_when_equal(self):
        crit = _crit('sg_bce')
        z = torch.randn(4, 57)
        mask = torch.ones(4, 57)
        self.assertAlmostEqual(float(crit.consistency_loss_sg(z, z.clone(), mask)), 0.0, places=5)

    def test_positive_when_different(self):
        crit = _crit('sg_bce')
        z = torch.zeros(2, 5)
        loss = crit.consistency_loss_sg(z, torch.full((2, 5), 3.0), torch.ones(2, 5))
        self.assertGreater(float(loss), 0.1)

    def test_stop_grad_on_clean(self):
        crit = _crit('sg_bce')
        z_c = torch.randn(3, 6, requires_grad=True)
        z_o = torch.randn(3, 6, requires_grad=True)
        crit.consistency_loss_sg(z_c, z_o, torch.ones(3, 6)).backward()
        self.assertIsNone(z_c.grad)
        self.assertGreater(float(z_o.grad.abs().sum()), 0.0)

    def test_mask(self):
        crit = _crit('sg_bce')
        z_c = torch.zeros(2, 4)
        z_o = torch.zeros(2, 4, requires_grad=True)
        with torch.no_grad():
            z_o[:, 3] = 5.0
        mask = torch.tensor([[1., 1., 1., 0.], [1., 1., 1., 0.]])
        loss = crit.consistency_loss_sg(z_c, z_o, mask)
        self.assertAlmostEqual(float(loss), 0.0, places=5)
        loss.backward()
        self.assertEqual(float(z_o.grad[:, 3].abs().sum()), 0.0)
        empty = crit.consistency_loss_sg(z_c, z_o, torch.zeros(2, 4))
        self.assertEqual(float(empty), 0.0)

    def test_extreme_fp16(self):
        crit = _crit('sg_bce')
        z_c = torch.tensor([[30.0, -30.0, 0.0]]).half()
        z_o = torch.tensor([[-30.0, 30.0, 0.0]]).half()
        loss = crit.consistency_loss_sg(z_c, z_o, torch.ones(1, 3))
        self.assertTrue(torch.isfinite(loss))

    def test_forward_dispatch_and_l1_regression(self):
        z_c = torch.randn(4, 7)
        z_o = torch.randn(4, 7)
        mask = (torch.rand(4, 7) > 0.3).float()
        labels = torch.zeros(4, 7)
        manual_l1 = ((torch.sigmoid(z_c) - torch.sigmoid(z_o)).abs() * mask).sum() / mask.sum()
        _, parts_l1 = _crit('l1')(z_c, labels, features={}, logits_occ=z_o, consist_mask=mask)
        self.assertAlmostEqual(float(parts_l1['L_consist']), float(manual_l1), places=5)
        crit_sg = _crit('sg_bce')
        _, parts_sg = crit_sg(z_c, labels, features={}, logits_occ=z_o, consist_mask=mask)
        self.assertAlmostEqual(
            float(parts_sg['L_consist']),
            float(crit_sg.consistency_loss_sg(z_c, z_o, mask)), places=5)

    def test_presets(self):
        from config import get_thesis_variant
        c4b = get_thesis_variant('C4b')
        self.assertEqual((c4b.lambda_v, c4b.lambda_c, c4b.consistency_type), (0.5, 0.2, 'sg_bce'))
        self.assertFalse(c4b.use_multiscale)
        c3ms = get_thesis_variant('C3_MS')
        self.assertEqual((c3ms.lambda_c, c3ms.use_multiscale), (0.0, True))
        c4bms = get_thesis_variant('C4b_MS')
        self.assertEqual((c4bms.consistency_type, c4bms.use_multiscale), ('sg_bce', True))
        self.assertEqual(get_thesis_variant('C4').consistency_type, 'l1')
        fpw = get_thesis_variant('C4b_MS_fpw')
        self.assertTrue(fpw.use_fixed_pos_weight and fpw.use_multiscale)
        self.assertEqual(fpw.consistency_type, 'sg_bce')
        self.assertTrue(fpw.attr_freq_path.endswith('attr_pos_weight_sqrt.json'))

    def test_fixed_pos_weight_missing_file_raises(self):
        from losses.climpar_loss import CLIMPPARLoss

        class Cfg:
            use_fixed_pos_weight = True
            attr_freq_path = '/nonexistent/pos_weight.json'

        with self.assertRaises(FileNotFoundError):
            CLIMPPARLoss(config=Cfg())


class TestCalibration(unittest.TestCase):
    def test_threshold_and_group_decoding(self):
        spec = importlib.util.spec_from_file_location(
            'calib', REPO / 'experiments' / 'aq_matrix' / 'calibrate_thresholds.py')
        cal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cal)
        rng = np.random.default_rng(1)
        n = 2000
        # Majority attr (97% positive) whose probs never drop below 0.5 → mA 0.5 at fixed 0.5.
        y = (rng.random(n) < 0.97).astype(np.float32)
        p = np.where(y == 1, rng.uniform(0.8, 0.99, n), rng.uniform(0.55, 0.8, n))
        t = cal.fit_thresholds(p[:, None], y[:, None])
        fixed = cal.balanced_acc((p >= 0.5).astype(np.int8), y)
        tuned = cal.balanced_acc((p >= t[0]).astype(np.int8), y)
        self.assertAlmostEqual(fixed, 0.5, places=3)
        self.assertGreater(tuned, 0.95)

        attrs = ['Fat', 'Normal', 'Thin', 'Other']
        labels = np.array([[0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1]], dtype=np.float32)
        groups = cal.exclusive_groups(attrs, labels)
        self.assertTrue(groups[0][3])
        probs = np.array([[0.6, 0.9, 0.7, 0.8], [0.9, 0.95, 0.1, 0.2], [0.2, 0.3, 0.8, 0.6]])
        thr = np.array([0.5, 0.9, 0.5, 0.5])
        pred = cal.decode(probs, thr, groups)
        self.assertEqual(pred[:, :3].sum(1).tolist(), [1, 1, 1])
        self.assertEqual(pred[1, :3].tolist(), [1, 0, 0])   # Fat margin 0.4 > Normal 0.05


def _load_file(name, rel):
    spec = importlib.util.spec_from_file_location(name, PKG / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(torch is not None, 'torch required')
class TestMultiScaleCPU(unittest.TestCase):
    def test_token_shapes_and_decoder(self):
        ms = _load_file('ms_mod', 'model/multiscale.py')
        aq = _load_file('aq_mod', 'model/attribute_query_decoder.py')
        self.assertEqual(ms.vmamba_stage_channels(96, 1), 384)
        self.assertEqual(ms.vmamba_stage_channels(96, 3), 768)
        tok = ms.MultiScaleTokenizer(in_channels=384, d_model=64, pool_hw=(16, 8))
        main = torch.randn(2, 32, 64)
        stage = torch.randn(2, 384, 28, 28)
        fused = tok.fuse(main, stage)
        self.assertEqual(tuple(fused.shape), (2, 32 + 128, 64))
        dec = aq.AttributeQueryDecoder(num_attrs=57, d_model=64, num_layers=2, num_heads=8)
        q = dec(fused, text_attr_init=torch.randn(2, 57, 64))
        self.assertEqual(tuple(q.shape), (2, 57, 64))

    def test_gradient_reaches_tokenizer(self):
        ms = _load_file('ms_mod2', 'model/multiscale.py')
        tok = ms.MultiScaleTokenizer(in_channels=8, d_model=16, pool_hw=(4, 2))
        out = tok.fuse(torch.randn(1, 3, 16), torch.randn(1, 8, 10, 6))
        out.sum().backward()
        self.assertGreater(float(tok.proj.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(tok.scale_emb_main.grad.abs().sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
