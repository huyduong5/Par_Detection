import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from eopar.datasets.interventions import InterventionDataset, region_masks
from eopar.eval.metrics import par_metrics, risk_coverage
from eopar.eval.natural import binary_auc
from eopar.losses.objective import masked_mean, observability_ranking, stable_consistency
from eopar.models.adapter import EOPAR, load_baseline
from eopar.train.runtime import atomic_save, capture_rng, restore_rng


class ToyBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x, tokens, attrs):
        logits = self.linear(x)
        return logits, {'query_feats': x[:, None].expand(-1, 3, -1)}


class ToyDataset:
    attributes = ['Hat', 'Shirt', 'Adult']
    def __len__(self):
        return 2
    def __getitem__(self, index):
        return Image.new('RGB', (24, 64), (100, 150, 200)), np.array([1., 0., 1.]), f'{index}.jpg', torch.zeros(32, 768)


class TestModelLoss(unittest.TestCase):
    def test_disabled_matches_baseline(self):
        base = ToyBaseline().eval()
        wrapped = EOPAR(base, 4, enabled=False).eval()
        x = torch.randn(2, 4)
        self.assertTrue(torch.equal(base(x, None, None)[0], wrapped(x, None, None)['attr_logits']))

    def test_heads_have_gradients(self):
        model = EOPAR(ToyBaseline(), 4)
        out = model(torch.randn(2, 4), None, None)
        self.assertEqual(out['observability_logits'].shape, (2, 3))
        (out['attr_logits'].sum() + out['observability_logits'].sum()).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_observability_never_multiplies_class_probability(self):
        model=EOPAR(ToyBaseline(),4).eval();x=torch.randn(2,4)
        before=model.predict(x,None,None)['attribute_probability']
        with torch.no_grad():
            for p in model.observability_head.parameters():p.zero_()
            model.observability_head[-1].bias.fill_(-100.)
        after=model.predict(x,None,None)
        self.assertTrue(torch.equal(before,after['attribute_probability']))
        self.assertTrue((after['observability_probability']<1e-10).all())

    def test_checkpoint_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.pth'
            torch.save({'model_state_dict': {}}, path)
            with self.assertRaises(RuntimeError):
                load_baseline(ToyBaseline(), path)

    def test_teacher_stop_gradient(self):
        clean, occ = torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        stable_consistency(clean, occ, torch.ones(2, 3)).backward()
        self.assertIsNone(clean.grad)
        self.assertGreater(float(occ.grad.abs().sum()), 0.)

    def test_empty_masks_finite(self):
        clean, occ = torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        loss = stable_consistency(clean, occ, torch.zeros(2, 3)) + observability_ranking(clean, occ, torch.zeros(2, 3))
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertEqual(float(occ.grad.abs().sum()), 0.)

    def test_ranking_direction(self):
        c, o = torch.tensor([0.], requires_grad=True), torch.tensor([0.], requires_grad=True)
        observability_ranking(c, o, torch.ones(1)).backward()
        self.assertLess(c.grad.item(), 0.)
        self.assertGreater(o.grad.item(), 0.)

    def test_masks_global_uncertain(self):
        args = (['Hat', 'Shirt', 'Adult'], {'Hat': 'head', 'Shirt': 'upper', 'Adult': 'global'}, 'head')
        valid, affected, intact = region_masks(*args)
        self.assertEqual(valid.tolist(), [1., 1., 0.])
        self.assertEqual(affected.tolist(), [1., 0., 0.])
        self.assertEqual(intact.tolist(), [0., 1., 0.])
        self.assertEqual(sum(float(m.sum()) for m in region_masks(*args, valid=False)), 0.)

    def test_tiny_data_overfit_and_head_noncollapse(self):
        torch.manual_seed(42)
        model = EOPAR(ToyBaseline(), 4)
        optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        x = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        labels = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
        losses = []
        for _ in range(150):
            out = model(x, None, None)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out['attr_logits'], labels)
            # Shared per-query toy features cannot distinguish attrs, so use
            # sample-level obs targets to validate learnability without fake tokens.
            target = torch.tensor([[1., 1., 1.], [0., 0., 0.]])
            loss += torch.nn.functional.binary_cross_entropy_with_logits(out['observability_logits'], target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss))
        self.assertLess(losses[-1], losses[0] * .1)
        self.assertGreater(float(out['observability_logits'].sigmoid().std()), .2)


class TestDataMetrics(unittest.TestCase):
    def test_pairs_labels_and_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = Path(tmp) / 'map.json'
            mapping.write_text(json.dumps({'Hat': 'head', 'Shirt': 'upper', 'Adult': 'global'}))
            ds = InterventionDataset(ToyDataset(), mapping, split='val', size=64)
            a, b = ds[0], ds[0]
            self.assertTrue(torch.equal(a['img_occ'], b['img_occ']))
            self.assertEqual(a['labels'].tolist(), [1., 0., 1.])
            self.assertFalse(torch.equal(a['img_clean'], a['img_occ']))
            # Original and intervention share padding and all pixels outside the band.
            changed = (a['img_clean'] != a['img_occ']).any(0)
            self.assertLess(int(changed.sum()), 64 * 24)

    def test_metric_instance_not_element(self):
        m = par_metrics([[.9, .9, .1, .1]], [[1, 0, 0, 0]])
        self.assertAlmostEqual(m['instance_accuracy'], .5)
        self.assertAlmostEqual(m['legacy_element_accuracy'], .75)

    def test_selective_ties_order_independent(self):
        p, y, s = np.array([[.9, .9, .9]]), np.array([[1, 0, 1]]), np.ones((1, 3))
        a, b = risk_coverage(p, y, s), risk_coverage(p[:, ::-1], y[:, ::-1], s)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a['aurc'], 1 / 3)
        self.assertEqual(a['at_coverage']['0.8']['coverage'], 1.)

    def test_metrics_reject_missing_labels(self):
        with self.assertRaises(ValueError):
            par_metrics([[.5]], [[-1]])

    def test_natural_auc_ties(self):
        self.assertEqual(binary_auc([1, 0], [.5, .5]), .5)
        self.assertEqual(binary_auc([1, 0], [.9, .1]), 1.)
        self.assertIsNone(binary_auc([1, 1], [.9, .1]))

    def test_resume_rng_optimizer_scheduler_scaler(self):
        generator = torch.Generator().manual_seed(12)
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        scaler = torch.amp.GradScaler('cpu', enabled=False)
        optimizer.zero_grad(); model(torch.ones(1, 2)).sum().backward(); optimizer.step(); scheduler.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'resume.pth'
            atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                             scaler=scaler.state_dict(), epoch=3, rng=capture_rng(generator)), path)
            expected = (random.random(), np.random.rand(), torch.rand(1), torch.rand(1, generator=generator))
            state = torch.load(path, weights_only=False)
            fresh = nn.Linear(2, 1); fresh.load_state_dict(state['model'], strict=True)
            opt = torch.optim.AdamW(fresh.parameters(), lr=1.)
            sch = torch.optim.lr_scheduler.StepLR(opt, 1)
            opt.load_state_dict(state['optimizer']); sch.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
            restore_rng(state['rng'], generator)
            actual = (random.random(), np.random.rand(), torch.rand(1), torch.rand(1, generator=generator))
            self.assertEqual(expected[:2], actual[:2])
            self.assertTrue(torch.equal(expected[2], actual[2]))
            self.assertTrue(torch.equal(expected[3], actual[3]))
            self.assertEqual(opt.param_groups[0]['lr'], optimizer.param_groups[0]['lr'])
            self.assertEqual(sch.last_epoch, scheduler.last_epoch)
            self.assertEqual(state['epoch'], 3)


if __name__ == '__main__':
    unittest.main()
