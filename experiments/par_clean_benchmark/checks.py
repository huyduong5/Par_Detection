"""Focused CPU checks for masking, metric parity, and clean decoding."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from PIL import Image
from aq_visibility.losses.climpar_loss import AsymmetricLoss
from eopar.eval.metrics import par_metrics
from eopar.datasets.interventions import encode
from .metrics import paired_bootstrap,scores
from .train import CleanDataset,MaskedClassification


class Checks(unittest.TestCase):
    def test_complete_loss_matches_original(self):
        x = torch.tensor([[.2,-.1],[.4,1.2],[-3,.7]])
        y = torch.tensor([[1.,0.],[0.,0.],[0.,1.]])
        pos = y.sum(0).clamp_min(1)
        ref = torch.nn.functional.binary_cross_entropy_with_logits(
            x,y,pos_weight=(len(y)-pos).clamp_min(1)/pos)+AsymmetricLoss()(x,y)
        self.assertTrue(torch.allclose(MaskedClassification()(x,y)[0],ref))

    def test_masked_entries_cannot_change_loss_or_scores(self):
        y = torch.tensor([[1.,-1.],[0.,1.]])
        x = torch.zeros_like(y,requires_grad=True)
        criterion = MaskedClassification()
        loss,_ = criterion(x,y)
        changed = x.detach().clone()
        changed[0,1] = 40
        self.assertEqual(loss.item(),criterion(changed,y)[0].item())
        loss.backward()
        self.assertEqual(x.grad[0,1].item(),0)
        p = x.detach().sigmoid().numpy()
        q = changed.sigmoid().numpy()
        mask = y.numpy()>=0
        targets = y.clamp_min(0).numpy()
        self.assertEqual(scores(p,targets,mask),scores(q,targets,mask))

    def test_metric_parity_and_ties(self):
        p = np.array([[.5,.7],[.5,.3],[.9,.2],[.2,.7]])
        y = np.array([[1,1],[0,0],[1,0],[0,1]])
        valid = np.ones_like(y,bool)
        result = scores(p,y,valid)
        ref = par_metrics(p,y)
        for key in ('mA','instance_accuracy','instance_precision','instance_recall','instance_f1_mean'):
            self.assertEqual(result[key],ref[key])
        permutation = [1,0,3,2]
        self.assertEqual(result,scores(p[permutation],y[permutation],valid[permutation]))
        self.assertEqual(paired_bootstrap(p,p,y,valid,draws=32)['ci95_pp'],[0.,0.])

    def test_clean_loader_and_input_hash(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'sample.png'
            image = Image.new('RGB',(15,30),(211,57,13))
            image.save(path)
            raw = path.read_bytes()
            row = dict(split='val',id='sample.png',path='sample.png',label=[1,0],valid=[1,0],
                       sha256=hashlib.sha256(raw).hexdigest())
            ds = CleanDataset(dict(rows=[row],counts={'val':1},image_root=tmp,attributes=['a','b']),'val',42)
            example = ds[0]
            self.assertTrue(torch.equal(example['img_clean'],encode(image,448)))
            self.assertEqual(example['labels'].tolist(),[1.,-1.])
            path.write_bytes(raw+b'changed')
            with self.assertRaisesRegex(ValueError,'Changed image'):
                ds[0]


if __name__=='__main__':
    unittest.main()
