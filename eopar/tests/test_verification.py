"""Production-path regression tests: independent formulas and failure cases."""
import copy
import json
import math
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from aq_visibility.losses.climpar_loss import CLIMPPARLoss
from eopar.losses.objective import objective, masked_mean, observability_ranking, stable_consistency
from eopar.models.weights import checked_state
from eopar.models.adapter import EOPAR
from eopar.eval.artifacts import validate_arrays, validate_thresholds, save_predictions
from eopar.train.runtime import atomic_save, digest, verify_pretrained, write_json
from eopar.train.run import load_config, train_epoch, evaluate


def criterion():
    return CLIMPPARLoss(SimpleNamespace(loss_type='weighted_bce_asl', lambda_v=.5, lambda_c=.2,
        use_visibility=True, consistency_type='sg_bce', global_contrastive_weight=0., finegrained_contrastive_weight=0.))


class TestProductionObjective(unittest.TestCase):
    def fixture(self, extreme=False):
        torch.manual_seed(14)
        def z():
            return (torch.randn(3, 4) * (50 if extreme else 1)).requires_grad_()
        clean = dict(attr_logits=z(), observability_logits=z(), legacy_visibility=z().sigmoid())
        occ = dict(attr_logits=z(), observability_logits=z(), legacy_visibility=z().sigmoid())
        batch = dict(labels=torch.randint(0, 2, (3, 4)).float(), vis_target=torch.rand(3, 4),
                     vis_valid=torch.tensor([[1, 0, 1, 0]] * 3).float(),
                     intact_mask=torch.tensor([[0, 1, 0, 1]] * 3).float(),
                     affected_mask=torch.tensor([[1, 0, 0, 0]] * 3).float(),
                     rank_weights=torch.tensor([[.25, 0, 0, 0], [.5, 0, 0, 0], [1., 0, 0, 0]]))
        cfg = dict(variant='P3', margin=.15, lambda_obs=.5, lambda_rank=.1)
        return clean, occ, batch, cfg

    def test_real_criterion_total_matches_independent_oracle(self):
        c, o, b, cfg = self.fixture()
        loss, parts = objective(c, o, b, criterion(), cfg)
        z, y = c['attr_logits'].double(), b['labels'].double()
        pos = y.sum(0).clamp(min=1); pw = (len(y) - pos).clamp(min=1) / pos
        bce = (-pw * y * F.logsigmoid(z) - (1-y) * F.logsigmoid(-z)).mean()
        p = z.sigmoid(); neg = (1 - p + .05).clamp(max=1)
        asl = (-(y * p.clamp(min=1e-8).log() + (1-y) * neg.clamp(min=1e-8).log()) *
               (1 - (p*y + neg*(1-y))) ** (y + 4*(1-y))).mean()
        def mean(v, w):
            w = w.double(); return (v*w).sum() / w.sum()
        vis = mean(-(b['vis_target'].double()*o['legacy_visibility'].double().log() +
                     (1-b['vis_target'].double())*(1-o['legacy_visibility'].double()).log()), b['vis_valid'])
        target = c['attr_logits'].detach().double().sigmoid()
        q = o['attr_logits'].double().sigmoid()
        kl = mean(target*(target/q).log() + (1-target)*((1-target)/(1-q)).log(), b['intact_mask'])
        obs = mean(-b['vis_target'].double()*F.logsigmoid(o['observability_logits'].double()) -
                   (1-b['vis_target'].double())*F.logsigmoid(-o['observability_logits'].double()), b['vis_valid'])
        rank = mean(torch.relu(.15-c['observability_logits'].double().sigmoid()+o['observability_logits'].double().sigmoid()), b['rank_weights'])
        oracle = bce + asl + .5*vis + .2*kl + .5*obs + .1*rank
        self.assertAlmostEqual(float(loss), float(oracle), places=5)
        self.assertAlmostEqual(parts['L_consist'], float(kl), places=6)
        self.assertAlmostEqual(parts['L_rank'], float(rank), places=6)

    def test_all_variants_finite_extremes_and_masks(self):
        for variant in ('B0', 'B1', 'P1', 'P2', 'P3'):
            for empty in (False, True):
                c, o, b, cfg = self.fixture(extreme=True); cfg['variant'] = variant
                if empty:
                    for key in ('vis_valid', 'intact_mask', 'affected_mask', 'rank_weights'): b[key].zero_()
                if variant.startswith('B'): c['observability_logits'] = o['observability_logits'] = None
                crit = criterion()
                if variant == 'B0': crit.lambda_c = 0.
                loss, _ = objective(c, o, b, crit, cfg)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for out in (c, o):
                    for value in out.values():
                        if value is not None and value.is_leaf and value.grad is not None:
                            self.assertTrue(torch.isfinite(value.grad).all(), variant)

    def test_teacher_has_classification_gradient_only(self):
        c, o, b, cfg = self.fixture()
        c['observability_logits'] = o['observability_logits'] = None
        loss, _ = objective(c, o, b, criterion(), cfg)
        actual = torch.autograd.grad(loss, c['attr_logits'], retain_graph=True)[0]
        crit = criterion(); crit.lambda_c = 0.; crit.lambda_v = 0.
        classification, _ = crit(c['attr_logits'], b['labels'])
        expected = torch.autograd.grad(classification, c['attr_logits'])[0]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_weighted_mask_preserves_severity(self):
        self.assertAlmostEqual(float(masked_mean(torch.tensor([1., 3.]), torch.tensor([.25, .75]))), 2.5)
        self.assertEqual(float(masked_mean(torch.tensor([float('nan')]), torch.zeros(1))), 0.)
        with self.assertRaises(ValueError): masked_mean(torch.ones(1), torch.tensor([-1.]))

    def test_differentiation_and_margin_boundary(self):
        c = torch.tensor([.1, -.2], dtype=torch.float64, requires_grad=True)
        o = torch.tensor([.2, .3], dtype=torch.float64, requires_grad=True)
        weights = torch.tensor([.25, 1.], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda x,y: observability_ranking(x,y,weights), (c,o)))
        self.assertTrue(torch.autograd.gradcheck(lambda y: stable_consistency(c,y,weights), (o,)))
        # Equal probs and zero margin lie exactly at the hinge: PyTorch uses 0.
        z = torch.zeros(1, requires_grad=True)
        observability_ranking(torch.zeros(1), z, torch.ones(1), margin=0.).backward()
        self.assertEqual(float(z.grad), 0.)
        self.assertEqual(float(observability_ranking(torch.tensor([10.]), torch.tensor([-10.]), torch.ones(1))), 0.)

    def test_loss_coefficient_ablation(self):
        c, o, b, cfg = self.fixture()
        total, parts = objective(c, o, b, criterion(), cfg)
        cfg['lambda_rank'] = 0.
        no_rank, _ = objective(c, o, b, criterion(), cfg)
        self.assertAlmostEqual(float(total-no_rank), .1*parts['L_rank'], places=6)
        cfg['lambda_obs'] = 0.
        no_obs, _ = objective(c, o, b, criterion(), cfg)
        self.assertAlmostEqual(float(no_rank-no_obs), .5*parts['L_obs'], places=6)


class MicroDataset(Dataset):
    attributes = ['a', 'b']
    epoch = 0
    def __init__(self, n): self.n=n
    def __len__(self): return self.n
    def __getitem__(self, i):
        return dict(img_clean=torch.tensor([i/10., 1.]), img_occ=torch.tensor([i/10., .5]),
                    labels=torch.tensor([i%2, (i+1)%2]).float(), domain_token=torch.zeros(1))


class MicroModel(nn.Module):
    def __init__(self): super().__init__(); self.linear = nn.Linear(2, 2)
    def forward(self, image, tokens, attrs):
        return dict(attr_logits=self.linear(image), observability_logits=None, legacy_visibility=None)


class TestTrainer(unittest.TestCase):
    def test_production_main_resume_new_process(self):
        raw=json.loads((Path(__file__).parents[1]/'configs/B1.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'images').mkdir()
            for key in ('pkl_path','region_map','pretrained_lock'):
                path=root/key; path.write_text('{}'); raw[key]=str(path)
            raw.update(img_dir=str(root/'images'),batch_size=2,eval_batch_size=2,workers=0,
                       epochs=2,schedule_epochs=30,num_attrs=2,max_train_batches=4,max_val_batches=2)
            config=root/'config.json'; config.write_text(json.dumps(raw))
            def launch(folder,*extra):
                return subprocess.run([sys.executable,'-m','eopar.verification.toy_process','--config',str(config),
                    '--output',str(root/folder),*extra],capture_output=True,text=True)
            for folder,args in [('continuous',()),('resumed',('--stop-after','1')),('resumed',('--resume',))]:
                result=launch(folder,*args); self.assertEqual(result.returncode,0,result.stderr)
            a,b=[torch.load(root/x/'latest.pth',weights_only=False) for x in ('continuous','resumed')]
            for key in a['model']: torch.testing.assert_close(a['model'][key],b['model'][key],atol=0,rtol=0)
            self.assertEqual(a['scheduler'],b['scheduler'])
            self.assertEqual(a['epoch'],b['epoch'])
            for key in ('torch','loader'): self.assertTrue(torch.equal(a['rng'][key],b['rng'][key]))
            self.assertEqual(launch('resumed','--resume').returncode,1)
            status=json.loads((root/'resumed/status.json').read_text())
            self.assertEqual(status['completed_epochs'],2)

    def test_accumulation_steps_remainder_and_parameter_oracle(self):
        # Exercise production trainer with a deterministic simple objective, then
        # separately exercise its real objective above. Clipping inactive here.
        def simple(c,o,b,crit,cfg):
            return F.binary_cross_entropy_with_logits(c['attr_logits'], b['labels']), {}
        for batches in (1,3,4,5,8):
            for accum in (1,2,4):
                torch.manual_seed(3)
                model=MicroModel(); ref=copy.deepcopy(model)
                ds=MicroDataset(2*batches-1); loader=DataLoader(ds,batch_size=2)
                optimizer=torch.optim.SGD(model.parameters(),lr=.01)
                scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
                cfg=dict(grad_accum=accum,max_train_batches=0)
                with patch('eopar.train.run.objective',simple):
                    train_epoch(model,loader,None,optimizer,scheduler,torch.amp.GradScaler('cpu',enabled=False),cfg,torch.device('cpu'),0)
                expected_optimizer=torch.optim.SGD(ref.parameters(),lr=.01)
                group=[]
                for i,b in enumerate(loader):
                    group.append(b)
                    if len(group)==accum or i+1==len(loader):
                        n=sum(len(x['labels']) for x in group)
                        loss=sum(F.binary_cross_entropy_with_logits(ref(x['img_clean'],None,None)['attr_logits'],x['labels'])*len(x['labels'])/n for x in group)
                        expected_optimizer.zero_grad(); loss.backward(); expected_optimizer.step(); group=[]
                self.assertEqual(scheduler.last_epoch, math.ceil(batches/accum))
                for a,b in zip(model.parameters(),ref.parameters()): torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)

    def test_frozen_mode_without_detaching_inputs(self):
        class Base(nn.Module):
            use_attr_queries=False
            def __init__(self):
                super().__init__(); self.vision_encoder=nn.Module(); self.text_encoder=nn.Module()
                for module in (self.vision_encoder,self.text_encoder):
                    module.backbone=nn.Sequential(nn.Linear(4,4),nn.Dropout(.8))
                    for p in module.backbone.parameters(): p.requires_grad_(False)
            def forward(self,x,t,a):
                feat=self.text_encoder.backbone(x)
                return feat[:,:2],{'query_feats':feat[:,None,:].expand(-1,2,-1)}
        model=EOPAR(Base(),4).train()
        self.assertFalse(model.baseline.text_encoder.backbone.training)
        x=torch.randn(2,4,requires_grad=True); model(x,None,None)['attr_logits'].sum().backward()
        self.assertIsNotNone(x.grad)


class TestValidation(unittest.TestCase):
    def test_direct_pilot_and_unbounded_diagnostics_blocked_before_build(self):
        raw=json.loads((Path(__file__).parents[1]/'configs/P3.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'images').mkdir()
            for key in ('pkl_path','region_map','pretrained_lock'):
                path=root/key;path.write_text('{}');raw[key]=str(path)
            raw['img_dir']=str(root/'images');config=root/'c.json';config.write_text(json.dumps(raw))
            for flags in ([],['--diagnostic-proxy']):
                result=subprocess.run([sys.executable,'-m','eopar.train.run','--config',str(config),
                    '--output',str(root/'out'),*flags],capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertFalse((root/'out').exists())

    def test_config_fail_early(self):
        raw=json.loads((Path(__file__).parents[1]/'configs/P3.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'images').mkdir()
            for key in ('pkl_path','region_map','pretrained_lock'):
                path=root/key; path.write_text('{}'); raw[key]=str(path)
            raw['img_dir']=str(root/'images'); config=root/'c.json'
            config.write_text(json.dumps(raw)); self.assertEqual(load_config(config)['variant'],'P3')
            for key,value in [('batch_size',0),('workers',-1),('epochs',-1),('lr',0),('lr',float('nan')),
                              ('size',31),('max_val_batches',-1),('grad_accum',True),('margin',2),('unknown',1)]:
                bad=dict(raw); bad[key]=value; config.write_text(json.dumps(bad))
                with self.assertRaises((ValueError,FileNotFoundError),msg=str((key,value))): load_config(config)
            bad=dict(raw); del bad['seed']; config.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): load_config(config)

    def test_strict_weights_and_checksum(self):
        model=nn.Linear(2,3)
        with self.assertRaises(ValueError): checked_state(model,{})
        bad=dict(model.state_dict()); bad['weight']=torch.zeros(1,2)
        with self.assertRaises(ValueError): checked_state(model,bad)
        checked_state(model,model.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'weights'; path.write_bytes(b'original')
            lock=Path(tmp)/'lock.json'; lock.write_text(json.dumps({str(path):digest(path)}))
            verify_pretrained(lock); path.write_bytes(b'corrupt')
            with self.assertRaises(ValueError): verify_pretrained(lock)

    def test_atomic_checkpoint_failure_retains_good_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'last.pth'; atomic_save({'epoch':1},path); before=path.read_bytes()
            with patch('eopar.train.runtime.torch.save',side_effect=OSError('disk full')):
                with self.assertRaises(OSError): atomic_save({'epoch':2},path)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(list(Path(tmp).glob('*.tmp')),[])
            path.write_bytes(b'truncated')
            with self.assertRaises(Exception): torch.load(path,weights_only=False)

    def test_artifact_and_threshold_validation(self):
        a=dict(names=np.array(['a','b']),attributes=np.array(['Hat']),clean=np.array([[.1],[.9]]),labels=np.array([[0],[1]]))
        validate_arrays(a)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'p.npz'; save_predictions(path,a)
            with np.load(path) as loaded: validate_arrays(loaded)
        fitted=dict(fitted_on='val',attributes=['Hat'],thresholds=[.5],validation_sha256='hash',names=['a'],checkpoint_sha256='ckpt')
        validate_thresholds(fitted,['Hat'],'ckpt',['b'],'test')
        for replacement in [dict(thresholds=[]),dict(thresholds=[float('nan')]),dict(fitted_on='test'),dict(checkpoint_sha256='other')]:
            with self.assertRaises(ValueError): validate_thresholds(dict(fitted,**replacement),['Hat'],'ckpt',['b'],'test')
        with self.assertRaises(ValueError): validate_thresholds(fitted,['Hat'],'ckpt',['a'],'test')

    def test_json_failure_retains_good_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.json'; write_json(path,{'state':'good'})
            with self.assertRaises(ValueError): write_json(path,{'bad':float('nan')})
            self.assertEqual(json.loads(path.read_text()),{'state':'good'})
