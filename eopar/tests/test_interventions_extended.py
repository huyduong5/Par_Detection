import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from eopar.datasets.interventions import InterventionDataset, BANDS, REGIONS


class Images:
    attributes=['Hat','Shirt','Pants','Shoes','Age']
    def __init__(self,mode='RGB',size=(24,64)): self.mode,self.size=mode,size
    def __len__(self): return 32
    def __getitem__(self,i):
        color=80 if self.mode=='L' else (80,120,170,255) if self.mode=='RGBA' else (80,120,170)
        return Image.new(self.mode,self.size,color),np.array([1,0,1,0,1]),str(i),torch.zeros(32,768)


class TestInterventionsExtended(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.path=Path(self.tmp.name)/'map.json'
        self.path.write_text(json.dumps(dict(zip(Images.attributes,[*REGIONS,'global']))))
    def tearDown(self): self.tmp.cleanup()
    def data(self,**kwargs): return InterventionDataset(Images(),self.path,size=64,**kwargs)

    def test_regions_severities_geometry_targets(self):
        for severity in (.25,.5,.75,1.):
            ds=self.data(split='val',severity=severity)
            found=set()
            for i in range(len(ds)):
                sample=ds[i]; region=sample['region']; found.add(region)
                j=REGIONS.index(region); lo=int(BANDS[j][0]*64)
                hi0=max(lo+1,int(BANDS[j][1]*64)); hi=min(64,lo+max(1,round((hi0-lo)*severity)))
                changed=(sample['img_clean']!=sample['img_occ']).any(0)
                expected=torch.zeros_like(changed); expected[lo:hi,20:44]=True
                self.assertTrue(torch.equal(changed,expected),(region,severity))
                self.assertEqual(sample['labels'].tolist(),[1,0,1,0,1])
                self.assertAlmostEqual(sample['vis_target'][j].item(),.85-.7*severity,places=6)
                self.assertAlmostEqual(sample['rank_weights'][j].item(),severity)
                self.assertEqual(sample['vis_valid'][-1].item(),0)
            self.assertEqual(found,set(REGIONS))

    def test_outside_targets_and_inside_ranking(self):
        ds=self.data(split='train',two_types=True)
        types=set()
        for i in range(len(ds)):
            sample=ds[i]; types.add(sample['intervention_type'])
            if sample['intervention_type']=='outside':
                self.assertNotEqual(sample['region'],sample['target_region'])
                self.assertEqual(float(sample['rank_weights'].sum()),0.)
                self.assertEqual(float(sample['intact_mask'].sum()),1.)
            else:
                self.assertGreater(float(sample['rank_weights'].sum()),0.)
        self.assertEqual(types,{'inside','outside'})

    def test_workers_epoch_and_order(self):
        ds=self.data(split='train')
        def read(workers):
            return {name:image.clone() for b in DataLoader(ds,batch_size=4,num_workers=workers)
                    for name,image in zip(b['imgname'],b['img_occ'])}
        a,b=read(0),read(4)
        self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
        ds.epoch=1; c=read(0)
        self.assertTrue(any(not torch.equal(a[k],c[k]) for k in a))
        val=self.data(split='val'); first=val[0]['img_occ']; val.epoch=99
        self.assertTrue(torch.equal(first,val[0]['img_occ']))
        self.assertTrue(torch.equal(ds[3]['img_occ'],ds[3]['img_occ']))

    def test_modes_extreme_sizes_and_too_small_rejected(self):
        for mode in ('RGB','RGBA','L'):
            for size in ((1,8),(8,1000),(1000,8),(448,448)):
                ds=InterventionDataset(Images(mode,size),self.path,split='val',size=32)
                self.assertEqual(tuple(ds[0]['img_clean'].shape),(3,32,32))
        with self.assertRaises(ValueError): InterventionDataset(Images(size=(8,1)),self.path,split='val')[0]

    def test_invalid_region_map_and_severity(self):
        for severity in (0,-1,1.1,float('nan')):
            with self.assertRaises(ValueError): self.data(severity=severity)
        self.path.write_text(json.dumps({'Hat':'head'}))
        with self.assertRaises(ValueError): self.data()

    def test_unverified_geometry_is_excluded_from_supervision(self):
        manifest=Path(self.tmp.name)/'geometry.json'
        manifest.write_text(json.dumps({'0':[]}))
        ds=InterventionDataset(Images(),self.path,size=64,split='val',geometry_manifest=manifest)
        for i in (0,1):
            sample=ds[i]
            self.assertFalse(sample['geometry_verified'])
            for key in ('vis_valid','intact_mask','rank_weights'):
                self.assertEqual(float(sample[key].sum()),0.)
        manifest.write_text(json.dumps({'0':list(REGIONS)}))
        ds=InterventionDataset(Images(),self.path,size=64,split='val',geometry_manifest=manifest)
        self.assertTrue(ds[0]['geometry_verified'])
        self.assertGreater(float(ds[0]['vis_valid'].sum()),0.)
