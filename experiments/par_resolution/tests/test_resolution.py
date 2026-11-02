import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
from PIL import Image
from experiments.par_resolution.core import degraded,sha,read_source,load_manifest,safe_relative,write_png
from experiments.par_resolution.generate import generate,stratified

class OperatorTests(unittest.TestCase):
    def test_reference_and_sizes(self):
        rng=np.random.default_rng(42)
        for mode,shape in [('RGB',(13,9,3)),('RGBA',(11,7,4)),('L',(5,3)),('RGB',(1,1,3))]:
            im=Image.fromarray(rng.integers(0,256,shape,dtype=np.uint8),mode)
            for s in (1,2,4,8):
                got,low=degraded(im,s);rgb=im.convert('RGB');expected=(max(1,im.width//s),max(1,im.height//s))
                ref=rgb if s==1 else rgb.resize(expected,resample=Image.Resampling.BICUBIC,reducing_gap=None).resize(im.size,resample=Image.Resampling.BICUBIC,reducing_gap=None)
                self.assertEqual(low,expected);self.assertEqual(got.size,im.size)
                np.testing.assert_array_equal(got,ref)
    def test_constant_and_direct_original(self):
        constant=Image.new('RGB',(17,31),(13,127,244))
        for s in (1,2,4,8):np.testing.assert_array_equal(degraded(constant,s)[0],constant)
        grid=(np.indices((33,29)).sum(0)%2*255).astype('uint8');im=Image.fromarray(grid)
        got,_=degraded(im,8);expected=im.convert('RGB').resize((3,4),Image.Resampling.BICUBIC).resize((29,33),Image.Resampling.BICUBIC)
        np.testing.assert_array_equal(got,expected)
    def test_bad_scale_and_paths(self):
        for s in (0,3,-2,2.,True):
            with self.assertRaises(ValueError):degraded(Image.new('RGB',(2,2)),s)
        for name in ('../a.jpg','/a.jpg','a/../../b','a\\b'):
            with self.assertRaises(ValueError):safe_relative(name)
    def test_format_not_only_pixels(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'fake.png';im=Image.new('RGB',(8,8),'black');im.save(p,format='JPEG')
            with self.assertRaises(ValueError):write_png(p,im)
            write_png(p,im,repair=True)
            with Image.open(p) as check:self.assertEqual(check.format,'PNG')

class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.root=Path(self.tmp.name);self.images=self.root/'images';self.images.mkdir()
        names=['folder/a.png','other/a.png','gray.png','alpha.png','tiny.png']
        for i,name in enumerate(names):
            p=self.images/name;p.parent.mkdir(exist_ok=True)
            mode='L' if i==2 else 'RGBA' if i==3 else 'RGB';size=(1,2) if i==4 else (11+i,17+i)
            im=Image.new(mode,size,i*23 if mode=='L' else ((i*23,100,210,100) if mode=='RGBA' else (i*23,100,210)));im.save(p)
        self.pkl=self.root/'data.pkl';self.pkl.write_bytes(pickle.dumps(SimpleNamespace(attributes=['hat','shoe'],image_name=names,label=np.array([[1,0],[0,1],[1,1],[0,0],[1,0]]),partition={'val':[np.array([0,1,2])],'test':[np.array([3,4])]})))
        self.output=self.root/'benchmark';self.ex=self.root/'ex.json';self.ex.write_text(json.dumps({'val':[names[1]],'test':[]}))
    def tearDown(self):self.tmp.cleanup()
    def run_generation(self,**kw):return generate(self.pkl,self.images,self.output,self.ex,**kw)
    def test_generation_resume_and_integrity(self):
        original={str(p):sha(p) for p in self.images.rglob('*.png')};idx=self.run_generation()
        self.assertEqual(idx['generated_images'],20);self.assertEqual(idx['splits']['val']['dedup_count'],2)
        _,_,rows=load_manifest(self.output/'manifests/index.json')
        expected={str(p.relative_to(self.output)):sha(p) for p in self.output.rglob('*.png')}
        for row in rows:
            for s in (1,2,4,8):
                v=row['versions'][str(s)]
                with Image.open(self.output/v['path']) as im:self.assertEqual(im.size,(row['width'],row['height']))
            with Image.open(row['source_path']) as a,Image.open(self.output/row['versions']['1']['path']) as b:np.testing.assert_array_equal(a.convert('RGB'),b)
        self.run_generation(resume=True)
        self.assertEqual(expected,{str(p.relative_to(self.output)):sha(p) for p in self.output.rglob('*.png')})
        self.assertEqual(original,{str(p):sha(p) for p in self.images.rglob('*.png')})
        target=self.output/rows[0]['versions']['2']['path'];target.write_bytes(b'broken')
        with self.assertRaises(ValueError):self.run_generation(resume=True)
        with self.assertRaises(ValueError):load_manifest(self.output/'manifests/index.json')
        self.run_generation(resume=True,repair=True)
        self.assertEqual(sha(target),rows[0]['versions']['2']['sha256'])
    def test_invalid_roots_and_exclusions(self):
        with self.assertRaises(ValueError):generate(self.pkl,self.images,self.images/'nested')
        self.ex.write_text('{"val":["missing"],"test":[]}')
        with self.assertRaises(ValueError):self.run_generation()
    def test_labels_and_overlap_rejected(self):
        data=pickle.loads(self.pkl.read_bytes());data.label[0,0]=-1;self.pkl.write_bytes(pickle.dumps(data))
        with self.assertRaises(ValueError):read_source(self.pkl)
    def test_dataset_clean_tensor_and_subset(self):
        import torch
        from experiments.par_resolution.dataset import ResolutionDataset
        from aq_visibility.dataset.clip_dataset import get_clip_transforms
        self.run_generation();_,tf=get_clip_transforms(32,32)
        for s in (1,2,4,8):
            ds=ResolutionDataset(self.output/'manifests/index.json','val',s,tf)
            self.assertEqual(tuple(ds[0]['image'].shape),(3,32,32));self.assertEqual(ds[0]['labels'].tolist(),[1,0])
        clean=ResolutionDataset(self.output/'manifests/index.json','val',1,tf)
        with Image.open(clean.rows[0]['source_path']) as im:self.assertTrue(torch.equal(clean[0]['image'],tf(im.convert('RGB'))))
        subset=self.root/'subset.json';subset.write_text('["missing"]')
        with self.assertRaises(ValueError):ResolutionDataset(self.output/'manifests/index.json','val',1,tf,subset)
        p=self.output/clean.rows[0]['versions']['1']['path'];p.write_bytes(b'bad')
        with self.assertRaises(ValueError):clean[0]
    def test_stratified_reproducible(self):
        rows=[dict(image_id=f'part{i%8}_{i}',width=i+1,height=i+1) for i in range(200)]
        a=stratified(rows,64);self.assertEqual(a,stratified(rows,64));self.assertEqual(len(set(a)),64)
        self.assertTrue(any(int(x.split('_')[1])<50 for x in a));self.assertTrue(any(int(x.split('_')[1])>=150 for x in a))

if __name__=='__main__':unittest.main()
