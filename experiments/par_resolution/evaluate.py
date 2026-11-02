"""Frozen checkpoint classification across clean/x2/x4/x8; no occlusion masks."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import resource
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from .core import ROOT,SCALES,atomic_json,load_manifest,sha
from .dataset import ResolutionDataset
from . import adapters
from eopar.eval.artifacts import save_predictions
from eopar.eval.metrics import evaluate_arrays,par_metrics


def evaluate_entry(entry,manifest,split,output,subset=None,batch_size=8,workers=2):
    if type(batch_size) is not int or batch_size<1 or type(workers) is not int or workers<0:raise ValueError('Invalid batch size/workers')
    root,index,rows=load_manifest(manifest)
    index=dict(index,protocol_split_sha256=json.loads((root/'protocol.json').read_text())['split_sha256'])
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model,config,transform=adapters.load(entry,index,output/'adapter_config')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
    versions={name:parameter._version for name,parameter in model.named_parameters()}
    results={};ref_names=None;ref_labels=None
    prototype=ResolutionDataset(manifest,split,1,transform,subset,
        config.domain_tokens_path if config.use_domain_tokens else None,config.embed_dim)
    # Reference parity: original decoded image, same checkpoint-specific transform.
    maximum=0.
    for i,record in enumerate(prototype.rows[:128]):
        if sha(record['source_path'])!=record['source_sha256']:raise ValueError('Original image changed since generation')
        with Image.open(record['source_path']) as image:original=transform(image.convert('RGB'))
        sample=prototype[i]
        if not torch.equal(original,sample['image']):raise ValueError('Clean PNG preprocessing differs from original')
        maximum=max(maximum,float((original-sample['image']).abs().max()))
    parity=dict(state='PASS',checked=min(128,len(prototype)),max_input_tensor_error=maximum,
                reasoning='Identical tensors through the same frozen model and preprocessing')
    for scale in SCALES:
        tag='clean' if scale==1 else f'x{scale}';marker=output/f'{split}_{tag}.json';predfile=marker.with_suffix('.npz')
        identity=dict(checkpoint_sha256=entry['checkpoint_sha256'],benchmark_index_sha256=sha(manifest),
                      subset_sha256=sha(subset) if subset else None,scale=scale,split=split)
        if marker.exists():
            result=json.loads(marker.read_text())
            if any(result.get(k)!=v for k,v in identity.items()) or sha(predfile)!=result['predictions_sha256']:
                raise ValueError('Existing evaluation has incompatible provenance')
            with np.load(predfile,allow_pickle=False) as previous:
                if ref_names is not None and (not np.array_equal(ref_names,previous['names']) or not np.array_equal(ref_labels,previous['labels'])):raise ValueError('Resumed predictions are not paired')
                ref_names,ref_labels=previous['names'].copy(),previous['labels'].copy()
            results[str(scale)]=result;continue
        if predfile.exists():raise FileExistsError(f'Orphan predictions: {predfile}; inspect before restarting')
        ds=ResolutionDataset(manifest,split,scale,transform,subset,
            config.domain_tokens_path if config.use_domain_tokens else None,config.embed_dim)
        loader=DataLoader(ds,batch_size=batch_size,num_workers=workers,shuffle=False,pin_memory=device.type=='cuda')
        predictions=[];labels=[];names=[];observability=[];times=[];wall_start=time.monotonic()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for batch in loader:
                images=batch['image'].to(device);tokens=batch['domain'].to(device)
                if device.type=='cuda':torch.cuda.synchronize()
                start=time.monotonic()
                with torch.amp.autocast(device.type,enabled=device.type=='cuda' and config.use_amp):
                    logits,obs=adapters.forward(model,entry['family'],images,tokens,ds.attributes)
                if device.type=='cuda':torch.cuda.synchronize()
                times.append(time.monotonic()-start)
                predictions.append(logits.float().sigmoid().cpu().numpy());labels.append(batch['labels'].numpy());names.extend(batch['names'])
                if obs is not None:observability.append(obs.float().sigmoid().cpu().numpy())
        arrays=dict(clean=np.concatenate(predictions),labels=np.concatenate(labels),names=np.asarray(names),attributes=np.asarray(ds.attributes))
        if scale==1:
            original_probabilities=[]
            with torch.inference_mode():
                for start in range(0,min(128,len(ds)),batch_size):
                    reference=ds.rows[start:min(start+batch_size,128)]
                    images=[]
                    for record in reference:
                        with Image.open(record['source_path']) as image:images.append(transform(image.convert('RGB')))
                    images=torch.stack(images).to(device);tokens=torch.stack([ds.token(r) for r in reference]).to(device)
                    with torch.amp.autocast(device.type,enabled=device.type=='cuda' and config.use_amp):
                        logits,_=adapters.forward(model,entry['family'],images,tokens,ds.attributes)
                    original_probabilities.append(logits.float().sigmoid().cpu().numpy())
            original_probabilities=np.concatenate(original_probabilities);count=len(original_probabilities)
            error=float(np.max(np.abs(original_probabilities-arrays['clean'][:count])))
            tolerance=.002 if device.type=='cuda' and config.use_amp else 1e-6
            if error>tolerance:raise ValueError('Original-image versus clean-PNG logits differ')
            original_metrics=par_metrics(original_probabilities,arrays['labels'][:count]);clean_metrics=par_metrics(arrays['clean'][:count],arrays['labels'][:count])
            if original_metrics!=clean_metrics:raise ValueError('Original-image versus clean-PNG classification metrics differ')
            parity.update(classification_checked=count,max_probability_error=error,probability_tolerance=tolerance,classification_metrics_equal=True)
        for key in ('width','height'):
            arrays[key]=np.asarray([r[key] for r in ds.rows])
        for key in ('low_width','low_height','clamped_to_one'):
            arrays[key]=np.asarray([r['versions'][str(scale)][key] for r in ds.rows])
        if observability:arrays['obs']=np.concatenate(observability)
        if ref_names is not None and (not np.array_equal(ref_names,arrays['names']) or not np.array_equal(ref_labels,arrays['labels'])):
            raise ValueError('Scale evaluations are not paired')
        ref_names,ref_labels=arrays['names'],arrays['labels']
        result=evaluate_arrays(arrays['clean'],arrays['labels'],arrays.get('obs'))
        # Same predictions, separate deduplicated supplementary metrics; no extra inference.
        idsfile=root/f'manifests/{split}_dedup_ids.json'
        if sha(idsfile)!=index['splits'][split]['dedup_ids_sha256']:raise ValueError('Dedup ID list changed')
        allowed=set(json.loads(idsfile.read_text()));mask=np.array([n in allowed for n in names])
        if mask.any():result['supplementary_dedup']=evaluate_arrays(arrays['clean'][mask],arrays['labels'][mask],arrays.get('obs')[mask] if 'obs' in arrays else None)
        result.update(identity,model_id=entry['id'],group=entry.get('group',entry['id']),seed=entry['seed'],family=entry['family'],samples=len(names),
            attributes=ds.attributes,preprocessing=dict(height=config.img_height,width=config.img_width,description=str(transform)),
            threshold=.5,artifact_state='complete',clean_reference_parity=parity,
            timings=dict(seconds=sum(times),forward_images_per_second=len(names)/sum(times),wall_seconds=time.monotonic()-wall_start,warmup_included=True),
            precision='cuda_autocast_float16' if device.type=='cuda' and config.use_amp else 'float32',
            peak_main_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_vram_bytes=torch.cuda.max_memory_reserved() if device.type=='cuda' else None,
            legacy_attribute_binding=entry.get('attribute_binding'),
            selection='Saved best clean-validation checkpoint; no degradation-specific tuning')
        save_predictions(predfile,arrays);result['predictions_sha256']=sha(predfile)
        # Recompute from the published artifact, rather than trusting in-memory metrics.
        with np.load(predfile,allow_pickle=False) as saved:
            check=evaluate_arrays(saved['clean'],saved['labels'],saved['obs'] if 'obs' in saved else None)
        if check['par']!=result['par']:raise ValueError('Prediction round-trip metric mismatch')
        if model.training or any(p.requires_grad for p in model.parameters()) or any(p._version!=versions[n] for n,p in model.named_parameters()):raise ValueError('Frozen inference model was updated')
        atomic_json(marker,result);results[str(scale)]=result
        print(entry['id'],split,tag,'mA',result['par']['mA'],flush=True)
    atomic_json(output/f'{split}_complete.json',dict(state='complete',model_id=entry['id'],scales=list(SCALES),
        results={k:dict(mA=v['par']['mA'],instance_f1=v['par']['instance_f1_mean']) for k,v in results.items()}))
    return results


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--registry',required=True);p.add_argument('--model-id',required=True)
    p.add_argument('--degradation-manifest',required=True);p.add_argument('--split',choices=['val','test'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--subset');p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--test-gate')
    a=p.parse_args()
    registry=json.loads(Path(a.registry).read_text());entry=next(r for r in registry['models'] if r['id']==a.model_id)
    from .registry import source_fingerprint
    if source_fingerprint()[0]!=registry['source_fingerprint']:raise ValueError('Runtime source differs from registered source')
    from eopar.train.runtime import environment_identity
    if environment_identity()!=registry['environment']:raise ValueError('Runtime environment differs from checkpoint registry')
    if a.split=='test':
        from .gate import validate
        if not a.test_gate:p.error('Test evaluation requires --test-gate from validation smoke')
        validate(a.test_gate,a.registry,a.degradation_manifest)
    evaluate_entry(entry,a.degradation_manifest,a.split,a.output,a.subset,a.batch_size,a.workers)


if __name__=='__main__':main()
