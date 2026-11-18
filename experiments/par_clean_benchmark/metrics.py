"""PAR metrics over valid annotations and image/identity paired bootstrap."""
import numpy as np


def scores(p, y, valid):
    if p.shape!=y.shape or p.shape!=valid.shape or not p.size or p.ndim!=2:
        raise ValueError('Invalid prediction shape')
    if not np.isfinite(p).all() or not np.isin(y,[0,1]).all():
        raise ValueError('Invalid scores or labels')
    valid = valid.astype(bool)
    pos,neg = (y==1)&valid,(y==0)&valid
    pred = (p>.5)&valid
    tp = (pred&pos)
    tn = (~pred&neg)
    positives,negatives = pos.sum(0),neg.sum(0)
    attribute_ma = .5*(tp.sum(0)/np.maximum(positives,1)+tn.sum(0)/np.maximum(negatives,1))
    supported = valid.any(0)
    rows = valid.any(1)
    if not supported.any():
        raise ValueError('No valid annotation')
    inter = tp.sum(1)[rows]
    predicted,truth = pred.sum(1)[rows],pos.sum(1)[rows]
    aps = []
    for c in range(y.shape[1]):
        rank = np.argsort(-p[valid[:,c],c],kind='stable')
        if not positives[c]:
            aps.append(None)
            continue
        targets = y[valid[:,c],c][rank]
        ranked = p[valid[:,c],c][rank]
        ends = np.r_[np.flatnonzero(np.diff(ranked)),len(rank)-1]
        hits = np.cumsum(targets)[ends]
        aps.append(float(np.sum(np.diff(np.r_[0,hits])*hits/(ends+1))/positives[c]))
    return dict(mA=float(attribute_ma[supported].mean()), per_attribute_mA=attribute_ma.tolist(),
                positive_support=positives.tolist(),negative_support=negatives.tolist(),
                instance_accuracy=float(np.mean(inter/np.maximum(predicted+truth-inter,1))),
                instance_precision=float(np.mean(inter/np.maximum(predicted,1))),
                instance_recall=float(np.mean(inter/np.maximum(truth,1))),
                instance_f1_mean=float(np.mean(2*inter/np.maximum(predicted+truth,1))),
                macro_ap=float(np.mean([a for a in aps if a is not None])) if positives.any() else None,
                per_attribute_ap=aps, valid_instances=int(rows.sum()), threshold=.5)


def paired_bootstrap(left, right, y, valid, draws=2000):
    pos,neg = (y==1)&valid,(y==0)&valid
    d = (right>.5).astype(np.int8)-(left>.5)
    packed = np.concatenate((d*pos,-d*neg,pos,neg),axis=1).astype(np.float64)
    rng = np.random.default_rng(42)
    active = valid.any(0)
    samples = []
    n = len(y)
    for offset in range(0,draws,32):
        weights = np.stack([np.bincount(rng.integers(n,size=n),minlength=n)
                            for _ in range(min(32,draws-offset))])
        tp,tn,p,q = np.split(weights@packed,4,axis=1)
        samples.extend(50*(tp[:,active]/np.maximum(p[:,active],1)+tn[:,active]/np.maximum(q[:,active],1)).mean(1))
    return dict(delta_mA_pp=100*(scores(right,y,valid)['mA']-scores(left,y,valid)['mA']),
                ci95_pp=np.quantile(samples,[.025,.975]).tolist(),draws=draws,unit='image',seed=42)
