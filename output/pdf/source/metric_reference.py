"""Explicit metric names; legacy element accuracy is not instance IoU."""
import numpy as np


def par_metrics(probability, labels, threshold=.5):
    p, y = np.asarray(probability), np.asarray(labels)
    if p.shape != y.shape or p.ndim != 2 or not p.size:
        raise ValueError('Expected equal nonempty [N,A] arrays')
    t = np.asarray(threshold)
    if t.shape not in ((), (p.shape[1],)) or not np.isfinite(t).all() or (t < 0).any() or (t > 1).any():
        raise ValueError('Invalid threshold shape/range')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected finite predictions and binary labels')
    pred, truth = p > threshold, y == 1
    tp = (pred & truth).sum(0)
    tn = (~pred & ~truth).sum(0)
    positive, negative = truth.sum(0), (~truth).sum(0)
    attr = .5 * (tp / np.maximum(positive, 1) + tn / np.maximum(negative, 1))
    itp = (pred & truth).sum(1)
    pp, gt = pred.sum(1), truth.sum(1)
    precision = np.mean(itp / np.maximum(pp, 1))
    recall = np.mean(itp / np.maximum(gt, 1))
    return dict(mA=float(attr.mean()), per_attribute_mA=attr.tolist(),
                positive_support=positive.tolist(), negative_support=negative.tolist(),
                instance_accuracy=float(np.mean(itp / np.maximum(pp + gt - itp, 1))),
                instance_precision=float(precision), instance_recall=float(recall),
                instance_f1_harmonic=float(2 * precision * recall / max(precision + recall, 1e-12)),
                instance_f1_mean=float(np.mean(2 * itp / np.maximum(pp + gt, 1))),
                legacy_element_accuracy=float((pred == truth).mean()),
                empty_positive_label_samples=int((gt == 0).sum()),
                threshold=np.asarray(threshold).tolist())


def risk_coverage(probability, labels, score):
    """Attribute-decision error, grouped by score ties for order invariance.

    Coverage excludes no decisions silently; zero-denominator groups are absent.
    AURC uses a right-continuous step sum, including the first tie group.
    """
    p, y, s = map(np.asarray, (probability, labels, score))
    if p.shape != y.shape or p.shape != s.shape or not p.size:
        raise ValueError('Scores must match nonempty predictions')
    if not np.isfinite(s).all() or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Nonfinite scores/predictions or invalid labels')
    error = ((p > .5) != y).ravel().astype(float)
    flat = s.ravel()
    order = np.argsort(-flat, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(flat[order]) != 0), len(order) - 1]
    counts = ends + 1
    risks = np.cumsum(error[order])[ends] / counts
    coverages = counts / len(order)
    aurc = float(np.sum(risks * np.diff(np.r_[0., coverages])))
    points = {}
    for target in (.8, .9, .95):
        i = min(np.searchsorted(coverages, target), len(ends) - 1)
        points[str(target)] = dict(coverage=float(coverages[i]), risk=float(risks[i]),
                                  accuracy=float(1 - risks[i]), threshold=float(flat[order][ends[i]]))
    return dict(aurc=aurc, unit='attribute_decision', tie_policy='accept_entire_score_group',
                decisions=int(len(order)), at_coverage=points)


def evaluate_arrays(probability, labels, observability=None):
    p = np.asarray(probability).clip(1e-7, 1 - 1e-7)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    selectors = {'binary_max_probability': np.maximum(p, 1 - p), 'negative_entropy': -entropy}
    if observability is not None:
        selectors['observability'] = observability
    return {'par': par_metrics(probability, labels),
            'selective': {name: risk_coverage(probability, labels, s) for name, s in selectors.items()}}
