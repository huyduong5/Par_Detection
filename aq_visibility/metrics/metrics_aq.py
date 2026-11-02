"""Metrics for clean / occluded PAR + visibility + consistency."""

import numpy as np


def per_attr_label_accuracy(predictions, labels, threshold=0.5):
    """Balanced accuracy per attribute, shape (num_attrs,). mA is its mean."""
    preds_binary = (predictions >= threshold).astype(np.float32)
    num_attrs = labels.shape[1]
    attr_accuracies = np.zeros(num_attrs, dtype=np.float64)
    for i in range(num_attrs):
        tp = ((preds_binary[:, i] == 1) & (labels[:, i] == 1)).sum()
        tn = ((preds_binary[:, i] == 0) & (labels[:, i] == 0)).sum()
        p = (labels[:, i] == 1).sum()
        n = (labels[:, i] == 0).sum()
        attr_accuracies[i] = (tp / max(p, 1) + tn / max(n, 1)) / 2.0
    return attr_accuracies


def per_attr_ma_dict(predictions, labels, attributes, threshold=0.5):
    accs = per_attr_label_accuracy(predictions, labels, threshold=threshold)
    return {str(a): float(v) for a, v in zip(attributes, accs)}


def compute_metrics(predictions, labels, threshold=0.5):
    preds_binary = (predictions >= threshold).astype(np.float32)
    mA = float(np.mean(per_attr_label_accuracy(predictions, labels, threshold=threshold)))

    tp = (preds_binary * labels).sum(axis=1)
    fp = (preds_binary * (1 - labels)).sum(axis=1)
    fn = ((1 - preds_binary) * labels).sum(axis=1)
    precision = (tp / (tp + fp + 1e-8)).mean()
    recall = (tp / (tp + fn + 1e-8)).mean()
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (preds_binary == labels).mean()
    return {
        'mA': float(mA),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }


def visibility_metrics(vis_pred, vis_target, vis_valid, threshold=0.5):
    mask = vis_valid.astype(bool)
    if mask.sum() < 1:
        return {'vis_mae': float('nan'), 'vis_acc': float('nan')}
    vp = vis_pred[mask]
    vt = vis_target[mask]
    mae = float(np.abs(vp - vt).mean())
    acc = float(((vp >= threshold) == (vt >= threshold)).mean())
    return {'vis_mae': mae, 'vis_acc': acc}


def consistency_gap(p_clean, p_occ, consist_mask):
    mask = consist_mask.astype(bool)
    if mask.sum() < 1:
        return {'consistency_gap': float('nan')}
    gap = np.abs(p_clean - p_occ)[mask].mean()
    return {'consistency_gap': float(gap)}


def per_region_attr_metrics(predictions, labels, attributes, region_map, region_name, threshold=0.5):
    idx = [i for i, a in enumerate(attributes) if region_map.get(a) == region_name]
    if not idx:
        return {}
    sub_p = predictions[:, idx]
    sub_l = labels[:, idx]
    return compute_metrics(sub_p, sub_l, threshold=threshold)
