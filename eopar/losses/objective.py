"""Separate existence, synthetic visibility and relative observability."""
import torch
import torch.nn.functional as F


def masked_mean(values, mask):
    weights = mask.to(values.dtype)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Mask weights must be finite and nonnegative')
    active = weights > 0
    safe = torch.where(active, values, torch.zeros_like(values))
    denominator = weights.sum()
    return (safe * weights).sum() / denominator.clamp_min(torch.finfo(values.dtype).eps)


def stable_consistency(clean, intervened, mask):
    target = clean.detach().sigmoid()
    entropy = F.binary_cross_entropy_with_logits(clean.detach(), target, reduction='none')
    return masked_mean(F.binary_cross_entropy_with_logits(
        intervened, target, reduction='none') - entropy, mask)


def observability_ranking(clean, intervened, mask, margin=0.15):
    # Both branches learn; detach is reserved for the attribute teacher.
    return masked_mean(F.relu(margin - clean.sigmoid() + intervened.sigmoid()), mask)


def objective(clean, occ, batch, base_criterion, config):
    # The old criterion retains classification, visibility and consistency losses.
    features = {'vis_pred': occ['legacy_visibility']}
    loss, parts = base_criterion(
        clean['attr_logits'], batch['labels'], features=features,
        vis_target=batch['vis_target'], vis_valid_mask=batch['vis_valid'],
        logits_occ=occ['attr_logits'], consist_mask=batch['intact_mask'])
    obs_c, obs_o = clean['observability_logits'], occ['observability_logits']
    if obs_o is not None:
        # Only synthetic intervened targets are anchored. Originals are not vis=1.
        target = torch.where(batch['vis_valid'].bool(), batch['vis_target'],
                             torch.zeros_like(batch['vis_target']))
        anchor = masked_mean(F.binary_cross_entropy_with_logits(
            obs_o.float(), target.float(), reduction='none'), batch['vis_valid'])
        rank = observability_ranking(obs_c, obs_o, batch.get('rank_weights', batch['affected_mask']), config['margin'])
        rank_weight = config['lambda_rank'] if config['variant'] in ('P2', 'P3') else 0.
        loss = loss + config['lambda_obs'] * anchor + rank_weight * rank
        parts.update(L_obs=anchor.detach(), L_rank=rank.detach())
    return loss, {k: float(v.detach()) for k, v in parts.items() if k != '_total'}
