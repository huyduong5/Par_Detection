# AQ Visibility — Experiment Protocol

Official thesis matrix (C0–C4 main table; C5 optional background ablation).

## Data

| Item | Value |
|------|--------|
| Split | Fixed `MSP60k/SUBMIT/dataset_random.pkl` (train/val/test) |
| Images | `MSP60k/SUBMIT/images` |
| Occlusion pairs | Built **after** split → `data/occlusion/occlusion_pairs_{split}.pkl` |
| Occlusion images | `data/occlusion/occlusion_images/{split}/` |
| Body map | `data/occlusion/body_region_map.json` |
| Manual audit | `data/occlusion/manual_visibility.csv` (holdout only, not in train loss) |

Rules:

- Never assume clean originals have `visibility=1`. Unpaired samples use `vis_valid=0`.
- Synth soft targets: occluded region attrs `0.15`, other evidence-region attrs `0.85`.
- Global / non-evidence attrs stay `vis_valid=0` (excluded from `L_vis`).
- Uncertain estimates (pose-lite / bad bbox) → `vis_valid=0`.

## Loss

```
L = L_attr + λ_v * L_vis + λ_c * L_consist
```

| Term | Definition |
|------|------------|
| `L_attr` | Weighted BCE (+ ASL) on attribute logits |
| `L_vis` | Masked BCE on `vis_pred` where `vis_valid=1`; predict from **occluded** image |
| `L_consist` | Mean \|σ(clean)−σ(occ)\| on `consist_mask` (attrs whose evidence region is intact) |

Fixed lambdas for thesis presets:

- C3: `λ_v=0.5`, `λ_c=0`
- C4: `λ_v=0.5`, `λ_c=0.2`

Contrastive / CLIP fuse / KD are **off** in this protocol.

## Variants

| ID | Model | Flags |
|----|--------|-------|
| C0 | `_my_model_8` GAP+MLP | base |
| C1 | `_my_model_8` + controlled occ aug | base + occlusion augmentation |
| C2 | AQ 57 queries | `use_attr_queries`, no vis/consist |
| C3 | AQ + VisHead | + `use_visibility`, `use_occlusion_pairs`, `λ_v=0.5` |
| C4 | AQ + vis + consist | + `λ_c=0.2` |
| C5 | optional | C4 + `use_background=True` (ablation only) |

### Extension matrix (2026-10)

Backbone frozen, `L_cls`, `L_vis`, `λ_v=0.5` unchanged.

| ID | Change vs C3 | Flags |
|----|--------------|-------|
| C4_reaudit | C4 retrained after consist_mask fix | `consistency_type='l1'`, `λ_c=0.2` |
| C4b | stop-grad consistency | `consistency_type='sg_bce'`, `λ_c=0.2` |
| C3_MS | multi-scale tokens into AQ | `use_multiscale=True`, `ms_stage_index=1` (28×28×384, stride 16) pooled to 16×8 = 128 extra tokens |
| C4b_MS | both | `sg_bce` + multi-scale |

`L_cons^sg = mean_mask KL(Bern(sg(σ(z_clean))) ‖ Bern(σ(z_occ)))`: clean is the teacher, only the occluded branch receives gradient, value 0 when they match.

Multi-scale tokens go only to the attribute-query decoder (key/value: 32 main + 128 stage tokens, each branch with its own learnable scale embedding); spatial cross-attention and cross-Mamba still see the 32 main tokens.

Batch: 16×4 when the GPU has ≥11 GB free at launch, else 8×8 (effective 64 in both cases); recorded in `output/<VAR>/batch.cfg`.

Runner: `experiments/aq_matrix/run_ext.sh` (smoke: `MODE=smoke`), status: `experiments/aq_matrix/check_ext.sh`.

## Training defaults

| Knob | Value |
|------|--------|
| Seed | 42 |
| Image size | 448×224 (or package default 448×448 — keep consistent per run) |
| Epochs | 30 (smoke: 1) |
| Batch | 4–28 depending on GPU |
| AMP | on |
| Multi-scale vision | deferred until Phase 6 analysis |

## Metrics (must print)

- Clean: mA, Acc, Precision, Recall, F1
- Occluded subset (pairs only): same five metrics
- Visibility: MAE, Acc@0.5 (valid attrs only)
- Consistency gap: mean \|σ(clean)−σ(occ)\| on consist mask
- Region mA: head / upper / lower / shoes

## Commands

```bash
# Build pairs (after split exists)
bash experiments/aq_matrix/build_data.sh

# Full matrix
bash experiments/aq_matrix/run_all.sh

# Aggregate
python experiments/aq_matrix/aggregate_results.py
```

Results go to `docs/aq_visibility/results.md` (measured JSON only).
