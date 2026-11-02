# aq_visibility — Attribute Queries + Visibility (thesis matrix)

Clean package for the official C0–C5 thesis ablation (Attribute Query + Visibility).
Scope of this repo: `aq_visibility` + `_my_model_8` (C0/C1 base).

## Variants

| ID | Config | Meaning |
|----|--------|---------|
| C0 | `configs/c0_base.yaml` | `_my_model_8` base |
| C1 | `configs/c1_base_occaug.yaml` | base + region occ aug |
| C2 | `configs/c2_aq.yaml` | 57 attribute queries |
| C3 | `configs/c3_aq_vis.yaml` | + visibility (`λ_v=0.5`) |
| C4 | `configs/c4_aq_vis_consist.yaml` | + consistency (`λ_c=0.2`) |
| C5 | `configs/c5_bg_ablation.yaml` | optional background ON |

Protocol: [`docs/aq_visibility/protocol.md`](../docs/aq_visibility/protocol.md).

## Quick start

```bash
# 1) Build occlusion pairs (after fixed split)
bash experiments/aq_matrix/build_data.sh

# 2) Full matrix on GPU host
BATCH_SIZE=16 EPOCHS=30 bash experiments/aq_matrix/run_all.sh

# 3) Results
# docs/aq_visibility/results.md
# docs/aq_visibility/gap_analysis.md
```

## Layout

```
aq_visibility/
  configs/     # YAML descriptors
  model/       # VMamba + text + AQ decoder (+ optional background)
  dataset/     # PAR + occlusion pairs + C1 occ aug
  losses/      # L_attr + L_vis + L_consist
  train/       # train.py (C2–C5), train_c0_c1.py
  eval/        # evaluate.py
  metrics/
data/occlusion/
experiments/aq_matrix/
```

## Critical correctness notes

- C3/C4 forward **occluded** images whenever `use_visibility` or `lambda_c > 0`.
- Test loaders return distinct `img_occ` when pairs exist.
- Clean originals are **not** assumed `vis=1` (`vis_valid=0` if unpaired).
