# AQ Visibility — code audit

Protocol: `docs/aq_visibility/protocol.md`. Dates: 2026-10-15 (audit/fix) · 2026-10-16 (C4 retrain + eval done).

## Checklist

| ID | Check | Result |
|----|--------|--------|
| T0.1 | C3 λv=0.5 λc=0; C4 λv=0.5 λc=0.2; loss adds L_attr + λv L_vis + λc L_consist | PASS |
| T0.2 | L_vis masked by vis_valid; vis_pred taken from occluded forward; VisHead is sigmoid | PASS |
| T0.3 | L_consist only on intact evidence-region attrs | FAIL then fixed |
| T0.4 | Unpaired samples keep vis_valid=0 (not vis=1) | PASS |
| T0.5 | C0/C1 train via `_my_model_8`; C1 swaps in RegionOcclusionAug | PASS |
| T0.6 | C0/C1 eval omitted region mA | FAIL then fixed (eval only) |
| T0.7 | `strict=False` hid missing query weights | FAIL then fixed (raise if query keys missing) |

## Defects fixed

1. **consist_mask included global attributes.** Builder used `~affected`, so global attrs entered L_consist. Protocol: only intact evidence regions. Loader recomputes mask from `body_region_map.json`. C4 retrained → `output/C4_reaudit/`.
2. **Test consistency_gap was mean-of-batch-means.** Now one masked mean after concat.
3. **C0/C1 JSON had no region mA.** `eval_test_json.py` now emits region metrics.
4. **Checkpoint load.** Eval prints missing keys and raises if query weights are missing.

## Retrain / eval status

| Item | Status |
|------|--------|
| Unit tests (sol1) | 11/11 passed |
| Re-eval C0–C4 original ckpts | Done (`EVAL_ALL_DONE`) |
| C4_reaudit train | Done — best val mA **0.7993** |
| C4_reaudit test eval | Done |

## Measured test (after audit)

| Variant | mA | occ_mA | consist_gap | shoes |
|---------|------|--------|-------------|-------|
| C0 | 0.7751 | — | — | 0.7770 |
| C1 | 0.7674 | — | — | 0.7696 |
| C2 | 0.8012 | — | — | 0.7840 |
| C3 | 0.8023 | 0.7443 | 0.1105 | 0.7831 |
| C4 (old ckpt) | 0.8011 | 0.7220 | 0.0461 | 0.7828 |
| **C4_reaudit** | **0.8027** | **0.7317** | **0.0435** | **0.7846** |

Use **C4_reaudit** as the official C4 row. Multi-scale gate (C0.shoes − C4.shoes > 0.04) still **not** fired → keep single-scale.

## Tests

```bash
python -m unittest aq_visibility.tests.test_occ_forward aq_visibility.tests.test_protocol
```

Tables: `docs/aq_visibility/results.md`, `docs/aq_visibility/gap_analysis.md`.
