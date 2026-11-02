# AQ Visibility — results

Protocol: `docs/aq_visibility/protocol.md`. Metrics from measured JSON after code audit (2026-10-15/16).

| Variant | mA | Acc | Prec | Rec | F1 | occ_mA | vis_mae | consist_gap |
|---------|-----|-----|------|-----|-----|--------|---------|-------------|
| C0 | 0.7751 | 0.9195 | 0.7005 | 0.8898 | 0.7839 | — | — | — |
| C1 | 0.7674 | 0.9189 | 0.6977 | 0.8893 | 0.7819 | — | — | — |
| C2 | 0.8012 | 0.9172 | 0.6933 | 0.9047 | 0.7850 | — | — | — |
| C3 | 0.8023 | 0.9248 | 0.7182 | 0.8997 | 0.7988 | 0.7443 | 0.0085 | 0.1105 |
| C4 | 0.8011 | 0.9248 | 0.7177 | 0.9011 | 0.7990 | 0.7220 | 0.0089 | 0.0461 |
| C4_reaudit | 0.8027 | 0.9250 | 0.7183 | 0.9013 | 0.7994 | 0.7317 | 0.0091 | 0.0435 |

## Region mA (clean)

| Variant | head | upper | lower | shoes |
|---------|------|-------|-------|-------|
| C0 | 0.8067 | 0.8226 | 0.8297 | 0.7770 |
| C1 | 0.7947 | 0.8134 | 0.8261 | 0.7696 |
| C2 | 0.8199 | 0.8419 | 0.8457 | 0.7840 |
| C3 | 0.8216 | 0.8420 | 0.8523 | 0.7831 |
| C4 | 0.8173 | 0.8420 | 0.8517 | 0.7828 |
| C4_reaudit | 0.8161 | 0.8406 | 0.8518 | 0.7846 |

## Notes

- C0–C4 rows: re-eval of original checkpoints with fixed eval (region mA for C0/C1; global consistency_gap).
- **C4_reaudit**: retrained after fixing consist_mask (evidence-region only). Prefer this for C4 in thesis tables.
- Source: `aq_visibility/output/eval_json/*_test.json`

