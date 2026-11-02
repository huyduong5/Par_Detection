# AQ Visibility — results

Protocol: `docs/aq_visibility/protocol.md`. Metrics from measured JSON only.

| Variant | mA | Acc | Prec | Rec | F1 | occ_mA | vis_mae | consist_gap |
|---------|-----|-----|------|-----|-----|--------|---------|-------------|
| C0 | 0.7751 | 0.9195 | 0.7005 | 0.8898 | 0.7839 | — | — | — |
| C1 | 0.7674 | 0.9189 | 0.6977 | 0.8893 | 0.7819 | — | — | — |
| C2 | 0.8012 | 0.9172 | 0.6933 | 0.9047 | 0.7850 | — | — | — |
| C3 | 0.8024 | 0.9248 | 0.7184 | 0.8995 | 0.7988 | 0.7444 | 0.0085 | 0.1105 |
| C4_reaudit | 0.8024 | 0.9250 | 0.7182 | 0.9013 | 0.7994 | 0.7317 | 0.0091 | 0.0435 |
| C4b | 0.7807 | 0.9349 | 0.7561 | 0.8917 | 0.8183 | 0.7191 | 0.0080 | 0.0467 |
| C3_MS | 0.8105 | 0.9266 | 0.7247 | 0.9043 | 0.8046 | 0.6998 | 0.0106 | 0.1093 |
| C4b_MS | 0.7886 | 0.9375 | 0.7648 | 0.8973 | 0.8258 | 0.7206 | 0.0074 | 0.0432 |

## Region mA (clean)

| Variant | head | upper | lower | shoes |
|---------|------|-------|-------|-------|
| C0 | 0.8067 | 0.8226 | 0.8297 | 0.7770 |
| C1 | 0.7947 | 0.8134 | 0.8261 | 0.7696 |
| C2 | 0.8199 | 0.8419 | 0.8457 | 0.7840 |
| C3 | 0.8216 | 0.8420 | 0.8521 | 0.7836 |
| C4_reaudit | 0.8163 | 0.8408 | 0.8517 | 0.7845 |
| C4b | 0.8068 | 0.8205 | 0.8382 | 0.7642 |
| C3_MS | 0.8283 | 0.8472 | 0.8527 | 0.8160 |
| C4b_MS | 0.8171 | 0.8324 | 0.8312 | 0.7801 |

## Source JSON

- `aq_visibility/output/eval_json/C0_test.json`
- `aq_visibility/output/eval_json/C1_test.json`
- `aq_visibility/output/eval_json/C2_test.json`
- `aq_visibility/output/eval_json/C3_test.json`
- `aq_visibility/output/eval_json/C4_reaudit_test.json`
- `aq_visibility/output/eval_json/C4b_test.json`
- `aq_visibility/output/eval_json/C3_MS_test.json`
- `aq_visibility/output/eval_json/C4b_MS_test.json`
