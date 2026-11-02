# AQ Visibility — gap analysis (Phase 6)

Multi-scale is **deferred** unless shoes/small-attr regions lag clearly.

| Variant | overall_mA | head | upper | lower | shoes |
|---------|------------|------|-------|-------|-------|
| C0 | 0.7751 | 0.8067 | 0.8226 | 0.8297 | 0.7770 |
| C1 | 0.7674 | 0.7947 | 0.8134 | 0.8261 | 0.7696 |
| C2 | 0.8012 | 0.8199 | 0.8419 | 0.8457 | 0.7840 |
| C3 | 0.8024 | 0.8216 | 0.8420 | 0.8521 | 0.7836 |
| C4_reaudit | 0.8024 | 0.8163 | 0.8408 | 0.8517 | 0.7845 |
| C4b | 0.7807 | 0.8068 | 0.8205 | 0.8382 | 0.7642 |
| C3_MS | 0.8105 | 0.8283 | 0.8472 | 0.8527 | 0.8160 |
| C4b_MS | 0.7886 | 0.8171 | 0.8324 | 0.8312 | 0.7801 |

## Multi-scale decision

- Evidence: C0.shoes_mA=0.7770, C4_reaudit.shoes_mA=0.7845, gap=-0.0075
- Threshold: 0.04
- **Decision: KEEP single-scale (deferred)**

