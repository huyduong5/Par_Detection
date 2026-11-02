# Post-hoc threshold calibration (no retraining)

Thresholds fitted on **val** only (per attribute, max balanced accuracy), applied to **test**. `fixed0.5` is the standard protocol number; calibrated rows must be reported as a separate decoding.

| Variant | decoding | mA | F1 | occ_mA | occ_F1 |
|---------|----------|----|----|--------|--------|
| C3_MS | fixed0.5 | 0.8104 | 0.8046 | 0.6997 | 0.7035 |
| C3_MS | thr_val | 0.8436 | 0.7079 | 0.7303 | 0.5168 |
| C3_MS | thr_val+grp | 0.8433 | 0.6958 | 0.7284 | 0.4907 |
| C4b_MS_bs16 | fixed0.5 | 0.8102 | 0.8029 | 0.7363 | 0.7843 |
| C4b_MS_bs16 | thr_val | 0.8442 | 0.6947 | 0.7977 | 0.6106 |
| C4b_MS_bs16 | thr_val+grp | 0.8431 | 0.6824 | 0.7971 | 0.5972 |
| C3 | fixed0.5 | 0.8024 | 0.7988 | 0.7444 | 0.7197 |
| C3 | thr_val | 0.8416 | 0.6971 | 0.7554 | 0.5153 |
| C3 | thr_val+grp | 0.8415 | 0.6919 | 0.7560 | 0.4974 |
| C3_MS_s43 | fixed0.5 | 0.8072 | 0.8018 | 0.7077 | 0.7024 |
| C3_MS_s43 | thr_val | 0.8394 | 0.7095 | 0.7289 | 0.5584 |
| C3_MS_s43 | thr_val+grp | 0.8381 | 0.6992 | 0.7288 | 0.5374 |
| C4b_MS_s43 | fixed0.5 | 0.8066 | 0.8022 | 0.7443 | 0.7890 |
| C4b_MS_s43 | thr_val | 0.8414 | 0.7031 | 0.7996 | 0.6329 |
| C4b_MS_s43 | thr_val+grp | 0.8402 | 0.6937 | 0.8003 | 0.6255 |
| C3_s43 | fixed0.5 | 0.8041 | 0.7961 | 0.7461 | 0.7427 |
| C3_s43 | thr_val | 0.8379 | 0.6959 | 0.7658 | 0.5207 |
| C3_s43 | thr_val+grp | 0.8377 | 0.6884 | 0.7647 | 0.5029 |
| C4_reaudit | fixed0.5 | 0.8024 | 0.7994 | 0.7317 | 0.7860 |
| C4_reaudit | thr_val | 0.8389 | 0.6946 | 0.7897 | 0.6705 |
| C4_reaudit | thr_val+grp | 0.8389 | 0.6870 | 0.7901 | 0.6656 |
| C4b_bs16 | fixed0.5 | 0.8030 | 0.7991 | 0.7330 | 0.7885 |
| C4b_bs16 | thr_val | 0.8436 | 0.6927 | 0.7970 | 0.6037 |
| C4b_bs16 | thr_val+grp | 0.8432 | 0.6848 | 0.7983 | 0.6018 |
| C4b_s43 | fixed0.5 | 0.8047 | 0.7976 | 0.7405 | 0.7862 |
| C4b_s43 | thr_val | 0.8377 | 0.6812 | 0.7984 | 0.5855 |
| C4b_s43 | thr_val+grp | 0.8366 | 0.6745 | 0.7974 | 0.5802 |

## Exclusive groups (label check on val)

- Fat, Normal, Thin: exactly-one rate 1.0000 → used
- Child, Adult, Elderly: exactly-one rate 1.0000 → used
- Front, Back, Side: exactly-one rate 0.9993 → used

## Attributes most changed by calibration (first variant, clean mA)

| Attribute | fixed0.5 | thr_val | thr_val+grp | threshold |
|-----------|----------|---------|-------------|-----------|
| Normal | 0.5000 | 0.6663 | 0.7237 | 0.88 |
| Paper bag | 0.5286 | 0.6820 | 0.6820 | 0.01 |
| Thin | 0.6124 | 0.7734 | 0.7328 | 0.01 |
| Scarf | 0.8025 | 0.9171 | 0.9171 | 0.01 |
| Others | 0.6427 | 0.7397 | 0.7397 | 0.01 |
| Smoking | 0.7265 | 0.8102 | 0.8102 | 0.01 |
| Other shoes | 0.7508 | 0.8344 | 0.8344 | 0.12 |
| Bald | 0.7175 | 0.8000 | 0.8000 | 0.06 |
| Suit | 0.7301 | 0.8090 | 0.8090 | 0.04 |
| Short skirt | 0.8298 | 0.9042 | 0.9042 | 0.04 |
| Vest | 0.7334 | 0.8052 | 0.8052 | 0.20 |
| Hands behind back | 0.7657 | 0.8303 | 0.8303 | 0.09 |

_(table above: C3_MS)_
