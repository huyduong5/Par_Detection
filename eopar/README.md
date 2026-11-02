> Verification update (2026-11-01): **pilot blocked** after 64-pair visual review
> found anatomically invalid fixed-band supervision on cropped/posed people.
> See `research/verification/20261101_v2/REPORT.md`. 48 CPU tests pass; real GPU
> diagnostics are pending GPU availability. Old direct-training examples below
> are historical and now fail closed without the new verification/gate flags.
> No full training is authorized by a pilot gate.

# EOPAR — Evidence-Observability PAR

Research prototype developed independently of the existing AQ baseline. The
8–12 week program and open gates are in `research/roadmap.md`. Novelty, Q1
eligibility, natural-image gains and SOTA are **not established**.

## Model and supervision
Wraps the existing frozen-backbone VMamba/query model, preserving classification
logits. A separate LayerNorm/MLP predicts observability. No probability product
and no negative relabeling under occlusion. Classification uses the original
weighted BCE+ASL, legacy synthetic visibility, and stop-gradient consistency on
intact evidence. P1 adds observability proxy supervision; P2 adds relative ranking
after evidence removal; P3 adds target-region outside interventions. P1/P2/B0/B1
share the same synthetic image policy; P3 changes the intervention distribution
explicitly. This is an ablation factor, not an identical-data comparison.

Original visibility is never assumed 1. Soft band targets (.85 intact, decreasing
with severity for affected regions) are **proxies**, not observed human labels.
Ranking does not guarantee that a clean image was visible. Natural annotation
validation is necessary. Bands/global exclusions and deterministic image pairing
are covered by tests. Pair-specific masks are created after a shared flip and
brightness transform; independent crops/random erasing are disabled.

## Execution (repository root, pedestrian-par environment)
```bash
python -m unittest discover -s eopar/tests -v
python -m eopar.scripts.reference_metrics --output eopar/research/metric_reference_check.json
python -m eopar.scripts.audit_data --pkl MSP60k/SUBMIT/dataset_random.pkl --images MSP60k/SUBMIT/images --output eopar/research/msp60k_split_audit.json
python -m eopar.scripts.collect_baselines --output eopar/research/baseline_summary.json
CUDA_VISIBLE_DEVICES=0 python -m eopar.train.run --config eopar/configs/smoke.json --output eopar/outputs/smoke_manual
CUDA_VISIBLE_DEVICES=0 python -m eopar.train.run --config eopar/configs/P3.json --output eopar/outputs/P3_s42 --stop-after 5
CUDA_VISIBLE_DEVICES=0 python -m eopar.train.run --config eopar/configs/P3.json --output eopar/outputs/P3_s42 --resume
CUDA_VISIBLE_DEVICES=0 python -m eopar.eval.run --checkpoint eopar/outputs/P3_s42/best.pth --split val --family noise --output eopar/outputs/P3_s42/val_noise.json
```

Queue commands (use a fresh output per worker):
```bash
nohup python -m eopar.scripts.queue --gpu 0 --variant B1 --run-pilot --output eopar/outputs/initial_B1 > /tmp/eopar_B1_queue.log 2>&1 < /dev/null &
nohup python -m eopar.scripts.queue --gpu 1 --variant P3 --run-pilot --output eopar/outputs/initial_P3 > /tmp/eopar_P3_queue.log 2>&1 < /dev/null &
```
Queue scripts never kill processes and stop after smoke plus a five-epoch pilot.
They do not automatically conduct the whole research program or notify anyone.

## Reproducibility and evaluation
- Pilot stops after epoch 5 of a fixed 30-epoch schedule; resume preserves horizon.
- Checkpoints restore model, optimizer, scheduler, scaler, epoch, RNG and loader
  state. Config/source mismatch fails closed; resume is at epoch boundaries.
- Manifest records source hashes (including dirty baseline files), data/region
  hashes, cached pretrained hashes, optional warmstart hash, environment and GPU.
  Exact reproducibility is not promised across different hardware/kernel builds.
- B0=C3_MS, B1=C4b_MS, batch 16 × accumulation 4; starts from pretrained backbones,
  not historical best checkpoints. Optional warmstarts must be reported separately.
- Standard metric uses strict `p > .5`, per-label balanced accuracy, instance IoU,
  precision/recall, **mean per-instance F1**. Harmonic-of-mean F1 and element
  accuracy retain explicit legacy names. Official reference parity is checked.
- Prediction NPZ stores names, attributes, labels, raw probabilities, observability
  and masks. Risk–coverage counts attribute decisions, accepts entire tied-score
  groups and reports actual coverage; score scales are not assumed calibrated.
- `eval.calibrate` requires a validation JSON/NPZ pair. Calibrated decoding remains
  separate from fixed-.5 tables. Test data must never select checkpoints/thresholds.
- `eval.natural` uses adjudicated human labels and the **clean single-image** obs
  output from NPZ. Partial/uncertain are reported and excluded only from binary
  observability AUC, never standard PAR classification metrics.

## Current limitations
PA100K/RAPv1 need licensed source data, region maps and explicit missing-label
adapters; binary MSP loader rejects missing labels. No automatic cross-dataset
label mapping. No training labels are fabricated from model outputs. Human labels,
fulltext novelty review, official nearest-method reproduction, statistical main
study, manuscript and JCR verification remain
separate tasks. See the ledger for precise status.

Natural annotation candidates: `research/annotation/annotator_a.csv` is a blank human sheet, not ground truth. `eval.run --exclude research/annotation/deduplicated_eval_exclusions.json` produces a separately named supplementary evaluation. `eval.run --thresholds <artifact.json>` adds a calibrated section without replacing standard metrics. `eval.bootstrap` compares paired saved predictions with optional cluster IDs.

The real VMamba/Mamba stack needs a visible CUDA driver even for module import (Triton); CPU unit tests use lightweight stand-ins. Full backbone forward/backward is validated by the queued GPU smoke, not by the CPU toy tests.

## Verified release workflow (v2)
The runtime uses an isolated backbone copy in `eopar.models.backbone`, leaving AQ
sources unchanged. Pretrained and tokenizer cache files must match
`configs/pretrained_lock.json` and are loaded offline with structural checks.
Run from the frozen release root rather than changing live training sources.

CPU/data verification:
```bash
python -m eopar.verification.cpu --output /absolute/verification/report
python -m eopar.verification.data --config eopar/configs/P3.json --output /absolute/verification/report
```
A bounded smoke additionally needs `--diagnostic-proxy` and <=4 train batches,
<=2 validation batches and <=2 epochs. It cannot authorize a pilot. Natural
anatomical validity cannot be inferred from these smoke tests. `geometry_manifest`
is required for a real P1/P2/P3 pilot; unverified bands are excluded from region
supervision. Never fill that file using unchecked body-part assumptions.

Pilot CLI now requires `--gate-report` and an explicit `--stop-after 5` or less.
The gate binds source snapshot, configuration, inputs and environment to completed
CPU/data/GPU reports. A future scientific full-training approval is separate.
Frozen-backbone eval is a deliberate training-policy change: historical scores
cannot substitute for newly matched B1 comparisons.
