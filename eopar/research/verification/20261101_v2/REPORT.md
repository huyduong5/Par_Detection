# EOPAR verification — 2026-11-01, release 20261101_v2

**Decision: NOT READY FOR PILOT OR LONG TRAINING.**

The two original waiting EOPAR workers were stopped after checking command lines,
waiting state and absence of children. Seed44 and Mamba-YOLO jobs were preserved.
The tested source is isolated on the server at:
`eopar/releases/20261101_v2/repo/`. Original AQ/VMamba source was not edited.

## Results and limits

| Phase | Result | Evidence / remaining requirement |
|---|---|---|
| 0: isolation | PASS | worker_stop.json, snapshot_before.json; source snapshot separated from live baseline jobs |
| 1: config/pretrained/environment | CPU checks PASS; GPU pending | Fail-closed offline loaders, cache lock, config negative tests; real CUDA construction/loading still requires idle GPU |
| 2: data | Structural PASS, anatomical FAIL | All 60,122 images decoded; no decode errors; 43 duplicate occurrences disclosed. 64 visual pairs reviewed; several cropped/posed images invalidate fixed-band anatomical supervision |
| 3: objective/gradients | CPU PASS; AMP pending | Production criterion independently recomputed; numerical gradients, mask weights, ablations, extremes and teacher detachment checked |
| 4: real model/branches | PENDING GPU | Real-backbone parity, graph effects, trainable gradient audit and strict pretrained integration are scheduled, not claimed passed |
| 5: trainer/resume | CPU PASS; real-backbone pending | 15 batch/accumulation combinations, partial batch oracle, new-process continuous-vs-resumed training, failed checkpoint write and corruption tests |
| 6: evaluator | CPU PASS | OpenPAR 9-case reference comparison; artifact/threshold validation, batch invariance, AUC/AURC hand cases and annotation checks |
| 7: GPU capacity/stability | PENDING; full epoch blocked by data | Both GPUs busy with existing jobs. Only bounded diagnostics will run when idle; no full verification epoch/pilot while anatomical validity fails |
| 8: launch gate | CPU guard tests PASS; authorization DENIED | Missing/stale gates and direct unbounded diagnostics/pilots are rejected; no pilot_gate.json PASS is issued |

48 CPU tests passed on the final candidate in the server environment. This count
does not include GPU tests. Parameterized tests additionally cover 15 accumulation
cases and 128 region/severity examples. Exact IDs, duration and runtime/test hashes
are in cpu.json; assertions and fixtures are preserved in the snapshot source.

## Anatomical failure found by visual inspection

The patch placement is mechanically aligned, but a cropped torso image can have
its bottom strip labeled `shoes`; bent or seated poses also break the fixed bands.
Examples and names are in visual_review.json, including sample indices 4, 17,
24, 25, 35 and 55. This is an inspection finding, not an independently annotated
natural-visibility benchmark. Contact sheets are pairs_00.png through pairs_07.png.

P1/P2/P3 pilots are therefore blocked until fixed-band validity is supported by
verified per-image metadata, or the region generator is replaced by a validated
anatomical-region pipeline. No model prediction is being treated as ground truth.

The new `geometry_manifest` format is an image-name → list-of-valid-fixed-bands
JSON mapping. It certifies the validity of the **exact band coordinates**, not
merely body-part presence. Unknown/unverified images or invalid affected bands
have visibility, consistency and ranking supervision masked out. Training and
validation must each cover all four regions; a validation-only manifest cannot
silently qualify the training split. The manifest and images are checksummed.

## Corrections made

- Independent EOPAR backbone copy with strict, offline pretrained loading; missing
  weights/keys/shapes/checksums fail instead of silently freezing random weights.
- Actual weight/tokenizer files and compiled-extension/environment identities are
  pinned; runtime source and test-source hashes are recorded separately.
- Frozen backbones remain eval during training without detaching trainable input
  gradients. Disconnected Cross-Mamba weights are excluded from optimization;
  its AQ-logit contribution is not claimed and is scheduled for GPU confirmation.
- Severity weights survive the masked reduction. Ranking validity and weights
  are separate; global/unverified evidence remains excluded.
- Partial accumulation windows weight microbatch losses by example count. This
  does not make per-microbatch weighted BCE mathematically equivalent to a
  full-batch BCE with separately estimated class weights.
- Validation precision is explicit and defaults to AMP on CUDA; no automatic
  change to training microbatch size. Batch16 approval still needs the GPU test.
- Checkpoints and JSON use temporary files plus atomic publication; resume checks
  data, geometry, attributes, source/environment, optimizer, scaler and RNG.
- Prediction arrays/calibration provenance are validated; JSON is the completion
  marker. Mean-instance F1 and legacy F1/element accuracy retain distinct names.
- Smoke uses the selected variant. Direct training and workers require the gate;
  diagnostic bypass is restricted to at most two four-batch epochs. Verification
  mode allows at most one full epoch. A pilot gate never authorizes 30 epochs.

## GPU follow-up

Supervisors wait for no compute processes, low utilization and >=22GB free, then
run bounded real-backbone tests: selective-scan backward, all variants, parity,
branch interventions, AMP/overflow handling, fixed-data learning, fresh-process
resume, memory profiles and 105-batch train/validation checks. They stop on error.
Because visual review failed, they do not run full verification epochs and do not
launch pilots. worker0.json and worker1.json are the live status records.

After the anatomical-data issue is resolved, rerun the affected data/model/GPU
checks on a new frozen release. Full one-epoch B1/P3 checks and both GPU reports
must pass before a matching pilot-only gate can be assembled. Scientific novelty,
natural-label validation and 30-epoch research approval remain separate gates.
