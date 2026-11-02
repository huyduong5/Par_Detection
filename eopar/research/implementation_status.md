# Implementation handoff — 2026-11-01

Deployed to `/mnt/data8tb/huydq/projects/Pedestrain-Attribute-Recognition/eopar/`
on sol1. Local source mirrors this new package only. Existing AQ code, historical
results, official partitions and running jobs were not changed.

## Verified
- 18 CPU unit/integration tests passed in the server's pedestrian-par environment.
  Includes lightweight tiny-data overfit, gradient flow, teacher detachment,
  empty/global/uncertain masks, deterministic image pairing, checkpoint strictness,
  RNG/optimizer/scheduler/scaler recovery, calibration provenance, annotation
  validation, metric semantics, AUC ties and risk–coverage ties.
- Official OpenPAR evaluator comparison: 9 cases passed at tolerance 1e-10,
  including threshold equality and empty positive/prediction cases.
- Real MSP60K validation loader: 6002 images, tensor shape 3×448×448, 57 labels.
- Full byte audit: 43 duplicate occurrences across partitions (7 train–val,
  30 train–test, 6 val–test). Official split preserved; supplementary exclusion
  JSON created. This does not rule out identity/near-duplicate leakage.
- Natural annotation screening pool: 400 validation + 1200 test candidates,
  blank labels and no predictions, after byte-duplicate exclusion. This is not
  an annotated naturally occluded dataset.

## Running service state at handoff
GPU0 worker B1 and GPU1 worker P3 are detached background processes with
`waiting_for_idle_gpu` status. They require no compute apps, <=5% utilization,
>=18GB free and a second idle check, then run smoke and a five-epoch pilot.
They expire after 72 hours, stop on failure, and do not kill existing jobs.
Inspect `outputs/initial_B1/queue_status.json` and
`outputs/initial_P3/queue_status.json`, plus `/tmp/eopar_B1_queue.log` and
`/tmp/eopar_P3_queue.log` on the server. Local copies are snapshots only.

## Not yet verified/completed
Full model CUDA forward/backward remains the queued GPU smoke gate. Importing
the real Mamba stack with CUDA hidden failed because Triton requires an active
driver; this is recorded in real_model_cpu_build.txt, not counted as a pass.
CPU toy tests are not a substitute for backbone integration or research results.

26 literature candidates registered, but 15 fulltext critical reviews, full AAR
loss comparison, nearest-method reproduction and JCR evidence remain open.
No full study is launched before those gates and the natural-data review.
PA100K/RAPv1 access, missing-label adapters where needed, real human annotation,
main experiments, statistical interpretation and manuscript are not marked done.
The package supplies experimental infrastructure; the multi-week research program
is not claimed complete and no improvement/Q1 acceptance is claimed.
