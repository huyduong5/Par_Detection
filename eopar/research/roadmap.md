# Execution ledger, started 2026-11-01

This is an 8–12 week research program, not a completed publication result.
See README for runnable commands. Do not overwrite baseline jobs or outputs.

| Phase / tasks | Current artifact | Remaining acceptance gate |
|---|---|---|
| 0: discover, review 15 full texts, register 25–35 works, compare nearest losses, check JCR | literature.json (26 candidates), novelty.md | Fulltext review, citation snowballing, overlap resolution, institutional JCR/category/year verification |
| 1: collect seeds, verify evaluator, audit training and splits, freeze B0/B1 | collector, reference metric checker, split auditor, new trainer | Seed44 completion; original-checkpoint standardized re-eval; byte/perceptual/identity leakage checks; old training audit |
| 2: MSP random/cross-domain, PA100K/RAPv1 access, interventions, natural labels | deterministic aligned intervention loader, annotation tools/rubric | External dataset authorization/download/adapters; 800 human labels; overlap/adjudication; test lock |
| 3: package, heads/losses, selective output, smoke/overfit, gradients | model/loss/eval/tests, smoke config | Real-backbone GPU smoke and tiny-data overfit; anti-collapse validation |
| 4: B0/B1/P1/P2/P3, 5-epoch pilots, rank weights .05/.1/.2, <=2 full candidates | five configs, idle-GPU queue, fixed 30-epoch scheduler horizon | Novelty/data gates; pilots; val-only candidate selection |
| 5: three datasets × three seeds, cross-domain, nearest methods, ablations, CI/cost | standardized prediction NPZ, confidence/entropy selectors, manifests | Runs and independent reproductions; cluster bootstrap, latency steady-state profiling |
| 6: severity/region/frequency, AURC, errors, background, failures | held-out noise/severity evaluation, natural-label evaluator | Real experimental analyses and background intervention tests |
| 7: manuscript, claims audit, reproducibility, tables, journal check | manuscript outline below | Measured tables, released compatible artifacts, complete manuscript and journal decision |

## Run order and resource budget
1. Existing seed44 and unrelated jobs retain priority. CPU checks use one thread.
2. Workers wait for no GPU compute processes, <=5% utilization and >=18GB free;
   use cooperative locks and recheck. No process is stopped to make room.
3. GPU0: B1 5-epoch integration pilot. GPU1: P3 5-epoch integration pilot. Both
   run P3 smoke first; pilot scores are exploratory until Phase 0/2 gates pass.
4. P1 and P2 follow after inspecting the initial paired pilot. Select lambda_rank
   from .05/.1/.2 only for the strongest validation configuration.
5. Main study: seeds 42/43/44; three datasets; identical image/batch/training
   budgets within each paired comparison. Cross-domain splits use training pairs
   generated from that split, never random-split pair files.
6. At most two full candidates. Screen for <=0.5pp clean mA regression and either
   >=1pp occluded mA gain or >=10% relative AURC reduction on validation; coverage
   and natural validation remain mandatory. These are internal goals, not results.
7. Budget 800–1200 GPU-hours, revised after measured epoch timings; no unlimited
   retry loop. Queue expires after 72h and stops on failure, leaving logs/status.

## Manuscript outline
Problem and distinct targets → related uncertainty/attention/visibility methods
→ intervention objective and assumptions → dataset/annotation protocol → paired
ablation and reproducibility → cross-domain and natural-image evaluation →
failure cases, resources and limits. Model architecture diagrams and measured
tables are produced only after experiments; no placeholder numbers as results.

## Human/external dependencies
Natural labels require real annotators; scripts cannot invent them. PA100K and
RAPv1 are not presently verified as available on this account. JCR evidence may
require the institution's subscription. A reproduction without official code is
explicitly labeled reimplementation, not official reproduction.

## Audit finding (2026-11-01)
Full SHA256 image audit found 43 cross-partition duplicate occurrences (7 train–val, 30 train–test, 6 val–test). Official splits are preserved. Supplementary exclusion manifests and blinded candidate sheets are generated under annotation/. These exclusions address byte identity only, not near-duplicates or shared identities.
