# Clean PAR benchmark

Train C3 and C3_MS on the official PA-100K split or PETA partition 0.
Both use ordinary 448 px preprocessing and the same training schedule.

Initialize the pinned backbone sources with `git submodule update --init`.
Use the CUDA environment with VMamba selective-scan and Mamba extensions,
plus Pillow, NumPy, SciPy and iterative-stratification for dataset preparation.
Set the local pretrained paths and hashes in `eopar/configs/pretrained_lock.json`
and the existing project paths in `eopar/configs/B0.json`. The trainer uses
that config's validated project paths and verifies the files actually loaded.

Prepare each dataset from its extracted image directory and original annotation:

```sh
python -m experiments.par_clean_benchmark.data --dataset pa100k \
  --annotation /path/to/annotation.mat --images /path/to/images --output /path/to/pa100k
python -m experiments.par_clean_benchmark.checks -v
python -m experiments.par_clean_benchmark.queue --output /path/to/outputs \
  /path/to/pa100k/manifest.json /path/to/peta/manifest.json
```

Use `--dataset peta` and `PETA.mat` for PETA. The queue runs smoke tests,
30-epoch training, test evaluation and report generation. It uses two GPUs
and checks the GPU locks before launching each run.

The original server checkpoints bind their runtime source hashes. Use the
archived server release when resuming those checkpoints. This branch contains
the reporting fix that distinguishes the training seed from the bootstrap seed.
