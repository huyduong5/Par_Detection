# Pedestrian Attribute Recognition

Multi-label pedestrian attribute recognition with an image backbone based on
VMamba and an attribute-query decoder conditioned by Mamba text embeddings.
C3 uses a single visual scale. C3_MS adds visual tokens at a second scale.
The clean benchmark compares both models on PA-100K and PETA.

| Path | Purpose |
|---|---|
| `aq_visibility/` | Attribute-query models and historical baseline training |
| `eopar/` | Evidence-observability prototype, configuration and checks |
| `experiments/par_clean_benchmark/` | Clean PA-100K and PETA training, evaluation and reporting |
| `experiments/par_resolution/` | Resolution benchmark scripts |
| `_my_model_8/` | Original fusion baseline |

## Setup

1. Clone with `git clone --branch main --recurse-submodules https://github.com/huyduong5/Par_Detection.git`.
   For an existing checkout, run `git submodule update --init`.
2. Use Python 3.10 and a CUDA-enabled PyTorch environment compatible with the
   pinned VMamba and Mamba sources. Build the selective-scan and Mamba
   extensions for that environment. The server runs used PyTorch 2.6.0 with
   CUDA 12.4.
3. Install the Python utilities with `pip install -r requirements.txt`.
   Install the CUDA dependencies using the instructions in the pinned backbone
   repositories.
4. Obtain the datasets and pretrained weights separately. Configure the
   weight paths and SHA-256 values in `eopar/configs/pretrained_lock.json`.
   For MSP60K experiments, also provide the images and the appropriate
   dataset split annotation at the paths in the experiment config.

Follow the [clean benchmark instructions](experiments/par_clean_benchmark/README.md)
to prepare the official splits and run the C3/C3_MS comparison. Resolution
benchmark entry points are described in
[its README](experiments/par_resolution/README.md).

Generated predictions, checkpoint weights and reports live in local output
directories. Checkpoints bind to their original source hashes for resume.
