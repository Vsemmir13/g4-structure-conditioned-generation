# G4 Structure-Conditioned Generation

This directory is a separate research copy of the original G4 generation project.
The original project lives in `../../quadruplex` and is intentionally left
untouched.

The old project generated G4 sequences conditioned on EndoQuad level
`{4, 5, 6}`. This research branch replaces that target with biologically more
interpretable conditions:

- `topology_label`: `parallel`, `antiparallel`, `hybrid`
- `predicted_tm`: continuous G4STAB melting temperature estimate

The old `analysis/`, `logs/`, `checkpoints/`, and `generated/` directories were
not copied. New runs create their own outputs inside this directory.

## What Changed From `quadruplex`

The copied baseline project was changed in these places:

| Area | Original `quadruplex` | This research branch |
| --- | --- | --- |
| Condition | EndoQuad level `4/5/6` | topology, stability, or joint topology+stability |
| Dataset | BED+FASTA loaded during training | processed CSV with sequence and external labels |
| Condition input | one categorical id | generic `ConditionEncoder` for one or more categorical conditions |
| Models | LSTM, VAE, DFM CNN, DFM Transformer | same models plus DDSM |
| DDSM | absent | official-noise DDSM score model adapted to conditional G4 generation |
| Preprocessing | sequence extraction only | G4 sequence extraction, G4ShapePredictor labels, G4STAB Tm labels |
| Splits | stratified by G4 level | stratified by joint topology+stability when possible |
| Evaluation | class-wise G4-level metrics | old metric scripts kept; topology/stability evaluation is the next extension |

The central code path is now:

```text
EndoQuad BED + hg38 FASTA
    -> original EndoQuad peak sequence
    -> topology by G4ShapePredictor
    -> Tm by G4STAB
    -> stability bins
    -> paired 512 bp context window
    -> processed CSV
    -> conditional generation models
```

## Directory Layout

```text
g4_structure_conditioning/
  main.py                         # train/test entrypoint
  generate.py                     # generation from a saved checkpoint
  README.md
  RESEARCH_PLAN.md

  models/
    condition_encoder.py          # categorical condition encoder
    lstm.py                       # conditional autoregressive baseline
    vae.py                        # conditional convolutional VAE
    dfm_model.py                  # DFM CNN/Transformer backbones
    dfm_module.py                 # Lightning module for DFM
    dfm_flow_utils.py             # Dirichlet flow utilities
    ddsm_module.py                # DDSM score model and Lightning module
    melanoma_model.py             # melanoma/FBD embedder model

  preprocessing/
    annotate_structure_conditions.py
                                  # short G4 sequence -> topology/Tm/stability
    run_g4shape_predictor.py      # headless G4ShapePredictor wrapper
    run_g4stab.py                 # G4STAB wrapper
    run_full_annotation_pipeline.sh
    README.md

  metrics/
    eval.py                       # old class-wise G4-level evaluation path
    pqsfinder.py                  # pqsfinder metrics for generated JSONL files
    pqsfinder_metrics.R

  utils/
    config.py                     # default hyperparameters
    data_utils.py                 # processed dataset and sequence utilities
    model_factory.py              # model construction/checkpoint loading
    gen_metrics_callback.py       # validation/test generative metrics callback
    model_utils.py
    logging_utils.py

  external_tools/
    G4ShapePredictor/             # cloned external predictor
    G4STAB/                       # cloned external predictor
    ddsm/                         # cloned official DDSM repository
    setup_external_envs.sh
```

## Models

All models receive integer-encoded categorical conditions.

For topology-only training:

```text
parallel     -> 0
antiparallel -> 1
hybrid       -> 2
```

For stability-only training:

```text
low    -> 0
medium -> 1
high   -> 2
```

For joint conditioning:

```text
parallel + high -> [0, 2]
hybrid + low    -> [2, 0]
```

The `ConditionEncoder` has one embedding table per condition and then combines
them through a small MLP. This is different from the original DFM code, where a
single `nn.Embedding(num_cls + 1, hidden_dim)` was enough for one label. The
backbones remain the same, but the conditioning interface is more general.

| Model | `--model_type` | Notes |
| --- | --- | --- |
| LSTM | `lstm` | autoregressive baseline, uses `typer="gen"` |
| VAE | `vae` | latent-variable reconstruction/generation baseline |
| DFM CNN | `dfm` | Dirichlet Flow Matching CNN backbone |
| DFM Transformer | `dfm_transformer` | DFM Transformer backbone |
| DDSM | `ddsm` | Dirichlet Diffusion Score Model baseline |

### DDSM Implementation

`models/ddsm_module.py` uses the official DDSM repository in
`external_tools/ddsm` for the paper-like path:

- `diffusion_fast_flatdirichlet`
- `gx_to_gv`
- `Euler_Maruyama_sampler`

The DDSM score network follows the official promoter-design ScoreNet structure:

- Gaussian Fourier time embedding
- 20 residual dilated convolution blocks
- simplex-valued DNA states
- zero-mean score outputs

For paper-like training, pass a precomputed Jacobi diffusion noise table. If no
noise table is passed, the model falls back to a lightweight direct Dirichlet
noising objective intended only for debugging/smoke tests.

## 1. Prepare Environment

From this directory:

```bash
cd /path/to/BIO/research/g4_structure_conditioning
```

Install normal project dependencies in your Python environment. The external
predictors need separate environments because they have old/specific dependency
requirements:

```bash
bash external_tools/setup_external_envs.sh
```

This creates:

```text
g4shape_predictor
g4stab_env
```

On a cluster, activate the environment you use for model training before running
`main.py`. Use the dedicated conda envs only for G4ShapePredictor/G4STAB.

## 2. Annotate Short G4 Sequences

The first preprocessing stage works only with the original short EndoQuad G4
sequence. It extracts or validates this short sequence, runs G4ShapePredictor,
runs G4STAB, and stores the continuous `predicted_tm`. It does not create
512 bp model windows and does not create train/validation/test splits.

```bash
python preprocessing/annotate_structure_conditions.py \
  --input_table ../../quadruplex/data/EQ_hg38_lifted.bed \
  --file_path_seq ../../quadruplex/data/hg38.fa \
  --output_csv data/processed/g4_structure_conditions.csv
```

The output table contains the original columns plus structure annotations:

```text
chrom
start
end
level_raw
score
strand
level
length
sequence                  # short original G4 sequence
topology_label
topology_prob_parallel
topology_prob_antiparallel
topology_prob_hybrid
topology_confidence
predicted_tm
g4stab_std
```

`predicted_tm` is kept as a continuous value. Temperature thresholds are not
assigned during annotation; they can be defined later in the dataloader or
experiment configuration.

## 3. Training Dataset Construction

The training code reads `data/processed/g4_structure_conditions.csv` directly.
No second processed dataset is created. During training, `utils/data_utils.py`
adds `stability_class` from `predicted_tm`, filters examples with
`g4stab_std > 5.0`, stratifies the split by
`stability_class + topology_label`, and extracts 512 bp model windows from the
reference genome using the original EndoQuad coordinates.

The default stability thresholds are:

```text
low:    predicted_tm < 52
medium: 52 <= predicted_tm < 65
high:   predicted_tm >= 65
```

To run the annotation stage:

```bash
bash preprocessing/run_full_annotation_pipeline.sh
```

## 4. Prepare DDSM Noise And Time Weights

DDSM paper-like training needs a precomputed Jacobi diffusion noise table.
Create it once and reuse it for all DDSM runs with the same diffusion settings.

```bash
python external_tools/ddsm/presample_noise.py \
  -n 100000 \
  -c 4 \
  -t 400 \
  --max_time 4 \
  --speed_balance \
  --out_path data/ddsm_noise
```

Expected output:

```text
data/ddsm_noise/steps400.cat4.speed_balance.time4.0.samples100000.pth
```

The official DDSM training recipe also estimates time-dependent loss weights
from the training data. For the closest match to the paper implementation,
estimate them after the processed dataset is built:

```bash
python preprocessing/estimate_ddsm_time_weights.py \
  --processed_csv data/processed/g4_structure_conditions.csv \
  --condition_mode topology \
  --noise_table_path data/ddsm_noise/steps400.cat4.speed_balance.time4.0.samples100000.pth \
  --output_path data/ddsm_noise/time_dependent_weights.pth \
  --batch_size 256 \
  --passes 1
```

The official DDSM promoter-design setup uses:

| Parameter | Value |
| --- | ---: |
| pre-sampled noise trajectories | `100000` |
| time steps | `400` |
| max time | `4.0` |
| speed balancing | `True` |
| learning rate | `5e-4` |
| batch size | `256` |
| epochs | `200` |

On A100, `--batch_size 512` is usually reasonable, but `256` is closer to the
official script.

## 5. Quick Sanity Checks

Run a fast model smoke test:

```bash
python tests/test_models.py \
  --file_path_quadruplex ../../quadruplex/data/EQ_hg38_lifted.bed \
  --file_path_seq ../../quadruplex/data/hg38.fa \
  --seq_len 128 \
  --batch_size 2 \
  --max_items 4
```

Run style and import checks:

```bash
python -m ruff check .
python -m py_compile \
  main.py generate.py \
  models/lstm.py models/vae.py models/dfm_model.py models/dfm_module.py models/ddsm_module.py \
  utils/data_utils.py utils/model_factory.py utils/config.py \
  preprocessing/annotate_structure_conditions.py preprocessing/annotate_structure_conditions.py
```

## 6. Train Models

All final runs should use the processed CSV:

```text
data/processed/g4_structure_conditions.csv
```

### LSTM

```bash
python main.py \
  --experiment_name topology_lstm \
  --model_type lstm \
  --processed_csv data/processed/g4_structure_conditions.csv \
  --condition_mode topology \
  --batch_size 256 \
  --max_epochs 200 \
  --max_steps 450000 \
  --num_workers 4 \
  --devices 1 \
  --progress_bar
```

### VAE

```bash
python main.py \
  --experiment_name topology_vae \
  --model_type vae \
  --processed_csv data/processed/g4_structure_conditions.csv \
  --condition_mode topology \
  --batch_size 256 \
  --max_epochs 200 \
  --max_steps 450000 \
  --num_workers 4 \
  --devices 1 \
  --progress_bar
```

### DFM CNN

```bash
python main.py \
  --experiment_name topology_dfm \
  --model_type dfm \
  --processed_csv data/processed/g4_structure_conditions.csv \
  --condition_mode topology \
  --batch_size 512 \
  --max_epochs 5000 \
  --max_steps 450000 \
  --num_workers 4 \
  --devices 1 \
  --guidance_mode probability_addition \
  --guidance_scale 3.0 \
  --progress_bar
```

### DFM Transformer

```bash
python main.py \
  --experiment_name topology_dfm_transformer \
  --model_type dfm_transformer \
  --processed_csv data/processed/g4_structure_conditions.csv \
  --condition_mode topology \
  --batch_size 512 \
  --max_epochs 5000 \
  --max_steps 450000 \
  --num_workers 4 \
  --devices 1 \
  --guidance_mode probability_addition \
  --guidance_scale 3.0 \
  --progress_bar
```

### DDSM

```bash
python main.py \
  --experiment_name topology_ddsm \
  --model_type ddsm \
  --processed_csv data/processed/g4_structure_conditions.csv \
  --condition_mode topology \
  --ddsm_noise_table_path data/ddsm_noise/steps400.cat4.speed_balance.time4.0.samples100000.pth \
  --ddsm_time_dependent_weights_path data/ddsm_noise/time_dependent_weights.pth \
  --batch_size 256 \
  --max_epochs 200 \
  --max_steps 450000 \
  --num_workers 4 \
  --devices 1 \
  --guidance_scale 1.0 \
  --progress_bar
```

## 7. SLURM Examples

### DFM on one GPU

```bash
#!/bin/bash
#SBATCH --job-name=g4_dfm_topology
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
cd /path/to/BIO/research/g4_structure_conditioning
mkdir -p logs/slurm

python main.py \
  --experiment_name topology_dfm \
  --model_type dfm \
  --processed_csv data/processed/g4_structure_conditions.csv \
  --condition_mode topology \
  --batch_size 512 \
  --max_epochs 5000 \
  --max_steps 450000 \
  --num_workers 4 \
  --devices 1 \
  --progress_bar
```

### DDSM on one A100

```bash
#!/bin/bash
#SBATCH --job-name=g4_ddsm_topology
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
cd /path/to/BIO/research/g4_structure_conditioning
mkdir -p logs/slurm data/ddsm_noise

NOISE=data/ddsm_noise/steps400.cat4.speed_balance.time4.0.samples100000.pth
if [ ! -f "$NOISE" ]; then
  python external_tools/ddsm/presample_noise.py \
    -n 100000 \
    -c 4 \
    -t 400 \
    --max_time 4 \
    --speed_balance \
    --out_path data/ddsm_noise
fi

python main.py \
  --experiment_name topology_ddsm \
  --model_type ddsm \
  --processed_csv data/processed/g4_structure_conditions.csv \
  --condition_mode topology \
  --ddsm_noise_table_path "$NOISE" \
  --batch_size 256 \
  --max_epochs 200 \
  --max_steps 450000 \
  --num_workers 4 \
  --devices 1 \
  --guidance_scale 1.0 \
  --progress_bar
```

Submit:

```bash
sbatch run_ddsm.sh
```

Follow:

```bash
squeue -u $USER
tail -f logs/slurm/g4_ddsm_topology_<JOBID>.out
```

## 8. Resume And Test

Resume training:

```bash
python main.py \
  --experiment_name topology_dfm \
  --model_type dfm \
  --processed_csv data/processed/g4_structure_conditions.csv \
  --condition_mode topology \
  --ckpt_path checkpoints/dfm/topology_dfm/last.ckpt \
  --batch_size 512 \
  --max_steps 450000 \
  --progress_bar
```

Evaluate a checkpoint without training:

```bash
python main.py \
  --experiment_name topology_dfm_test \
  --model_type dfm \
  --processed_csv data/processed/g4_structure_conditions.csv \
  --condition_mode topology \
  --run_mode test \
  --ckpt_path checkpoints/dfm/topology_dfm/last.ckpt \
  --batch_size 512 \
  --num_workers 4 \
  --devices 1 \
  --progress_bar
```

For DDSM checkpoints trained with official noise:

```bash
python main.py \
  --experiment_name topology_ddsm_test \
  --model_type ddsm \
  --processed_csv data/processed/g4_structure_conditions.csv \
  --condition_mode topology \
  --run_mode test \
  --ckpt_path checkpoints/ddsm/topology_ddsm/last.ckpt \
  --ddsm_noise_table_path data/ddsm_noise/steps400.cat4.speed_balance.time4.0.samples100000.pth \
  --batch_size 256 \
  --devices 1 \
  --progress_bar
```

## 9. Generate Sequences

Generate 2000 sequences for a topology condition:

```bash
python generate.py \
  --model_type dfm \
  --ckpt_path checkpoints/dfm/topology_dfm/last.ckpt \
  --condition_mode topology \
  --topology parallel \ \
  --num_samples 2000 \
  --batch_size 64 \
  --guidance_scale 3.0 \
  --output_jsonl generated/dfm_parallel.jsonl
```

DDSM generation:

```bash
python generate.py \
  --model_type ddsm \
  --ckpt_path checkpoints/ddsm/topology_ddsm/last.ckpt \
  --condition_mode topology \
  --topology parallel \ \
  --ddsm_noise_table_path data/ddsm_noise/steps400.cat4.speed_balance.time4.0.samples100000.pth \
  --num_samples 2000 \
  --batch_size 64 \
  --guidance_scale 1.0 \
  --output_jsonl generated/ddsm_parallel.jsonl
```

## 10. Outputs

Training creates:

```text
logs/<model_type>/<experiment_name>/
checkpoints/<model_type>/<experiment_name>/
examples/<model_type>/<experiment_name>.jsonl
```

The checkpoint callback monitors:

```text
val_perplexity
```

The generative metrics callback logs:

```text
novelty
melanoma_fbd
hyenadna_fbd
g4hunter_real_mean
g4hunter_gen_mean
g4hunter_gap
g4_real_frac
g4_gen_frac
g4_frac_gap
```

HyenaDNA FBD requires `transformers` and may download model weights. If the
embedder is unavailable, the callback logs a warning and skips that FBD metric.

## 11. Important Notes

- The original `../../quadruplex` code is not modified by this project.
- The processed dataset must exist before training.
- G4ShapePredictor and G4STAB are external predictors and should be treated as
  annotation tools, not train-time dependencies.
- DDSM paper-like training requires `--ddsm_noise_table_path`; for the closest
  official recipe also pass `--ddsm_time_dependent_weights_path`.
- The old `metrics/eval.py` still computes class-wise metrics for EndoQuad
  levels. Topology/stability-specific evaluation should be added as a separate
  next step.
- Warnings about macOS matplotlib/fontconfig cache are local environment noise
  and do not indicate model failure.
