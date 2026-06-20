# G4 Structure-Conditioned Generation

This directory is a separate research copy of the G4 generation project. The original project in `../../quadruplex` is intentionally left unchanged.

The goal of this branch is to replace EndoQuad level conditioning with biologically interpretable conditioning targets:

- G4 topology: `parallel`, `antiparallel`, `hybrid`
- thermal stability: predicted melting temperature and categorical stability class

The initial copy reuses the existing model code, training loop, generation logic, and metric infrastructure. Heavy artifacts were not copied:

- `data/`
- `checkpoints/`
- `generated/`
- `logs/`
- `.git/`

Use the original dataset paths unless a local copy is created:

```text
../../quadruplex/data/EQ_hg38_lifted.bed
../../quadruplex/data/hg38.fa
```

## Proposed Directory Layout

```text
g4_structure_conditioning/
  main.py
  models/
  utils/
  metrics/
  analysis/
  preprocessing/
    annotate_structure_conditions.py
  data/
    processed/
      g4_structure_conditions.csv
  reports/
    dataset_statistics/
    topology_eval/
    stability_eval/
```

## Reusable Components

The following parts should be reused with minimal changes:

- `utils/data_utils.py`: sequence extraction from BED and FASTA
- `models/lstm.py`: conditional autoregressive baseline
- `models/vae.py`: conditional VAE baseline
- `models/dfm_model.py`, `models/dfm_module.py`, `models/dfm_flow_utils.py`: DFM generators
- `utils/model_factory.py`: model construction from config/checkpoint
- `metrics/eval.py`: generation and class-wise metrics
- `metrics/pqsfinder.py`: pqsfinder-based validation

## Required Refactoring

1. Introduce a generic condition interface.

Current code assumes:

```text
condition = EndoQuad level - 4
num_cls = 3
```

New code should support:

```text
condition_name = topology
condition_classes = parallel, antiparallel, hybrid
```

and later after dataloader-side thresholding:

```text
condition_name = predicted_tm_bin
condition_classes = low, medium, high
```

2. Keep single-condition training first.

The code should be designed so that multi-condition generation is possible later:

```text
p(x | topology, stability)
```

but the first implementation should train only:

```text
p(x | topology)
```

3. Separate annotation from training.

Topology and stability predictors should run once and save a processed table. Training should read the processed table and should not rerun external predictors.

## External Dependencies

### pqsfinder

Used to extract the strongest putative quadruplex sequence from each DNA window.

Needed outputs:

- hit start/end
- strand
- score
- strongest PQS sequence

### G4ShapePredictor

Used to predict topology from the strongest PQS hit.

Needed outputs:

- predicted topology label
- topology probabilities

Risk: availability and command-line/API format must be checked. If the official implementation is not usable locally, use an equivalent topology predictor only if its assumptions are documented.

### G4STAB

Used to predict thermal stability.

Needed outputs:

- predicted melting temperature, `predicted_tm`
- optional confidence/metadata if available

Risk: G4STAB input format may expect only the PQS core rather than a full 512bp genomic window.

## Processed Dataset Schema

The processed dataset should contain one row per usable sequence:

```text
sample_id
chrom
start
end
strand
sequence
strongest_pqs_sequence
strongest_pqs_start
strongest_pqs_end
strongest_pqs_score
topology_label
topology_prob_parallel
topology_prob_antiparallel
topology_prob_hybrid
topology_confidence
predicted_tm
split
```

Optional filters:

- remove sequences without pqsfinder hits
- remove low-confidence topology predictions
- remove invalid G4STAB predictions

## Dataset Statistics

Generate and save:

- topology distribution
- predicted Tm distribution
- stability class distribution
- topology vs Tm table
- topology vs stability class heatmap
- class balance by split

## Training Plan

1. Build topology-conditioned dataset.
2. Train LSTM conditioned on topology.
3. Train VAE conditioned on topology.
4. Train DFM conditioned on topology.
5. Train DFM large if topology labels are balanced enough.

The first training runs should keep the same architecture sizes as the final G4-level experiments. Only the condition vocabulary changes.

## Evaluation Plan

Keep existing metrics:

- novelty against the full train set
- HyenaDNA FBD
- G4Hunter
- pqsfinder metrics

Add topology-control metrics:

- generated topology accuracy
- topology confusion matrix
- per-class precision and recall
- generated topology distribution vs target distribution

Add stability metrics later:

- predicted Tm mean absolute error to target bin midpoint
- generated Tm distribution by stability class
- stability class confusion matrix
- calibration by stability class

## First Implementation Steps

1. Add `preprocessing/annotate_structure_conditions.py`.
2. Add a simple two-condition mapping in `utils/data_utils.py`.
3. Update dataset loading to read the annotated CSV directly.
4. Update `main.py` to train with stability and topology conditions.
5. Update metrics to report topology- and stability-specific control metrics.
6. Run a small CPU/MPS smoke test on 100 samples.
7. Run full annotation.
10. Train topology-conditioned LSTM as the first baseline.
