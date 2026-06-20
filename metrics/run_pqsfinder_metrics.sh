#!/usr/bin/env bash
set -euo pipefail

SAMPLES_ROOT=${SAMPLES_ROOT:-generated/classwise_metrics/val}
BED=${BED:-data/EQ_hg38_lifted.bed}
FASTA=${FASTA:-data/hg38.fa}
OUT=${OUT:-generated/classwise_metrics/val/pqsfinder_metrics.csv}
SAMPLE_GLOB=${SAMPLE_GLOB:-*.jsonl}
N_REAL=${N_REAL:-2000}
SPLIT=${SPLIT:-val}
MIN_SCORE=${MIN_SCORE:-42}
STRAND=${STRAND:-"*"}

python -m metrics.pqsfinder \
  --samples_root "$SAMPLES_ROOT" \
  --sample_glob "$SAMPLE_GLOB" \
  --file_path_quadruplex "$BED" \
  --file_path_seq "$FASTA" \
  --output_csv "$OUT" \
  --split "$SPLIT" \
  --classes 4 5 6 \
  --num_real "$N_REAL" \
  --min_score "$MIN_SCORE" \
  --strand "$STRAND"
