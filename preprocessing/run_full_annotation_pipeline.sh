#!/usr/bin/env bash
set -euo pipefail

INPUT_TABLE=${INPUT_TABLE:-../../quadruplex/data/EQ_hg38_lifted.bed}
GENOME_FASTA=${GENOME_FASTA:-../../quadruplex/data/hg38.fa}
CONDITIONS_CSV=${CONDITIONS_CSV:-data/processed/g4_structure_conditions.csv}
OUTPUT_CSV=${OUTPUT_CSV:-data/processed/g4_structure_dataset.csv}

python preprocessing/annotate_structure_conditions.py \
  --input_table "$INPUT_TABLE" \
  --file_path_seq "$GENOME_FASTA" \
  --output_csv "$CONDITIONS_CSV"

python preprocessing/build_structure_dataset.py \
  --annotated_csv "$CONDITIONS_CSV" \
  --file_path_seq "$GENOME_FASTA" \
  --output_csv "$OUTPUT_CSV"
