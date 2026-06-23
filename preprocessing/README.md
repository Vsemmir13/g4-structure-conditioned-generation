# Structure Preprocessing

The preprocessing creates the annotated short-sequence table used by training.

## Annotate Short G4 Sequences

This stage works only with the original short EndoQuad G4 sequence. It does not
create 512 bp model windows and does not create train/validation/test splits.

Input columns:

```text
chrom
start
end
level_raw
score
strand
```

The short sequence is extracted from `hg38.fa` using `chrom:start-end`.

Run:

```bash
python preprocessing/annotate_structure_conditions.py \
  --input_table data/EQ_hg38_lifted.bed \
  --file_path_seq data/hg38.fa \
  --output_csv data/processed/g4_structure_conditions.csv
```

Output:

```text
sample_id
chrom
start
end
level_raw
score
strand
level
length
sequence                  # short G4 sequence
topology_label
topology_prob_parallel
topology_prob_antiparallel
topology_prob_hybrid
topology_confidence
predicted_tm
g4stab_std
```

`topology_label` is predicted by G4ShapePredictor. `predicted_tm` is predicted
by G4STAB and is kept as a continuous temperature value. Stability thresholds
are intentionally not assigned during annotation.

Training reads this CSV directly. `QuadDataset` derives `stability_class` from
`predicted_tm`, applies the `g4stab_std` quality filter, creates stratified
train/validation/test splits, and extracts 512 bp model windows from the
reference genome at training time.
