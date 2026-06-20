import argparse
import logging
import os
import random

import pandas as pd
from pyfaidx import Fasta
from sklearn.model_selection import train_test_split

SEQ_LEN = 512
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Build the 512 bp model dataset from annotated G4 conditions.")
    parser.add_argument("--annotated_csv", default="data/processed/g4_structure_conditions.csv")
    parser.add_argument("--file_path_seq", default="../../quadruplex/data/hg38.fa")
    parser.add_argument("--output_csv", default="data/processed/g4_structure_dataset.csv")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_context(genome, row, rng):
    chrom_sequence = genome[row["chrom"]]
    start = int(row["start"])
    end = int(row["end"])
    min_start_pos = max(0, end - SEQ_LEN)
    max_start_pos = min(start, len(chrom_sequence) - SEQ_LEN)
    if max_start_pos < min_start_pos:
        return None
    context_start = rng.randint(min_start_pos, max_start_pos)
    context_end = context_start + SEQ_LEN
    sequence = chrom_sequence[context_start:context_end].seq.upper()
    if any(base not in "ACGT" for base in sequence):
        return None
    return context_start, context_end, sequence


def add_model_sequences(df, file_path_seq):
    rng = random.Random(SEED)
    genome = Fasta(file_path_seq)
    rows = []
    for _, row in df.iterrows():
        context = extract_context(genome, row, rng)
        if context is None:
            continue
        context_start, context_end, model_sequence = context
        out = row.to_dict()
        out["context_start"] = int(context_start)
        out["context_end"] = int(context_end)
        out["model_sequence"] = model_sequence
        rows.append(out)
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No valid 512 bp model sequences were extracted")
    logging.info("Prepared %d records with 512 bp model_sequence", len(out))
    return out


def valid_stratification(stratify_values, df_size):
    num_classes = stratify_values.nunique()
    rest_size = int(round(df_size * 0.2))
    return (
        num_classes >= 2
        and stratify_values.value_counts().min() >= 3
        and rest_size >= num_classes * 2
        and df_size - rest_size >= num_classes
    )


def add_splits(df):
    candidates = [df["topology_label"].astype(str)]
    stratify_values = None
    for candidate in candidates:
        if valid_stratification(candidate, len(df)):
            stratify_values = candidate
            break

    if stratify_values is None:
        logging.warning("Could not create stratified split; assigning all records to train")
        df = df.copy()
        df["split"] = "train"
        return df

    train_df, rest_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=stratify_values
    )
    rest_stratify = stratify_values.loc[rest_df.index]
    val_df, test_df = train_test_split(
        rest_df, test_size=0.5, random_state=SEED, stratify=rest_stratify
    )
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def save_stats(df, output_csv):
    stats_dir = os.path.join(os.path.dirname(output_csv), "stats")
    os.makedirs(stats_dir, exist_ok=True)
    df["topology_label"].value_counts(dropna=False).rename_axis("topology_label").reset_index(
        name="count"
    ).to_csv(os.path.join(stats_dir, "topology_distribution.csv"), index=False)
    df.groupby("topology_label")["predicted_tm"].describe().reset_index().to_csv(
        os.path.join(stats_dir, "tm_by_topology.csv"), index=False
    )
    df["predicted_tm"].describe().to_frame(name="predicted_tm").to_csv(
        os.path.join(stats_dir, "tm_distribution.csv")
    )


def main():
    args = parse_args()
    setup_logging()
    df = pd.read_csv(args.annotated_csv)
    if args.limit:
        df = df.head(args.limit)
    df = add_model_sequences(df, args.file_path_seq)
    df = add_splits(df)
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    save_stats(df, args.output_csv)
    logging.info("Saved model dataset to %s", args.output_csv)
    logging.info("Final records: %d", len(df))


if __name__ == "__main__":
    main()
