import argparse
import logging
import os
import shlex
import subprocess
import tempfile

import pandas as pd
from pyfaidx import Fasta

TOPOLOGY_CLASSES = ["parallel", "antiparallel", "hybrid"]
G4SHAPE_COMMAND = "/opt/anaconda3/envs/github-g4predictor/bin/python"
G4SHAPE_TOOL_DIR = "external_tools/G4ShapePredictor/g4sp application code"
G4SHAPE_MODEL_NAME = "RandomForest (default)"
G4SHAPE_PAD_LENGTH = 100
G4STAB_COMMAND = "/opt/anaconda3/envs/g4stab_env/bin/python"
G4STAB_TOOL_DIR = "external_tools/G4STAB"
SALT_K = 100.0
SALT_NA = 0.0
SALT_OTHER = 0.0
PH = 7.0

def parse_args():
    parser = argparse.ArgumentParser(description="Annotate short G4 sequences with topology and stability.")
    parser.add_argument("--input_table", default="../../quadruplex/data/EQ_hg38_lifted.bed")
    parser.add_argument("--file_path_seq", default="../../quadruplex/data/hg38.fa")
    parser.add_argument("--output_csv", default="data/processed/g4_structure_conditions.csv")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def read_input_table(path):
    cols = ["chrom", "start", "end", "level_raw", "score", "strand"]
    df = pd.read_csv(path, sep="\t", names=cols)
    df = df.rename(columns={column: str(column).strip() for column in df.columns})
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["level"] = df["level_raw"].astype(str).str.extract(r"(\d+)").astype(int)
    df["length"] = df["end"] - df["start"]
    return df

def extract_sequence(genome, row):
    sequence = genome[row["chrom"]][int(row["start"]) : int(row["end"])].seq.upper()
    if not sequence or any(base not in "ACGT" for base in sequence):
        return None
    return sequence

def build_sequence_table(args):
    df = read_input_table(args.input_table).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    genome = Fasta(args.file_path_seq)
    rows = []
    max_quadruplex_length = df["length"].quantile(0.99)
    df = df[df["length"] <= max_quadruplex_length]
    for idx, row in df.iterrows():
        sequence = extract_sequence(genome, row)
        if sequence is None:
            continue
        if row["length"] <= max_quadruplex_length:
            rows.append(
                {
                    "sample_id": f"g4_{idx}",
                    "chrom": row["chrom"],
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "level_raw": row["level_raw"],
                    "score": row["score"],
                    "strand": row["strand"],
                    "level": int(row["level"]),
                    "length": int(row["length"]),
                    "sequence": sequence,
                }
            )
    out = pd.DataFrame(rows)
    logging.info("Prepared %d short G4 sequences", len(out))
    return out

def annotate_topology(df):
    with tempfile.TemporaryDirectory() as tmpdir:
        records_csv = os.path.join(tmpdir, "records.csv")
        topology_csv = os.path.join(tmpdir, "g4shape_predictions.csv")
        df[["sample_id", "sequence"]].rename(columns={"sequence": "g4_sequence"}).to_csv(
            records_csv, index=False
        )
        script = os.path.join(os.path.dirname(__file__), "run_g4shape_predictor.py")
        cmd = [
            *shlex.split(G4SHAPE_COMMAND),
            script,
            "--records_csv",
            records_csv,
            "--output_csv",
            topology_csv,
            "--tool_dir",
            G4SHAPE_TOOL_DIR,
            "--model_name",
            G4SHAPE_MODEL_NAME,
            "--pad_length",
            str(G4SHAPE_PAD_LENGTH),
        ]
        logging.info("Running G4ShapePredictor on %d short sequences", len(df))
        subprocess.run(cmd, check=True)
        topology = pd.read_csv(topology_csv)
    out = df.merge(topology.drop(columns=["g4_sequence"], errors="ignore"), on="sample_id")
    out = out[out["topology_label"].isin(TOPOLOGY_CLASSES)].copy()
    logging.info("Annotated topology for %d records", len(out))
    return out

def annotate_stability(df):
    with tempfile.TemporaryDirectory() as tmpdir:
        records_csv = os.path.join(tmpdir, "records.csv")
        stability_csv = os.path.join(tmpdir, "g4stab_predictions.csv")
        df[["sample_id", "sequence"]].rename(columns={"sequence": "g4_sequence"}).to_csv(
            records_csv, index=False
        )
        script = os.path.join(os.path.dirname(__file__), "run_g4stab.py")
        cmd = [
            *shlex.split(G4STAB_COMMAND),
            script,
            "--records_csv",
            records_csv,
            "--output_csv",
            stability_csv,
            "--tool_dir",
            G4STAB_TOOL_DIR,
            "--salt_k",
            str(SALT_K),
            "--salt_na",
            str(SALT_NA),
            "--salt_other",
            str(SALT_OTHER),
            "--ph",
            str(PH),
        ]
        logging.info("Running G4STAB on %d short sequences", len(df))
        subprocess.run(cmd, check=True)
        stability = pd.read_csv(stability_csv)
    out = df.merge(stability.drop(columns=["g4_sequence"], errors="ignore"), on="sample_id")
    logging.info("Annotated stability for %d records", len(out))
    return out

def main():
    args = parse_args()
    setup_logging()
    df = build_sequence_table(args)
    df = annotate_topology(df)
    df = annotate_stability(df)
    df["annotation_method"] = "G4ShapePredictor+G4STAB"
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logging.info("Saved annotated short-sequence conditions to %s", args.output_csv)
    logging.info("Final records: %d", len(df))

if __name__ == "__main__":
    main()
