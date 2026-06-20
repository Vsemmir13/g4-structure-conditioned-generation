import argparse
import csv
import json
import logging
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from utils.config import CFG
from utils.data_utils import QuadDataset, decode_seq, split_data
from utils.logging_utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute pqsfinder metrics for already generated JSONL samples."
    )
    parser.add_argument("--samples_root", required=True)
    parser.add_argument("--sample_glob", default="*.jsonl")
    parser.add_argument("--file_path_quadruplex", required=True)
    parser.add_argument("--file_path_seq", required=True)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--classes", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--num_real", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_score", type=int, default=42)
    parser.add_argument("--strand", default="*", choices=["+", "-", "*"])
    parser.add_argument("--overlapping", action="store_true")
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--keep_per_sequence", action="store_true")
    return parser.parse_args()


def sample_real_sequences(df, file_path_seq, classes, num_real, seed):
    rows = []
    for class_level in classes:
        class_df = df[df["level"] == class_level]
        sample_size = min(len(class_df), num_real + 500)
        class_df = class_df.sample(n=sample_size, random_state=seed + int(class_level))
        random.seed(seed + int(class_level))
        dataset = QuadDataset(
            class_df,
            file_path_seq=file_path_seq,
            typer="rec",
            seq_len=CFG["seq_len"],
            level_offset=CFG["level_offset"],
        )
        if len(dataset.encoded_seqs) < num_real:
            raise RuntimeError(
                f"Class {class_level}: need {num_real} real sequences, "
                f"got {len(dataset.encoded_seqs)}"
            )
        for idx, seq_ids in enumerate(dataset.encoded_seqs[:num_real]):
            rows.append(
                {
                    "row_id": f"real_{class_level}_{idx}",
                    "source": "real",
                    "model": "real",
                    "generation": "real",
                    "class_level": int(class_level),
                    "samples_path": "real",
                    "seq": decode_seq(seq_ids.tolist()),
                }
            )
        logging.info("Loaded real class %s for pqsfinder: %d", class_level, num_real)
    return rows


def iter_generated_jsonl(samples_root, sample_glob):
    root = Path(samples_root)
    for path in sorted(root.rglob(sample_glob)):
        if ".ipynb_checkpoints" not in path.parts:
            yield path


def generated_rows(samples_root, sample_glob):
    rows = []
    for path in iter_generated_jsonl(samples_root, sample_glob):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                seq = record.get("seq") or record.get("generation_seq")
                if seq is None and "ids" in record:
                    seq = decode_seq(record["ids"])
                if seq is None:
                    raise ValueError(f"Cannot find sequence field in {path}")
                model = record.get("model") or path.parts[-3]
                generation = record.get("generation") or path.stem
                class_level = int(record.get("class_level") or path.parts[-2].replace("class_", ""))
                rows.append(
                    {
                        "row_id": f"gen_{len(rows)}",
                        "source": "generated",
                        "model": model,
                        "generation": generation,
                        "class_level": class_level,
                        "samples_path": str(path),
                        "seq": seq,
                    }
                )
    if not rows:
        raise RuntimeError(f"No JSONL samples found under {samples_root}")
    logging.info("Loaded generated sequences for pqsfinder: %d", len(rows))
    return rows


def write_tsv(rows, path):
    fieldnames = ["row_id", "source", "model", "generation", "class_level", "samples_path", "seq"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run_pqsfinder(args, input_tsv, per_sequence_csv):
    if shutil.which(args.rscript) is None:
        raise RuntimeError(
            f"Cannot find {args.rscript}. Install R and pqsfinder, or pass --rscript /path/to/Rscript."
        )
    script_path = Path(__file__).with_name("pqsfinder_metrics.R")
    cmd = [
        args.rscript,
        str(script_path),
        str(input_tsv),
        str(per_sequence_csv),
        str(args.min_score),
        args.strand,
        "TRUE" if args.overlapping else "FALSE",
    ]
    logging.info("Running pqsfinder via Rscript")
    subprocess.run(cmd, check=True)


def aggregate(per_sequence_csv):
    df = pd.read_csv(per_sequence_csv)
    df["pqs_has_hit"] = (df["pqs_count"] > 0).astype(float)
    metrics = {
        "pqs_frac": ("pqs_has_hit", "mean"),
        "pqs_count_mean": ("pqs_count", "mean"),
        "pqs_max_score_mean": ("pqs_max_score", "mean"),
        "pqs_mean_score_mean": ("pqs_mean_score", "mean"),
        "pqs_total_score_mean": ("pqs_total_score", "mean"),
        "pqs_max_width_mean": ("pqs_max_width", "mean"),
        "pqs_mean_width_mean": ("pqs_mean_width", "mean"),
    }
    grouped = df.groupby(["source", "model", "generation", "class_level", "samples_path"]).agg(**metrics)
    summary = grouped.reset_index()

    real = summary[summary["source"] == "real"].set_index("class_level")
    generated = summary[summary["source"] == "generated"].copy()
    for metric in metrics:
        generated[f"real_{metric}"] = generated["class_level"].map(real[metric])
        generated[f"{metric}_gap"] = (generated[metric] - generated[f"real_{metric}"]).abs()
    return summary, generated


def main():
    args = parse_args()
    setup_logging()
    output_csv = args.output_csv
    if output_csv is None:
        output_csv = str(Path(args.samples_root) / "pqsfinder_metrics.csv")

    split_df = split_data(
        args.file_path_quadruplex,
        split=CFG["split"],
        val_split=CFG["val_split"],
        seed=args.seed,
    )[args.split]
    rows = sample_real_sequences(
        split_df,
        args.file_path_seq,
        args.classes,
        args.num_real,
        args.seed,
    )
    rows.extend(generated_rows(args.samples_root, args.sample_glob))

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_sequence_csv = output_path.with_name(output_path.stem + "_per_sequence.csv")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_tsv = Path(tmpdir) / "pqsfinder_input.tsv"
        write_tsv(rows, input_tsv)
        run_pqsfinder(args, input_tsv, per_sequence_csv)

    summary, generated = aggregate(per_sequence_csv)
    summary_path = output_path.with_name(output_path.stem + "_summary_all.csv")
    summary.to_csv(summary_path, index=False)
    generated.to_csv(output_path, index=False)
    logging.info("Saved generated-vs-real pqsfinder metrics to %s", output_path)
    logging.info("Saved all pqsfinder summaries to %s", summary_path)
    if not args.keep_per_sequence:
        per_sequence_csv.unlink(missing_ok=True)
    else:
        logging.info("Saved per-sequence pqsfinder metrics to %s", per_sequence_csv)


if __name__ == "__main__":
    main()
