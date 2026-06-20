import argparse
import csv
import json
import logging
import os
import random

import numpy as np
import torch

from utils.config import CFG
from utils.data_utils import QuadDataset, decode_seq, split_data
from utils.gen_metrics_callback import (
    DEFAULT_HYENADNA_MODEL,
    GenerativeMetricsCallback,
    HyenaDNAEmbedder,
    MelanomaEmbedder,
    _frechet_distance,
)
from utils.logging_utils import setup_logging
from utils.metric_utils import encode_sequences
from utils.model_factory import build_model_from_checkpoint
from utils.model_utils import count_trainable_params

G4_THRESHOLD = 1.5


class ModelSpec:
    def __init__(self, name, model_type, ckpt_path):
        self.name = name
        self.model_type = model_type
        self.ckpt_path = ckpt_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate class-wise samples and compute metrics per class."
    )
    parser.add_argument("--file_path_quadruplex", required=True)
    parser.add_argument("--file_path_seq", required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec: name:type:ckpt_path, where type is lstm/vae/dfm/dfm_transformer/ddsm.",
    )
    parser.add_argument("--output_dir", default="generated/classwise_metrics")
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--classes", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--g4hunter_window", type=int, default=CFG["g4hunter_window"])
    parser.add_argument("--g4_threshold", type=float, default=G4_THRESHOLD)
    parser.add_argument("--guidance_scales", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0])
    parser.add_argument(
        "--guidance_modes",
        nargs="+",
        default=["probability_addition"],
        choices=[
            "score",
            "score_free",
            "probability_addition",
            "probability_tilt",
            "vectorfield_addition",
            "logit",
        ],
    )
    parser.add_argument(
        "--embedders",
        nargs="+",
        default=["melanoma", "hyenadna"],
        choices=["melanoma", "hyenadna"],
    )
    parser.add_argument("--hyenadna_model", default=DEFAULT_HYENADNA_MODEL)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate samples even if the corresponding JSONL file already exists.",
    )
    return parser.parse_args()


def parse_model_spec(value):
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise ValueError("--model must have format name:type:ckpt_path")
    name, model_type, ckpt_path = parts
    if model_type not in {"lstm", "vae", "dfm", "dfm_transformer", "ddsm"}:
        raise ValueError(f"Unsupported model type: {model_type}")
    return ModelSpec(name=name, model_type=model_type, ckpt_path=ckpt_path)


def sample_real_by_class(df, file_path_seq, levels, num_samples, seed, log_label="real"):
    real = {}
    for level in levels:
        class_df = df[df["level"] == level]
        sample_size = min(len(class_df), num_samples + 500)
        class_df = class_df.sample(n=sample_size, random_state=seed + int(level))
        random.seed(seed + int(level))
        dataset = QuadDataset(
            class_df,
            file_path_seq=file_path_seq,
            typer="rec",
            seq_len=CFG["seq_len"],
            level_offset=CFG["level_offset"],
        )
        if len(dataset.encoded_seqs) < num_samples:
            raise RuntimeError(
                f"Class {level}: need {num_samples} valid sequences, got {len(dataset.encoded_seqs)}"
            )
        real[level] = torch.stack(dataset.encoded_seqs[:num_samples])
        logging.info("Loaded %s class %s: %d sequences", log_label, level, real[level].size(0))
    return real


def train_sets_by_class(train_df, file_path_seq, levels):
    train_all_set = set()
    train_class_sets = {}
    for level in levels:
        class_df = train_df[train_df["level"] == level]
        dataset = QuadDataset(
            class_df,
            file_path_seq=file_path_seq,
            typer="rec",
            seq_len=CFG["seq_len"],
            level_offset=CFG["level_offset"],
        )
        class_set = {tuple(row.tolist()) for row in dataset.encoded_seqs}
        train_class_sets[level] = class_set
        train_all_set.update(class_set)
        logging.info(
            "Loaded full train novelty class %s: %d unique sequences",
            level,
            len(class_set),
        )
    logging.info("Loaded full train novelty set: %d unique sequences", len(train_all_set))
    return train_all_set, train_class_sets


def output_path(output_dir, run_name, model_name, class_level, guidance_mode=None, guidance_scale=None):
    class_dir = os.path.join(output_dir, run_name, model_name, f"class_{class_level}")
    os.makedirs(class_dir, exist_ok=True)
    if guidance_mode is None:
        filename = "samples.jsonl"
    else:
        scale = str(guidance_scale).replace(".", "p")
        filename = f"cfg_{guidance_mode}_scale_{scale}.jsonl"
    return os.path.join(class_dir, filename)


def write_sequences(path, seq_ids, class_level, model_name, generation_name):
    with open(path, "w", encoding="utf-8") as handle:
        for idx, row in enumerate(seq_ids):
            record = {
                "id": idx,
                "model": model_name,
                "generation": generation_name,
                "class_level": int(class_level),
                "cond": int(class_level) - CFG["level_offset"],
                "seq": decode_seq(row.tolist()),
                "ids": [int(token) for token in row.tolist()],
            }
            handle.write(json.dumps(record) + "\n")


def read_sequences(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            rows.append(torch.tensor(json.loads(line)["ids"], dtype=torch.long))
    return torch.stack(rows)


@torch.no_grad()
def generate_sequences(model, model_type, cond, seq_len, guidance_scale=None, batch_size=32):
    generated = []
    for start in range(0, cond.size(0), batch_size):
        end = min(start + batch_size, cond.size(0))
        cond_batch = cond[start:end]
        if model_type == "lstm":
            batch = model.generate(cond_batch, seq_len=seq_len)
        elif model_type == "vae":
            batch = model.generate(cond_batch)
        else:
            batch = model.generate(cond_batch, guidance_scale=guidance_scale)
        generated.append(batch.detach().cpu().long())
    return torch.cat(generated, dim=0)


def ids_to_seqs(seq_ids):
    return [GenerativeMetricsCallback._ids_to_seq(row) for row in seq_ids]


def g4_metrics(real_ids, gen_ids, window, threshold):
    real_g4 = GenerativeMetricsCallback._g4hunter_scores(ids_to_seqs(real_ids), window=window)
    gen_g4 = GenerativeMetricsCallback._g4hunter_scores(ids_to_seqs(gen_ids), window=window)
    real_frac = float(np.mean(real_g4 > threshold))
    gen_frac = float(np.mean(gen_g4 > threshold))
    return {
        "g4_real_mean": float(np.mean(real_g4)),
        "g4_gen_mean": float(np.mean(gen_g4)),
        "g4_mean_gap": float(abs(np.mean(real_g4) - np.mean(gen_g4))),
        "g4_paired_gap": float(np.mean(np.abs(real_g4 - gen_g4))),
        "g4_real_frac": real_frac,
        "g4_gen_frac": gen_frac,
        "g4_frac_gap": float(abs(real_frac - gen_frac)),
    }


def novelty_metrics(gen_ids, train_all_set, train_class_set):
    gen_tuples = [tuple(row.tolist()) for row in gen_ids]
    return {
        "novelty_all_train": float(np.mean([row not in train_all_set for row in gen_tuples])),
        "novelty_class_train": float(np.mean([row not in train_class_set for row in gen_tuples])),
    }


def make_embedders(names, device, hyenadna_model, seq_len):
    embedders = {}
    if "melanoma" in names:
        embedders["melanoma"] = MelanomaEmbedder(device)
    if "hyenadna" in names:
        embedders["hyenadna"] = HyenaDNAEmbedder(device, model_name=hyenadna_model, seq_len=seq_len)
    return embedders


def generation_jobs_for_model(args, spec, model, device):
    jobs = []
    for class_level in args.classes:
        cond_value = int(class_level) - CFG["level_offset"]
        cond = torch.full((args.num_samples,), cond_value, dtype=torch.long, device=device)

        if spec.model_type in {"dfm", "dfm_transformer"}:
            for guidance_mode in args.guidance_modes:
                model.hparams.guidance_mode = guidance_mode
                for guidance_scale in args.guidance_scales:
                    generation_name = f"{guidance_mode}_scale_{guidance_scale:g}"
                    path = output_path(
                        args.output_dir,
                        args.split,
                        spec.name,
                        class_level,
                        guidance_mode=guidance_mode,
                        guidance_scale=guidance_scale,
                    )
                    jobs.append(
                        {
                            "class_level": class_level,
                            "generation_name": generation_name,
                            "path": path,
                            "cond": cond,
                            "guidance_scale": guidance_scale,
                        }
                    )
        elif spec.model_type == "ddsm":
            for guidance_scale in args.guidance_scales:
                generation_name = f"cfg_scale_{guidance_scale:g}"
                path = output_path(
                    args.output_dir,
                    args.split,
                    spec.name,
                    class_level,
                    guidance_mode="ddsm",
                    guidance_scale=guidance_scale,
                )
                jobs.append(
                    {
                        "class_level": class_level,
                        "generation_name": generation_name,
                        "path": path,
                        "cond": cond,
                        "guidance_scale": guidance_scale,
                    }
                )
        else:
            path = output_path(args.output_dir, args.split, spec.name, class_level)
            jobs.append(
                {
                    "class_level": class_level,
                    "generation_name": "sample",
                    "path": path,
                    "cond": cond,
                    "guidance_scale": None,
                }
            )
    return jobs


def generate_or_load(args, spec, model, job):
    if os.path.exists(job["path"]) and not args.overwrite:
        logging.info(
            "Reusing existing samples model=%s class=%s generation=%s path=%s",
            spec.name,
            job["class_level"],
            job["generation_name"],
            job["path"],
        )
        return read_sequences(job["path"])

    logging.info(
        "Generating model=%s class=%s generation=%s",
        spec.name,
        job["class_level"],
        job["generation_name"],
    )
    gen = generate_sequences(
        model,
        spec.model_type,
        job["cond"],
        CFG["seq_len"],
        guidance_scale=job["guidance_scale"],
        batch_size=args.batch_size,
    )
    write_sequences(job["path"], gen, job["class_level"], spec.name, job["generation_name"])
    return gen


def main():
    args = parse_args()
    setup_logging()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    specs = [parse_model_spec(value) for value in args.model]
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logging.info("Using device=%s", device)

    split_dfs = split_data(
        args.file_path_quadruplex,
        split=CFG["split"],
        val_split=CFG["val_split"],
        seed=args.seed,
    )
    eval_df = split_dfs[args.split]
    train_df = split_dfs["train"] if args.split != "train" else split_dfs["train"]

    real_by_class = sample_real_by_class(
        eval_df,
        args.file_path_seq,
        args.classes,
        args.num_samples,
        args.seed,
        log_label=f"{args.split} real",
    )
    train_all_set, train_class_sets = train_sets_by_class(
        train_df,
        args.file_path_seq,
        args.classes,
    )

    embedders = make_embedders(args.embedders, device, args.hyenadna_model, CFG["seq_len"])
    real_embeddings = {
        embedder_name: {
            class_level: encode_sequences(embedder, real_ids, args.batch_size)
            for class_level, real_ids in real_by_class.items()
        }
        for embedder_name, embedder in embedders.items()
    }

    rows = []
    for spec in specs:
        logging.info("Loading model %s (%s)", spec.name, spec.model_type)
        fallback_args = argparse.Namespace(
            model_type=spec.model_type,
            guidance_scale=1.0,
            guidance_mode="probability_addition",
        )
        model = build_model_from_checkpoint(spec.model_type, spec.ckpt_path, fallback_args).to(device)
        model.eval()
        params = count_trainable_params(model)
        jobs = generation_jobs_for_model(args, spec, model, device)

        for job in jobs:
            gen = generate_or_load(args, spec, model, job)
            class_level = job["class_level"]
            real = real_by_class[class_level]

            base_row = {
                "model": spec.name,
                "model_type": spec.model_type,
                "generation": job["generation_name"],
                "class_level": class_level,
                "num_samples": args.num_samples,
                "params": params,
                "samples_path": job["path"],
            }
            base_row.update(g4_metrics(real, gen, args.g4hunter_window, args.g4_threshold))
            base_row.update(novelty_metrics(gen, train_all_set, train_class_sets[class_level]))

            for embedder_name, embedder in embedders.items():
                gen_emb = encode_sequences(embedder, gen, args.batch_size)
                fbd = _frechet_distance(real_embeddings[embedder_name][class_level], gen_emb)
                row = dict(base_row)
                row["embedder"] = embedder_name
                row["fbd"] = fbd
                rows.append(row)
                logging.info(
                    "%s | %s | class=%s | %s | FBD=%.6f | G4 mean gap=%.6f | frac gap=%.6f",
                    embedder_name,
                    spec.name,
                    class_level,
                    job["generation_name"],
                    fbd,
                    row["g4_mean_gap"],
                    row["g4_frac_gap"],
                )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_path = os.path.join(args.output_dir, args.split, "classwise_metrics.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    fieldnames = [
        "embedder",
        "model",
        "model_type",
        "generation",
        "class_level",
        "num_samples",
        "params",
        "fbd",
        "g4_real_mean",
        "g4_gen_mean",
        "g4_mean_gap",
        "g4_paired_gap",
        "g4_real_frac",
        "g4_gen_frac",
        "g4_frac_gap",
        "novelty_all_train",
        "novelty_class_train",
        "samples_path",
    ]
    with open(results_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Saved metrics to %s", results_path)


if __name__ == "__main__":
    main()
