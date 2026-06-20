import json
import logging
import os
import random

import pandas as pd
import torch
from pyfaidx import Fasta
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3}
VOCAB_SIZE = len(VOCAB)
ID2BASE = {v: k for k, v in VOCAB.items()}
BOS_TOKEN_ID = VOCAB_SIZE
STABILITY_CLASSES = ["low", "medium", "high"]
TOPOLOGY_CLASSES = ["parallel", "hybrid", "antiparallel"]
CONDITION_NAMES = ["stability_class", "topology_label"]
CONDITION_CLASSES = [STABILITY_CLASSES, TOPOLOGY_CLASSES]


def load_data(
    data_path,
    max_g4stab_std=5.0,
    stability_low_tm=52.0,
    stability_high_tm=65.0,
    topology_classes=None,
):
    if str(data_path).endswith(".bed"):
        cols = ["chrom", "start", "end", "level_raw", "score", "strand"]
        df = pd.read_csv(data_path, sep="\t", names=cols)
        df["level"] = df["level_raw"].str.extract(r"(\d+)").astype(int)
        df["length"] = df["end"] - df["start"]
        df = df[df["length"] <= df["length"].quantile(0.99)]
        return df[df["level"] > 3].reset_index(drop=True)

    df = pd.read_csv(data_path)
    df["stability_class"] = pd.cut(
        df["predicted_tm"],
        bins=[float("-inf"), stability_low_tm, stability_high_tm, float("inf")],
        labels=STABILITY_CLASSES,
        include_lowest=True,
    ).astype(str)
    if max_g4stab_std is not None:
        df = df[df["g4stab_std"] <= float(max_g4stab_std)].copy()
    classes = TOPOLOGY_CLASSES if topology_classes is None else [str(item) for item in topology_classes]
    df = df[df["topology_label"].astype(str).isin(classes)].copy()
    return df.reset_index(drop=True)


def split_data(
    data_path,
    split=0.8,
    val_split=0.1,
    seed=42,
    condition_names=None,
    condition_classes=None,
    max_g4stab_std=5.0,
    stability_low_tm=52.0,
    stability_high_tm=65.0,
    log_sizes=False,
):
    condition_names = CONDITION_NAMES if condition_names is None else normalize_condition_names(condition_names)
    topology_classes = None
    if condition_classes is not None and "topology_label" in condition_names:
        topology_idx = condition_names.index("topology_label")
        topology_classes = condition_classes[topology_idx]
    df = load_data(
        data_path,
        max_g4stab_std=max_g4stab_std,
        stability_low_tm=stability_low_tm,
        stability_high_tm=stability_high_tm,
        topology_classes=topology_classes,
    ).sample(frac=1, random_state=seed).reset_index(drop=True)
    if set(condition_names).issubset(df.columns):
        if condition_classes is not None:
            for condition_name, class_group in zip(condition_names, condition_classes, strict=True):
                df = df[df[condition_name].astype(str).isin([str(item) for item in class_group])].copy()
        stratify = df[condition_names].astype(str).agg("|".join, axis=1)
    else:
        stratify = df["level"]
    train_df, rest_df = train_test_split(
        df,
        test_size=1.0 - split,
        stratify=stratify,
        random_state=seed,
    )
    rest_stratify = rest_df[condition_names].astype(str).agg("|".join, axis=1) if set(condition_names).issubset(rest_df.columns) else rest_df["level"]
    test_df, val_df = train_test_split(
        rest_df,
        test_size=val_split / (1.0 - split),
        stratify=rest_stratify,
        random_state=seed,
    )
    if log_sizes:
        logging.info("Data size: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))
        if set(condition_names).issubset(df.columns):
            logging.info("Joint condition counts:\n%s", df.groupby(condition_names).size())
    return {"train": train_df, "val": val_df, "test": test_df, "all": df}


def normalize_condition_names(condition_names):
    if isinstance(condition_names, str):
        return [condition_names]
    return [str(name) for name in condition_names]


def condition_classes_from_data(df, condition_name, condition_classes=None):
    if condition_classes:
        return [str(item) for item in condition_classes]
    values = sorted(str(value) for value in df[condition_name].dropna().unique())
    if not values:
        raise ValueError(f"No condition classes found for {condition_name}")
    return values


class QuadDataset(Dataset):

    def __init__(
        self,
        df,
        file_path_seq=None,
        typer="rec",
        seq_len=512,
        level_offset=4,
        condition_names=None,
        condition_classes=None,
    ):
        self.file_path_seq = file_path_seq
        self.seq_len = seq_len
        self.genome = Fasta(file_path_seq) if file_path_seq else None
        self.typer = typer
        assert self.typer in ["rec", "gen"]
        self.level_offset = int(level_offset)
        self.sequence_column = "model_sequence" if "model_sequence" in df.columns else None
        self.condition_names = CONDITION_NAMES if condition_names is None else normalize_condition_names(condition_names)
        self.condition_classes = CONDITION_CLASSES if condition_classes is None else [
            [str(item) for item in group] for group in condition_classes
        ]
        self.condition_to_id = [
            {label: idx for idx, label in enumerate(class_group)}
            for class_group in self.condition_classes
        ]
        self.encoded_seqs = []
        self.conditions = []
        for _, row in df.iterrows():
            seq = self.get_sequence(row)
            cond = self.get_condition(row)
            if seq is None or cond is None:
                continue
            self.encoded_seqs.append(self.encode_seq(seq))
            self.conditions.append(cond)

    def __len__(self):
        return len(self.encoded_seqs)

    def encode_seq(self, s):
        ids = []
        for ch in s.upper():
            ids.append(VOCAB[ch])
        return torch.tensor(ids, dtype=torch.long)

    def get_sequence(self, row):
        if self.sequence_column is not None:
            seq = str(row[self.sequence_column]).upper()
            if len(seq) == self.seq_len and all(base in VOCAB for base in seq):
                return seq
            return None
        if self.genome is None:
            return None
        return self.generate_full_sequence(row["start"], row["end"], row["chrom"])

    def get_condition(self, row):
        if set(self.condition_names).issubset(row.index):
            ids = []
            for condition_name, mapping in zip(
                self.condition_names, self.condition_to_id, strict=True
            ):
                value = str(row[condition_name])
                if value not in mapping:
                    return None
                ids.append(mapping[value])
            return ids
        if "level" in row.index:
            return [int(row["level"]) - self.level_offset]
        return None

    def generate_full_sequence(self, start, end, chrom):
        chrom_sequence = self.genome[chrom]
        min_start_pos = max(0, end - self.seq_len)
        max_start_pos = min(start, len(chrom_sequence) - self.seq_len)
        if max_start_pos < min_start_pos:
            return None
        start_pos = random.randint(min_start_pos, max_start_pos)
        full_seq = chrom_sequence[start_pos : start_pos + self.seq_len].seq
        if "N" in full_seq:
            return None
        return full_seq

    def __getitem__(self, idx):
        encoded_seq = self.encoded_seqs[idx]
        if self.typer == "rec":
            x = encoded_seq
            y = encoded_seq
        else:
            bos = torch.tensor([BOS_TOKEN_ID], dtype=torch.long)
            x = torch.cat([bos, encoded_seq[:-1]], dim=0)
            y = encoded_seq
        cond = torch.tensor(self.conditions[idx], dtype=torch.long)
        if cond.numel() == 1:
            cond = cond.view(())
        return x, y, cond


def decode_seq(ids):
    return "".join(ID2BASE.get(int(i), "N") for i in ids)


def save_examples(predictions, output_path, max_examples=20, *, compact=False):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    saved = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for batch_out in predictions:
            x = batch_out["x"]
            cond = batch_out["levels"]
            recon = batch_out["recon"]
            gen = batch_out["gen"]
            batch_size = x.size(0)
            for i in range(batch_size):
                cond_value = cond[i].tolist()
                if compact:
                    row = {
                        "id": saved,
                        "cond": cond_value,
                        "generation_seq": decode_seq(gen[i].tolist()),
                    }
                else:
                    row = {
                        "id": saved,
                        "cond": cond_value,
                        "test_x": x[i].tolist(),
                        "reconstruction": recon[i].tolist(),
                        "generation": gen[i].tolist(),
                        "test_x_seq": decode_seq(x[i].tolist()),
                        "reconstruction_seq": decode_seq(recon[i].tolist()),
                        "generation_seq": decode_seq(gen[i].tolist()),
                    }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                saved += 1
                if saved >= max_examples:
                    return
