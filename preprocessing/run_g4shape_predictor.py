import argparse
import os
import pickle

import numpy as np
import pandas as pd

TOPOLOGY_LABELS = {
    0: "parallel",
    1: "antiparallel",
    2: "hybrid",
    -1: "mixed",
    -2: "invalid",
}


def sequence_column(df):
    for column in ("g4_sequence", "sequence"):
        if column in df.columns:
            return column
    raise ValueError("Input table must contain g4_sequence or sequence")


def parse_args():
    p = argparse.ArgumentParser(description="Run G4ShapePredictor headlessly on G4 sequences.")
    p.add_argument("--records_csv", default="data/processed/g4_structure_records.csv")
    p.add_argument("--output_csv", default="data/processed/g4shape_predictions.csv")
    p.add_argument(
        "--tool_dir",
        default="external_tools/G4ShapePredictor/g4sp application code",
        help="Directory containing G4ShapePredictor pickle models.",
    )
    p.add_argument("--model_name", default="RandomForest (default)")
    p.add_argument("--pad_length", type=int, default=100)
    return p.parse_args()


def validate_sequence(sequence):
    return all(character in "ATCGN" for character in sequence)


def convert_sequence(sequence, pad_length):
    seqsdict = {"A": 1, "T": 2, "C": 3, "G": 4, "N": 0}
    sequence = str(sequence).upper()
    if not validate_sequence(sequence):
        return None
    values = [seqsdict[character] for character in sequence]
    while len(values) < pad_length:
        values.insert(0, 0)
        values.append(0)
    while len(values) > pad_length:
        values.pop()
    return np.array(values)


def load_model(tool_dir, model_name):
    model_path = os.path.join(tool_dir, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except ValueError as exc:
        raise RuntimeError(
            "Failed to load the G4ShapePredictor pickle model. "
            "This usually means scikit-learn is incompatible with the upstream pickle. "
            "Use scikit-learn==1.0.2, or create the upstream environment with: "
            "conda env create -f external_tools/G4ShapePredictor/environment.yml"
        ) from exc


def main():
    args = parse_args()
    records_df = pd.read_csv(args.records_csv)
    seq_col = sequence_column(records_df)
    records_df = records_df[records_df[seq_col].notna()].copy()
    if records_df.empty:
        raise ValueError(f"No rows with non-empty {seq_col}")
    model = load_model(args.tool_dir, args.model_name)
    features = []
    valid_sample_ids = []
    valid_sequences = []
    invalid_rows = []
    for _, row in records_df.iterrows():
        converted = convert_sequence(row[seq_col], args.pad_length)
        if converted is None:
            invalid_rows.append(row["sample_id"])
            continue
        features.append(converted)
        valid_sample_ids.append(row["sample_id"])
        valid_sequences.append(row[seq_col])
    if not features:
        raise ValueError("No valid G4 sequences for G4ShapePredictor")
    x = np.stack(features)
    probs = model.predict_proba(x)
    preds = np.argmax(probs, axis=1)
    out = pd.DataFrame(
        {
            "sample_id": valid_sample_ids,
            "g4_sequence": valid_sequences,
            "topology_label": [TOPOLOGY_LABELS[int(pred)] for pred in preds],
            "topology_prob_parallel": probs[:, 0],
            "topology_prob_antiparallel": probs[:, 1],
            "topology_prob_hybrid": probs[:, 2],
            "topology_confidence": probs.max(axis=1),
            "g4shape_model": args.model_name,
        }
    )
    if invalid_rows:
        invalid_df = pd.DataFrame(
            {
                "sample_id": invalid_rows,
                "g4_sequence": "",
                "topology_label": "invalid",
                "topology_prob_parallel": 0.0,
                "topology_prob_antiparallel": 0.0,
                "topology_prob_hybrid": 0.0,
                "topology_confidence": 0.0,
                "g4shape_model": args.model_name,
            }
        )
        out = pd.concat([out, invalid_df], ignore_index=True)
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved G4ShapePredictor predictions to {args.output_csv}")

if __name__ == "__main__":
    main()
