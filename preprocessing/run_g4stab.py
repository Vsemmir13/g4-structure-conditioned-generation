import argparse
import os
import subprocess
import sys
import tempfile

import pandas as pd


def sequence_column(df):
    for column in ("g4_sequence", "sequence"):
        if column in df.columns:
            return column
    raise ValueError("Input table must contain g4_sequence or sequence")


def parse_args():
    p = argparse.ArgumentParser(description="Run G4STAB and normalize predictions for this project.")
    p.add_argument("--records_csv", default="data/processed/g4_structure_records.csv")
    p.add_argument("--output_csv", default="data/processed/g4stab_predictions.csv")
    p.add_argument("--tool_dir", default="external_tools/G4STAB")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--salt_k", type=float, default=100.0)
    p.add_argument("--salt_na", type=float, default=0.0)
    p.add_argument("--salt_other", type=float, default=0.0)
    p.add_argument("--ph", type=float, default=7.0)
    return p.parse_args()


def main():
    args = parse_args()
    records_df = pd.read_csv(args.records_csv)
    seq_col = sequence_column(records_df)
    records_df = records_df[records_df[seq_col].notna()].copy()
    if records_df.empty:
        raise ValueError(f"No rows with non-empty {seq_col}")
    tool_dir = os.path.abspath(args.tool_dir)
    predictor_path = os.path.join(tool_dir, "g4stab_predictor.py")
    models_dir = os.path.join(tool_dir, "trained_models")
    if not os.path.exists(predictor_path):
        raise FileNotFoundError(predictor_path)
    if not os.path.exists(models_dir):
        raise FileNotFoundError(models_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = os.path.join(tmpdir, "matplotlib")
        env["XDG_CACHE_HOME"] = os.path.join(tmpdir, "cache")
        os.makedirs(env["MPLCONFIGDIR"], exist_ok=True)
        os.makedirs(env["XDG_CACHE_HOME"], exist_ok=True)

        input_csv = os.path.join(tmpdir, "g4stab_input.csv")
        raw_output_csv = os.path.join(tmpdir, "g4stab_raw_output.csv")
        input_df = pd.DataFrame(
            {
                "sample_id": records_df["sample_id"].values,
                "sequence": records_df[seq_col].values,
                "salt_k": args.salt_k,
                "salt_na": args.salt_na,
                "salt_other": args.salt_other,
                "ph": args.ph,
            }
        )
        input_df.to_csv(input_csv, index=False)
        cmd = [
            args.python,
            predictor_path,
            "-f",
            input_csv,
            "-o",
            raw_output_csv,
            "--models-dir",
            models_dir,
        ]
        subprocess.run(cmd, check=True, cwd=tool_dir, env=env)
        raw = pd.read_csv(raw_output_csv)

    if len(raw) != len(input_df):
        raise RuntimeError(
            f"G4STAB returned {len(raw)} rows for {len(input_df)} input rows; cannot restore sample_id safely."
        )
    out = pd.DataFrame(
        {
            "sample_id": input_df["sample_id"].values,
            "g4_sequence": input_df["sequence"].values,
            "predicted_tm": raw["ensemble_mean"].values,
            "g4stab_std": raw["ensemble_std"].values,
            "salt_k": raw["salt_k"].values,
            "salt_na": raw["salt_na"].values,
            "salt_other": raw["salt_other"].values,
            "ph": raw["ph"].values,
        }
    )
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved G4STAB predictions to {args.output_csv}")

if __name__ == "__main__":
    main()
