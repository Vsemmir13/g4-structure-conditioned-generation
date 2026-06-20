import argparse
import json
import os

import torch

from utils.config import CFG
from utils.data_utils import decode_seq
from utils.model_factory import build_model_from_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Generate G4 sequences from a conditioned checkpoint.")
    p.add_argument(
        "--model_type",
        required=True,
        choices=["lstm", "vae", "dfm", "dfm_transformer", "ddsm"],
    )
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument(
        "--condition_mode",
        default=CFG["condition_mode"],
        choices=sorted(CFG["condition_specs"].keys()),
    )
    p.add_argument("--stability", choices=["low", "medium", "high"])
    p.add_argument("--topology", choices=["parallel", "hybrid", "antiparallel"])
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--guidance_mode", default="probability_addition")
    p.add_argument("--ddsm_noise_table_path")
    p.add_argument("--ddsm_time_dependent_weights_path")
    return p.parse_args()


def condition_spec(args):
    spec = CFG["condition_specs"][args.condition_mode]
    values = []
    for name in spec["names"]:
        if name == "topology_label":
            if args.topology is None:
                raise ValueError("--topology is required for topology or joint conditioning")
            values.append(args.topology)
        elif name == "stability_class":
            if args.stability is None:
                raise ValueError("--stability is required for stability or joint conditioning")
            values.append(args.stability)
        else:
            raise ValueError(f"Unsupported condition name: {name}")
    return spec, values


def encode_condition(spec, values, batch_size, device):
    ids = []
    for value, classes in zip(values, spec["classes"], strict=True):
        ids.append(classes.index(value))
    cond = torch.tensor(ids, dtype=torch.long, device=device)
    if len(ids) == 1:
        return cond.repeat(batch_size)
    return cond[None, :].repeat(batch_size, 1)


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate_batch(model, model_type, cond):
    if model_type == "lstm":
        return model.generate(cond, seq_len=CFG["seq_len"])
    return model.generate(cond)


def main():
    args = parse_args()
    spec, values = condition_spec(args)
    args.num_cls = [len(classes) for classes in spec["classes"]]
    if len(args.num_cls) == 1:
        args.num_cls = args.num_cls[0]

    device = resolve_device()
    model = build_model_from_checkpoint(args.model_type, args.ckpt_path, fallback_args=args).to(device)
    model.eval()

    out_dir = os.path.dirname(args.output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    written = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        while written < args.num_samples:
            batch_size = min(args.batch_size, args.num_samples - written)
            cond = encode_condition(spec, values, batch_size, device)
            with torch.no_grad():
                gen = generate_batch(model, args.model_type, cond)
            for row in gen.detach().cpu():
                record = {
                    "id": written,
                    "condition_mode": args.condition_mode,
                    "generation_seq": decode_seq(row.tolist()),
                }
                for name, value in zip(spec["names"], values, strict=True):
                    record[name] = value
                f.write(json.dumps(record) + "\n")
                written += 1


if __name__ == "__main__":
    main()
