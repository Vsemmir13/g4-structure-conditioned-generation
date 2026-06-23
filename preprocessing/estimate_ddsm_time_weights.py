import argparse
import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from external_tools.ddsm.ddsm import (
    UnitStickBreakingTransform,
    diffusion_fast_flatdirichlet,
    gx_to_gv,
)
from utils.config import CFG
from utils.data_utils import QuadDataset, split_data
from utils.model_utils import torch_load


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate DDSM time-dependent loss weights.")
    parser.add_argument("--processed_csv", default="data/processed/g4_structure_conditions.csv")
    parser.add_argument("--file_path_seq", default="data/hg38.fa")
    parser.add_argument(
        "--condition_mode",
        default=CFG["condition_mode"],
        choices=sorted(CFG["condition_specs"].keys()),
    )
    parser.add_argument("--noise_table_path", required=True)
    parser.add_argument("--output_path", default="data/ddsm_noise/time_dependent_weights.pth")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_noise_table(path):
    table = torch_load(path, map_location="cpu")
    if len(table) == 5:
        v_one, _, v_one_loggrad, _, timepoints = table
    elif len(table) == 3:
        v_one, v_one_loggrad, timepoints = table
    else:
        raise ValueError("Expected a DDSM noise table with 3 or 5 tensors")
    return v_one.float().cpu(), v_one_loggrad.float().cpu(), timepoints.float().cpu()


def estimate_weights(loader, v_one, v_one_loggrad, timepoints, device, passes):
    num_steps = int(timepoints.numel())
    vocab_size = int(v_one.shape[-1] + 1)
    counts = torch.zeros(num_steps, device=device)
    cums = torch.zeros(num_steps, device=device)
    sb = UnitStickBreakingTransform()
    speed = 2 / (
        torch.ones(vocab_size - 1, device=device)
        + torch.arange(vocab_size - 1, 0, -1, device=device).float()
    )

    for pass_idx in range(int(passes)):
        logging.info("DDSM weight estimation pass %d/%d", pass_idx + 1, passes)
        for seq, _, _ in loader:
            seq = seq.to(device)
            one_hot = F.one_hot(seq, num_classes=vocab_size).float()
            time_inds = torch.randint(0, num_steps, (seq.size(0),), device=device)
            perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(
                one_hot.detach().cpu(),
                time_inds.detach().cpu(),
                v_one,
                v_one_loggrad,
                symmetrize=False,
            )
            perturbed_x = perturbed_x.to(device)
            perturbed_x_grad = perturbed_x_grad.to(device)
            perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()
            contribution = (
                perturbed_v
                * (1 - perturbed_v)
                * speed[(None,) * (one_hot.ndim - 1)]
                * gx_to_gv(perturbed_x_grad, perturbed_x).pow(2)
            )
            contribution = contribution.view(seq.size(0), -1).mean(dim=1).detach()
            cums.scatter_add_(0, time_inds, contribution)
            counts.scatter_add_(0, time_inds, torch.ones_like(contribution))

    missing = counts == 0
    if missing.any():
        logging.warning("Filling %d empty time bins with the mean observed weight", int(missing.sum()))
    weights = cums / counts.clamp_min(1)
    mean_observed = weights[~missing].mean() if (~missing).any() else torch.tensor(1.0, device=device)
    weights = torch.where(missing, mean_observed, weights)
    return weights / weights.mean().clamp_min(1e-12)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.manual_seed(args.seed)
    spec = CFG["condition_specs"][args.condition_mode]
    split_dfs = split_data(
        args.processed_csv,
        max_g4stab_std=CFG["max_g4stab_std"],
        stability_low_tm=CFG["stability_low_tm"],
        stability_high_tm=CFG["stability_high_tm"],
        log_sizes=True,
    )
    dataset = QuadDataset(
        split_dfs["train"],
        file_path_seq=args.file_path_seq,
        condition_names=spec["names"],
        condition_classes=spec["classes"],
        typer="rec",
        seq_len=CFG["seq_len"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device=%s for %d training sequences", device, len(dataset))
    v_one, v_one_loggrad, timepoints = load_noise_table(args.noise_table_path)
    weights = estimate_weights(loader, v_one, v_one_loggrad, timepoints, device, args.passes).cpu()

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save({"time_dependent_weights": weights, "timepoints": timepoints}, args.output_path)
    logging.info("Saved DDSM time-dependent weights to %s", args.output_path)


if __name__ == "__main__":
    main()
