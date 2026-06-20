import argparse
import logging
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from utils.config import CFG
from utils.data_utils import (
    QuadDataset,
    save_examples,
    split_data,
)
from utils.gen_metrics_callback import GenerativeMetricsCallback
from utils.logging_utils import setup_logging
from utils.model_factory import build_model
from utils.model_utils import count_trainable_params, load_weights


def parse_args():
    p = argparse.ArgumentParser(description="Train/test G4 DNA generative models.")
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--file_path_quadruplex", default="../../quadruplex/data/EQ_hg38_lifted.bed")
    p.add_argument("--file_path_seq", default="../../quadruplex/data/hg38.fa")
    p.add_argument("--processed_csv", default="data/processed/g4_structure_conditions.csv")
    p.add_argument(
        "--condition_mode",
        default=CFG["condition_mode"],
        choices=sorted(CFG["condition_specs"].keys()),
    )
    p.add_argument(
        "--model_type",
        default="dfm",
        choices=["lstm", "vae", "dfm", "dfm_transformer", "ddsm"],
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_epochs", "--epochs", type=int, default=100000)
    p.add_argument("--max_steps", type=int, default=450000)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--devices", default="auto")
    p.add_argument("--run_mode", default="train", choices=["train", "test"])
    p.add_argument("--ckpt_path")
    p.add_argument(
        "--metric_samples", "--val_metrics_sample_size", type=int, default=CFG["metric_samples"]
    )
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument("--ddsm_noise_table_path")
    p.add_argument("--ddsm_time_dependent_weights_path")
    p.add_argument(
        "--guidance_mode",
        default="probability_addition",
        choices=[
            "score",
            "score_free",
            "probability_addition",
            "probability_tilt",
            "vectorfield_addition",
            "logit",
        ],
    )
    p.add_argument("--progress_bar", action="store_true")
    args = p.parse_args()
    spec = CFG["condition_specs"][args.condition_mode]
    args.condition_names = spec["names"]
    args.condition_classes = spec["classes"]
    if args.run_mode == "test" and not args.ckpt_path:
        raise ValueError("--ckpt_path is required for --run_mode test")
    return args


def make_loaders(args):
    typer = "gen" if args.model_type == "lstm" else "rec"
    split_dfs = split_data(
        args.processed_csv,
        split=CFG["split"],
        val_split=CFG["val_split"],
        seed=42,
        condition_names=args.condition_names,
        condition_classes=args.condition_classes,
        max_g4stab_std=CFG["max_g4stab_std"],
        stability_low_tm=CFG["stability_low_tm"],
        stability_high_tm=CFG["stability_high_tm"],
        log_sizes=True,
    )
    args.num_cls = [len(class_group) for class_group in args.condition_classes]
    if len(args.num_cls) == 1:
        args.num_cls = args.num_cls[0]
    datasets = [
        QuadDataset(
            df,
            file_path_seq=args.file_path_seq,
            condition_names=args.condition_names,
            condition_classes=args.condition_classes,
            typer=typer,
            seq_len=CFG["seq_len"],
        )
        for df in (split_dfs["train"], split_dfs["val"], split_dfs["test"])
    ]
    train_ds, val_ds, test_ds = datasets
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    return train_ds, train_loader, val_loader, test_loader


def accelerator_and_strategy(devices_arg):
    if torch.cuda.is_available():
        devices = torch.cuda.device_count() if devices_arg == "auto" else int(devices_arg)
        if devices > torch.cuda.device_count():
            raise ValueError(
                f"--devices={devices}, but torch sees only {torch.cuda.device_count()} CUDA device(s)"
            )
        strategy = DDPStrategy(find_unused_parameters=False) if devices > 1 else "auto"
        return "gpu", devices, strategy
    if torch.backends.mps.is_available():
        return "mps", 1, "auto"
    return "cpu", 1, "auto"


def make_trainer(args, callbacks):
    accelerator, devices, strategy = accelerator_and_strategy(args.devices)
    logging.info(
        "Init trainer on accelerator=%s devices=%s strategy=%s", accelerator, devices, strategy
    )
    return pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        enable_progress_bar=args.progress_bar,
        callbacks=callbacks,
        logger=TensorBoardLogger(f"logs/{args.model_type}", name=args.experiment_name),
        default_root_dir=os.environ.get(
            "MODEL_DIR", f"logs/{args.model_type}/{args.experiment_name}"
        ),
        check_val_every_n_epoch=CFG["check_val_every_n_epoch"],
    )


def make_callbacks(args, train_ds):
    callbacks = [
        GenerativeMetricsCallback(
            train_sequences=train_ds.encoded_seqs,
            seq_len=CFG["seq_len"],
            sample_size=args.metric_samples,
            g4hunter_window=CFG["g4hunter_window"],
        )
    ]
    if args.run_mode == "train":
        callbacks.insert(
            0,
            ModelCheckpoint(
                dirpath=os.environ.get(
                    "MODEL_DIR", f"checkpoints/{args.model_type}/{args.experiment_name}"
                ),
                save_top_k=CFG["checkpoint_save_top_k"],
                save_last=True,
                monitor="val_perplexity",
                mode="min",
            ),
        )
    return callbacks


def train_or_test(args, model, trainer, train_loader, val_loader, test_loader):
    if args.run_mode == "train":
        fit_kwargs = {"ckpt_path": args.ckpt_path} if args.ckpt_path else {}
        if args.ckpt_path:
            logging.info("Resuming training from %s", args.ckpt_path)
        trainer.fit(model, train_loader, val_loader, **fit_kwargs)
    else:
        logging.info("Skipping training because run_mode=test")

    results = trainer.test(model, dataloaders=test_loader)
    logging.info("Test results: %s", results)
    predictions = trainer.predict(model, dataloaders=test_loader)
    save_examples(
        predictions,
        f"examples/{args.model_type}/{args.experiment_name}.jsonl",
        max_examples=30,
        compact=True,
    )


def main():
    args = parse_args()
    setup_logging()
    logging.info("Loading data and dataloaders")
    train_ds, train_loader, val_loader, test_loader = make_loaders(args)

    model = build_model(args)
    logging.info("Model trainable parameters: %s", f"{count_trainable_params(model):,}")
    if args.run_mode == "test":
        load_weights(model, args.ckpt_path)

    trainer = make_trainer(args, make_callbacks(args, train_ds))
    train_or_test(args, model, trainer, train_loader, val_loader, test_loader)
    logging.info("Finish")


if __name__ == "__main__":
    main()
