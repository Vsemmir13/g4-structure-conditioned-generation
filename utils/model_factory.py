from models.ddsm_module import QuadDDSMModule
from models.dfm_module import QuadDFMModule
from models.lstm import QuadLSTM
from models.vae import DNAConvVAE
from utils.config import CFG
from utils.model_utils import (
    checkpoint_hparams,
    checkpoint_state_dict,
    signature_kwargs,
    torch_load,
)


def build_model(args):
    num_cls = getattr(args, "num_cls", CFG["num_cls"])
    if args.model_type == "lstm":
        return QuadLSTM(vocab_size=5, num_cls=num_cls, lr=CFG["lr"], **CFG["lstm"])

    if args.model_type == "vae":
        return DNAConvVAE(
            seq_len=CFG["seq_len"], num_cls=num_cls, lr=CFG["lr"], **CFG["vae"]
        )

    if args.model_type == "ddsm":
        ddsm_cfg = dict(CFG["ddsm"])
        noise_table_path = getattr(args, "ddsm_noise_table_path", None)
        time_weights_path = getattr(args, "ddsm_time_dependent_weights_path", None)
        if noise_table_path:
            ddsm_cfg["noise_table_path"] = noise_table_path
        if time_weights_path:
            ddsm_cfg["time_dependent_weights_path"] = time_weights_path
        return QuadDDSMModule(
            seq_len=CFG["seq_len"],
            vocab_size=4,
            num_cls=num_cls,
            lr=CFG["lr"],
            guidance_scale=args.guidance_scale,
            **ddsm_cfg,
        )

    return QuadDFMModule(
        backbone="transformer" if args.model_type == "dfm_transformer" else "cnn",
        seq_len=CFG["seq_len"],
        vocab_size=4,
        num_cls=num_cls,
        lr=CFG["lr"],
        guidance_scale=args.guidance_scale,
        guidance_mode=args.guidance_mode,
        **CFG["dfm"],
    )


def build_model_from_checkpoint(model_type, ckpt_path, fallback_args=None):
    ckpt = torch_load(ckpt_path, map_location="cpu")
    hparams = checkpoint_hparams(ckpt)
    state_dict = checkpoint_state_dict(ckpt)

    if model_type == "lstm":
        kwargs = signature_kwargs(QuadLSTM, hparams)
        if not kwargs:
            num_cls = getattr(fallback_args, "num_cls", CFG["num_cls"])
            kwargs = dict(vocab_size=5, num_cls=num_cls, lr=CFG["lr"], **CFG["lstm"])
        model = QuadLSTM(**kwargs)
    elif model_type == "vae":
        kwargs = signature_kwargs(DNAConvVAE, hparams)
        if not kwargs:
            num_cls = getattr(fallback_args, "num_cls", CFG["num_cls"])
            kwargs = dict(seq_len=CFG["seq_len"], num_cls=num_cls, lr=CFG["lr"], **CFG["vae"])
        model = DNAConvVAE(**kwargs)
    elif model_type == "ddsm":
        kwargs = signature_kwargs(QuadDDSMModule, hparams)
        if not kwargs:
            num_cls = getattr(fallback_args, "num_cls", CFG["num_cls"])
            guidance_scale = getattr(fallback_args, "guidance_scale", 1.0)
            ddsm_cfg = dict(CFG["ddsm"])
            noise_table_path = getattr(fallback_args, "ddsm_noise_table_path", None)
            time_weights_path = getattr(fallback_args, "ddsm_time_dependent_weights_path", None)
            if noise_table_path:
                ddsm_cfg["noise_table_path"] = noise_table_path
            if time_weights_path:
                ddsm_cfg["time_dependent_weights_path"] = time_weights_path
            kwargs = dict(
                seq_len=CFG["seq_len"],
                vocab_size=4,
                num_cls=num_cls,
                lr=CFG["lr"],
                guidance_scale=guidance_scale,
                **ddsm_cfg,
            )
        model = QuadDDSMModule(**kwargs)
    else:
        kwargs = signature_kwargs(QuadDFMModule, hparams)
        if kwargs:
            model = QuadDFMModule(**kwargs)
        elif fallback_args is not None:
            model = build_model(fallback_args)
        else:
            raise ValueError("DFM checkpoint has no hparams; fallback_args is required")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
