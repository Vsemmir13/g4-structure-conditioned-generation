import inspect
import logging

import torch


def torch_load(path, map_location="cpu"):
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def checkpoint_hparams(ckpt):
    if isinstance(ckpt, dict):
        hparams = ckpt.get("hyper_parameters") or ckpt.get("hparams")
        if hparams:
            return dict(hparams)
    return {}


def checkpoint_state_dict(ckpt):
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def load_weights(model, ckpt_path, strict=True):
    ckpt = torch_load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint_state_dict(ckpt), strict=strict)
    logging.info("Loaded checkpoint weights from %s", ckpt_path)


def signature_kwargs(callable_obj, values):
    allowed = inspect.signature(callable_obj).parameters
    return {key: value for key, value in values.items() if key in allowed}


def count_trainable_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
