import inspect
import logging
import math
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch

from models.melanoma_model import MelanomaCNNModel
from utils.model_utils import torch_load

DEFAULT_HYENADNA_MODEL = "LongSafari/hyenadna-tiny-1k-seqlen-hf"


def _load_melanoma_fbd_checkpoint(map_location):
    rel_path = os.path.join("checkpoints", "melanoma_fbd", "epoch=9-step=5540.ckpt")
    project_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), rel_path)
    for path in (rel_path, project_path):
        if os.path.exists(path):
            return torch_load(path, map_location=map_location)
    return torch_load(rel_path, map_location=map_location)


def _frechet_distance(real_emb, gen_emb, eps=1e-6):
    mu_r = np.mean(real_emb, axis=0)
    mu_g = np.mean(gen_emb, axis=0)
    cov_r = np.cov(real_emb, rowvar=False)
    cov_g = np.cov(gen_emb, rowvar=False)
    if cov_r.ndim == 0:
        cov_r = np.array([[float(cov_r)]], dtype=np.float64)
    if cov_g.ndim == 0:
        cov_g = np.array([[float(cov_g)]], dtype=np.float64)
    cov_r = cov_r + np.eye(cov_r.shape[0]) * eps
    cov_g = cov_g + np.eye(cov_g.shape[0]) * eps
    cov_prod = cov_r @ cov_g
    cov_prod = 0.5 * (cov_prod + cov_prod.T)
    eigvals = np.linalg.eigvalsh(cov_prod)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    tr_covmean = float(np.sum(np.sqrt(eigvals)))
    delta = mu_r - mu_g
    fbd = float(delta @ delta + np.trace(cov_r) + np.trace(cov_g) - 2.0 * tr_covmean)
    return max(fbd, 0.0)


class MelanomaEmbedder:

    def __init__(self, device):
        self.device = device
        self.model = MelanomaCNNModel(
            vocab_size=4,
            hidden_dim=128,
            num_cnn_stacks=4,
            p_dropout=0.2,
            num_classes=47,
            classifier=True,
            clean_data=True,
        ).to(device)
        state = _load_melanoma_fbd_checkpoint(map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError("Unsupported checkpoint format for clean classifier embedder")
        cleaned = {}
        for k, v in state.items():
            nk = k
            for prefix in ("model.", "cls_model.", "clean_cls_model."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix) :]
            cleaned[nk] = v
        self.model.load_state_dict(cleaned, strict=False)
        self.model.eval()

    @torch.no_grad()
    def encode(self, seq_ids):
        t = torch.zeros(seq_ids.size(0), device=self.device)
        _, emb = self.model(seq_ids.to(self.device), t=t, return_embedding=True)
        return emb.detach().cpu().numpy()


class HyenaDNAEmbedder:
    def __init__(self, device, model_name=DEFAULT_HYENADNA_MODEL, seq_len=512):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install transformers to use HyenaDNA FBD: pip install transformers"
            ) from exc

        self.device = device
        self.seq_len = int(seq_len)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(
            device
        )
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

    @staticmethod
    def _ids_to_seq_batch(seq_ids):
        alphabet = "ACGT"
        seqs = []
        for row in seq_ids.detach().cpu():
            seqs.append("".join(alphabet[int(t)] if 0 <= int(t) < 4 else "N" for t in row.tolist()))
        return seqs

    @torch.no_grad()
    def encode(self, seq_ids):
        seqs = self._ids_to_seq_batch(seq_ids)
        tokens = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.seq_len,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        outputs = self.model(**tokens, output_hidden_states=True)
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise RuntimeError("HyenaDNA model did not return hidden states")
        hidden = outputs.hidden_states[-1]
        mask = tokens.get("attention_mask")
        if mask is None:
            emb = hidden.mean(dim=1)
        else:
            mask = mask[:, : hidden.size(1)].unsqueeze(-1).to(hidden.dtype)
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return emb.detach().cpu().numpy()


class GenerativeMetricsCallback(pl.Callback):
    def __init__(
        self,
        *,
        train_sequences,
        seq_len,
        sample_size=256,
        log_prefix="val_",
        g4hunter_window=25,
        hyenadna_model_name=None,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.sample_size = int(sample_size)
        self.log_prefix = str(log_prefix)
        self._train_set = {tuple(s.tolist()) for s in train_sequences}
        self._g4hunter_window = int(g4hunter_window)
        self._last_val_batch = None
        self._last_test_batch = None
        self._melanoma = None
        self._hyenadna = None
        self._fb = None
        self._hyenadna_model_name = hyenadna_model_name or os.environ.get(
            "HYENADNA_FBD_MODEL", DEFAULT_HYENADNA_MODEL
        )
        self._warned_fbd = set()
        self._gen_chunk_size = 32

    @staticmethod
    def _ids_to_seq(ids):
        alphabet = "ACGT"
        return "".join(alphabet[int(t)] if 0 <= int(t) < 4 else "N" for t in ids.tolist())

    @staticmethod
    def _g4hunter_base_scores(seq):
        scores = np.zeros(len(seq), dtype=np.float32)
        i = 0
        while i < len(seq):
            ch = seq[i]
            j = i + 1
            while j < len(seq) and seq[j] == ch:
                j += 1
            run_len = min(4, j - i)
            if ch == "G":
                scores[i:j] = float(run_len)
            elif ch == "C":
                scores[i:j] = float(-run_len)
            i = j
        return scores

    @classmethod
    def _g4hunter_seq_score(cls, seq, window):
        base = cls._g4hunter_base_scores(seq)
        if len(base) == 0:
            return 0.0
        if window <= 1:
            return float(np.max(np.abs(base)))
        if len(base) < window:
            return 0.0
        kernel = np.ones(window, dtype=np.float32) / float(window)
        smooth = np.convolve(base, kernel, mode="valid")
        return float(np.max(np.abs(smooth)))

    @classmethod
    def _g4hunter_scores(cls, seqs, window):
        return np.array([cls._g4hunter_seq_score(s, window=window) for s in seqs], dtype=np.float32)

    def _run_generative_metrics(
        self, trainer, pl_module, _x, y, cond, log_prefix, loss_metric_names
    ):
        device = pl_module.device
        n = min(self.sample_size, int(cond.size(0)))
        if n <= 0:
            return
        cond = cond[:n].to(device)
        y = y[:n].to(device)

        # --- generation ---
        if not hasattr(pl_module, "generate") or not callable(pl_module.generate):
            return
        gen_chunks = []
        with torch.no_grad():
            for start in range(0, n, self._gen_chunk_size):
                end = min(n, start + self._gen_chunk_size)
                cond_chunk = cond[start:end]
                gen_chunk = self._generate_with_signature(
                    pl_module, cond_chunk, seq_len=int(y.size(1))
                )
                gen_chunks.append(gen_chunk.long().detach().cpu())
        gen = torch.cat(gen_chunks, dim=0)
        real = y.long().detach().cpu()

        # --- perplexity from epoch loss (val_loss / test loss) ---
        loss_tensor = None
        for name in loss_metric_names:
            v = trainer.callback_metrics.get(name)
            if isinstance(v, torch.Tensor):
                loss_tensor = v
                break
        if loss_tensor is not None:
            ppl = float(math.exp(float(loss_tensor.detach().cpu().item())))
            pl_module.log(
                log_prefix + "perplexity",
                ppl,
                prog_bar=True,
                logger=True,
                on_epoch=True,
                sync_dist=True,
            )

        # --- novelty ---
        novelty = float(np.mean([tuple(s.tolist()) not in self._train_set for s in gen]))
        pl_module.log(
            log_prefix + "novelty",
            novelty,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            sync_dist=True,
        )

        # --- FBD metrics ---
        self._log_fbd(
            pl_module,
            log_prefix + "melanoma_fbd",
            "melanoma",
            lambda: MelanomaEmbedder(device),
            real,
            gen,
        )
        self._log_fbd(
            pl_module,
            log_prefix + "hyenadna_fbd",
            "hyenadna",
            lambda: HyenaDNAEmbedder(
                device, model_name=self._hyenadna_model_name, seq_len=self.seq_len
            ),
            real,
            gen,
        )

        # --- G4Hunter similarity ---
        real_seqs = [self._ids_to_seq(s) for s in real]
        gen_seqs = [self._ids_to_seq(s) for s in gen]
        real_g4 = self._g4hunter_scores(real_seqs, window=self._g4hunter_window)
        gen_g4 = self._g4hunter_scores(gen_seqs, window=self._g4hunter_window)
        pl_module.log(
            log_prefix + "g4hunter_real_mean",
            float(np.mean(real_g4)),
            prog_bar=False,
            logger=True,
            on_epoch=True,
            sync_dist=True,
        )
        pl_module.log(
            log_prefix + "g4hunter_gen_mean",
            float(np.mean(gen_g4)),
            prog_bar=False,
            logger=True,
            on_epoch=True,
            sync_dist=True,
        )
        pl_module.log(
            log_prefix + "g4hunter_gap",
            float(np.mean(np.abs(real_g4 - gen_g4))),
            prog_bar=True,
            logger=True,
            on_epoch=True,
            sync_dist=True,
        )

        G4_THRESHOLD = 1.5
        real_g4_frac = np.mean(real_g4 > G4_THRESHOLD)
        gen_g4_frac = np.mean(gen_g4 > G4_THRESHOLD)
        pl_module.log(
            log_prefix + "g4_real_frac",
            real_g4_frac,
            prog_bar=False,
            logger=True,
            on_epoch=True,
            sync_dist=True,
        )
        pl_module.log(
            log_prefix + "g4_gen_frac",
            gen_g4_frac,
            prog_bar=False,
            logger=True,
            on_epoch=True,
            sync_dist=True,
        )
        pl_module.log(
            log_prefix + "g4_frac_gap",
            np.abs(real_g4_frac - gen_g4_frac),
            prog_bar=False,
            logger=True,
            on_epoch=True,
            sync_dist=True,
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        self._val_cond, self._val_y = [], []
        self._val_seen = 0

    def _log_fbd(self, pl_module, metric_name, embedder_name, embedder_factory, real, gen):
        attr = f"_{embedder_name}"
        try:
            embedder = getattr(self, attr)
            if embedder is None:
                embedder = embedder_factory()
                setattr(self, attr, embedder)
            real_emb = embedder.encode(real.to(pl_module.device))
            gen_emb = embedder.encode(gen.to(pl_module.device))
            pl_module.log(
                metric_name,
                _frechet_distance(real_emb, gen_emb),
                prog_bar=False,
                logger=True,
                on_epoch=True,
                sync_dist=True,
            )
        except Exception as exc:
            if embedder_name not in self._warned_fbd:
                logging.warning(f"Skipping {metric_name}: {exc}")
                self._warned_fbd.add(embedder_name)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, *args):
        _, y, cond = batch
        self._val_seen = self._reservoir_update(
            self._val_cond, self._val_y, self._val_seen, cond, y
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._val_cond:
            return

        self._run_generative_metrics(
            trainer,
            pl_module,
            None,
            y=torch.stack(self._val_y),
            cond=torch.stack(self._val_cond),
            log_prefix=self.log_prefix,
            loss_metric_names=("val_loss",),
        )

    def on_test_epoch_start(self, trainer, pl_module):
        self._test_cond, self._test_y = [], []
        self._test_seen = 0

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, *args):
        _, y, cond = batch
        self._test_seen = self._reservoir_update(
            self._test_cond, self._test_y, self._test_seen, cond, y
        )

    def on_test_epoch_end(self, trainer, pl_module):
        if not self._test_cond:
            return

        self._run_generative_metrics(
            trainer,
            pl_module,
            None,
            y=torch.stack(self._test_y),
            cond=torch.stack(self._test_cond),
            log_prefix="test_",
            loss_metric_names=("test_loss",),
        )

    def _reservoir_update(self, store_cond, store_y, seen, cond, y):
        cond_cpu = cond.detach().cpu()
        y_cpu = y.detach().cpu()

        for i in range(len(cond_cpu)):
            seen += 1
            if len(store_cond) < self.sample_size:
                store_cond.append(cond_cpu[i])
                store_y.append(y_cpu[i])
            else:
                j = random.randint(0, seen - 1)
                if j < self.sample_size:
                    store_cond[j] = cond_cpu[i]
                    store_y[j] = y_cpu[i]
        return seen

    @staticmethod
    def _generate_with_signature(pl_module, cond, seq_len):
        generate = pl_module.generate
        try:
            sig = inspect.signature(generate)
            params = sig.parameters
            kwargs = {}
            if "seq_len" in params:
                kwargs["seq_len"] = seq_len
            return generate(cond, **kwargs)
        except Exception:
            try:
                return generate(cond)
            except TypeError:
                return generate(cond, seq_len=seq_len)
