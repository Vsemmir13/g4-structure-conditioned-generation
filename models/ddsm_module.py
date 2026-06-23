import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.condition_encoder import ConditionEncoder
from models.dfm_flow_utils import GaussianFourierProjection, simplex_proj

try:
    from external_tools.ddsm.ddsm import (
        Euler_Maruyama_sampler,
        UnitStickBreakingTransform,
        diffusion_fast_flatdirichlet,
        gx_to_gv,
    )
except ImportError:
    Euler_Maruyama_sampler = None
    UnitStickBreakingTransform = None
    diffusion_fast_flatdirichlet = None
    gx_to_gv = None


def _torch_load(path, map_location="cpu"):
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


class DDSMScoreCNN(nn.Module):
    def __init__(
        self,
        *,
        vocab_size=4,
        num_cls=3,
        hidden_dim=256,
        num_layers=20,
        dropout=0.1,
        time_embed_scale=30.0,
        classifier_free_guidance=False,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.classifier_free_guidance = bool(classifier_free_guidance)

        self.in_proj = nn.Conv1d(self.vocab_size, self.hidden_dim, kernel_size=9, padding=4)
        self.time_embedder = nn.Sequential(
            GaussianFourierProjection(embedding_dim=self.hidden_dim, scale=time_embed_scale),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.cond_embedder = ConditionEncoder(
            num_cls,
            self.hidden_dim,
            out_dim=self.hidden_dim,
            dropout=dropout,
        )
        dilations = [1, 1, 4, 16, 64]
        self.convs = nn.ModuleList()
        self.time_layers = nn.ModuleList()
        self.cond_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for idx in range(self.num_layers):
            dilation = dilations[idx % len(dilations)]
            padding = 4 * dilation
            self.convs.append(
                nn.Conv1d(
                    self.hidden_dim,
                    self.hidden_dim,
                    kernel_size=9,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.time_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.cond_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.norms.append(nn.GroupNorm(1, self.hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(self.hidden_dim, self.vocab_size, kernel_size=1),
        )

    def forward(self, xt, t, cond=None, cond_drop_mask=None, force_uncond=False):
        if not self.classifier_free_guidance and (cond is None or cond_drop_mask is not None or force_uncond):
            raise ValueError("Unconditional DDSM calls require classifier_free_guidance=True")

        time_emb = F.silu(self.time_embedder(t / 2.0))
        cond_emb = self.cond_embedder(
            cond,
            batch_size=xt.size(0),
            cond_drop_mask=cond_drop_mask if self.classifier_free_guidance else None,
            force_uncond=force_uncond if self.classifier_free_guidance else False,
        )

        h = F.silu(self.in_proj(xt.permute(0, 2, 1)))
        for conv, time_layer, cond_layer, norm in zip(
            self.convs,
            self.time_layers,
            self.cond_layers,
            self.norms,
            strict=True,
        ):
            z = self.dropout(h)
            z = z + time_layer(time_emb)[:, :, None]
            z = z + cond_layer(cond_emb)[:, :, None]
            z = F.silu(conv(norm(z)))
            h = h + z if h.shape == z.shape else z

        score = self.out(h).permute(0, 2, 1)
        return score - score.mean(dim=-1, keepdim=True)


class QuadDDSMModule(LightningModule):
    def __init__(
        self,
        *,
        seq_len=512,
        vocab_size=4,
        num_cls=3,
        hidden_dim=256,
        num_layers=20,
        dropout=0.1,
        lr=5e-4,
        min_time=0.01,
        max_time=4.0,
        dirichlet_scale=24.0,
        score_clip=50.0,
        denoising_loss_weight=0.0,
        num_sampling_steps=100,
        sampling_noise=0.02,
        classifier_free_guidance=False,
        cond_drop_prob=0.3,
        guidance_scale=1.0,
        noise_table_path=None,
        time_dependent_weights_path=None,
        use_official_noise=True,
        official_reverse_sampler=True,
        time_importance_sampling=True,
        random_order=False,
        speed_balanced=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = DDSMScoreCNN(
            vocab_size=vocab_size,
            num_cls=num_cls,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            classifier_free_guidance=classifier_free_guidance,
        )
        self.test_losses = []
        self.test_recons = []
        self._load_official_noise_table(noise_table_path)
        self._load_time_dependent_weights(time_dependent_weights_path)

    def _load_official_noise_table(self, noise_table_path):
        self.noise_table_path = noise_table_path
        self.v_one = None
        self.v_one_loggrad = None
        self.timepoints = None
        if not noise_table_path:
            if bool(self.hparams.use_official_noise):
                raise ValueError(
                    "Official DDSM training requires noise_table_path. "
                    "Create it with external_tools/ddsm/presample_noise.py or set use_official_noise=False."
                )
            return
        if diffusion_fast_flatdirichlet is None or gx_to_gv is None or UnitStickBreakingTransform is None:
            raise ImportError("Cannot import official DDSM utilities from external_tools/ddsm/ddsm.py")
        table = _torch_load(noise_table_path, map_location="cpu")
        if len(table) == 5:
            v_one, _v_zero, v_one_loggrad, _v_zero_loggrad, timepoints = table
        elif len(table) == 3:
            v_one, v_one_loggrad, timepoints = table
        else:
            raise ValueError(
                "DDSM noise table must contain either "
                "(v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints) "
                "or (v_one, v_one_loggrad, timepoints)"
            )
        self.v_one = v_one.float().cpu()
        self.v_one_loggrad = v_one_loggrad.float().cpu()
        self.timepoints = timepoints.float().cpu()

    def _load_time_dependent_weights(self, path):
        self.time_dependent_weights = None
        if not path:
            return
        weights = _torch_load(path, map_location="cpu")
        if isinstance(weights, dict):
            weights = weights.get("time_dependent_weights")
        if weights is None:
            raise ValueError("Cannot read time_dependent_weights from provided path")
        self.time_dependent_weights = weights.float().cpu()

    def _sample_cond_drop_mask(self, batch_size, device):
        if not bool(self.hparams.classifier_free_guidance):
            return None
        drop_prob = float(self.hparams.cond_drop_prob)
        if drop_prob <= 0:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        if drop_prob >= 1:
            return torch.ones(batch_size, device=device, dtype=torch.bool)
        return torch.rand(batch_size, device=device) < drop_prob

    def _sample_time(self, batch_size, device):
        min_t = float(self.hparams.min_time)
        max_t = float(self.hparams.max_time)
        return torch.rand(batch_size, device=device) * (max_t - min_t) + min_t

    def _sample_time_indices(self, batch_size, device):
        if self.timepoints is None:
            raise RuntimeError(
                "Official DDSM noise mode requires noise_table_path. "
                "Create it with external_tools/ddsm/presample_noise.py."
            )
        n_steps = int(self.timepoints.numel())
        if (
            bool(self.hparams.time_importance_sampling)
            and self.time_dependent_weights is not None
            and self.time_dependent_weights.numel() == n_steps
        ):
            probs = torch.sqrt(self.time_dependent_weights.clamp_min(1e-12))
            probs = probs / probs.sum()
            return torch.multinomial(probs, batch_size, replacement=True).to(device)
        return torch.randint(0, n_steps, (batch_size,), device=device)

    def _dirichlet_noising(self, seq, t):
        one_hot = F.one_hot(seq, num_classes=int(self.hparams.vocab_size)).float()
        k = int(self.hparams.vocab_size)
        signal = torch.exp(-t).view(-1, 1, 1)
        mean = signal * one_hot + (1.0 - signal) / k
        concentration = 1.0 + float(self.hparams.dirichlet_scale) * mean
        xt = torch.distributions.Dirichlet(concentration).rsample()
        target_score = (concentration - 1.0) / xt.clamp_min(torch.finfo(xt.dtype).eps)
        target_score = target_score - target_score.mean(dim=-1, keepdim=True)
        target_score = target_score.clamp(
            -float(self.hparams.score_clip),
            float(self.hparams.score_clip),
        )
        return xt, target_score

    def _denoised_logits(self, xt, score, t):
        scale = t.view(-1, 1, 1).clamp_min(float(self.hparams.min_time))
        logits = xt.clamp_min(torch.finfo(xt.dtype).eps).log() + scale * score
        return logits - logits.mean(dim=-1, keepdim=True)

    def _step_loss(self, seq, cond):
        if bool(self.hparams.use_official_noise) and self.v_one is not None:
            return self._official_step_loss(seq, cond)
        t = self._sample_time(seq.size(0), seq.device)
        xt, target_score = self._dirichlet_noising(seq, t)
        cond_drop_mask = self._sample_cond_drop_mask(cond.size(0), cond.device)
        pred_score = self.model(xt, t=t, cond=cond, cond_drop_mask=cond_drop_mask)
        score_loss = F.mse_loss(pred_score, target_score)
        logits = self._denoised_logits(xt, pred_score, t)
        recon = F.cross_entropy(logits.reshape(-1, int(self.hparams.vocab_size)), seq.reshape(-1))
        loss = score_loss + float(self.hparams.denoising_loss_weight) * recon
        return loss, score_loss, recon

    def _official_step_loss(self, seq, cond):
        one_hot = F.one_hot(seq, num_classes=int(self.hparams.vocab_size)).float()
        time_inds = self._sample_time_indices(seq.size(0), seq.device)
        perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(
            one_hot.detach().cpu(),
            time_inds.detach().cpu(),
            self.v_one,
            self.v_one_loggrad,
            symmetrize=False,
        )
        perturbed_x = perturbed_x.to(seq.device)
        perturbed_x_grad = perturbed_x_grad.to(seq.device)
        random_timepoints = self.timepoints[time_inds.detach().cpu()].to(seq.device)

        cond_drop_mask = self._sample_cond_drop_mask(cond.size(0), cond.device)
        score = self.model(
            perturbed_x,
            t=random_timepoints,
            cond=cond,
            cond_drop_mask=cond_drop_mask,
        )

        sb = UnitStickBreakingTransform()
        perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()
        if bool(self.hparams.speed_balanced):
            s = 2 / (
                torch.ones(int(self.hparams.vocab_size) - 1, device=seq.device)
                + torch.arange(
                    int(self.hparams.vocab_size) - 1,
                    0,
                    -1,
                    device=seq.device,
                ).float()
            )
        else:
            s = torch.ones(int(self.hparams.vocab_size) - 1, device=seq.device)

        pred_v_score = gx_to_gv(score, perturbed_x, create_graph=True)
        target_v_score = gx_to_gv(perturbed_x_grad, perturbed_x)
        weight = s[(None,) * (perturbed_v.ndim - 1)] * perturbed_v * (1 - perturbed_v)
        max_time_ind = int(time_inds.max().detach().cpu())
        if self.time_dependent_weights is not None and self.time_dependent_weights.numel() > max_time_ind:
            tdw = torch.sqrt(self.time_dependent_weights.to(seq.device)[time_inds]).clamp_min(1e-12)
            weight = weight / tdw[(...,) + (None,) * (perturbed_v.ndim - 1)]
        score_loss = (weight * (pred_v_score - target_v_score) ** 2).mean()

        logits = self._denoised_logits(perturbed_x, score, random_timepoints)
        recon = F.cross_entropy(logits.reshape(-1, int(self.hparams.vocab_size)), seq.reshape(-1))
        loss = score_loss + float(self.hparams.denoising_loss_weight) * recon
        return loss, score_loss, recon

    def training_step(self, batch, batch_idx):
        x, _, cond = batch
        loss, score_loss, recon = self._step_loss(x, cond)
        self.log_dict(
            {
                "train_loss": loss,
                "train_score_loss": score_loss,
                "train_recon": recon,
                "train_perplexity": torch.exp(recon.detach()),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, cond = batch
        with torch.enable_grad():
            loss, score_loss, recon = self._step_loss(x, cond)
        self.log_dict(
            {
                "val_loss": loss,
                "val_score_loss": score_loss,
                "val_recon": recon,
                "val_perplexity": torch.exp(recon.detach()),
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        x, _, cond = batch
        with torch.enable_grad():
            loss, score_loss, recon = self._step_loss(x, cond)
        self.test_losses.append(loss.detach())
        self.test_recons.append(recon.detach())
        self.log_dict(
            {
                "test_loss": loss,
                "test_score_loss": score_loss,
                "test_recon": recon,
                "test_perplexity": torch.exp(recon.detach()),
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

    def on_test_epoch_end(self):
        if self.test_losses:
            avg_loss = torch.stack(self.test_losses).mean()
            self.log("avg_test_loss", avg_loss, prog_bar=True, logger=True, sync_dist=True)
            self.test_losses.clear()
        if self.test_recons:
            avg_recon = torch.stack(self.test_recons).mean()
            self.log("avg_test_recon", avg_recon, logger=True, sync_dist=True)
            self.log(
                "avg_test_perplexity",
                torch.exp(avg_recon),
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.test_recons.clear()

    def _guided_score(self, xt, t, cond, guidance_scale=None):
        score_cond = self.model(xt, t=t, cond=cond)
        if not bool(self.hparams.classifier_free_guidance):
            return score_cond
        score_uncond = self.model(xt, t=t, cond=cond, force_uncond=True)
        scale = float(self.hparams.guidance_scale if guidance_scale is None else guidance_scale)
        return score_uncond + scale * (score_cond - score_uncond)

    @torch.no_grad()
    def generate(self, cond, guidance_scale=None):
        cond = cond.to(self.device)
        if (
            bool(self.hparams.official_reverse_sampler)
            and Euler_Maruyama_sampler is not None
            and self.v_one is not None
        ):
            return self._official_generate(cond, guidance_scale=guidance_scale)
        batch_size = cond.size(0)
        seq_len = int(self.hparams.seq_len)
        vocab_size = int(self.hparams.vocab_size)
        xt = torch.distributions.Dirichlet(
            torch.ones(batch_size, seq_len, vocab_size, device=self.device)
        ).sample()
        time_steps = torch.linspace(
            float(self.hparams.max_time),
            float(self.hparams.min_time),
            int(self.hparams.num_sampling_steps),
            device=self.device,
        )
        log_xt = xt.clamp_min(torch.finfo(xt.dtype).eps).log()
        for idx, t in enumerate(time_steps):
            t_batch = t.expand(batch_size)
            score = self._guided_score(xt, t_batch, cond, guidance_scale=guidance_scale)
            step = 1.0 / max(1, int(self.hparams.num_sampling_steps) - 1)
            log_xt = log_xt + step * score
            if idx < len(time_steps) - 1 and float(self.hparams.sampling_noise) > 0:
                noise_scale = float(self.hparams.sampling_noise) * math.sqrt(float(t / time_steps[0]))
                log_xt = log_xt + noise_scale * torch.randn_like(log_xt)
            xt = torch.softmax(log_xt, dim=-1)
            xt = simplex_proj(xt)
            log_xt = xt.clamp_min(torch.finfo(xt.dtype).eps).log()
        return torch.argmax(xt, dim=-1)

    def _official_generate(self, cond, guidance_scale=None):
        class GuidedScoreWrapper(nn.Module):
            def __init__(self, parent, condition, scale):
                super().__init__()
                self.parent = parent
                self.condition = condition
                self.scale = scale

            def forward(self, x, t):
                return self.parent._guided_score(x, t, self.condition, guidance_scale=self.scale)

        score_model = GuidedScoreWrapper(self, cond, guidance_scale)
        samples = Euler_Maruyama_sampler(
            score_model,
            sample_shape=(int(self.hparams.seq_len), int(self.hparams.vocab_size)),
            batch_size=cond.size(0),
            max_time=float(self.hparams.max_time),
            min_time=float(self.hparams.min_time),
            num_steps=int(self.hparams.num_sampling_steps),
            device=str(self.device),
            random_order=bool(self.hparams.random_order),
            speed_balanced=bool(self.hparams.speed_balanced),
            eps=1e-5,
        )
        return torch.argmax(samples, dim=-1)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _, cond = batch
        gen = self.generate(cond)
        return {
            "x": x.detach().cpu(),
            "conditions": cond.detach().cpu(),
            "recon": x.detach().cpu(),
            "gen": gen.detach().cpu(),
        }

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=float(self.hparams.lr))
        sched = {
            "scheduler": ReduceLROnPlateau(opt, mode="min", patience=5),
            "monitor": "val_loss",
            "strict": False,
        }
        return [opt], [sched]
