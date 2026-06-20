import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .dfm_flow_utils import (
    DirichletConditionalFlow,
    expand_simplex,
    sample_cond_prob_path,
    simplex_proj,
)
from .dfm_model import QuadCondCNN, QuadCondTransformer


class QuadDFMModule(LightningModule):
    def __init__(
        self,
        *,
        backbone="cnn",
        seq_len=512,
        vocab_size=4,
        num_cls=3,
        hidden_dim=256,
        num_cnn_stacks=2,
        num_transformer_layers=6,
        num_attention_heads=4,
        transformer_ff_mult=4,
        dropout=0.1,
        lr=1e-3,
        alpha_max=8.0,
        alpha_scale=2.0,
        fix_alpha=None,
        prior_pseudocount=2.0,
        num_integration_steps=64,
        flow_temp=1.0,
        classifier_free_guidance=False,
        cond_drop_prob=0.3,
        guidance_scale=0.5,
        guidance_mode="score",
        cls_free_guidance=None,
        cls_free_noclass_ratio=None,
        score_free_guidance=False,
        probability_addition=False,
        adaptive_prob_add=False,
        probability_tilt=False,
        vectorfield_addition=False,
        allow_nan_cfactor=False,
    ):
        super().__init__()
        if cls_free_guidance is not None:
            classifier_free_guidance = bool(cls_free_guidance)
        if cls_free_noclass_ratio is not None:
            cond_drop_prob = float(cls_free_noclass_ratio)
        if score_free_guidance:
            guidance_mode = "score_free"
        elif probability_addition:
            guidance_mode = "probability_addition"
        elif probability_tilt:
            guidance_mode = "probability_tilt"
        elif vectorfield_addition:
            guidance_mode = "vectorfield_addition"
        self.save_hyperparameters(
            {
                "backbone": backbone,
                "seq_len": seq_len,
                "vocab_size": vocab_size,
                "num_cls": num_cls,
                "hidden_dim": hidden_dim,
                "num_cnn_stacks": num_cnn_stacks,
                "num_transformer_layers": num_transformer_layers,
                "num_attention_heads": num_attention_heads,
                "transformer_ff_mult": transformer_ff_mult,
                "dropout": dropout,
                "lr": lr,
                "alpha_max": alpha_max,
                "alpha_scale": alpha_scale,
                "fix_alpha": fix_alpha,
                "prior_pseudocount": prior_pseudocount,
                "num_integration_steps": num_integration_steps,
                "flow_temp": flow_temp,
                "classifier_free_guidance": classifier_free_guidance,
                "cond_drop_prob": cond_drop_prob,
                "guidance_scale": guidance_scale,
                "guidance_mode": guidance_mode,
                "cls_free_guidance": cls_free_guidance,
                "cls_free_noclass_ratio": cls_free_noclass_ratio,
                "score_free_guidance": score_free_guidance,
                "probability_addition": probability_addition,
                "adaptive_prob_add": adaptive_prob_add,
                "probability_tilt": probability_tilt,
                "vectorfield_addition": vectorfield_addition,
                "allow_nan_cfactor": allow_nan_cfactor,
            }
        )
        if backbone == "cnn":
            self.model = QuadCondCNN(
                alphabet_size=vocab_size,
                num_cls=num_cls,
                hidden_dim=hidden_dim,
                num_cnn_stacks=num_cnn_stacks,
                dropout=dropout,
                expanded_simplex=True,
                classifier_free_guidance=classifier_free_guidance,
            )
        elif backbone == "transformer":
            self.model = QuadCondTransformer(
                seq_len=seq_len,
                alphabet_size=vocab_size,
                num_cls=num_cls,
                hidden_dim=hidden_dim,
                num_layers=num_transformer_layers,
                num_heads=num_attention_heads,
                ff_mult=transformer_ff_mult,
                dropout=dropout,
                expanded_simplex=True,
                classifier_free_guidance=classifier_free_guidance,
            )
        else:
            raise ValueError(f"Unsupported DFM backbone: {backbone}")
        self.condflow = DirichletConditionalFlow(
            k=vocab_size, alpha_max=alpha_max, alpha_spacing=0.001
        )
        self.test_losses = []

    def training_step(self, batch, batch_idx):
        x, _, cond = batch
        loss, recon = self._step_loss(x, cond)
        self.log_dict(
            {
                "train_loss": loss,
                "train_recon": recon,
                "train_perplexity": torch.exp(loss.detach()),
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
        loss, recon = self._step_loss(x, cond)
        self.log_dict(
            {
                "val_loss": loss,
                "val_recon": recon,
                "val_perplexity": torch.exp(loss.detach()),
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        x, _, cond = batch
        loss, _ = self._step_loss(x, cond)
        self.test_losses.append(loss.detach())
        self.log("test_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log(
            "test_perplexity", torch.exp(loss.detach()), prog_bar=True, logger=True, sync_dist=True
        )

    def on_test_epoch_end(self):
        if self.test_losses:
            avg = torch.stack(self.test_losses).mean()
            self.log("avg_test_loss", avg, prog_bar=True, logger=True, sync_dist=True)
            self.test_losses.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _, cond = batch
        gen = self.generate(cond)
        return {
            "x": x.detach().cpu(),
            "levels": cond.detach().cpu(),
            "recon": x.detach().cpu(),
            "gen": gen.detach().cpu(),
        }

    def _step_loss(self, seq, cond):
        xt, alphas = sample_cond_prob_path(
            seq,
            self.hparams.vocab_size,
            alpha_scale=self.hparams.alpha_scale,
            alpha_max=self.hparams.alpha_max,
            fix_alpha=self.hparams.fix_alpha,
        )
        xt_inp, _ = expand_simplex(xt, alphas, self.hparams.prior_pseudocount)
        cond_drop_mask = self._sample_cond_drop_mask(cond.size(0), cond.device)
        logits = self.model(xt_inp, t=alphas, cond=cond, cond_drop_mask=cond_drop_mask)
        recon = F.cross_entropy(logits.reshape(-1, self.hparams.vocab_size), seq.reshape(-1))
        return recon, recon

    def _sample_cond_drop_mask(self, batch_size, device):
        if not bool(self.hparams.classifier_free_guidance):
            return None
        drop_prob = float(self.hparams.cond_drop_prob)
        if drop_prob <= 0:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        if drop_prob >= 1:
            return torch.ones(batch_size, device=device, dtype=torch.bool)
        return torch.rand(batch_size, device=device) < drop_prob

    def _score_guided_probs(self, xt, alpha, logits_uncond, logits_cond, guidance_scale):
        b, seq_len, k = xt.shape
        alpha = alpha.view(b, 1, 1, 1)
        probs_uncond = torch.softmax(logits_uncond, dim=-1)
        probs_cond = torch.softmax(logits_cond, dim=-1)
        eye = torch.eye(k, device=xt.device, dtype=xt.dtype).view(1, 1, k, k)
        cond_scores = (alpha - 1) * eye / xt.unsqueeze(-1).clamp_min(torch.finfo(xt.dtype).tiny)
        cond_scores = cond_scores - cond_scores.mean(2, keepdim=True)

        score_uncond = torch.einsum("blkc,blc->blk", cond_scores, probs_uncond)
        score_cond = torch.einsum("blkc,blc->blk", cond_scores, probs_cond)
        score_guided = (1 - guidance_scale) * score_uncond + guidance_scale * score_cond

        q_mats = cond_scores.clone()
        q_mats[:, :, -1, :] = torch.ones((b, seq_len, k), device=xt.device, dtype=xt.dtype)
        score_guided = score_guided.clone()
        score_guided[:, :, -1] = torch.ones((b, seq_len), device=xt.device, dtype=xt.dtype)
        return torch.linalg.solve(q_mats, score_guided)

    def _guided_flow_probs(self, xt, xt_exp, t, cond, guidance_scale=None):
        logits_cond = self.model(xt_exp, t=t, cond=cond)
        probs_cond = torch.softmax(logits_cond / float(self.hparams.flow_temp), dim=-1)
        if not bool(self.hparams.classifier_free_guidance):
            return probs_cond, probs_cond, None

        mode = str(self.hparams.guidance_mode)
        scale = float(self.hparams.guidance_scale if guidance_scale is None else guidance_scale)
        if mode == "score_free":
            return probs_cond, probs_cond, None

        logits_uncond = self.model(xt_exp, t=t, cond=cond, force_uncond=True)
        probs_uncond = torch.softmax(logits_uncond / float(self.hparams.flow_temp), dim=-1)

        if mode == "score":
            flow_probs = self._score_guided_probs(xt, t + 1e-4, logits_uncond, logits_cond, scale)
        elif mode == "probability_addition":
            if bool(self.hparams.adaptive_prob_add):
                potential_scales = probs_cond / (probs_cond - probs_uncond)
                max_guide_scale = potential_scales.min(-1).values
                flow_probs = (
                    probs_cond * (1 - max_guide_scale[..., None])
                    + probs_uncond * max_guide_scale[..., None]
                )
            else:
                flow_probs = probs_cond * scale + probs_uncond * (1 - scale)
        elif mode == "probability_tilt":
            eps = torch.finfo(probs_cond.dtype).tiny
            flow_probs = (
                probs_cond.clamp_min(eps) ** (1 - scale) * probs_uncond.clamp_min(eps) ** scale
            )
            flow_probs = flow_probs / flow_probs.sum(-1, keepdim=True).clamp_min(eps)
        elif mode == "vectorfield_addition":
            flow_probs = probs_cond
        elif mode == "logit":
            logits_guided = logits_uncond + scale * (logits_cond - logits_uncond)
            flow_probs = torch.softmax(logits_guided / float(self.hparams.flow_temp), dim=-1)
        else:
            raise ValueError(f"Unsupported guidance_mode: {mode}")
        return flow_probs, probs_cond, probs_uncond

    @torch.no_grad()
    def generate(self, cond, guidance_scale=None):
        cond = cond.to(self.device)
        b = cond.size(0)
        seq_len = int(self.hparams.seq_len)
        k = int(self.hparams.vocab_size)
        xt = torch.distributions.Dirichlet(torch.ones(b, seq_len, k, device=self.device)).sample()
        eye = torch.eye(k, device=self.device)
        t_span = torch.linspace(
            1.0,
            float(self.hparams.alpha_max),
            int(self.hparams.num_integration_steps),
            device=self.device,
        )
        for s, t in zip(t_span[:-1], t_span[1:], strict=False):
            s_batch = s[None].expand(b)
            xt_exp, _ = expand_simplex(xt, s_batch, float(self.hparams.prior_pseudocount))
            flow_probs, probs_cond, probs_uncond = self._guided_flow_probs(
                xt, xt_exp, s_batch, cond, guidance_scale=guidance_scale
            )
            if (
                not torch.allclose(
                    flow_probs.sum(-1), torch.ones_like(flow_probs[..., 0]), atol=1e-4
                )
            ) or (flow_probs < 0).any():
                flow_probs = simplex_proj(flow_probs)
            c_factor = self.condflow.c_factor(xt.detach().cpu().numpy(), float(s.item()))
            c_factor = torch.from_numpy(c_factor).to(xt).float()
            if torch.isnan(c_factor).any():
                if bool(self.hparams.allow_nan_cfactor):
                    c_factor = torch.nan_to_num(c_factor)
                else:
                    raise RuntimeError(
                        f"NAN c_factor during DFM inference: xt.min()={xt.min().item()}, "
                        f"flow_probs.min()={flow_probs.min().item()}"
                    )
            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            if (
                bool(self.hparams.classifier_free_guidance)
                and str(self.hparams.guidance_mode) == "vectorfield_addition"
                and probs_uncond is not None
            ):
                scale = float(
                    self.hparams.guidance_scale if guidance_scale is None else guidance_scale
                )
                flow_cond = (probs_cond.unsqueeze(-2) * cond_flows).sum(-1)
                flow_uncond = (probs_uncond.unsqueeze(-2) * cond_flows).sum(-1)
                flow = flow_cond * scale + flow_uncond * (1 - scale)
            else:
                flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)
            xt = xt + flow * (t - s)
            if (not torch.allclose(xt.sum(-1), torch.ones_like(xt[..., 0]), atol=1e-4)) or (
                xt < 0
            ).any():
                xt = simplex_proj(xt)
        final_t = t_span[-1][None].expand(b)
        final_xt = expand_simplex(xt, final_t, float(self.hparams.prior_pseudocount))[0]
        final_logits = self.model(final_xt, t=final_t, cond=cond)
        return torch.argmax(final_logits, dim=-1)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), float(self.hparams.lr))
        sched = {
            "scheduler": ReduceLROnPlateau(opt, mode="min", patience=5),
            "monitor": "val_loss",
            "strict": False,
        }
        return [opt], [sched]
