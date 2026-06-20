import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.condition_encoder import ConditionEncoder


class ConvResBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=7, padding=3),
            nn.Dropout(dropout),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=7, padding=3),
        )

    def forward(self, x):
        return x + self.net(x)


class DNAConvVAE(LightningModule):
    def __init__(
        self,
        seq_len=512,
        vocab_size=4,
        latent_dim=64,
        num_cls=3,
        hidden_dim=256,
        num_res_blocks=3,
        dropout=0.1,
        sample_temperature=1.0,
        lr=1e-3,
        beta=0.1,
        beta_warmup_steps=2000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.sample_temperature = float(sample_temperature)
        self.cond_emb = ConditionEncoder(num_cls, hidden_dim, out_dim=hidden_dim, dropout=dropout)
        self.enc_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dec_cond_proj = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * (self.seq_len // 4))

        self.encoder = nn.Sequential(
            nn.Conv1d(vocab_size, hidden_dim // 2, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            *[ConvResBlock(hidden_dim, dropout=dropout) for _ in range(num_res_blocks)],
        )

        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_blocks = nn.Sequential(
            *[ConvResBlock(hidden_dim, dropout=dropout) for _ in range(num_res_blocks)],
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            ConvResBlock(hidden_dim // 2, dropout=dropout),
            nn.ConvTranspose1d(
                hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1
            ),
            nn.SiLU(),
            ConvResBlock(hidden_dim // 4, dropout=dropout),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 4, vocab_size, kernel_size=1),
        )

        self.lr = lr
        self.beta = beta
        self.beta_warmup_steps = beta_warmup_steps
        self.test_losses = []
        self.test_recons = []

    def one_hot(self, x):
        return F.one_hot(x, num_classes=self.vocab_size).float()

    def encode(self, x, cond):
        x = self.one_hot(x)
        x = x.permute(0, 2, 1)
        h = self.encoder(x)
        cond_emb = self._cond_embedding(cond)
        h = h + self.enc_cond_proj(cond_emb)[:, :, None]
        h = h.mean(dim=-1)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def _cond_embedding(self, cond):
        return self.cond_emb(cond, batch_size=cond.size(0))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        cond_emb = self._cond_embedding(cond)
        z = z + self.dec_cond_proj(cond_emb)
        h = self.latent_to_hidden(z)
        h = h.view(z.size(0), self.hparams.hidden_dim, self.seq_len // 4)
        h = self.decoder_blocks(h)
        logits = self.decoder(h)
        logits = logits.permute(0, 2, 1)
        return logits[:, : self.seq_len, :]

    def forward(self, x, cond):
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, cond)
        return logits, mu, logvar

    def loss_fn(self, logits, targets, mu, logvar):
        recon = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        warm = (
            min(1.0, float(self.global_step) / float(self.beta_warmup_steps))
            if self.beta_warmup_steps
            else 1.0
        )
        beta_eff = self.beta * warm
        return recon + beta_eff * kld, recon, kld

    def training_step(self, batch, batch_idx):
        x, y, cond = batch
        logits, mu, logvar = self(x, cond)
        loss, recon, kld = self.loss_fn(logits, y, mu, logvar)
        self.log_dict(
            {
                "train_loss": loss,
                "train_recon": recon,
                "train_kld": kld,
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
        x, y, cond = batch
        logits, mu, logvar = self(x, cond)
        loss, recon, kld = self.loss_fn(logits, y, mu, logvar)
        self.log_dict(
            {
                "val_loss": loss,
                "val_recon": recon,
                "val_kld": kld,
                "val_perplexity": torch.exp(recon.detach()),
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        x, y, cond = batch
        logits, mu, logvar = self(x, cond)
        loss, recon, kld = self.loss_fn(logits, y, mu, logvar)
        self.test_losses.append(loss.detach())
        self.test_recons.append(recon.detach())
        self.log_dict(
            {
                "test_loss": loss,
                "test_recon": recon,
                "test_kld": kld,
                "test_perplexity": torch.exp(recon.detach()),
            },
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

    def on_test_epoch_end(self):
        avg = torch.stack(self.test_losses).mean()
        self.log("avg_test_loss", avg, logger=True, sync_dist=True)
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
        self.test_losses.clear()
        self.test_recons.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, cond = batch
        logits, mu, logvar = self(x, cond)
        recon = torch.argmax(logits, dim=-1)
        gen = self.generate(cond)
        return {
            "x": x.detach().cpu(),
            "levels": cond.detach().cpu(),
            "recon": recon.detach().cpu(),
            "gen": gen.detach().cpu(),
            "mu": mu.detach().cpu(),
            "logvar": logvar.detach().cpu(),
        }

    def generate(self, cond, z=None, greedy=False, temperature=None):
        if z is None:
            z = torch.randn(
                cond.size(0),
                self.hparams.latent_dim,
                device=cond.device,
            )
        elif z.dim() == 3:
            z = z.mean(dim=-1)
        logits = self.decode(z, cond)
        if greedy:
            return torch.argmax(logits, dim=-1)
        temp = self.sample_temperature if temperature is None else float(temperature)
        probs = torch.softmax(logits / max(temp, 1e-6), dim=-1)
        return torch.distributions.Categorical(probs=probs).sample()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = {
            "scheduler": ReduceLROnPlateau(opt, mode="min", patience=5),
            "monitor": "val_loss",
            "strict": False,
        }
        return [opt], [sched]
