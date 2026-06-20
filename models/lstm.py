import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.condition_encoder import ConditionEncoder


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class QuadLSTM(LightningModule):
    def __init__(
        self,
        vocab_size=5,
        emb_dim=128,
        level_dim=8,
        num_cls=3,
        hidden_dim=256,
        num_layers=2,
        mlp_layers=2,
        dropout=0.2,
        sample_temperature=1.0,
        top_k=0,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = int(vocab_size)
        self.sample_temperature = float(sample_temperature)
        self.top_k = int(top_k)
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.condition_encoder = ConditionEncoder(num_cls, level_dim, out_dim=level_dim, dropout=dropout)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(emb_dim + level_dim),
            nn.Linear(emb_dim + level_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.mlp_blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(mlp_layers)]
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, vocab_size),
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.test_losses = []
        self.lr = lr

    def forward(self, x, levels):
        B, T = x.shape
        token_emb = self.emb(x)
        level_emb = self.condition_encoder(levels, batch_size=B)
        level_emb = level_emb.unsqueeze(1).expand(B, T, -1)
        inp = torch.cat([token_emb, level_emb], dim=-1)
        inp = self.input_proj(inp)
        out, _ = self.lstm(inp)
        out = self.mlp_blocks(self.out_norm(out))
        logits = self.fc_out(out)
        return logits

    @torch.no_grad()
    def generate(self, levels, seq_len, greedy=False, temperature=None, top_k=None):
        device = levels.device
        bsz = levels.size(0)
        level_emb = self.condition_encoder(levels, batch_size=bsz).unsqueeze(1)
        token = torch.full((bsz, 1), 4, dtype=torch.long, device=device)
        hidden = None
        out_tokens = []
        for _ in range(int(seq_len)):
            token_emb = self.emb(token)
            inp = self.input_proj(torch.cat([token_emb, level_emb], dim=-1))
            out, hidden = self.lstm(inp, hidden)
            out = self.mlp_blocks(self.out_norm(out))
            logits = self.fc_out(out)[:, -1, :4]
            if greedy:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                temp = self.sample_temperature if temperature is None else float(temperature)
                logits = logits / max(temp, 1e-6)
                k = self.top_k if top_k is None else int(top_k)
                if k > 0 and k < logits.size(-1):
                    values, indices = torch.topk(logits, k=k, dim=-1)
                    filtered = torch.full_like(logits, float("-inf"))
                    logits = filtered.scatter(-1, indices, values)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            out_tokens.append(next_token)
            token = next_token
        return torch.cat(out_tokens, dim=1)

    def training_step(self, batch, batch_idx):
        x, y, levels = batch
        logits = self(x, levels)
        loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log_dict(
            {
                "train_loss": loss,
                "train_perplexity": torch.exp(loss.detach()),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, levels = batch
        logits = self(x, levels)
        loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log_dict(
            {
                "val_loss": loss,
                "val_perplexity": torch.exp(loss.detach()),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5),
            "monitor": "val_loss",
            "strict": False,
        }
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx):
        x, y, levels = batch
        logits = self(x, levels)
        loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.test_losses.append(loss.detach())
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_losses).mean()
        self.log("avg_test_loss", avg_loss, prog_bar=True, logger=True, sync_dist=True)
        perplexity = math.exp(avg_loss.item())
        self.log("test_perplexity", perplexity, prog_bar=True, logger=True, sync_dist=True)
        self.test_losses.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, levels = batch
        logits = self(x, levels)
        recon = torch.argmax(logits, dim=-1)
        gen = self.generate(levels, seq_len=y.size(1))
        return {
            "x": x.detach().cpu(),
            "levels": levels.detach().cpu(),
            "recon": recon.detach().cpu(),
            "gen": gen.detach().cpu(),
        }
