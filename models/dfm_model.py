import copy

import torch.nn as nn
import torch.nn.functional as F

from .condition_encoder import ConditionEncoder
from .dfm_flow_utils import GaussianFourierProjection


class Dense(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


class QuadCondCNN(nn.Module):

    def __init__(
        self,
        *,
        alphabet_size=4,
        num_cls=3,
        hidden_dim=256,
        num_cnn_stacks=2,
        dropout=0.1,
        expanded_simplex=True,
        time_embed_scale=30.0,
        classifier_free_guidance=False,
    ):
        super().__init__()
        self.alphabet_size = int(alphabet_size)
        self.num_cls = num_cls
        self.hidden_dim = int(hidden_dim)
        self.num_cnn_stacks = int(num_cnn_stacks)
        self.dropout_p = float(dropout)
        self.expanded_simplex = bool(expanded_simplex)
        self.classifier_free_guidance = bool(classifier_free_guidance)

        inp_size = self.alphabet_size * (2 if self.expanded_simplex else 1)

        self.in_proj = nn.Conv1d(inp_size, self.hidden_dim, kernel_size=9, padding=4)

        self.time_embedder = nn.Sequential(
            GaussianFourierProjection(embedding_dim=self.hidden_dim, scale=time_embed_scale),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.cls_embedder = ConditionEncoder(
            num_cls, self.hidden_dim, out_dim=self.hidden_dim, dropout=dropout
        )

        base_convs = [
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, padding=4),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, padding=4),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=64, padding=256),
        ]
        self.num_layers = len(base_convs) * self.num_cnn_stacks
        self.convs = nn.ModuleList(
            [copy.deepcopy(layer) for layer in base_convs for _ in range(self.num_cnn_stacks)]
        )
        self.time_layers = nn.ModuleList(
            [Dense(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)]
        )
        self.cls_layers = nn.ModuleList(
            [Dense(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(self.dropout_p)

        self.out = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.alphabet_size, kernel_size=1),
        )

    def forward(self, xt, t, cond=None, cond_drop_mask=None, force_uncond=False):
        # xt: [B, L, C] -> [B, C, L]
        time_emb = F.relu(self.time_embedder(t))
        if not self.classifier_free_guidance and (cond is None or cond_drop_mask is not None or force_uncond):
            raise ValueError("Unconditional DFM calls require classifier_free_guidance=True")
        cls_emb = self.cls_embedder(
            cond,
            batch_size=xt.size(0),
            cond_drop_mask=cond_drop_mask if self.classifier_free_guidance else None,
            force_uncond=force_uncond if self.classifier_free_guidance else False,
        )

        h = xt.permute(0, 2, 1)
        h = F.relu(self.in_proj(h))

        for i in range(self.num_layers):
            z = self.dropout(h.clone())
            z = z + self.time_layers[i](time_emb)[:, :, None]
            z = z + self.cls_layers[i](cls_emb)[:, :, None]
            z = self.norms[i](z.permute(0, 2, 1)).permute(0, 2, 1)
            z = F.relu(self.convs[i](z))
            h = h + z if z.shape == h.shape else z

        y = self.out(h)  # [B, K, L]
        return y.permute(0, 2, 1)  # [B, L, K]


class QuadCondTransformer(nn.Module):
    def __init__(
        self,
        *,
        seq_len,
        alphabet_size=4,
        num_cls=3,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        ff_mult=4,
        dropout=0.1,
        expanded_simplex=True,
        time_embed_scale=30.0,
        classifier_free_guidance=False,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.alphabet_size = int(alphabet_size)
        self.num_cls = num_cls
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ff_mult = int(ff_mult)
        self.dropout_p = float(dropout)
        self.expanded_simplex = bool(expanded_simplex)
        self.classifier_free_guidance = bool(classifier_free_guidance)

        inp_size = self.alphabet_size * (2 if self.expanded_simplex else 1)
        self.embedder = nn.Linear(inp_size, self.hidden_dim)

        self.time_embedder = nn.Sequential(
            GaussianFourierProjection(embedding_dim=self.hidden_dim, scale=time_embed_scale),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.cls_embedder = ConditionEncoder(
            num_cls, self.hidden_dim, out_dim=self.hidden_dim, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout_p,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.out = nn.Linear(self.hidden_dim, self.alphabet_size)

    def forward(self, xt, t, cond=None, cond_drop_mask=None, force_uncond=False):
        feat = self.embedder(xt)
        time_embed = F.relu(self.time_embedder(t))
        feat = feat + time_embed[:, None, :]
        if not self.classifier_free_guidance and (cond is None or cond_drop_mask is not None or force_uncond):
            raise ValueError("Unconditional DFM calls require classifier_free_guidance=True")
        cls_emb = self.cls_embedder(
            cond,
            batch_size=xt.size(0),
            cond_drop_mask=cond_drop_mask if self.classifier_free_guidance else None,
            force_uncond=force_uncond if self.classifier_free_guidance else False,
        )
        feat = feat + cls_emb[:, None, :]
        feat = self.transformer(feat)
        return self.out(feat)
