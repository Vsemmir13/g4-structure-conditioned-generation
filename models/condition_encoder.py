import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    def __init__(self, num_classes, embedding_dim, out_dim=None, dropout=0.0):
        super().__init__()
        if isinstance(num_classes, int):
            num_classes = [num_classes]
        self.num_classes = [int(value) for value in num_classes]
        self.num_conditions = len(self.num_classes)
        self.embedding_dim = int(embedding_dim)
        self.out_dim = int(out_dim or embedding_dim)
        if self.out_dim != self.embedding_dim:
            raise ValueError("Embedding-based ConditionEncoder requires out_dim == embedding_dim")
        self.embedders = nn.ModuleList(
            [nn.Embedding(num_cls + 1, self.embedding_dim) for num_cls in self.num_classes]
        )

    def _normalize_cond(self, cond, batch_size, device, force_uncond=False):
        if cond is None or force_uncond:
            values = [
                torch.full((batch_size,), num_cls, device=device, dtype=torch.long)
                for num_cls in self.num_classes
            ]
            return torch.stack(values, dim=1)

        cond = cond.to(device)
        if cond.dim() == 0:
            cond = cond.view(1, 1).expand(batch_size, self.num_conditions)
        elif cond.dim() == 1:
            if self.num_conditions == 1:
                cond = cond.view(batch_size, 1)
            else:
                cond = cond.view(batch_size, self.num_conditions)
        elif cond.dim() == 2:
            if cond.size(1) != self.num_conditions:
                raise ValueError(
                    f"Expected {self.num_conditions} condition columns, got {cond.size(1)}"
                )
        else:
            raise ValueError(f"Unsupported condition shape: {tuple(cond.shape)}")

        cond = torch.round(cond).long()
        clipped = []
        for idx, num_cls in enumerate(self.num_classes):
            clipped.append(cond[:, idx].clamp(0, num_cls - 1))
        return torch.stack(clipped, dim=1)

    def forward(self, cond, batch_size=None, cond_drop_mask=None, force_uncond=False):
        if batch_size is None:
            if cond is None:
                raise ValueError("batch_size is required when cond is None")
            batch_size = cond.size(0)
        device = self.embedders[0].weight.device
        cond_ids = self._normalize_cond(cond, batch_size, device, force_uncond=force_uncond)

        if cond_drop_mask is not None:
            cond_drop_mask = cond_drop_mask.to(device=device, dtype=torch.bool).view(batch_size, 1)
            null_cols = [
                torch.full((batch_size,), num_cls, device=device, dtype=torch.long)
                for num_cls in self.num_classes
            ]
            null_ids = torch.stack(null_cols, dim=1)
            cond_ids = torch.where(cond_drop_mask, null_ids, cond_ids)

        embeddings = [
            embedder(cond_ids[:, idx]) for idx, embedder in enumerate(self.embedders)
        ]
        return torch.stack(embeddings, dim=0).sum(dim=0)
