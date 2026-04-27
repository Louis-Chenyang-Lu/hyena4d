from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyena4d import Hyena4DPixelBlock


class Hyena4DBackbone(nn.Module):
    """
    A simple 4D backbone built from Hyena4DPixelBlocks.

    Expected input:  [B, C_in, T, Z, H, W]

    The backbone:
      1. Projects the input channels to `embed_dim` using a 3D 1x1x1 conv
         over a merged depth dimension D = T * Z.
      2. Applies a stack of Hyena4DPixelBlocks that operate on 4D volumes.
      3. Globally pools over (T, Z, H, W) to obtain a [B, embed_dim] feature.
      4. Applies a classifier head to produce logits [B, num_classes].

    For contrastive learning / representation learning, you can instead
    call :meth:`forward_features` to obtain the pooled features without
    the classifier, or use :class:`Hyena4DContrastiveEncoder` below.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_blocks: int,
        num_classes: int,
        kernel_size: Tuple[int, int, int, int] = (3, 3, 7, 7),
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        t_gap: float = 1.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.kernel_size = kernel_size

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=1,
            stride=1,
        )

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                Hyena4DPixelBlock(
                    d_model=embed_dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    voxel_size=voxel_size,
                    t_gap=t_gap,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.classifier_norm = nn.LayerNorm(embed_dim)
        self.classifier_head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract pooled 4D features without applying the classifier.

        Args:
            x: Tensor [B, C_in, T, Z, H, W]

        Returns:
            feats: Tensor [B, embed_dim]
        """
        if x.dim() != 6:
            raise ValueError(
                f"Hyena4DBackbone expects input [B, C, T, Z, H, W], "
                f"but got tensor with shape {x.shape} (dim={x.dim()})."
            )
        B, C_in, T, Z, H, W = x.shape
        if C_in != self.in_channels:
            raise ValueError(
                f"Input channels {C_in} != configured in_channels {self.in_channels}."
            )

        x = x.contiguous()

        D = T * Z
        x_3d = x.view(B, C_in, D, H, W)  # [B, C_in, D=T*Z, H, W]

        x_3d = self.proj(x_3d)  # [B, embed_dim, D, H, W]
        Bp, embed_dim, Dp, Hp, Wp = x_3d.shape
        if Bp != B or Dp != D or Hp != H or Wp != W:
            raise RuntimeError(
                f"Unexpected shape after projection: got {x_3d.shape}, "
                f"expected [B, embed_dim, D=T*Z, H, W] = [{B}, {embed_dim}, {D}, {H}, {W}]."
            )

        x = x_3d.view(B, embed_dim, T, Z, H, W) # [B, embed_dim, T, Z, H, W]

        for blk in self.blocks:
            x = blk(x)  # [B, embed_dim, T, Z, H, W]

        feats = x.mean(dim=[2, 3, 4, 5])  # [B, embed_dim]
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard classification forward.

        Args:
            x: Tensor [B, C_in, T, Z, H, W]

        Returns:
            logits: Tensor [B, num_classes]
        """
        feats = self.forward_features(x)         # [B, embed_dim]
        feats = self.classifier_norm(feats)      # [B, embed_dim]
        logits = self.classifier_head(feats)     # [B, num_classes]
        return logits


class Hyena4DContrastiveEncoder(Hyena4DBackbone):
    """
    Encoder for contrastive learning.

    Uses the Hyena4D backbone to obtain pooled features and then applies
    a projection head to obtain embeddings for contrastive loss.

    Expected input: [B, C_in, T, Z, H, W], where each item is a subsequence.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_blocks: int,
        projection_dim: int = 128,
        kernel_size: Tuple[int, int, int, int] = (3, 3, 7, 7),
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        normalize: bool = True,
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        t_gap: float = 1.0,
    ):

        super().__init__(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            num_classes=projection_dim,
            kernel_size=kernel_size,
            mlp_ratio=mlp_ratio,
            drop=drop,
            voxel_size=voxel_size,
            t_gap=t_gap,
        )

        self.projection_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, projection_dim),
        )
        self.projection_dim = projection_dim
        self.normalize_embeddings = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, C_in, T, Z, H, W], each entry is a subsequence.

        Returns:
            z: Tensor [B, projection_dim], normalized if normalize_embeddings=True
        """
        feats = self.forward_features(x)         # [B, embed_dim]
        z = self.projection_head(feats)         # [B, projection_dim]
        if self.normalize_embeddings:
            z = F.normalize(z, dim=-1)
        return z


def subject_contrastive_loss(
    z: torch.Tensor,
    subject_ids: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Contrastive loss over subsequences, using subject IDs to define positives.

    Args:
        z:           [B, D] embeddings (L2-normalized is recommended)
        subject_ids: [B] int labels, same subject => positives
        temperature: scalar temperature for InfoNCE

    Returns:
        loss: scalar tensor
    """
    device = z.device
    B, D = z.shape

    if subject_ids.shape[0] != B:
        raise ValueError(
            f"subject_ids must have shape [B], but got {subject_ids.shape} "
            f"for batch size B={B}."
        )

    sim = torch.matmul(z, z.T) / temperature

    logits_mask = torch.ones_like(sim, dtype=torch.bool, device=device)
    logits_mask.fill_(True)
    logits_mask.fill_diagonal_(False)

    subject_ids = subject_ids.view(-1, 1)
    pos_mask = (subject_ids == subject_ids.T) & logits_mask  # [B, B]

    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    exp_sim = torch.exp(sim) * logits_mask  
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(dim=1)
    pos_log_prob = (log_prob * pos_mask).sum(dim=1) / (pos_count + 1e-12)

    valid_anchors = (pos_count > 0).float()
    if valid_anchors.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss = - (pos_log_prob * valid_anchors).sum() / (valid_anchors.sum() + 1e-12)
    return loss
