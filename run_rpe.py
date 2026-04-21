"""
run_rpe.py
----------
Convenience wrapper around PermInvariantPatchEmbedding4D.

Accepts a raw fMRI volume in any of the following shapes:
    [H, W, Z, T]           → treated as batch=1, channels=1
    [B, H, W, Z, T]        → treated as channels=1
    [B, C, H, W, Z, T]     → used as-is

Returns patch embeddings: [B, out_dim, H_p, W_p, Z_p, T_p]
"""

import torch

from invariant_patching_4d import PermInvariantPatchEmbedding4D


def rpe_embed_volume(
    data: torch.Tensor,
    patch_size: tuple = (8, 8, 4, 10),
    hidden_dim: int = 64,
    out_dim: int = 128,
    patch_chunk: int = 64,
    voxel_chunk: int | None = None,
) -> torch.Tensor:
    """
    Embed a raw fMRI volume into patch tokens.

    Args:
        data:        Volume tensor — [H,W,Z,T], [B,H,W,Z,T], or [B,C,H,W,Z,T].
        patch_size:  (ph, pw, pz, pt). Must evenly divide each spatial/temporal dim.
        hidden_dim:  Hidden size of the phi MLP inside DeepSets.
        out_dim:     Output embedding dimension per patch.
        patch_chunk: Number of patches processed at once. Lower = less peak RAM.
        voxel_chunk: Optional second-level chunking over voxels within each patch.

    Returns:
        Tensor [B, out_dim, H_p, W_p, Z_p, T_p]
    """
    if data.dim() == 4:
        data = data.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W, Z, T]
    elif data.dim() == 5:
        data = data.unsqueeze(1)                # [B, 1, H, W, Z, T]

    _, C, _, _, _, _ = data.shape

    model = PermInvariantPatchEmbedding4D(
        in_channels=C,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        patch_size=patch_size,
        patch_chunk=patch_chunk,
        voxel_chunk=voxel_chunk,
    ).to(device=data.device, dtype=data.dtype)

    return model(data)
