"""
invariant_patching_4d.py
------------------------
Memory-efficient, permutation-invariant 4D patch embedding for fMRI volumes.

Architecture (DeepSets):
    For each non-overlapping patch [N_vox, C]:
        phi:  MLP per voxel  → [N_vox, hidden_dim]
        agg:  mean over voxels (permutation-invariant)
        rho:  linear          → [out_dim]

Memory control:
    patch_chunk  — number of patches processed simultaneously (default 256).
                   Lower → less peak RAM, slower.
    voxel_chunk  — if set, splits the voxel loop within each patch chunk.
                   Use when individual patches are too large for RAM.

Input:  [B, C, H, W, Z, T]
Output: [B, out_dim, H_p, W_p, Z_p, T_p]
"""

import torch
import torch.nn as nn


class PermInvariantPatchEmbedding4D(nn.Module):
    """Permutation-invariant 4D patch embedding via a chunked DeepSets encoder."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_dim: int,
        patch_size,
        agg: str = "mean",
        patch_chunk: int = 256,
        voxel_chunk: int | None = None,
    ):
        """
        Args:
            in_channels: Number of input channels C.
            hidden_dim:  Hidden size for the phi MLP.
            out_dim:     Output embedding dimension per patch.
            patch_size:  (ph, pw, pz, pt) — patch size along (H, W, Z, T).
            agg:         Aggregation type; currently only "mean" is supported.
            patch_chunk: Patches processed simultaneously. Lower = less peak RAM.
            voxel_chunk: If set, further splits the voxel loop inside each chunk.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.patch_size = tuple(patch_size)
        self.agg = agg
        self.patch_chunk = patch_chunk
        self.voxel_chunk = voxel_chunk

        self.phi = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.rho = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W, Z, T]

        Returns:
            out: [B, out_dim, H_p, W_p, Z_p, T_p]
        """
        B, C, H, W, Z, T = x.shape
        ph, pw, pz, pt = self.patch_size

        H_p, W_p, Z_p, T_p = H // ph, W // pw, Z // pz, T // pt
        N_vox = ph * pw * pz * pt

        # Reshape into patches: [B * H_p * W_p * Z_p * T_p, N_vox, C]
        x = x.view(B, C, H_p, ph, W_p, pw, Z_p, pz, T_p, pt)
        x = x.permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1).contiguous()
        x = x.view(B * H_p * W_p * Z_p * T_p, N_vox, C)

        N_patches = x.shape[0]
        out = x.new_empty((N_patches, self.out_dim))

        for p0 in range(0, N_patches, self.patch_chunk):
            p1 = min(N_patches, p0 + self.patch_chunk)
            xp = x[p0:p1]  # [P, N_vox, C]
            P = xp.shape[0]

            if self.voxel_chunk is None:
                h = self.phi(xp.reshape(P * N_vox, C))
                h = h.view(P, N_vox, self.hidden_dim).mean(dim=1)
            else:
                h = xp.new_zeros((P, self.hidden_dim))
                count = 0
                for v0 in range(0, N_vox, self.voxel_chunk):
                    v1 = min(N_vox, v0 + self.voxel_chunk)
                    hv = self.phi(xp[:, v0:v1].reshape(P * (v1 - v0), C))
                    hv = hv.view(P, v1 - v0, self.hidden_dim)
                    h += hv.sum(dim=1)
                    count += (v1 - v0)
                h /= count

            out[p0:p1] = self.rho(h)

        out = out.view(B, H_p, W_p, Z_p, T_p, self.out_dim)
        out = out.permute(0, 5, 1, 2, 3, 4).contiguous()  # [B, out_dim, H_p, W_p, Z_p, T_p]
        return out
