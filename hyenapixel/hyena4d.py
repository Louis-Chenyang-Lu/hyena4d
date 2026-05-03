
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_4d_positional_features(
    kernel_size: Tuple[int, int, int, int],
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    t_gap: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build rotation-invariant 4D positional features for a kernel of size (Kt, Kz, Kh, Kw).

    Rotation invariance in the 3D spatial axes (z, y, x) means kernel values must
    depend only on scalars that are invariant under SO(3) rotations of (z, y, x).

    Physical distances (true distances accounting for resolution):
        deltaX_phys = patch_offset_x * patch_size_x * sx   [mm]
        deltaY_phys = patch_offset_y * patch_size_y * sy   [mm]
        deltaZ_phys = patch_offset_z * patch_size_z * sz   [mm]
        deltaT_phys = patch_offset_t * patch_size_t * t_gap  [seconds]

    Within the kernel grid these simplify to:
        deltaX_phys = X * sx,  deltaY_phys = Y * sy,  deltaZ_phys = Z * sz
        deltaT_phys = T * t_gap  (all slices assumed simultaneous, so |deltaT| used)

    Args:
        kernel_size : (Kt, Kz, Kh, Kw)
        voxel_size  : (sz, sy, sx) — voxel side lengths in mm along (z, y, x).
                      Default (1, 1, 1) = voxel-index units (no physical scaling).
        t_gap       : time between consecutive timepoints in seconds (e.g. TR).
                      Default 1.0 = timepoint-index units (no physical scaling).
        device      : optional torch device

    Returns:
        pos_feats : Tensor of shape [Kt * Kz * Kh * Kw, 6]
                    Columns: [t_abs, r_3d_phys, r_4d_phys,
                               decay_t, decay_3d, decay_4d]
                    All distance/time values are in physical units when
                    voxel_size and t_gap are set correctly.
    """
    Kt, Kz, Kh, Kw = kernel_size
    sz, sy, sx = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])
    tg = float(t_gap)

    t_coords = torch.arange(Kt, dtype=torch.float32) - (Kt - 1) / 2.0
    z_coords = torch.arange(Kz, dtype=torch.float32) - (Kz - 1) / 2.0
    y_coords = torch.arange(Kh, dtype=torch.float32) - (Kh - 1) / 2.0
    x_coords = torch.arange(Kw, dtype=torch.float32) - (Kw - 1) / 2.0

    if device is not None:
        t_coords = t_coords.to(device)
        z_coords = z_coords.to(device)
        y_coords = y_coords.to(device)
        x_coords = x_coords.to(device)

    T, Z, Y, X = torch.meshgrid(
        t_coords, z_coords, y_coords, x_coords, indexing="ij"
    )  # [Kt, Kz, Kh, Kw]

    # True physical distances (considering resolution):
    #   deltaT_phys = |deltaT_index| * t_gap  [seconds]
    #   deltaZ_phys = deltaZ_index  * sz       [mm]
    #   deltaY_phys = deltaY_index  * sy       [mm]
    #   deltaX_phys = deltaX_index  * sx       [mm]
    t_abs = T.abs() * tg   # physical seconds (|deltaT_phys|)
    Zp = Z * sz            # physical mm
    Yp = Y * sy            # physical mm
    Xp = X * sx            # physical mm

    # --- Rotation-invariant scalars (in physical units) ---
    r_3d  = torch.sqrt(Zp**2 + Yp**2 + Xp**2)               # physical mm
    r_4d  = torch.sqrt(t_abs**2 + Zp**2 + Yp**2 + Xp**2)    # mixed physical units

    # Closed-form decays — these are the "built-in" non-learnable priors.
    # Learnable Gaussian decay is applied separately in Hyena4DFilter.
    decay_t   = 1.0 / (1.0 + t_abs)    # temporal locality prior
    decay_3d  = 1.0 / (1.0 + r_3d)     # spatial locality prior
    decay_4d  = 1.0 / (1.0 + r_4d)     # isotropic 4D locality prior

    feats = torch.stack(
        [t_abs, r_3d, r_4d, decay_t, decay_3d, decay_4d],
        dim=-1,  # [Kt, Kz, Kh, Kw, 6]
    )

    feats = feats.reshape(-1, feats.shape[-1])  # [N_pos, 6]
    return feats


class Hyena4DFilter(nn.Module):
    """
    Parameterizes a depthwise 4D Hyena-style kernel via a small MLP over
    rotation-invariant 4D positional features, plus a learnable anisotropic
    Gaussian decay that distinguishes the spatial (z,y,x) and temporal (t) axes.

    Kernel generation:
        1. MLP(pos_feats) -> kernel_base  [N_pos, C]   (rotation-invariant by construction)
        2. decay_mask = exp(-alpha * (deltaX^2 + deltaY^2 + deltaZ^2) - beta * deltaT^2)
           where deltaX, deltaY, deltaZ are true physical distances in mm,
           deltaT is true physical time in seconds (considering scanner resolution),
           and alpha, beta >= 0 are learned via softplus.
        3. kernel = kernel_base * decay_mask[:, None]

    Physical distances accounting for resolution:
        deltaX_phys = x_index_offset * sx  (mm),  where sx is voxel size in x
        deltaY_phys = y_index_offset * sy  (mm),  where sy is voxel size in y
        deltaZ_phys = z_index_offset * sz  (mm),  where sz is voxel size in z
        deltaT_phys = t_index_offset * t_gap  (seconds)

    Positional features are computed in physical units when voxel_size and
    t_gap are provided (e.g. voxel_size=(0.5, 0.5, 0.5) mm, t_gap=2.0 s).
    This makes alpha and beta physically interpretable across scanners:
        alpha  ~ 1 / (2 * spatial_correlation_length^2)  [mm^{-2}]
        beta   ~ 1 / (2 * temporal_correlation_time^2)   [s^{-2}]

    For efficiency, the 4D conv is implemented as a 3D depthwise conv by merging
    the T and Z axes:  D = T * Z,  K_D = K_t * K_z.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: Tuple[int, int, int, int],
        hidden_dim: int = 64,
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        t_gap: float = 1.0,
    ):
        """
        Args:
            d_model     : number of channels (C)
            kernel_size : (K_t, K_z, K_h, K_w)
            hidden_dim  : hidden size for the MLP generating kernels
            voxel_size  : (sz, sy, sx) voxel side lengths in mm — scales spatial
                          positional features to physical distances
            t_gap       : time between consecutive timepoints in seconds (e.g. TR)
        """
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        pos_feats = make_4d_positional_features(
            kernel_size, voxel_size=voxel_size, t_gap=t_gap
        )  # [N_pos, 6]
        self.register_buffer("pos_feats", pos_feats, persistent=False)

        # Recompute individual physical displacements for the Gaussian decay.
        # We need deltaX_phys^2 + deltaY_phys^2 + deltaZ_phys^2 and deltaT_phys^2
        # separately so the decay is exp(-alpha * r3d_sq - beta * dt_sq).
        Kt, Kz, Kh, Kw = kernel_size
        sz, sy, sx = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])
        tg = float(t_gap)

        t_coords = torch.arange(Kt, dtype=torch.float32) - (Kt - 1) / 2.0
        z_coords = torch.arange(Kz, dtype=torch.float32) - (Kz - 1) / 2.0
        y_coords = torch.arange(Kh, dtype=torch.float32) - (Kh - 1) / 2.0
        x_coords = torch.arange(Kw, dtype=torch.float32) - (Kw - 1) / 2.0
        T_g, Z_g, Y_g, X_g = torch.meshgrid(
            t_coords, z_coords, y_coords, x_coords, indexing="ij"
        )
        # True physical distances (resolution-aware):
        #   r3d_sq = (deltaZ * sz)^2 + (deltaY * sy)^2 + (deltaX * sx)^2  [mm^2]
        #   dt_sq  = (|deltaT| * t_gap)^2                                  [s^2]
        r3d_sq = (Z_g * sz) ** 2 + (Y_g * sy) ** 2 + (X_g * sx) ** 2   # [Kt,Kz,Kh,Kw]
        dt_sq  = (T_g.abs() * tg) ** 2                                   # [Kt,Kz,Kh,Kw]

        self.register_buffer("r3d_sq_flat", r3d_sq.reshape(-1), persistent=False)  # [N_pos]
        self.register_buffer("dt_sq_flat",  dt_sq.reshape(-1),  persistent=False)  # [N_pos]

        d_pos = pos_feats.shape[-1]
        self.mlp = nn.Sequential(
            nn.Linear(d_pos, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

        # Learnable Gaussian decay rates (stored in log-space; softplus -> >= 0)
        # alpha: spatial decay rate  [mm^{-2}]  — controls how fast the kernel
        #        loses sensitivity to spatially distant voxels
        # beta:  temporal decay rate [s^{-2}]   — controls temporal sensitivity decay
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.log_beta  = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, C, T, Z, H, W]

        Returns:
            y: Tensor of shape [B, C, T, Z, H, W]
        """
        if x.dim() != 6:
            raise ValueError(f"Expected [B, C, T, Z, H, W], got {x.shape}")
        B, C, T, Z, H, W = x.shape
        if C != self.d_model:
            raise ValueError(f"Input channels {C} != d_model {self.d_model}.")

        Kt, Kz, Kh, Kw = self.kernel_size
        KD = Kt * Kz  # merged depth-kernel size

        dtype = x.dtype

        # --- 1. Generate base kernel values (rotation-invariant) ---
        pos_feats = self.pos_feats.to(dtype=dtype)          # [N_pos, 6]
        kernel_vals = self.mlp(pos_feats)                   # [N_pos, C]

        # --- 2. Learnable Gaussian decay (physically meaningful, resolution-aware) ---
        # New formula: exp(-alpha * (deltaX^2 + deltaY^2 + deltaZ^2) - beta * deltaT^2)
        # where all deltas are TRUE physical distances:
        #   deltaX_phys = x_index_offset * sx  [mm]
        #   deltaY_phys = y_index_offset * sy  [mm]
        #   deltaZ_phys = z_index_offset * sz  [mm]
        #   deltaT_phys = |t_index_offset| * t_gap  [seconds]
        alpha    = F.softplus(self.log_alpha)               # spatial  decay rate > 0  [mm^{-2}]
        beta     = F.softplus(self.log_beta)                # temporal decay rate > 0  [s^{-2}]
        r3d_sq   = self.r3d_sq_flat.to(dtype=dtype)         # [N_pos]  (deltaX^2+deltaY^2+deltaZ^2) in mm^2
        dt_sq    = self.dt_sq_flat.to(dtype=dtype)          # [N_pos]  (deltaT_phys)^2 in s^2

        # Gaussian decay: rotation-invariant in 3D space and independent in time
        decay_mask = torch.exp(-alpha * r3d_sq - beta * dt_sq)  # [N_pos]
        kernel_vals = kernel_vals * decay_mask.unsqueeze(1)      # [N_pos, C]

        # --- 3. Reshape into depthwise conv3d kernel ---
        kernel = kernel_vals.t().contiguous()               # [C, N_pos]
        kernel = kernel.view(C, 1, KD, Kh, Kw)             # [C, 1, KD, Kh, Kw]

        # --- 4. Apply as depthwise 3D conv over merged T×Z depth axis ---
        x_3d = x.view(B, C, T * Z, H, W)                   # [B, C, D=T*Z, H, W]

        y_3d = F.conv3d(
            x_3d,
            kernel,
            padding=(KD // 2, Kh // 2, Kw // 2),
            groups=C,                                       # depthwise: O(C) not O(C²)
        )  # [B, C, D, H, W]

        y = y_3d.view(B, C, T, Z, H, W)
        return y


class Hyena4DPixelBlock(nn.Module):
    """
    A single Hyena4D block in the spirit of MetaFormer / HyenaPixel,
    extended to 4D volumes:

        x -> norm -> Hyena4DFilter (spatial-temporal mixing) -> +residual
          -> norm -> channel MLP -> +residual

    Layout:
        Input / Output: [B, C, T, Z, H, W]
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: Tuple[int, int, int, int],
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        t_gap: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.hyena4d = Hyena4DFilter(
            d_model, kernel_size, voxel_size=voxel_size, t_gap=t_gap
        )

        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(drop),
        )

    def _apply_ln(self, x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
        """LayerNorm over channel dim (last) while input is [B, C, T, Z, H, W]."""
        B, C, T, Z, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 4, 5, 1)   # [B, T, Z, H, W, C]
        x_norm = ln(x_perm)
        return x_norm.permute(0, 5, 1, 2, 3, 4).contiguous()   # [B, C, T, Z, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, Z, H, W]
        Returns:
            Tensor: [B, C, T, Z, H, W]
        """
        # --- Hyena mixing sub-block ---
        x = x + self.hyena4d(self._apply_ln(x, self.norm1))

        # --- Channel MLP sub-block ---
        B, C, T, Z, H, W = x.shape
        shortcut = x
        x_norm = self._apply_ln(x, self.norm2)
        # flatten spatial/temporal dims, apply MLP channel-wise, restore
        x_flat = x_norm.permute(0, 2, 3, 4, 5, 1).reshape(B * T * Z * H * W, C)
        x_flat = self.mlp(x_flat)
        x_out  = x_flat.view(B, T, Z, H, W, C).permute(0, 5, 1, 2, 3, 4).contiguous()
        x = shortcut + x_out
        return x
