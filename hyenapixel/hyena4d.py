
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_4d_positional_features(
    kernel_size: Tuple[int, int, int, int],
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    t_gap: float = 1.0,
    tr_slice: float = 0.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build rotation-invariant 4D positional features for a kernel of size (Kt, Kz, Kh, Kw).

    Rotation invariance in the 3D spatial axes (z, y, x) means kernel values must
    depend only on scalars that are invariant under SO(3) rotations of (z, y, x).

    Physical-unit temporal distance with slice-timing correction:
        t_phys = |Δt * t_gap + Δz * tr_slice|

    This captures the fact that slice z is acquired at physical time
    t * t_gap + z * tr_slice within the scan, so the true temporal separation
    between voxels at offsets (Δt, Δz) is Δt * t_gap + Δz * tr_slice.

    Args:
        kernel_size : (Kt, Kz, Kh, Kw)
        voxel_size  : (sz, sy, sx) — voxel side lengths in mm along (z, y, x).
                      Default (1, 1, 1) = voxel-index units (no physical scaling).
        t_gap       : time between consecutive timepoints in seconds (e.g. TR).
                      Default 1.0 = timepoint-index units (no physical scaling).
        tr_slice    : time between consecutive slice acquisitions within one TR,
                      in seconds.  Set to TR / num_slices for a typical
                      sequential acquisition.  Default 0.0 disables slice-timing
                      correction (reduces to |Δt| * t_gap).
        device      : optional torch device

    Returns:
        pos_feats : Tensor of shape [Kt * Kz * Kh * Kw, 6]
                    Columns: [t_phys, r_3d_phys, r_4d_phys,
                               decay_t, decay_3d, decay_4d]
                    All distance/time values are in physical units when
                    voxel_size, t_gap, and tr_slice are set correctly.
    """
    Kt, Kz, Kh, Kw = kernel_size
    sz, sy, sx = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])
    tg  = float(t_gap)
    trs = float(tr_slice)

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

    # --- Physical temporal distance with slice-timing correction ---
    # t_phys = |Δt * t_gap + Δz * tr_slice|
    # When tr_slice=0 this reduces to |Δt| * t_gap (no correction).
    t_abs = (T * tg + Z * trs).abs()   # physical seconds

    # --- Spatial distances in physical mm ---
    Zp = Z * sz
    Yp = Y * sy
    Xp = X * sx

    # --- Rotation-invariant scalars (in physical units) ---
    r_3d  = torch.sqrt(Zp**2 + Yp**2 + Xp**2)               # physical mm
    r_4d  = torch.sqrt(t_abs**2 + Zp**2 + Yp**2 + Xp**2)    # mixed physical units

    # Closed-form decays — these are the "built-in" non-learnable priors.
    # Learnable anisotropic decay is applied separately in Hyena4DFilter.
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
    4D decay that distinguishes the spatial (z,y,x) and temporal (t) axes.

    Kernel generation:
        1. MLP(pos_feats) -> kernel_base  [N_pos, C]   (rotation-invariant by construction)
        2. decay_mask = exp(-alpha_s * r_3d - alpha_t * |t|)  [N_pos]
           where alpha_s, alpha_t >= 0 are learned via softplus
        3. kernel = kernel_base * decay_mask[:, None]

    Positional features are computed in physical units when voxel_size and
    t_gap are provided (e.g. voxel_size=(0.5, 0.5, 0.5) mm, t_gap=2.0 s).
    This makes alpha_s and alpha_t physically interpretable across scanners:
        alpha_s  ~ 1 / correlation length in mm
        alpha_t  ~ 1 / correlation time in seconds

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
        tr_slice: float = 0.0,
    ):
        """
        Args:
            d_model     : number of channels (C)
            kernel_size : (K_t, K_z, K_h, K_w)
            hidden_dim  : hidden size for the MLP generating kernels
            voxel_size  : (sz, sy, sx) voxel side lengths in mm — scales spatial
                          positional features to physical distances
            t_gap       : time between consecutive timepoints in seconds (e.g. TR)
            tr_slice    : time between consecutive slice acquisitions within one TR
                          (seconds).  Set to TR / num_slices for sequential acquisition.
                          Default 0.0 disables slice-timing correction.
        """
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        pos_feats = make_4d_positional_features(
            kernel_size, voxel_size=voxel_size, t_gap=t_gap, tr_slice=tr_slice
        )  # [N_pos, 6]
        self.register_buffer("pos_feats", pos_feats, persistent=False)

        # Extract the rotation-invariant scalars needed for the learnable decay.
        # pos_feats columns: [0]=|t|, [1]=r_3d, [2]=r_4d, [3..5]=decays
        # Store separately so forward() can cast to input dtype without re-computing.
        self.register_buffer("t_abs_flat", pos_feats[:, 0].clone(), persistent=False)
        self.register_buffer("r_3d_flat",  pos_feats[:, 1].clone(), persistent=False)

        d_pos = pos_feats.shape[-1]
        self.mlp = nn.Sequential(
            nn.Linear(d_pos, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

        # Learnable anisotropic 4D decay rates (stored in log-space; softplus -> >= 0)
        # alpha_s: how fast the kernel decays with 3D spatial distance
        # alpha_t: how fast the kernel decays with temporal distance
        self.log_alpha_s = nn.Parameter(torch.zeros(1))
        self.log_alpha_t = nn.Parameter(torch.zeros(1))

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

        # --- 2. Learnable anisotropic 4D decay ---
        alpha_s = F.softplus(self.log_alpha_s)              # spatial decay rate > 0
        alpha_t = F.softplus(self.log_alpha_t)              # temporal decay rate > 0
        r_3d    = self.r_3d_flat.to(dtype=dtype)            # [N_pos]
        t_abs   = self.t_abs_flat.to(dtype=dtype)           # [N_pos]

        # exp(-alpha_s * r_3d - alpha_t * |t|): anisotropic, rotation-invariant
        decay_mask = torch.exp(-alpha_s * r_3d - alpha_t * t_abs)  # [N_pos]
        kernel_vals = kernel_vals * decay_mask.unsqueeze(1)         # [N_pos, C]

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
        tr_slice: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.hyena4d = Hyena4DFilter(
            d_model, kernel_size, voxel_size=voxel_size, t_gap=t_gap, tr_slice=tr_slice
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
