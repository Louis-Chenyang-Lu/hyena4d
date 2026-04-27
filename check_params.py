import os; os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from hyenapixel.hyena4d_model import Hyena4DContrastiveEncoder
from invariant_patching_4d import PermInvariantPatchEmbedding4D

# Settings matching real pre-training run
PATCH_SIZE   = (9, 4, 5, 10)
RPE_HIDDEN   = 64
PROJ_DIM     = 128
EMBED_DIM    = 64
NUM_BLOCKS   = 4
KERNEL_SIZE  = (3, 3, 3, 3)

# ---------------------------------------------------------------
# 1. RPE
# ---------------------------------------------------------------
rpe = PermInvariantPatchEmbedding4D(
    in_channels=1, hidden_dim=RPE_HIDDEN, out_dim=PROJ_DIM,
    patch_size=PATCH_SIZE,
)

print("=" * 60)
print("RPE (PermInvariantPatchEmbedding4D)")
print("=" * 60)
for n, p in rpe.named_parameters():
    print(f"  {n:45s}  {list(p.shape)}  =  {p.numel():>8,}")
rpe_total = sum(p.numel() for p in rpe.parameters())
print(f"  {'TOTAL':45s}           {rpe_total:>8,}")

# Manual calculation
voxels_per_patch = PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2] * PATCH_SIZE[3]
phi_in  = 1           # scalar voxel value
phi_h   = RPE_HIDDEN
phi_out = RPE_HIDDEN
rho_in  = RPE_HIDDEN
rho_out = PROJ_DIM

phi_w1 = phi_in  * phi_h  + phi_h   # Linear(1, 64)  + bias
phi_w2 = phi_h   * phi_h  + phi_h   # Linear(64, 64) + bias  (if present)
phi_w3 = phi_h   * phi_out + phi_out # Linear(64, 64) + bias
rho_w  = rho_in  * rho_out + rho_out # Linear(64, 128) + bias

print()
print("  [Manual check]")
print(f"  phi MLP: Linear(1→{phi_h}) + Linear({phi_h}→{phi_h}) + Linear({phi_h}→{phi_out})")
print(f"    = (1×{phi_h}+{phi_h}) + ({phi_h}×{phi_h}+{phi_h}) + ({phi_h}×{phi_out}+{phi_out})")
print(f"    = {1*phi_h+phi_h} + {phi_h*phi_h+phi_h} + {phi_h*phi_out+phi_out}")
print(f"  rho:    Linear({rho_in}→{rho_out}) = {rho_in}×{rho_out}+{rho_out} = {rho_w}")

# ---------------------------------------------------------------
# 2. Encoder
# ---------------------------------------------------------------
enc = Hyena4DContrastiveEncoder(
    in_channels=PROJ_DIM, embed_dim=EMBED_DIM, num_blocks=NUM_BLOCKS,
    projection_dim=PROJ_DIM, kernel_size=KERNEL_SIZE,
)

print()
print("=" * 60)
print("Encoder (Hyena4DContrastiveEncoder)")
print("=" * 60)

# Per-block breakdown (all blocks are identical)
block = enc.blocks[0]
block_params = {n: p.numel() for n, p in block.named_parameters()}
print(f"\n  One Hyena4DPixelBlock  (× {NUM_BLOCKS} blocks)")
for n, v in block_params.items():
    print(f"    {n:45s}  {v:>8,}")
print(f"    {'SUBTOTAL one block':45s}  {sum(block_params.values()):>8,}")
print(f"    {'× {NUM_BLOCKS} blocks':45s}  {sum(block_params.values())*NUM_BLOCKS:>8,}")

# Other encoder parts
other = {
    "proj (Conv3d 1×1×1  in→embed)": sum(p.numel() for n,p in enc.named_parameters() if n.startswith("proj")),
    "projection_head (LN+Linear+GELU+Linear)": sum(p.numel() for n,p in enc.named_parameters() if n.startswith("projection_head")),
    "classifier_norm + classifier_head": sum(p.numel() for n,p in enc.named_parameters() if n.startswith("classifier")),
}
print()
for k, v in other.items():
    print(f"  {k:55s}  {v:>8,}")

enc_total = sum(p.numel() for p in enc.parameters())
print(f"\n  {'TOTAL Encoder':55s}  {enc_total:>8,}")

# ---------------------------------------------------------------
# 3. Manual kernel calculation
# ---------------------------------------------------------------
Kt, Kz, Kh, Kw = KERNEL_SIZE
N_pos   = Kt * Kz * Kh * Kw   # kernel positions
d_pos   = 6                    # positional feature dim
h_mlp   = 64                   # MLP hidden (default in Hyena4DFilter)
C       = EMBED_DIM

mlp_w1  = d_pos * h_mlp + h_mlp   # Linear(6, 64)
mlp_w2  = h_mlp * C     + C       # Linear(64, 64)
decay   = 2                        # log_alpha_s, log_alpha_t

print()
print("=" * 60)
print("Kernel MLP detail (Hyena4DFilter, per block)")
print("=" * 60)
print(f"  Kernel positions N_pos = {Kt}×{Kz}×{Kh}×{Kw} = {N_pos}")
print(f"  Positional features d_pos = {d_pos}  [t,r3d,r4d,dec_t,dec_3d,dec_4d]")
print(f"  MLP: Linear({d_pos}→{h_mlp}) + GELU + Linear({h_mlp}→{C})")
print(f"    = ({d_pos}×{h_mlp}+{h_mlp}) + ({h_mlp}×{C}+{C})")
print(f"    = {mlp_w1} + {mlp_w2} = {mlp_w1+mlp_w2}")
print(f"  Learnable decay params: log_alpha_s + log_alpha_t = {decay}")
print(f"  Total per filter: {mlp_w1+mlp_w2+decay}")

# ---------------------------------------------------------------
# 4. Grand total
# ---------------------------------------------------------------
grand = rpe_total + enc_total
print()
print("=" * 60)
print("GRAND TOTAL")
print("=" * 60)
print(f"  RPE     : {rpe_total:>8,}")
print(f"  Encoder : {enc_total:>8,}")
print(f"  TOTAL   : {grand:>8,}")
