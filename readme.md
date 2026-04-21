# Hyena4D — fMRI Subject Embedding Pipeline

Self-supervised contrastive learning for 4D fMRI data.  Processes NIfTI volumes
`[H, W, Z, T]` into per-subject embeddings using a permutation-invariant patch
encoder (DeepSets RPE) followed by a rotation-aware Hyena4D spatiotemporal
encoder.  Physical scanner parameters (voxel size in mm, TR in seconds, and
slice acquisition time) are built into the positional features so that learned
decay rates are physically interpretable across scanners.

---

## Project Structure

```
Pack-Hyena4d_v2/
│
├── train_hyena.py               # Pre-training: contrastive loss → checkpoints
├── run_hyena.py                 # Inference: load checkpoint → subject embeddings
├── run_rpe.py                   # Standalone RPE wrapper (patch embedding only)
├── invariant_patching_4d.py     # PermInvariantPatchEmbedding4D (DeepSets, chunked)
│
├── hyenapixel/
│   ├── hyena4d.py               # Hyena4DFilter + Hyena4DPixelBlock (core operators)
│   ├── hyena4d_model.py         # Hyena4DBackbone + Hyena4DContrastiveEncoder + loss
│   └── __init__.py
│
├── envs/
│   ├── linux_env.yaml           # GPU environment (CUDA 12.1)
│   └── os_env.yaml              # macOS environment (CPU / no MPS)
│
├── input/                       # Place .nii files here for training
└── output/                      # Checkpoints and final embeddings
```

---

## Workflow

The full pipeline has two stages — pre-training on a large unlabelled cohort,
then prediction-only on a target cohort using the frozen trained weights.
**No fine-tuning step is needed**: the contrastive pre-training objective directly
optimises subject-discriminative representations.

### Data formats

Both `.nii` (uncompressed) and `.nii.gz` (gzip-compressed) NIfTI files are
supported.  Voxel size and TR are **auto-detected from the NIfTI header** by
default — no need to specify `--voxel-size` or `--t-gap` unless you want to
override the header values.

### Stage 1 — Pre-training (contrastive, `.nii.gz` cohort)

Recommended settings for the pre-training dataset (shape 90 × 20 × 50 × 1000,
~170 mice, TR ≈ 1.0 s):

```bash
python train_hyena.py \
    --input-dir    pretrain_data/ \
    --output-dir   output/ \
    --subject-field 0 \          # "sub-001" from "sub-001_ses-1_..._bold.nii.gz"
    --patch-size   9 4 5 10 \    # divides (90,20,50), T_sub=50 → 10×5×10×5 tokens
    --t-sub        50 \
    --batch-size   32 \          # larger batches = more positive pairs per step
    --epochs       50 \
    --tr-slice     0.02          # TR/num_slices = 1.0s/50 slices
```

Physical parameters (voxel size, TR) are read automatically from the `.nii.gz`
header.  Saves a checkpoint every epoch and keeps `best_model.pt` for the
lowest-loss epoch.  Subject-level embeddings of the training cohort are also
saved at the end of training.

**Resuming**: the model does NOT automatically detect checkpoints in `--output-dir`.
Without `--resume` it always starts from random weights:

```bash
python train_hyena.py ... --resume output/best_model.pt
```

### Stage 2 — Prediction only (`.nii` target cohort)

Recommended settings for the target dataset (shape 64 × 64 × 20 × 500,
~20 mice, TR ≈ 1.5 s):

```bash
python run_hyena.py \
    --input-dir    target_data/ \
    --output-dir   output/ \
    --checkpoint   output/best_model.pt \
    --subject-field 1 \          # "B16" from "driftRespCardlessBOLD_B16_..."
    --patch-size   8 8 4 10 \    # divides (64,64,20), T=500 → 8×8×5×50 tokens
    --tr-slice     0.075         # TR/num_slices = 1.5s/20 slices
```

Physical parameters are read automatically from the `.nii` header.  Loads the
pre-trained encoder weights and produces `subject_embeddings.pt` for the
target cohort — **no re-training or fine-tuning occurs**.

> **`--tr-slice` note**: this parameter cannot be read from the header and must
> be set manually.  Use `TR / num_slices` for sequential acquisition (e.g.
> 1.5 s / 20 slices = 0.075 s).  Omit or set to 0 to disable slice-timing
> correction.

---

## Architecture

```
NIfTI  [H, W, Z, T]
  │
  ├─ temporal crop → subsequences [H, W, Z, T_sub]        (train_hyena only)
  │
  └─→ PermInvariantPatchEmbedding4D  (DeepSets, chunked, CPU)
        patches of size (ph × pw × pz × pt)
        phi MLP per voxel → mean pool over patch → rho linear
        └─→ [B, proj_dim, H_p, W_p, Z_p, T_p]
              │
              └─→ permute → [B, proj_dim, T_p, Z_p, H_p, W_p]
                    │
                    └─→ Hyena4DContrastiveEncoder  (GPU/CPU)
                          │
                          ├─ 1×1×1 Conv3d channel projection
                          │
                          ├─ × num_blocks:
                          │     LayerNorm → Hyena4DFilter → + residual
                          │     LayerNorm → channel MLP   → + residual
                          │
                          ├─ global mean pool (T, Z, H, W) → [B, embed_dim]
                          │
                          └─ projection head (LN → Linear → GELU → Linear)
                               L2 normalise → [B, projection_dim]
                                    │
                                    └─→ mean-pool per subject
                                          └─→ subject_embeddings.pt
```

---

## Key Design Decisions

### Rotation-invariant kernel (Hyena4DFilter)

Kernel weights at each 4D offset `(Δt, Δz, Δy, Δx)` are generated by an MLP
whose inputs are rotation-invariant scalars:

| Feature | Formula | Unit |
|---|---|---|
| `t_abs` | `\|Δt\| × t_gap` | seconds |
| `r_3d` | `sqrt((sz·Δz)² + (sy·Δy)² + (sx·Δx)²)` | mm |
| `r_4d` | `sqrt(t_abs² + r_3d²)` | mixed |
| `decay_t` | `1 / (1 + t_abs)` | — |
| `decay_3d` | `1 / (1 + r_3d)` | — |
| `decay_4d` | `1 / (1 + r_4d)` | — |

An additional **learnable anisotropic decay** is multiplied on top:

```
decay_mask = exp(−α_s · r_3d − α_t · t_abs)
```

`α_s` and `α_t` are learned via softplus and carry physical meaning:
- `α_s ≈ 1 / spatial correlation length (mm)`
- `α_t ≈ 1 / temporal correlation time (s)`

### Physical-unit scaling (SR / TR / slice timing)

Three parameters bring all positional features into physical units:

- **`--voxel-size sz sy sx`** (mm): spatial distances become `r_3d = sqrt((sz·Δz)² + (sy·Δy)² + (sx·Δx)²)` in mm.
- **`--t-gap`** (s): the inter-volume TR.
- **`--tr-slice`** (s): the inter-slice acquisition time within one TR (typically `TR / num_slices`). This implements **slice-timing correction** directly in the kernel: the physical temporal distance between two voxels at offset `(Δt, Δz)` is

  `t_phys = |Δt · t_gap + Δz · tr_slice|`

  rather than the naïve `|Δt| · t_gap`.  Setting `tr_slice=0` (default) disables this correction.

Together these make the learned decay rates `α_s` (mm⁻¹) and `α_t` (s⁻¹) scanner-independent.

### DeepSets RPE (PermInvariantPatchEmbedding4D)

Each spatial patch is encoded independently by applying a shared MLP `φ` to
every voxel, averaging across the patch, and then projecting with `ρ`.  This is
a standard DeepSets operation and is permutation-invariant within a patch.
Because patches sit on a fixed spatial grid, the encoding is **not** invariant
to spatial rotations (the same voxels fall into different patches after rotation),
which is the intended behaviour for learning spatially structured representations.

### T×Z depth axis merging

The T and Z axes are merged into a single depth axis `D = T×Z` for the Conv3d
call.  Combined with the Hyena4D kernel's radially-symmetric spatial decay, this
makes the filter effectively treat `(t, z)` as the depth dimension while
preserving `(y, x)` as the spatial plane — an efficient approximation to a true
4D convolution.

### Contrastive loss (InfoNCE)

Same-mouse subsequences are positive pairs; all other pairs within the batch are
negatives.  The temperature parameter controls the hardness of the loss.

---

## Setup

**macOS (CPU):**
```bash
conda env create -f envs/os_env.yaml
conda activate hyena4d_osx
```

**Linux (CUDA 12.1):**
```bash
conda env create -f envs/linux_env.yaml
conda activate hyena4d_linux
```

> MPS (Apple Silicon GPU) is not supported — PyTorch MPS does not implement Conv3d.

---

## Training arguments (`train_hyena.py`)

| Argument | Default | Description |
|---|---|---|
| `--input-dir` | `input` | Directory with `.nii` / `.nii.gz` files |
| `--output-dir` | `output` | Directory for checkpoints |
| `--t-sub` | `50` | Temporal subsequence length (timepoints) |
| `--stride` | `t_sub` | Sliding window stride (default = non-overlapping) |
| `--subject-field` | `1` | Token index in filename (split on `_`) for subject ID; use `0` for BIDS-style `sub-001_...` names |
| `--patch-size H W Z T` | `8 8 4 10` | RPE patch size — must evenly divide each volume dimension |
| `--rpe-hidden` | `64` | RPE phi MLP hidden dim |
| `--embed-dim` | `64` | Backbone embedding dim |
| `--num-blocks` | `4` | Number of Hyena4DPixelBlocks |
| `--projection-dim` | `128` | Projection head output dim |
| `--kernel-size T Z H W` | `3 3 3 3` | Hyena4D filter kernel size |
| `--voxel-size SZ SY SX` | auto | Voxel side lengths in mm (z, y, x); auto-detected from NIfTI header if omitted |
| `--t-gap` | auto | TR / time between timepoints in seconds; auto-detected from NIfTI header if omitted |
| `--tr-slice` | `0.0` | Time between consecutive slice acquisitions within one TR (s); use `TR/num_slices` for sequential acquisition; `0` = disabled |
| `--patch-chunk` | `16` | Patches per RPE forward chunk |
| `--voxel-chunk` | `512` | Voxels per inner RPE chunk (0 = disabled) |
| `--epochs` | `50` | Training epochs |
| `--batch-size` | `8` | Subsequences per batch |
| `--lr` | `1e-3` | Adam learning rate |
| `--temperature` | `0.1` | InfoNCE temperature |
| `--save-every` | `1` | Save numbered checkpoint every N epochs |
| `--resume` | — | Path to checkpoint to resume from |

---

## Inference arguments (`run_hyena.py`)

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | — | Path to `best_model.pt` from `train_hyena.py` |
| `--input-dir` | `input` | Directory with `.nii` / `.nii.gz` files |
| `--output-dir` | `output` | Directory for output `.pt` file |
| `--output-name` | `subject_embeddings.pt` | Output filename |
| `--no-rpe` | — | Skip RPE; raw voxels as single-channel input |
| `--patch-size H W Z T` | `8 8 4 10` | Must match training patch size |
| `--patch-chunk` | `16` | Patches per RPE forward chunk |
| `--voxel-chunk` | `512` | Voxels per inner chunk (0 = disabled) |
| `--embed-dim` | `64` | Must match training embed-dim |
| `--num-blocks` | `4` | Must match training num-blocks |
| `--projection-dim` | `128` | Must match training projection-dim |
| `--subject-field` | `1` | Token index in filename (split on `_`) for subject ID; use `0` for BIDS-style `sub-001_...` names |
| `--voxel-size SZ SY SX` | auto | Voxel side lengths in mm (z, y, x); auto-detected from NIfTI header if omitted |
| `--t-gap` | auto | TR / time between timepoints in seconds; auto-detected from NIfTI header if omitted |
| `--tr-slice` | `0.0` | Must match the value used during training; `TR/num_slices` for sequential acquisition |

---

## Output format

Both scripts save a `.pt` dict:

```python
# Both train_hyena.py and run_hyena.py produce the same format:
data = torch.load("output/subject_embeddings.pt")
embeddings  = data["embeddings"]   # Tensor [N_subjects, projection_dim]
subject_ids = data["subject_ids"]  # list[str]
```

Checkpoint format (`best_model.pt`, `checkpoint_epoch*.pt`):
```python
{
    "epoch":                int,
    "model_state_dict":     dict,   # Hyena4DFullPipeline (RPE + encoder)
    "optimizer_state_dict": dict,
    "loss":                 float,
}
```
