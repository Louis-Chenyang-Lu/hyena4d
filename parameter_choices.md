# Hyena4D — Parameter & Architecture Choices

This document explains how each parameter was chosen and why, grounded in the
two datasets used in this project.

---

## Datasets

| Dataset | Shape (H × W × Z × T) | Subjects | Role |
|---|---|---|---|
| Pre-training (`.nii.gz`) | 90 × 20 × 50 × 1000 | ~170 mice | Contrastive pre-training |
| Prediction (`.nii`) | 64 × 64 × 20 × 500 | ~20 mice | Embedding extraction only |

All parameter choices below follow directly from these shapes and dataset sizes.

---

## Patch Size (`--patch-size H W Z T`)

The RPE patch encoder divides the volume into non-overlapping patches.
The hard constraint is: **each patch dimension must evenly divide the
corresponding volume dimension**.

**Pre-training dataset** (H=90, W=20, Z=50, T_sub=50):

| Axis | Volume dim | Patch dim | Num patches |
|---|---|---|---|
| H | 90 | 9 | 10 |
| W | 20 | 4 | 5 |
| Z | 50 | 5 | 10 |
| T | 50 | 10 | 5 |

Recommended: `--patch-size 9 4 5 10` → token grid `10 × 5 × 10 × 5` = 2500 tokens per sample.

**Prediction dataset** (H=64, W=64, Z=20, T=500, processed in full):

| Axis | Volume dim | Patch dim | Num patches |
|---|---|---|---|
| H | 64 | 8 | 8 |
| W | 64 | 8 | 8 |
| Z | 20 | 4 | 5 |
| T | 500 | 10 | 50 |

Recommended: `--patch-size 8 8 4 10` → token grid `8 × 8 × 5 × 50` = 16000 tokens per sample.

The two datasets use different patch sizes and produce different token grid
sizes — this is fine because the global mean pool at the end of the backbone
collapses all spatiotemporal dimensions regardless of grid size.  The RPE
weights also transfer across datasets because the phi MLP maps each individual
scalar voxel value (not a fixed-size patch) to a vector, so it is
dimension-agnostic.

---

## Temporal Subsequence Length (`--t-sub`)

T=1000 timepoints is too long to process in one forward pass (memory).
A sliding window approach splits each scan into shorter subsequences.

**Chosen value: `--t-sub 50`**

- At TR ≈ 1 s, 50 timepoints = 50 seconds — long enough to capture the full
  haemodynamic response function (HRF peaks at ~5 s, returns to baseline
  by ~25–30 s).
- 1000 / 50 = 20 non-overlapping windows per mouse.
- 170 mice × 20 windows = **3400 training samples** total.
- Patch size T=10 divides T_sub=50 evenly (50 / 10 = 5 tokens along T).

---

## Batch Size (`--batch-size`)

The InfoNCE contrastive loss requires **positive pairs to appear in the same
batch** (two windows from the same mouse).  If the batch is too small, most
batches contain no positive pairs and the loss carries no useful signal.

Expected same-mouse pairs per batch of size B drawn from N=3400 samples
(170 mice × 20 windows each):

```
E[pairs] ≈ B × (B-1) × (20/3400) / 2
```

| Batch size | Expected same-mouse pairs |
|---|---|
| 8 | ~0.16 |
| 16 | ~0.67 |
| 32 | ~2.77 |
| 64 | ~11.9 |

**Recommended: `--batch-size 32`** — gives ~3 positive pairs per step on
average, enough for a meaningful contrastive gradient, without requiring
excessive GPU memory.

---

## Physical Parameters

### Voxel size (`--voxel-size SZ SY SX`)

Spatial distances in the Hyena4D kernel are computed as:

```
r_3d = sqrt((sz·Δz)² + (sy·Δy)² + (sx·Δx)²)   [mm]
```

Without physical units, `r_3d` is in voxels and the learned decay rate `α_s`
is scanner-dependent (changes if you resample the data).  With voxel sizes in
mm, `α_s` has units mm⁻¹ and is physically interpretable across scanners.

**Auto-detected** from the NIfTI header via `nibabel`:
```python
zooms = hdr.get_zooms()
sz = zooms[2]   # Z axis (third spatial dim)
sy = zooms[0]   # H axis (first spatial dim)
sx = zooms[1]   # W axis (second spatial dim)
```

### TR / time gap (`--t-gap`)

Temporal distances in the kernel:
```
t_abs = |Δt · t_gap|   [seconds]
```

**Auto-detected** from `zooms[3]` (the 4th zoom = repetition time).
A common scanner bug stores TR in milliseconds but labels the unit as `'sec'`.
The fix applied:
```python
if time_unit == "msec" or (time_unit == "sec" and tr > 100):
    tr /= 1000.0
```
This correctly converts TR=1500.0 (mislabelled as sec) → 1.5 s.

### Slice acquisition time (`--tr-slice`)

Cannot be read from the header — must be set manually.

```
tr_slice = TR / num_slices
```

| Dataset | TR | num_slices | tr_slice |
|---|---|---|---|
| Pre-training | 1.0 s | 50 | 0.02 s |
| Prediction | 1.5 s | 20 | 0.075 s |

This enables **slice-timing correction** inside the kernel:
```
t_phys = |Δt · t_gap + Δz · tr_slice|
```
Slice z is physically acquired at time `t·t_gap + z·tr_slice`, so two voxels
at the same nominal timepoint but different slices have a real temporal
separation of `|Δz · tr_slice|`.  Setting `tr_slice=0` disables the
correction (no effect on learned weights).

---

## Network Depth (`--num-blocks`)

Each Hyena4DPixelBlock adds one Hyena4D filter layer and one channel MLP,
both with residual connections.  Deeper networks are more expressive but:

- Harder to optimise on small datasets (risk of overfitting or vanishing
  gradients).
- Slower per forward pass.

**Chosen value: `--num-blocks 4`**

With ~170 mice this is a small dataset by deep-learning standards.  Four
blocks provide enough depth to learn non-trivial spatiotemporal structure
without over-parameterising.

---

## Embedding and Projection Dimensions

### `--embed-dim 64`

The internal feature dimension of the Hyena4D backbone.  Kept small (64)
because the dataset is small — a larger hidden dimension would risk memorising
rather than generalising.

### `--projection-dim 128`

The output dimension of the contrastive projection head (the vector that goes
into the InfoNCE loss).  Standard practice in contrastive learning (SimCLR,
MoCo, etc.) is to use a projection head with output dimension 2× the backbone
dimension.  The subject embeddings saved to disk are these 128-d vectors after
mean-pooling over subsequences.

---

## Subject Field (`--subject-field`)

Filenames are split on `_` and the N-th token is used as the subject ID.

| Dataset | Example filename | Field | Subject ID |
|---|---|---|---|
| Pre-training (BIDS) | `sub-001_ses-1_task-rest_acq-EPI_bold.nii.gz` | `0` | `sub-001` |
| Prediction | `driftRespCardlessBOLD_B16_Final_scan5.nii` | `1` | `B16` |

Use `--subject-field 0` for BIDS-style names, `--subject-field 1` (default)
for the prediction dataset naming convention.
