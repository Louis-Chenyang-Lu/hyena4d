# Hyena4D Model Description
## Architecture, Mathematics, and Intuition

---

## 0. Big Picture

Hyena4D is a **self-supervised spatiotemporal encoder** for 4D fMRI data.
The input is a 4D NIfTI volume of shape $[H, W, Z, T]$ (height × width ×
slices × timepoints).  The output is a fixed-length embedding vector per
subject that can be used for downstream tasks such as clustering, classification,
or inter-subject comparison.

The pipeline has two stages:

1. **Pre-training** (contrastive): The encoder is trained on a cohort so that
   subsequences from the *same* mouse produce similar embeddings and subsequences
   from *different* mice produce dissimilar ones.  No labels are needed — subject
   identity (parsed from the filename) is the only supervision.

2. **Inference** (subject embedding extraction): The trained encoder is run on
   any set of NIfTI files.  All subsequences belonging to one subject are
   mean-pooled into a single representative embedding vector.

The architecture is built from two modules:

- **PermInvariantPatchEmbedding4D (RPE)** — tokenises the volume into a grid of
  patch embeddings using a DeepSets MLP.
- **Hyena4DContrastiveEncoder** — processes the token grid with a stack of
  Hyena4DPixelBlocks (each containing a spatiotemporal filter + channel MLP),
  then projects to a unit-norm embedding.

---

## 1. Input Data

### 1.1 Volume format

Each scan is stored as a NIfTI file and loaded as a float32 array of shape
$[H, W, Z, T]$, where:

- $H, W$ are the in-plane spatial dimensions (e.g. $64 \times 64$ pixels),
- $Z$ is the number of axial slices (e.g. 20),
- $T$ is the number of timepoints in the full scan (e.g. 500).

### 1.2 Temporal subsequencing (training only)

Processing the full temporal dimension is memory-intensive because the Hyena4D
filter operates over a merged $T \times Z$ depth axis.  During training, the
scan is divided into non-overlapping windows of length $T_\text{sub}$:

$$\text{window}_k = \text{vol}[\ldots,\, k \cdot s \,:\, k \cdot s + T_\text{sub}]$$

where $s$ is the stride (default $s = T_\text{sub}$, giving non-overlapping
windows).  Each window has shape $[H, W, Z, T_\text{sub}]$ and is treated as an
independent training sample labelled with its subject ID.

**Lazy loading**: The NIfTI file is memory-mapped; only the requested temporal
slice $[\ldots, t_\text{start} : t_\text{start} + T_\text{sub}]$ is read from
disk per sample, keeping RAM usage bounded regardless of scan length.

### 1.3 Physical-unit scaling

fMRI voxels have physical dimensions (in mm) and timepoints are separated by the
repetition time TR (in seconds).  These are scanner-dependent and vary across
studies.  Two parameters capture this:

- **`voxel_size` = $(s_z, s_y, s_x)$ mm**: the side lengths of one voxel along
  each spatial axis.
- **`t_gap` = $\Delta T$ s**: the time between consecutive timepoints (full-brain TR).
- **`tr_slice` = $\Delta T_\text{slice}$ s**: the time between the acquisition of
  consecutive slices within a single TR.  For a sequential acquisition of $N_z$
  slices this equals $\Delta T / N_z$.  Default 0 disables the correction.

These are passed at model construction time and baked into the kernel's
positional features (see §4).  Using physical units makes the learned decay
rates scanner-independent.

---

## 2. PermInvariantPatchEmbedding4D (RPE)

### 2.1 Purpose and intuition

A raw fMRI volume has $H \times W \times Z \times T$ scalar values — tens of
millions of numbers.  Feeding these directly into a sequence model would be
prohibitively expensive.  The RPE compresses the volume into a much smaller
grid of **patch tokens** by treating each local patch as a *set* of voxels and
encoding it with a DeepSets network.

### 2.2 Patch partitioning

The volume $[H, W, Z, T]$ is divided into non-overlapping patches of size
$(p_H, p_W, p_Z, p_T)$.  This yields a regular grid of

$$H_p = H/p_H, \quad W_p = W/p_W, \quad Z_p = Z/p_Z, \quad T_p = T/p_T$$

patches (integer division required).  The patch size must evenly divide each
volume dimension.

### 2.3 DeepSets encoding

Each patch contains $N_p = p_H \times p_W \times p_Z \times p_T$ scalar voxel
values.  The DeepSets framework encodes a set $\{v_1, \ldots, v_{N_p}\}$ via:

$$\mathbf{e}_\text{patch} = \rho\!\left(\frac{1}{N_p} \sum_{i=1}^{N_p} \phi(v_i)\right)$$

where:

- $\phi : \mathbb{R} \to \mathbb{R}^{d_\text{hidden}}$ is a small MLP applied
  **independently** to each voxel value (the *inner* or *phi* network),
- the sum is a mean pool over all $N_p$ voxel embeddings in the patch,
- $\rho : \mathbb{R}^{d_\text{hidden}} \to \mathbb{R}^{d_\text{out}}$ is a
  linear layer (the *outer* or *rho* network) that maps the pooled embedding
  to the final patch token dimension.

**Why DeepSets?** The mean pool over $\phi$-embeddings is a universal
permutation-invariant function (Zaheer et al., 2017).  This means the patch
embedding is unchanged if the voxels within the patch are reordered — the
encoding depends only on the *multiset* of values, not their order.

### 2.4 Output layout

After encoding all $H_p \times W_p \times Z_p \times T_p$ patches, the tokens
are stacked into a tensor of shape

$$[B,\; d_\text{out},\; H_p,\; W_p,\; Z_p,\; T_p]$$

This is then permuted to the Hyena4D layout:

$$[B,\; d_\text{out},\; T_p,\; Z_p,\; H_p,\; W_p]$$

### 2.5 Chunked computation

Patch encoding is done in chunks of `patch_chunk` patches at a time to keep
peak RAM bounded.  For large volumes (e.g. $8 \times 8 \times 5 \times 1 = 320$
patches at $64^2 \times 20 \times 500$ voxels) this is essential.

### 2.6 Rotation-breaking behaviour

The RPE assigns each voxel to a patch based on its position on a fixed
axis-aligned grid.  After a spatial rotation, the same voxels land in
*different* grid cells, producing a different token grid.  The RPE therefore
**breaks rotation equivariance**, which is intentional: the encoder downstream
learns spatially structured representations that are tied to the coordinate frame
of the scanner.

---

## 3. Axis Permutation

After the RPE, the token grid is reshaped:

$$[B,\; C,\; H_p,\; W_p,\; Z_p,\; T_p] \;\longrightarrow\; [B,\; C,\; T_p,\; Z_p,\; H_p,\; W_p]$$

This reordering places the temporal dimension first inside the spatial block,
matching the `[B, C, T, Z, H, W]` convention expected by the Hyena4D encoder.
No data is copied — this is a `.permute().contiguous()` call.

---

## 4. Hyena4DFilter — Spatiotemporal Kernel

### 4.1 Design goal

The Hyena4D filter is a **depthwise, data-independent, learnable convolutional
kernel** whose weights are generated by a small MLP conditioned on
**rotation-invariant positional features**.  The MLP output + learnable decay
together define a kernel that:

- has the same value at all spatial offsets with the same 3D Euclidean distance
  from the origin (radial symmetry → rotation equivariance),
- decays toward zero as spatial or temporal distance increases (locality prior),
- is parameterised in **physical units** (mm, seconds) when `voxel_size`,
  `t_gap`, and `tr_slice` are provided.

### 4.2 Positional feature construction

For a kernel of size $(K_t, K_z, K_h, K_w)$, define integer offset grids
centred at zero:

$$\Delta t \in \left[-\tfrac{K_t-1}{2},\; \tfrac{K_t-1}{2}\right], \quad
  \Delta z \in \left[-\tfrac{K_z-1}{2},\; \tfrac{K_z-1}{2}\right], \quad
  \Delta y, \Delta x \; \text{analogously}$$

Scale to physical units, including **slice-timing correction**:

$$t_\text{phys} = |\Delta t \cdot \Delta T + \Delta z \cdot \Delta T_\text{slice}|, \qquad
  \mathbf{r} = (s_z \Delta z,\; s_y \Delta y,\; s_x \Delta x)$$

**Slice-timing correction** accounts for the fact that in a typical fMRI
acquisition, each axial slice is not acquired simultaneously: slice $z$ is
measured at physical time $t \cdot \Delta T + z \cdot \Delta T_\text{slice}$
within the scan.  The true physical temporal separation between two voxels at
kernel offsets $(\Delta t, \Delta z)$ is therefore
$\Delta t \cdot \Delta T + \Delta z \cdot \Delta T_\text{slice}$, not
just $|\Delta t| \cdot \Delta T$.  For a sequential acquisition of $N_z$ slices,
$\Delta T_\text{slice} = \Delta T / N_z$.  Setting $\Delta T_\text{slice} = 0$
(default) disables the correction and reduces to $t_\text{phys} = |\Delta t| \cdot \Delta T$.

where $\Delta T$ is `t_gap`, $\Delta T_\text{slice}$ is `tr_slice`, and
$(s_z, s_y, s_x)$ is `voxel_size`.  Compute the rotation-invariant scalars:

$$r_{3d} = \|\mathbf{r}\|_2 = \sqrt{(s_z\Delta z)^2 + (s_y\Delta y)^2 + (s_x\Delta x)^2}
\quad \text{[mm]}$$

$$r_{4d} = \sqrt{t_\text{phys}^2 + r_{3d}^2}
\quad \text{[mixed units]}$$

Six features are assembled per kernel position:

$$\mathbf{f} = \bigl[\;t_\text{phys},\; r_{3d},\; r_{4d},\;
\underbrace{\tfrac{1}{1+t_\text{phys}}}_{\text{decay}_t},\;
\underbrace{\tfrac{1}{1+r_{3d}}}_{\text{decay}_{3d}},\;
\underbrace{\tfrac{1}{1+r_{4d}}}_{\text{decay}_{4d}}\;\bigr]
\in \mathbb{R}^6$$

These features are computed **once at model construction** and cached as
non-persistent buffers.  They are fixed for the lifetime of the model.

**Why only these features?**  Raw signed coordinates $(\Delta t, \Delta z,
\Delta y, \Delta x)$ are NOT included because they encode direction, which
changes under rotation.  Functions of $r_{3d}$ and $t_\text{phys}$ are
invariant under any rotation of the $(z, y, x)$ axes, so the resulting kernel
is the same in all orientations.

### 4.3 Base kernel generation (MLP)

A two-layer MLP with GELU activation maps the 6-dimensional feature vector to a
kernel weight for each of the $C$ channels:

$$\mathbf{k}_\text{base}(\mathbf{f}) = \text{Linear}_{h \to C}\!\left(\text{GELU}\!\left(\text{Linear}_{6 \to h}(\mathbf{f})\right)\right)
\in \mathbb{R}^C$$

This produces a tensor of shape $[N_\text{pos}, C]$ where
$N_\text{pos} = K_t K_z K_h K_w$ is the total number of kernel positions.  The
MLP is shared across all kernel positions; each position is distinguished solely
by its feature vector $\mathbf{f}$.

### 4.4 Learnable anisotropic decay

The MLP alone cannot guarantee that the kernel decays to zero at large offsets.
A **learnable anisotropic decay mask** is multiplied elementwise:

$$d(\Delta t, \Delta z, \Delta y, \Delta x)
= \exp\!\bigl(-\alpha_s \cdot r_{3d} - \alpha_t \cdot t_\text{phys}\bigr)$$

The two scalar rates $\alpha_s, \alpha_t \geq 0$ are learned parameters stored
in log-space and mapped via softplus to ensure non-negativity:

$$\alpha_s = \text{softplus}(\log\alpha_s), \qquad \alpha_t = \text{softplus}(\log\alpha_t)$$

**Interpretation**: $\alpha_s$ controls how quickly the filter loses sensitivity
to voxels farther away in 3D space (in mm); $\alpha_t$ controls temporal
sensitivity decay (in seconds).  A large $\alpha_s$ means the filter is a
sharp local spatial operator; a small $\alpha_s$ allows long-range spatial
mixing.  These two rates are learned separately, reflecting the fact that BOLD
signal correlations decay differently in space and time.

The final kernel weight at position $(\Delta t, \Delta z, \Delta y, \Delta x)$
for channel $c$ is:

$$k_c(\Delta t, \Delta z, \Delta y, \Delta x)
= k_\text{base,c}(\mathbf{f}) \cdot \exp(-\alpha_s r_{3d} - \alpha_t t_\text{phys})$$

### 4.5 Depthwise Conv3d execution (T×Z merging)

A true 4D convolution is expensive.  The Hyena4D filter avoids it by **merging
the $T$ and $Z$ axes** into a single depth dimension:

$$D = T \times Z, \qquad K_D = K_t \times K_z$$

The input tensor is reshaped:

$$[B,\; C,\; T,\; Z,\; H,\; W] \;\longrightarrow\; [B,\; C,\; D,\; H,\; W]$$

The kernel is reshaped to $[C,\; 1,\; K_D,\; K_h,\; K_w]$ and applied as a
standard PyTorch `F.conv3d` with `groups=C` (depthwise: each channel has its
own kernel, no cross-channel mixing here).

**Why this is valid**: the spatial-radial symmetry of the kernel already treats
$(\Delta z)$ and $(\Delta y, \Delta x)$ symmetrically within the 3D spatial
part.  Merging $T$ and $Z$ is an approximation that treats temporal position and
slice position uniformly along the depth axis — the kernel still has separate
decay rates for $t_\text{phys}$ and $r_{3d}$, but within the Conv3d call they
share the same `D` axis.  The primary computational benefit is that PyTorch's
highly-optimised 3D conv handles this case efficiently.

After the convolution, the output is reshaped back:

$$[B,\; C,\; D,\; H,\; W] \;\longrightarrow\; [B,\; C,\; T,\; Z,\; H,\; W]$$

---

## 5. Hyena4DPixelBlock

### 5.1 Structure

Each block follows the **MetaFormer** pattern: two sequential sub-blocks with
pre-norm and residual connections.

```
input x  [B, C, T, Z, H, W]
  │
  ├─ LN(x) → Hyena4DFilter → + x   (token mixing sub-block)
  │
  └─ LN(x) → channel MLP   → + x   (channel mixing sub-block)
```

### 5.2 LayerNorm placement

Standard LayerNorm normalises over the *channel* dimension.  Here the input is
$[B, C, T, Z, H, W]$ so the spatial/temporal axes are permuted to the last
position before applying `nn.LayerNorm(C)`, then permuted back:

$$[B,\;C,\;T,\;Z,\;H,\;W]
\;\xrightarrow{\text{permute}}\;
[B,\;T,\;Z,\;H,\;W,\;C]
\;\xrightarrow{\text{LN}}\;
\;\xrightarrow{\text{permute}}\;
[B,\;C,\;T,\;Z,\;H,\;W]$$

This normalises each spatial/temporal position's channel vector independently.

### 5.3 Channel MLP (FFN)

A two-layer MLP with GELU and dropout, applied identically to every
spatiotemporal position:

$$\text{FFN}(\mathbf{h}) = W_2 \cdot \text{GELU}(W_1 \mathbf{h} + b_1) + b_2$$

Hidden dimension is $4C$ by default (`mlp_ratio=4`).  The MLP operates on the
$[B \cdot T \cdot Z \cdot H \cdot W,\; C]$ flattened view for efficiency, then
is reshaped back.

### 5.4 Role of residual connections

Residuals preserve gradient flow across many blocks and ensure the filter output
is added to rather than replacing the input representation.  This allows shallow
blocks to act as near-identity operations early in training and gradually
specialise as training proceeds.

---

## 6. Hyena4DBackbone — Full Encoder

### 6.1 Channel projection

The first operation after receiving the RPE output $[B, C_\text{in}, T_p, Z_p,
H_p, W_p]$ is a $1 \times 1 \times 1$ Conv3d (applied over the merged $D_p =
T_p \times Z_p$ axis) that projects $C_\text{in} \to d_\text{model}$
(= `embed_dim`).  This decouples the RPE output dimension from the backbone
width.

### 6.2 Stacked blocks

The projected tensor passes through `num_blocks` Hyena4DPixelBlocks in sequence.
Each block:
- mixes information across spatiotemporal positions via Hyena4DFilter,
- mixes information across channels via the FFN.

Typical configuration: `num_blocks=4`, `embed_dim=64`.

### 6.3 Global mean pooling

After the last block, the feature map $[B, C, T, Z, H, W]$ is reduced to a
fixed-size vector by averaging over all spatiotemporal positions:

$$\mathbf{h} = \frac{1}{T \cdot Z \cdot H \cdot W}
\sum_{t,z,h,w} \mathbf{f}_{t,z,h,w} \in \mathbb{R}^{d_\text{model}}$$

**Rotation invariance**: because the Hyena4DFilter is radially symmetric in the
$(z, y, x)$ axes (§4.2), applying it to a spatially rotated input produces a
rotated feature map.  The global mean pool collapses this rotation-equivariant
feature map to a single vector that is invariant to the rotation.
Formally, for any rotation $R \in SO(3)$:

$$\text{MeanPool}(f(R \cdot \mathbf{x})) = \text{MeanPool}(R \cdot f(\mathbf{x}))
= \text{MeanPool}(f(\mathbf{x}))$$

where the second equality holds because mean pooling commutes with permutations
(which rotations induce on a discrete grid).

---

## 7. Hyena4DContrastiveEncoder — Projection Head

### 7.1 Architecture

The backbone features $\mathbf{h} \in \mathbb{R}^{d_\text{model}}$ are mapped to
an embedding $\mathbf{z} \in \mathbb{R}^{d_\text{proj}}$ by:

$$\mathbf{z} = \text{Linear}_{d_\text{model} \to d_\text{proj}}\!\left(
  \text{GELU}\!\left(
    \text{Linear}_{d_\text{model} \to d_\text{model}}\!\left(
      \text{LayerNorm}(\mathbf{h})
    \right)
  \right)
\right)$$

### 7.2 L2 normalisation

The output is projected onto the unit hypersphere:

$$\hat{\mathbf{z}} = \frac{\mathbf{z}}{\|\mathbf{z}\|_2}$$

L2 normalisation ensures that cosine similarity between two embeddings equals
their dot product, which simplifies the contrastive loss and makes embedding
magnitudes uninformative (only direction carries information).

---

## 8. Contrastive Loss (InfoNCE, subject-level)

### 8.1 Motivation

The contrastive loss teaches the encoder to produce embeddings such that
subsequences from the *same* subject cluster together on the unit sphere and
those from *different* subjects are spread apart.  This is a form of
self-supervised learning: subject identity is automatically available from the
filename without any manual annotation.

### 8.2 Similarity matrix

For a batch of $B$ subsequence embeddings $\hat{\mathbf{z}}_1, \ldots,
\hat{\mathbf{z}}_B \in \mathbb{R}^{d_\text{proj}}$ (each L2-normalised), compute
the pairwise similarity matrix:

$$S_{ij} = \frac{\hat{\mathbf{z}}_i \cdot \hat{\mathbf{z}}_j}{\tau}$$

where $\tau$ is the temperature hyperparameter (default 0.1).  Because all
embeddings are unit-normalised, $\hat{\mathbf{z}}_i \cdot \hat{\mathbf{z}}_j$
is the cosine similarity.

### 8.3 Positive and negative masks

For each anchor $i$, define:
- **positive set** $\mathcal{P}_i$: all $j \neq i$ with the same subject ID,
- **negative set**: all $j \neq i$ with a different subject ID.

The diagonal ($j = i$, self-similarity) is excluded from both sets.

### 8.4 InfoNCE loss

For each anchor $i$ that has at least one positive:

$$\ell_i = -\frac{1}{|\mathcal{P}_i|} \sum_{j \in \mathcal{P}_i}
\left[S_{ij} - \log\sum_{k \neq i} e^{S_{ik}}\right]$$

The final loss averages over all valid anchors (those with at least one
positive in the batch):

$$\mathcal{L} = \frac{1}{|\{i : |\mathcal{P}_i| > 0\}|}
\sum_{i : |\mathcal{P}_i| > 0} \ell_i$$

**Numerical stability**: before computing $e^{S_{ij}}$, the row maximum is
subtracted from all similarities (log-sum-exp trick):

$$\tilde{S}_{ij} = S_{ij} - \max_k S_{ik}$$

### 8.5 Temperature interpretation

- **Low $\tau$** (e.g. 0.05): sharp peaks — the loss strongly penalises any
  mismatch; hard negatives dominate.
- **High $\tau$** (e.g. 0.5): soft peaks — more uniform gradients; easier to
  train early on.

A value of $\tau = 0.1$ is a standard default for contrastive pre-training.

---

## 9. Training Loop

### 9.1 Model structure during training

`Hyena4DFullPipeline` wraps both the RPE and the encoder into a single
`nn.Module`.  During the forward pass, the RPE is run **one sample at a time**
(a Python loop over the batch dimension) so that peak memory is bounded to one
raw fMRI volume at a time.  After all RPE calls, the resulting embeddings are
concatenated into a batch and fed to the encoder together.

```
[B, H, W, Z, T]
  for i in range(B):
      xi = [1, 1, H, W, Z, T]
      ei = RPE(xi)             → [1, proj, H_p, W_p, Z_p, T_p]
      ei = permute(ei)         → [1, proj, T_p, Z_p, H_p, W_p]
  cat → [B, proj, T_p, Z_p, H_p, W_p]
  encoder → [B, projection_dim]
```

### 9.2 Optimiser

Adam with a fixed learning rate (default $10^{-3}$).  All parameters of both
RPE and encoder are jointly optimised.

### 9.3 Checkpointing

A numbered checkpoint (`checkpoint_epoch{N:03d}.pt`) is saved every
`--save-every` epochs.  `best_model.pt` is always overwritten when a new
lowest-loss epoch is reached.  Checkpoints store:
- `model_state_dict` — full pipeline weights (RPE + encoder),
- `optimizer_state_dict` — Adam state for exact resumption,
- `epoch` and `loss`.

---

## 10. Subject-Level Embedding Extraction (Post-Training)

After training, the encoder is run on all subsequences of the training (or
target) dataset with `torch.inference_mode()`.  For each subject $s$:

$$\mathbf{e}_s = \frac{1}{N_s} \sum_{k=1}^{N_s} \hat{\mathbf{z}}_{s,k}$$

where $\hat{\mathbf{z}}_{s,k}$ is the embedding of the $k$-th subsequence of
subject $s$.

**Note**: the mean is computed on the **raw projection head outputs** (before
re-normalising the mean), so $\mathbf{e}_s$ is typically close to but not
exactly on the unit sphere.  For cosine-similarity comparisons downstream,
normalise $\mathbf{e}_s$ again.

The result is a matrix of shape $[N_\text{subjects},\; d_\text{proj}]$ saved
to `subject_embeddings.pt`.

---

## 11. Summary of Trainable Parameters

For the default configuration (`embed_dim=64`, `num_blocks=4`,
`projection_dim=128`, `kernel_size=(3,3,3,3)`, `rpe_hidden=64`,
`patch_size=(8,8,4,10)`):

| Module | Description | Parameters |
|---|---|---|
| RPE `φ` MLP | `Linear(1→64) + GELU + Linear(64→64)` | 4,224 |
| RPE `ρ` linear | `Linear(64→128)` | 8,320 |
| Channel projection | `Conv3d(128→64, 1×1×1)` | 8,256 |
| Each Hyena4DPixelBlock | Filter MLP + decay (2) + norm (2) + FFN | ~47,000 |
| 4 blocks total | | ~188,000 |
| Projection head | `LN + Linear(64→64) + GELU + Linear(64→128)` | 12,800 |
| **Total** | | **~222,000** |

The exact count depends on configuration.  Run
`sum(p.numel() for p in model.parameters() if p.requires_grad)` to confirm.

---

## 12. Design Trade-offs and Limitations

| Aspect | Current Choice | Trade-off |
|---|---|---|
| **RPE patch grid** | Fixed axis-aligned | Breaks rotation invariance; computationally cheap |
| **T×Z axis merging** | Conv3d over merged D=T×Z | Efficient; treats (Δt,Δz) depth uniformly |
| **Time ≠ 4th spatial dim** | Separate α_s, α_t decay rates | Biologically motivated; not purely symmetric |
| **Slice-timing correction** | `t_phys = \|Δt·T_gap + Δz·TR_slice\|` | Accounts for within-TR slice acquisition order; disabled by default (tr_slice=0) |
| **Contrastive positive definition** | Same subject = positive | No within-subject structure; assumes intra-subject similarity |
| **Global mean pool** | Averages over all positions | Throws away spatial layout; needed for rotation invariance |
| **No fine-tuning step** | Pre-train → direct embedding | Suitable for unsupervised downstream analysis |
