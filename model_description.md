# Hyena4D Model Description
## Architecture, Mathematics, and Notation

---

## 0. Notation and Conventions

Throughout this document the following notation is used.

- $B$ — batch size.
- $H, W, Z, T$ — spatial height, width, depth (slices), and temporal length of an fMRI
  volume in voxel units.
- $s_x, s_y, s_z$ — physical voxel side lengths in mm along the $x$, $y$, $z$ axes
  (e.g. $s_x = s_y = 0.5$ mm, $s_z = 0.75$ mm).  Together they form
  `voxel_size` $= (s_z, s_y, s_x)$.
- $\Delta T$ — the repetition time (TR) in seconds, i.e. the temporal gap between
  consecutive timepoints (`t_gap`).
- $p_H, p_W, p_Z, p_T$ — patch sizes along each axis; $N_p = p_H p_W p_Z p_T$.
- $H_p = H/p_H,\; W_p = W/p_W,\; Z_p = Z/p_Z,\; T_p = T/p_T$ — patch grid dimensions
  (integer division required).
- $C, d_\text{hidden}, d_\text{out}, d_\text{model}, d_\text{proj}$ — various channel /
  embedding widths.
- $K_t, K_z, K_h, K_w$ — kernel sizes; $N_\text{pos} = K_t K_z K_h K_w$.
- $\text{GELU}(u) = u\,\Phi(u)$ where $\Phi$ is the standard normal CDF.
- $\text{softplus}(u) = \ln(1 + e^u) > 0$.
- $\text{LN}$ — LayerNorm over the channel dimension.

---

## 1. Input Data

### 1.1 Volume format

Each scan is loaded from a NIfTI file as a float32 array of shape $[H, W, Z, T]$.
All slices within a volume are assumed to be acquired simultaneously (no slice-timing
correction).

### 1.2 Temporal subsequencing (training only)

During training, the full temporal dimension is split into non-overlapping windows.
Given stride $s$ (default $s = T_\text{sub}$):

$$\text{window}_k = \text{vol}\!\left[\ldots,\; k \cdot s \;:\; k \cdot s + T_\text{sub}\right],
\quad k = 0, 1, \ldots, \left\lfloor\frac{T - T_\text{sub}}{s}\right\rfloor$$

Each window has shape $[H, W, Z, T_\text{sub}]$ and is treated as an independent
training sample labelled with its subject ID.  The NIfTI file is memory-mapped, so
only the bytes covering $[\ldots, t_\text{start} : t_\text{start} + T_\text{sub}]$
are read per sample.

### 1.3 Physical unit scaling

Two scanner-dependent parameters are passed at model construction:

$$\text{voxel\_size} = (s_z, s_y, s_x)\ \text{mm}, \qquad \text{t\_gap} = \Delta T\ \text{s}$$

These convert voxel-index offsets to true physical distances and are baked into the
kernel's positional buffers (see §4.2).

---

## 2. PermInvariantPatchEmbedding4D (RPE)

### 2.1 Input and patch partitioning

Input tensor: $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W \times Z \times T}$
(here $C = 1$ for a single-channel fMRI volume).

Define patch-grid dimensions:
$$H_p = \lfloor H / p_H \rfloor, \quad
  W_p = \lfloor W / p_W \rfloor, \quad
  Z_p = \lfloor Z / p_Z \rfloor, \quad
  T_p = \lfloor T / p_T \rfloor$$

The volume is partitioned into $N_\text{patches} = B \cdot H_p W_p Z_p T_p$
non-overlapping patches.  Explicitly, using the index notation
$i_H \in \{0,\ldots,H_p{-}1\}$ and similarly for the other axes, patch
$(i_H, i_W, i_Z, i_T)$ of sample $b$ contains the voxel set:

$$\mathcal{V}_{b,i_H,i_W,i_Z,i_T} = \bigl\{
  X_{b,c,\, i_H p_H + \delta_h,\; i_W p_W + \delta_w,\; i_Z p_Z + \delta_z,\; i_T p_T + \delta_t}
  \;\Big|\;
  \delta_h \in [p_H],\; \delta_w \in [p_W],\; \delta_z \in [p_Z],\; \delta_t \in [p_T]
\bigr\}$$

where $[n] = \{0,\ldots,n{-}1\}$.  This is implemented as a single reshape-and-permute:

$$\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W \times Z \times T}
\;\xrightarrow{\text{view}}\;
\mathbb{R}^{B \times C \times H_p \times p_H \times W_p \times p_W \times Z_p \times p_Z \times T_p \times p_T}
\;\xrightarrow{\text{permute}}\;
\mathbb{R}^{B H_p W_p Z_p T_p \times N_p \times C}$$

giving a matrix of shape $[N_\text{patches},\; N_p,\; C]$ where
$N_p = p_H p_W p_Z p_T$ is the number of voxels per patch.

### 2.2 Per-voxel embedding: the phi MLP

Each voxel value $v \in \mathbb{R}^C$ is lifted to a hidden vector by:

$$\phi(v) = \text{GELU}\!\Bigl(W_2\,\text{GELU}(W_1 v + b_1) + b_2\Bigr)
\;\in\; \mathbb{R}^{d_\text{hidden}}$$

where $W_1 \in \mathbb{R}^{d_\text{hidden} \times C}$,
$b_1 \in \mathbb{R}^{d_\text{hidden}}$,
$W_2 \in \mathbb{R}^{d_\text{hidden} \times d_\text{hidden}}$,
$b_2 \in \mathbb{R}^{d_\text{hidden}}$ are shared across all voxels and patches.

This is computed efficiently as:

$$\Phi = \phi\!\left(\mathbf{X}_\text{patches}\right)
\in \mathbb{R}^{N_\text{patches} \times N_p \times d_\text{hidden}}$$

by reshaping the patch matrix to $[N_\text{patches} \cdot N_p,\; C]$, applying the
linear layers, then reshaping back.

### 2.3 Permutation-invariant aggregation (mean pooling)

The $N_p$ per-voxel embeddings within each patch are aggregated by mean pooling:

$$\bar{\phi}_p = \frac{1}{N_p} \sum_{i=1}^{N_p} \phi(v_i^{(p)})
\;\in\; \mathbb{R}^{d_\text{hidden}}, \qquad p = 1,\ldots,N_\text{patches}$$

This operation is a sufficient statistic for universal permutation-invariant
functions over the voxel multiset (Zaheer et al., 2017), and is implemented as
`.mean(dim=1)` over the $N_p$ axis, yielding shape
$[N_\text{patches},\; d_\text{hidden}]$.

### 2.4 Patch-level projection: the rho linear

The pooled embedding is projected to the final patch token dimension by:

$$\mathbf{e}_p = \rho(\bar{\phi}_p) = W_\rho\,\bar{\phi}_p + b_\rho
\;\in\; \mathbb{R}^{d_\text{out}}$$

where $W_\rho \in \mathbb{R}^{d_\text{out} \times d_\text{hidden}}$,
$b_\rho \in \mathbb{R}^{d_\text{out}}$.

The full DeepSets encoding is therefore:

$$\mathbf{e}_p = W_\rho \left(\frac{1}{N_p}\sum_{i=1}^{N_p}\phi(v_i^{(p)})\right) + b_\rho$$

### 2.5 Output layout

After encoding all patches, the embeddings are collected and reshaped:

$$[N_\text{patches},\; d_\text{out}]
= [B \cdot H_p W_p Z_p T_p,\; d_\text{out}]
\;\xrightarrow{\text{view}}\;
[B,\; H_p,\; W_p,\; Z_p,\; T_p,\; d_\text{out}]
\;\xrightarrow{\text{permute}}\;
[B,\; d_\text{out},\; H_p,\; W_p,\; Z_p,\; T_p]$$

### 2.6 Chunked computation

Encoding is done in blocks of `patch_chunk` patches at a time to bound peak RAM.
Within each block, all $N_p$ voxel embeddings are computed simultaneously unless
`voxel_chunk` is also set (for extremely large patches).

---

## 3. Axis Permutation

The RPE outputs $[B,\; d_\text{out},\; H_p,\; W_p,\; Z_p,\; T_p]$.
The Hyena4D encoder expects $[B,\; C,\; T,\; Z,\; H,\; W]$, so the axes are
permuted:

$$[B,\; d_\text{out},\; H_p,\; W_p,\; Z_p,\; T_p]
\;\xrightarrow{\text{permute}(0,1,5,4,2,3)}\;
[B,\; d_\text{out},\; T_p,\; Z_p,\; H_p,\; W_p]$$

No data is copied — this is a metadata-only `.permute().contiguous()` call.

---

## 4. Hyena4DFilter — Spatiotemporal Kernel

### 4.1 Design goal

The filter is a **depthwise, data-independent convolutional kernel** whose weights
are generated by a small MLP conditioned on rotation-invariant positional features.
The resulting kernel:

1. Has the same weight at all spatial positions equidistant from the origin (radially
   symmetric in $z, y, x$ physical space).
2. Is modulated by a learnable Gaussian decay ensuring locality.
3. Is parameterised in **physical units** (mm, seconds) so that learned decay rates
   generalise across scanners.

### 4.2 Positional feature construction

For a kernel of size $(K_t, K_z, K_h, K_w)$, define integer index offset grids
centred at zero:

$$\Delta t \in \left\{-\tfrac{K_t-1}{2},\; \ldots,\; \tfrac{K_t-1}{2}\right\}, \quad
  \Delta z \in \left\{-\tfrac{K_z-1}{2},\; \ldots,\; \tfrac{K_z-1}{2}\right\}$$

and analogously $\Delta y$, $\Delta x$ for $K_h$ and $K_w$.  These are half-integer
when the kernel dimension is even, ensuring a symmetric grid.

**True physical distances** (accounting for scanner resolution):

$$\delta X = \Delta x \cdot s_x \;[\text{mm}], \quad
  \delta Y = \Delta y \cdot s_y \;[\text{mm}], \quad
  \delta Z = \Delta z \cdot s_z \;[\text{mm}], \quad
  \delta T = |\Delta t| \cdot \Delta T \;[\text{s}]$$

where $s_x, s_y, s_z$ are voxel side lengths and $\Delta T$ is `t_gap`.  The
absolute value is used for $\delta T$ because the decay should be symmetric in time.

**Rotation-invariant derived scalars:**

$$r_{3d} = \sqrt{(\delta Z)^2 + (\delta Y)^2 + (\delta X)^2} \;[\text{mm}]$$

$$r_{4d} = \sqrt{(\delta T)^2 + (\delta Z)^2 + (\delta Y)^2 + (\delta X)^2}
\;[\text{mixed units}]$$

**Six positional features assembled per kernel position:**

$$\mathbf{f} =
\left[
  \delta T,\;
  r_{3d},\;
  r_{4d},\;
  \frac{1}{1+\delta T},\;
  \frac{1}{1+r_{3d}},\;
  \frac{1}{1+r_{4d}}
\right]
\in \mathbb{R}^6$$

The first three capture physical scale; the last three are monotone-decreasing
locality priors.  The full tensor has shape $[N_\text{pos},\; 6]$ where
$N_\text{pos} = K_t K_z K_h K_w$.

These features are computed **once at model construction** and registered as a
non-persistent buffer, so they are fixed throughout training.

Raw signed offsets $(\Delta t, \Delta z, \Delta y, \Delta x)$ are deliberately
excluded: they encode direction and change under spatial rotations, breaking the
radial-symmetry property.

**Additional decay buffers** (also computed at construction and cached):

$$r^2_{3d} = (\delta Z)^2 + (\delta Y)^2 + (\delta X)^2 \;[\text{mm}^2]
\quad \longrightarrow \text{buffer } \mathtt{r3d\_sq\_flat} \in \mathbb{R}^{N_\text{pos}}$$

$$(\delta T)^2 = (|\Delta t| \cdot \Delta T)^2 \;[\text{s}^2]
\quad \longrightarrow \text{buffer } \mathtt{dt\_sq\_flat} \in \mathbb{R}^{N_\text{pos}}$$

### 4.3 Base kernel generation (MLP)

A two-layer MLP with GELU activation maps each 6-dimensional feature vector to a
weight for each of the $C$ channels:

$$\mathbf{k}_\text{base}(\mathbf{f}) =
W_2^{(k)}\,\text{GELU}\!\bigl(W_1^{(k)}\,\mathbf{f} + b_1^{(k)}\bigr) + b_2^{(k)}
\;\in\; \mathbb{R}^C$$

where $W_1^{(k)} \in \mathbb{R}^{h \times 6}$, $W_2^{(k)} \in \mathbb{R}^{C \times h}$,
and $h$ is `hidden_dim`.

Applied to all $N_\text{pos}$ positions simultaneously:

$$K_\text{base} = \text{MLP}(\mathbf{F}) \in \mathbb{R}^{N_\text{pos} \times C}$$

where $\mathbf{F} \in \mathbb{R}^{N_\text{pos} \times 6}$ is the stacked feature matrix.

### 4.4 Learnable anisotropic Gaussian decay

A **learnable Gaussian decay mask** is multiplied element-wise into the base kernel
to enforce locality.  The formula, in terms of true physical distances, is:

$$d(\Delta t, \Delta z, \Delta y, \Delta x) =
\exp\!\Bigl(-\alpha\,\bigl[(\delta X)^2 + (\delta Y)^2 + (\delta Z)^2\bigr]
            -\beta\,(\delta T)^2\Bigr)$$

$$= \exp\!\bigl(-\alpha\; r^2_{3d} - \beta\; (\delta T)^2\bigr)$$

where:
- $r^2_{3d} = (\delta X)^2 + (\delta Y)^2 + (\delta Z)^2$ with
  $\delta X = \Delta x \cdot s_x$, etc. — **true spatial squared distance in mm$^2$**,
- $(\delta T)^2 = (|\Delta t| \cdot \Delta T)^2$ — **true temporal squared distance in
  s$^2$**,
- $\alpha \geq 0$ — spatial Gaussian decay rate $[\text{mm}^{-2}]$,
- $\beta \geq 0$ — temporal Gaussian decay rate $[\text{s}^{-2}]$.

The scalar rates are learned parameters stored in log-space and constrained to be
non-negative via softplus:

$$\alpha = \text{softplus}(\log\tilde{\alpha}), \qquad
  \beta  = \text{softplus}(\log\tilde{\beta})$$

where $\log\tilde{\alpha}$ and $\log\tilde{\beta}$ are the actual `nn.Parameter`
objects (initialised to 0, so $\alpha = \beta = \ln 2$ at init).

**Physical interpretation:**
- A Gaussian decay with rate $\alpha$ is equivalent to a Gaussian kernel with
  standard deviation $\sigma_s = 1/\sqrt{2\alpha}$ mm.  Large $\alpha$ means the
  filter is sharply local in space.
- Similarly, $\sigma_t = 1/\sqrt{2\beta}$ seconds sets the temporal locality scale.
- Unlike the previous linear-decay formula $e^{-\alpha_s r_{3d} - \alpha_t |\delta T|}$
  (Laplacian), the Gaussian decay $e^{-\alpha r^2_{3d} - \beta (\delta T)^2}$ is
  differentiable at the origin and has faster tail decay, providing a stronger
  locality prior.

The decay mask has shape $[N_\text{pos}]$ and is computed in `forward()` at each
call so that gradient flow through $\alpha, \beta$ is preserved.

The final kernel weight for channel $c$ at position
$(\Delta t, \Delta z, \Delta y, \Delta x)$ is:

$$k_c(\Delta t, \Delta z, \Delta y, \Delta x) =
k_{\text{base},c}(\mathbf{f}) \cdot \exp\!\bigl(-\alpha\; r^2_{3d} - \beta\; (\delta T)^2\bigr)$$

In matrix form:

$$K = K_\text{base} \odot d\,\mathbf{1}^\top \in \mathbb{R}^{N_\text{pos} \times C}$$

where $d \in \mathbb{R}^{N_\text{pos}}$ is the decay vector and $\odot$ denotes
broadcasting.

### 4.5 Depthwise Conv3d execution (T×Z merging)

A true 6D convolution over $[B, C, T, Z, H, W]$ is not natively available in PyTorch.
Instead, the $T$ and $Z$ axes are merged into a single depth axis:

$$D = T \cdot Z, \qquad K_D = K_t \cdot K_z$$

**Input reshape:**
$$[B,\; C,\; T,\; Z,\; H,\; W] \;\xrightarrow{\text{view}}\; [B,\; C,\; D,\; H,\; W]$$

**Kernel reshape:** the kernel matrix $K \in \mathbb{R}^{N_\text{pos} \times C}$ is
transposed and reshaped:

$$K^\top \in \mathbb{R}^{C \times N_\text{pos}}
\;\xrightarrow{\text{view}}\;
\mathbb{R}^{C \times 1 \times K_D \times K_h \times K_w}$$

**Depthwise convolution** with `groups=C`:

$$\mathbf{Y}_{3d} = \text{Conv3d}\!\left(\mathbf{X}_{3d},\; K_\text{dw},\;
  \text{padding}=\left(\tfrac{K_D}{2},\, \tfrac{K_h}{2},\, \tfrac{K_w}{2}\right),\;
  \text{groups}=C\right)
\;\in\; \mathbb{R}^{B \times C \times D \times H \times W}$$

Each channel $c$ is convolved with its own kernel slice $K_\text{dw}[c, 0, :, :, :]$,
with no cross-channel mixing (that role belongs to the FFN in the PixelBlock).

**Output reshape:**
$$[B,\; C,\; D,\; H,\; W] \;\xrightarrow{\text{view}}\; [B,\; C,\; T,\; Z,\; H,\; W]$$

---

## 5. Hyena4DPixelBlock

### 5.1 Block structure

Each block follows the MetaFormer pattern with two sub-blocks, each using pre-norm
and a residual connection.  Let $\mathbf{x} \in \mathbb{R}^{B \times C \times T \times Z \times H \times W}$:

$$\mathbf{x} \leftarrow \mathbf{x} + \text{Hyena4DFilter}\!\bigl(\text{LN}(\mathbf{x})\bigr)$$

$$\mathbf{x} \leftarrow \mathbf{x} + \text{FFN}\!\bigl(\text{LN}(\mathbf{x})\bigr)$$

### 5.2 LayerNorm over the channel dimension

`nn.LayerNorm(C)` normalises a vector of length $C$.  Because the input layout is
$[B, C, T, Z, H, W]$ (channel-first), the axes must be permuted before applying LN:

$$[B,\;C,\;T,\;Z,\;H,\;W]
\;\xrightarrow{\text{permute}(0,2,3,4,5,1)}\;
[B,\;T,\;Z,\;H,\;W,\;C]$$

Apply $\text{LN}$ along the last axis $C$:

$$\text{LN}(\mathbf{h}) = \frac{\mathbf{h} - \mu_\mathbf{h}}{\sigma_\mathbf{h} + \epsilon} \odot \gamma + \beta_\text{LN}$$

where $\mu_\mathbf{h}, \sigma_\mathbf{h}$ are the mean and standard deviation over
the $C$ elements of position $\mathbf{h}$, and $\gamma, \beta_\text{LN} \in \mathbb{R}^C$
are learned affine parameters.  Then permute back:

$$[B,\;T,\;Z,\;H,\;W,\;C]
\;\xrightarrow{\text{permute}(0,5,1,2,3,4)}\;
[B,\;C,\;T,\;Z,\;H,\;W]$$

This normalises each spatial/temporal position's channel vector independently.

### 5.3 Channel FFN (Feed-Forward Network)

A two-layer MLP with GELU activation and optional dropout is applied identically
to every spatiotemporal position.  Let $\mathbf{h} \in \mathbb{R}^C$ be one position:

$$\text{FFN}(\mathbf{h}) =
  W_4\,\text{Drop}\!\left(\text{GELU}\!\left(W_3\,\mathbf{h} + b_3\right)\right) + b_4$$

where $W_3 \in \mathbb{R}^{d_\text{ff} \times C}$, $W_4 \in \mathbb{R}^{C \times d_\text{ff}}$,
$d_\text{ff} = \lfloor C \cdot r \rfloor$ with expansion ratio $r$ (`mlp_ratio`, default 4).

For efficiency, the positions are flattened before applying the linear layers:

$$[B,\;C,\;T,\;Z,\;H,\;W]
\;\xrightarrow{\text{permute},\,\text{reshape}}\;
[B T Z H W,\; C]
\;\xrightarrow{\text{FFN}}\;
[B T Z H W,\; C]
\;\xrightarrow{\text{view},\,\text{permute}}\;
[B,\;C,\;T,\;Z,\;H,\;W]$$

### 5.4 Residual connections

Both sub-blocks use pre-norm residuals:

$$\mathbf{x}^{(l+1/2)} = \mathbf{x}^{(l)} + \text{Mixer}\!\bigl(\text{LN}(\mathbf{x}^{(l)})\bigr)$$

$$\mathbf{x}^{(l+1)} = \mathbf{x}^{(l+1/2)} + \text{FFN}\!\bigl(\text{LN}(\mathbf{x}^{(l+1/2)})\bigr)$$

---

## 6. Hyena4DBackbone — Full Encoder

### 6.1 Channel projection ($1\times 1 \times 1$ Conv3d)

The RPE output $[B,\; d_\text{out},\; T_p,\; Z_p,\; H_p,\; W_p]$ is first reshaped
to merge $T_p$ and $Z_p$:

$$[B,\; d_\text{out},\; T_p,\; Z_p,\; H_p,\; W_p]
\;\xrightarrow{\text{view}}\;
[B,\; d_\text{out},\; D_p,\; H_p,\; W_p], \quad D_p = T_p \cdot Z_p$$

A $1\times1\times1$ convolution projects $d_\text{out} \to d_\text{model}$:

$$\mathbf{X}_\text{proj} = W_\text{proj} * \mathbf{X}_{3d}
\;\in\; \mathbb{R}^{B \times d_\text{model} \times D_p \times H_p \times W_p}$$

where $W_\text{proj} \in \mathbb{R}^{d_\text{model} \times d_\text{out} \times 1 \times 1 \times 1}$.
This is reshaped back to $[B,\; d_\text{model},\; T_p,\; Z_p,\; H_p,\; W_p]$.

### 6.2 Stacked Hyena4DPixelBlocks

The tensor passes through $L$ blocks in sequence ($L =$ `num_blocks`):

$$\mathbf{x}^{(0)} = \mathbf{X}_\text{proj}$$

$$\mathbf{x}^{(l+1)} = \text{Hyena4DPixelBlock}_l\!\left(\mathbf{x}^{(l)}\right),
\quad l = 0,\ldots,L-1$$

Each block maintains the shape $[B,\; d_\text{model},\; T_p,\; Z_p,\; H_p,\; W_p]$.

### 6.3 Global mean pooling

After the last block, the feature map is reduced to a fixed-size vector by averaging
over all spatiotemporal positions:

$$\mathbf{h} = \frac{1}{T_p Z_p H_p W_p}
\sum_{t=0}^{T_p-1}\sum_{z=0}^{Z_p-1}\sum_{h=0}^{H_p-1}\sum_{w=0}^{W_p-1}
\mathbf{x}^{(L)}_{:,\,t,z,h,w}
\;\in\; \mathbb{R}^{B \times d_\text{model}}$$

Implemented as `.mean(dim=[2,3,4,5])` over the last four axes.

---

## 7. Hyena4DContrastiveEncoder — Projection Head

### 7.1 Architecture

The backbone features $\mathbf{h} \in \mathbb{R}^{B \times d_\text{model}}$ are
mapped to embeddings $\mathbf{z} \in \mathbb{R}^{B \times d_\text{proj}}$ by:

$$\mathbf{h}' = \text{LN}(\mathbf{h}), \quad \mathbf{h}' \in \mathbb{R}^{d_\text{model}}$$

$$\mathbf{u} = W_b\,\mathbf{h}' + b_b, \quad W_b \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$$

$$\mathbf{z} = W_c\,\text{GELU}(\mathbf{u}) + b_c, \quad W_c \in \mathbb{R}^{d_\text{proj} \times d_\text{model}}$$

Tensor shape at each step: $[B, d_\text{model}] \to [B, d_\text{model}] \to [B, d_\text{proj}]$.

### 7.2 L2 normalisation

The projection is normalised onto the unit hypersphere:

$$\hat{\mathbf{z}}_i = \frac{\mathbf{z}_i}{\|\mathbf{z}_i\|_2}, \quad
\hat{\mathbf{z}}_i \in \mathbb{R}^{d_\text{proj}}, \quad
\|\hat{\mathbf{z}}_i\|_2 = 1$$

After normalisation, the cosine similarity between any two embeddings equals their
dot product: $\cos(\hat{\mathbf{z}}_i, \hat{\mathbf{z}}_j) = \hat{\mathbf{z}}_i \cdot \hat{\mathbf{z}}_j$.

---

## 8. Contrastive Loss (InfoNCE, subject-level)

### 8.1 Similarity matrix

For a batch of $B$ L2-normalised embeddings
$\hat{\mathbf{z}}_1,\ldots,\hat{\mathbf{z}}_B \in \mathbb{R}^{d_\text{proj}}$,
form the scaled cosine-similarity matrix:

$$S_{ij} = \frac{\hat{\mathbf{z}}_i \cdot \hat{\mathbf{z}}_j}{\tau},
\quad S \in \mathbb{R}^{B \times B}$$

where $\tau$ is the temperature (default $\tau = 0.1$).

### 8.2 Positive and negative masks

Let $\text{id}_i$ denote the subject ID of sample $i$.  Define:

$$\text{pos\_mask}_{ij} =
\begin{cases}
1 & \text{if } \text{id}_i = \text{id}_j \text{ and } i \neq j \\
0 & \text{otherwise}
\end{cases}$$

$$\text{logits\_mask}_{ij} = \mathbf{1}[i \neq j]$$

The diagonal ($i = j$, self-similarity) is excluded from all computations.

### 8.3 InfoNCE loss with numerical stabilisation

For each sample $i$, subtract the row maximum for numerical stability
(log-sum-exp trick):

$$\tilde{S}_{ij} = S_{ij} - \max_{k \neq i} S_{ik}$$

Compute log-probabilities over all non-self entries:

$$\log p_{ij} = \tilde{S}_{ij} - \log\!\left(\sum_{k \neq i} e^{\tilde{S}_{ik}} + \epsilon\right)$$

where $\epsilon = 10^{-12}$ prevents $\log 0$.  For each anchor $i$ with at least one
positive ($|\mathcal{P}_i| > 0$), the per-anchor loss is the average negative
log-probability over positives:

$$\ell_i = -\frac{1}{|\mathcal{P}_i|}
\sum_{j \in \mathcal{P}_i} \log p_{ij}$$

The batch loss averages over valid anchors:

$$\mathcal{L} = \frac{1}{\sum_i \mathbf{1}[|\mathcal{P}_i| > 0]}
\sum_{i:\,|\mathcal{P}_i|>0} \ell_i$$

If no anchor has a positive (all $B$ samples are from different subjects),
the loss is defined as $0$.

### 8.4 Temperature interpretation

$$\tau \to 0: \text{ sharper peaks, harder negatives, larger gradients}$$
$$\tau \to \infty: \text{ uniform distribution, vanishing gradients}$$

A value of $\tau = 0.1$ is a standard default for self-supervised contrastive
pre-training.

---

## 9. Training Loop

### 9.1 Full-pipeline forward pass

`Hyena4DFullPipeline` wraps the RPE and the encoder.  Because a raw fMRI volume is
large, the RPE is run one sample at a time:

$$\text{for } b = 0, \ldots, B-1:$$
$$\quad \mathbf{x}_b \in \mathbb{R}^{1 \times C \times H \times W \times Z \times T_\text{sub}}$$
$$\quad \mathbf{e}_b = \text{RPE}(\mathbf{x}_b) \in \mathbb{R}^{1 \times d_\text{out} \times H_p \times W_p \times Z_p \times T_p}$$
$$\quad \mathbf{e}_b = \text{permute}(\mathbf{e}_b) \in \mathbb{R}^{1 \times d_\text{out} \times T_p \times Z_p \times H_p \times W_p}$$

$$\mathbf{E} = \text{cat}(\mathbf{e}_0,\ldots,\mathbf{e}_{B-1},\;\dim=0)
\in \mathbb{R}^{B \times d_\text{out} \times T_p \times Z_p \times H_p \times W_p}$$

$$\hat{\mathbf{Z}} = \text{Encoder}(\mathbf{E})
\in \mathbb{R}^{B \times d_\text{proj}}$$

$$\mathcal{L} = \text{InfoNCE}(\hat{\mathbf{Z}},\; \text{subject\_ids})$$

### 9.2 Optimiser

Adam with fixed learning rate $\eta = 10^{-3}$.  All parameters of RPE and encoder
are jointly optimised.

### 9.3 Checkpointing

A numbered checkpoint is saved every `--save-every` epochs.  `best_model.pt` is
overwritten whenever the epoch loss is the lowest seen so far.  Each checkpoint
contains `model_state_dict`, `optimizer_state_dict`, `epoch`, and `loss`.

---

## 10. Subject-Level Embedding Extraction

After training, run inference with `torch.inference_mode()`.  For each subject $s$
with $N_s$ subsequences, the subject embedding is the mean over all subsequence
embeddings:

$$\mathbf{e}_s = \frac{1}{N_s} \sum_{k=1}^{N_s} \hat{\mathbf{z}}_{s,k}
\;\in\; \mathbb{R}^{d_\text{proj}}$$

This mean is taken over **L2-normalised** embeddings, so $\mathbf{e}_s$ is not
necessarily unit-norm.  For downstream cosine-similarity comparisons, normalise:

$$\hat{\mathbf{e}}_s = \frac{\mathbf{e}_s}{\|\mathbf{e}_s\|_2}$$

The result matrix has shape $[N_\text{subjects},\; d_\text{proj}]$ and is saved to
`subject_embeddings.pt`.

---

## 11. Summary of Trainable Parameters

For the default configuration
(`embed_dim=64`, `num_blocks=4`, `projection_dim=128`,
`kernel_size=(3,3,3,3)`, `rpe_hidden=64`, `patch_size=(8,8,4,10)`):

| Module | Formula | Parameters |
|---|---|---|
| RPE $\phi$: Linear(1→64) | $64 \cdot 1 + 64$ | 128 |
| RPE $\phi$: Linear(64→64) | $64 \cdot 64 + 64$ | 4,160 |
| RPE $\rho$: Linear(64→128) | $128 \cdot 64 + 128$ | 8,320 |
| Channel projection Conv3d(128→64, 1×1×1) | $64 \cdot 128 + 64$ | 8,256 |
| Each block: Filter MLP Linear(6→64) | $64 \cdot 6 + 64$ | 448 |
| Each block: Filter MLP Linear(64→64) | $64 \cdot 64 + 64$ | 4,160 |
| Each block: $\log\tilde{\alpha}, \log\tilde{\beta}$ | 2 scalars | 2 |
| Each block: LN1, LN2 ($\gamma, \beta$, dim=64) | $2 \times 2 \times 64$ | 256 |
| Each block: FFN Linear(64→256) | $256 \cdot 64 + 256$ | 16,640 |
| Each block: FFN Linear(256→64) | $64 \cdot 256 + 64$ | 16,448 |
| Each block total | | ~37,954 |
| 4 blocks | | ~151,816 |
| Projection head LN | $2 \times 64$ | 128 |
| Projection head Linear(64→64) | $64 \cdot 64 + 64$ | 4,160 |
| Projection head Linear(64→128) | $128 \cdot 64 + 128$ | 8,320 |
| **Total** | | **~185,288** |

Run `sum(p.numel() for p in model.parameters() if p.requires_grad)` to get the
exact count for your configuration.

---

## 12. Design Trade-offs and Limitations

| Aspect | Current choice | Trade-off |
|---|---|---|
| **Gaussian vs Laplacian decay** | $\exp(-\alpha r^2_{3d} - \beta (\delta T)^2)$ | Differentiable at origin, faster tail decay, physically motivated (RBF); Laplacian has heavier tails |
| **RPE patch grid** | Fixed axis-aligned grid | Breaks rotation equivariance; computationally cheap and standard for fMRI |
| **T×Z axis merging** | Conv3d over $D = T \cdot Z$ | Efficient; treats (Δt,Δz) along the same depth axis — kernel decay rates still separate $r^2_{3d}$ and $(\delta T)^2$ |
| **Simultaneous slice assumption** | $\delta T = \|\Delta t\| \cdot \Delta T$ | All slices treated as acquired within the same TR; no slice-timing correction |
| **Rotation invariance** | Radially symmetric kernel + global mean pool | Spatial layout discarded; invariance achieved at the cost of spatial structure in the embedding |
| **Contrastive positive definition** | Same subject ID = positive | No within-subject temporal structure exploited; assumes intra-subject similarity |
| **Global mean pool** | Averages over all $T_p Z_p H_p W_p$ positions | Throws away spatial layout; enables permutation/rotation invariance at the embedding level |
| **No fine-tuning** | Pre-train → direct embedding | Suitable for unsupervised downstream analysis; may be suboptimal for supervised tasks |
