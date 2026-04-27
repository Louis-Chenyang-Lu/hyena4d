"""
train_hyena.py
--------------
Contrastive training for the Hyena4D fMRI embedding pipeline.

Pipeline:
    NIfTI files  →  non-overlapping temporal subsequences
               →  RPE (PermInvariantPatchEmbedding4D)
               →  Hyena4DContrastiveEncoder
               →  subject_contrastive_loss (same mouse = positive pair)

After training, subject-level embeddings are produced by mean-pooling all
subsequence embeddings that belong to the same mouse.

Memory design:
    - NIfTI files are memory-mapped (lazy dataobj access) — only the
      requested temporal slice is loaded per sample.
    - RPE is run one sample at a time inside the model forward pass,
      keeping peak GPU/CPU usage bounded to a single raw volume at a time.
    - Subject embeddings are accumulated on CPU in a running sum, so the
      GPU only ever holds one batch.

Checkpointing:
    - A checkpoint is saved every --save-every epochs.
    - The best checkpoint (lowest loss) is always kept as best_model.pt.
    - Use --resume PATH to continue from any saved checkpoint; the
      optimizer state and epoch counter are restored exactly.

Subject ID parsing:
    Subject ID is extracted from the filename by splitting on "_" and
    taking the token at --subject-field (default 1).
    e.g. "driftRespCardlessBOLD_B16_Final_scan5.nii"  →  "B16"
    Note: run_hyena.py uses field 0 (experiment prefix). Adjust
    --subject-field to match your naming convention.

Usage:
    python train_hyena.py [options]

    --input-dir     DIR   Directory with .nii files        (default: input)
    --output-dir    DIR   Directory for checkpoints        (default: output)
    --t-sub         N     Subsequence length in timepoints (default: 50)
    --stride        N     Window stride (default: t_sub, non-overlapping)
    --epochs        N     Training epochs                  (default: 50)
    --batch-size    N     Subsequences per batch           (default: 8)
    --lr            F     Learning rate                    (default: 1e-3)
    --temperature   F     Contrastive loss temperature     (default: 0.1)
    --resume        PATH  Checkpoint to resume from
    --save-every    N     Save checkpoint every N epochs   (default: 1)
"""

import argparse
import os
import sys
from collections import defaultdict

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from hyenapixel.hyena4d_model import Hyena4DContrastiveEncoder, subject_contrastive_loss
from invariant_patching_4d import PermInvariantPatchEmbedding4D


# ---------------------------------------------------------------------------
# NIfTI utilities
# ---------------------------------------------------------------------------

def _is_nifti(fname: str) -> bool:
    return fname.endswith(".nii") or fname.endswith(".nii.gz")


def nifti_basename(fname: str) -> str:
    """Strip .nii or .nii.gz extension to get the bare filename stem."""
    if fname.endswith(".nii.gz"):
        return fname[:-7]
    if fname.endswith(".nii"):
        return fname[:-4]
    return fname


def read_nifti_params(fpath: str) -> tuple[tuple[float, float, float], float]:
    """
    Read voxel size (mm) and TR (seconds) from a NIfTI header.

    Returns:
        voxel_size : (sz, sy, sx) — size in mm along (Z, H, W) axes,
                     matching our [H, W, Z, T] volume convention where
                     NIfTI axes 0, 1, 2 correspond to H, W, Z respectively.
        t_gap      : TR in seconds.  Applies a heuristic to correct headers
                     that store TR in ms but label the unit as 'sec'
                     (TR > 100 with 'sec' label is treated as ms).
    """
    hdr = nib.load(fpath).header
    zooms = hdr.get_zooms()

    # NIfTI axis 0 = H, 1 = W, 2 = Z in our convention
    sz = float(zooms[2]) if len(zooms) > 2 else 1.0   # Z voxel size
    sy = float(zooms[0]) if len(zooms) > 0 else 1.0   # H voxel size
    sx = float(zooms[1]) if len(zooms) > 1 else 1.0   # W voxel size

    t_gap = 1.0
    if len(zooms) >= 4 and float(zooms[3]) > 0:
        tr = float(zooms[3])
        time_unit = hdr.get_xyzt_units()[1]
        # Some headers store TR in ms but label unit as 'sec'.
        # TR > 100 in 'sec' units is physically unreasonable → treat as ms.
        if time_unit == "msec" or (time_unit == "sec" and tr > 100):
            tr /= 1000.0
        t_gap = tr

    return (sz, sy, sx), t_gap


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FMRISubsequenceDataset(Dataset):
    """
    Generates temporal subsequences from all .nii files in a directory.

    Each sample is (vol [H, W, Z, T_sub], subject_id_int).

    By default stride = t_sub, giving non-overlapping windows. Set
    stride < t_sub for overlapping windows (more training data, but
    correlated samples).

    Files where T < t_sub are silently skipped.
    """

    def __init__(
        self,
        input_dir: str,
        t_sub: int = 50,
        stride: int | None = None,
        subject_field: int = 1,
    ):
        self.t_sub = t_sub
        stride = stride if stride is not None else t_sub   # non-overlapping default

        self.samples: list[tuple[str, int, int]] = []      # (filepath, t_start, sid_int)
        self.subject_to_int: dict[str, int] = {}
        self.int_to_subject: dict[int, str] = {}

        skipped = 0
        for fname in sorted(f for f in os.listdir(input_dir) if _is_nifti(f)):
            fpath = os.path.join(input_dir, fname)
            parts = nifti_basename(fname).split("_")
            subject = parts[subject_field] if len(parts) > subject_field else parts[0]

            if subject not in self.subject_to_int:
                idx = len(self.subject_to_int)
                self.subject_to_int[subject] = idx
                self.int_to_subject[idx] = subject

            sid = self.subject_to_int[subject]
            T = nib.load(fpath).shape[3]

            if T < t_sub:
                skipped += 1
                continue

            for t_start in range(0, T - t_sub + 1, stride):
                self.samples.append((fpath, t_start, sid))

        if skipped:
            print(f"  [Dataset] Skipped {skipped} file(s) shorter than t_sub={t_sub}")
        print(
            f"  [Dataset] {len(self.samples)} subsequences | "
            f"{len(self.subject_to_int)} subjects | "
            f"t_sub={t_sub} | stride={stride}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        fpath, t_start, sid = self.samples[idx]
        # dataobj gives memory-mapped access — only loads the requested slice
        arr = nib.load(fpath).dataobj[..., t_start : t_start + self.t_sub]
        vol = np.array(arr, dtype=np.float32)             # [H, W, Z, T_sub] — copy makes it writable
        return torch.from_numpy(vol), sid


# ---------------------------------------------------------------------------
# Full pipeline model (RPE + Encoder)
# ---------------------------------------------------------------------------

class Hyena4DFullPipeline(nn.Module):
    """
    End-to-end trainable module: RPE → Hyena4DContrastiveEncoder.

    RPE is run one sample at a time to keep peak memory bounded to a
    single raw fMRI volume. The encoder then processes the full batch of
    (much smaller) patch embeddings together.

    Input  : [B, H, W, Z, T]
    Output : [B, projection_dim]  (L2-normalised)
    """

    def __init__(
        self,
        patch_size: tuple,
        rpe_hidden_dim: int,
        projection_dim: int,
        embed_dim: int,
        num_blocks: int,
        kernel_size: tuple,
        patch_chunk: int,
        voxel_chunk: int | None,
        voxel_size: tuple = (1.0, 1.0, 1.0),
        t_gap: float = 1.0,
    ):
        super().__init__()
        self.rpe = PermInvariantPatchEmbedding4D(
            in_channels=1,
            hidden_dim=rpe_hidden_dim,
            out_dim=projection_dim,
            patch_size=patch_size,
            patch_chunk=patch_chunk,
            voxel_chunk=voxel_chunk,
        )
        self.encoder = Hyena4DContrastiveEncoder(
            in_channels=projection_dim,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            projection_dim=projection_dim,
            kernel_size=kernel_size,
            voxel_size=voxel_size,
            t_gap=t_gap,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, Z, T]
        Returns:
            z: [B, projection_dim]
        """
        # RPE: loop over samples to bound memory to one volume at a time.
        # Peak memory = one raw volume + B × (small RPE output).
        rpe_outs = []
        for i in range(x.shape[0]):
            xi = x[i : i + 1].unsqueeze(1)                 # [1, 1, H, W, Z, T]
            emb = self.rpe(xi)                              # [1, proj, H_p, W_p, Z_p, T_p]
            emb = emb.permute(0, 1, 5, 4, 2, 3).contiguous()  # [1, proj, T_p, Z_p, H_p, W_p]
            rpe_outs.append(emb)

        rpe_batch = torch.cat(rpe_outs, dim=0)              # [B, proj, T_p, Z_p, H_p, W_p]
        return self.encoder(rpe_batch)                      # [B, projection_dim]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: Hyena4DFullPipeline,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    temperature: float,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch:03d}/{num_epochs}",
        leave=False,
        unit="batch",
    )
    for vols, subject_ids in pbar:
        vols        = vols.to(device)           # [B, H, W, Z, T]
        subject_ids = subject_ids.to(device)    # [B]

        optimizer.zero_grad()
        z    = model(vols)                      # [B, projection_dim]
        loss = subject_contrastive_loss(z, subject_ids, temperature=temperature)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches    += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Subject-level embeddings (inference)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def compute_subject_embeddings(
    model: Hyena4DFullPipeline,
    dataset: FMRISubsequenceDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """
    Stream all subsequences through the model and mean-pool per subject.

    GPU memory stays bounded: embeddings are moved to CPU immediately
    after each batch and accumulated there as a running sum.

    Returns:
        {subject_str: embedding [projection_dim]}  (on CPU)
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    emb_sum:   dict[int, torch.Tensor] = {}
    emb_count: dict[int, int]          = defaultdict(int)

    for vols, sids in tqdm(loader, desc="  Subject embeddings", leave=False):
        z = model(vols.to(device)).cpu()        # [B, proj] — to CPU immediately
        for i, sid in enumerate(sids.tolist()):
            emb_sum[sid]   = emb_sum.get(sid, torch.zeros_like(z[i])) + z[i]
            emb_count[sid] += 1

    return {
        dataset.int_to_subject[sid]: emb_sum[sid] / emb_count[sid]
        for sid in sorted(emb_sum.keys())
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: Hyena4DFullPipeline,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
) -> None:
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss":                 loss,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Hyena4DFullPipeline,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    """Load checkpoint and return (epoch, loss) so training can resume."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  Resumed from '{path}'  (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")
    return ckpt["epoch"], ckpt["loss"]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hyena4D contrastive training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # I/O
    p.add_argument("--input-dir",      default="input",
                   help="Directory with .nii files")
    p.add_argument("--output-dir",     default="output",
                   help="Directory for checkpoints and final embeddings")
    # Data
    p.add_argument("--t-sub",          type=int,   default=50,
                   help="Temporal subsequence length (timepoints)")
    p.add_argument("--stride",         type=int,   default=None,
                   help="Sliding window stride (default: t_sub → non-overlapping)")
    p.add_argument("--subject-field",  type=int,   default=1,
                   help="Index of subject ID token when filename is split on '_'")
    # Model
    p.add_argument("--patch-size",     nargs=4, type=int, default=[8, 8, 4, 10],
                   metavar=("H", "W", "Z", "T"),
                   help="RPE patch size — must evenly divide each volume dim")
    p.add_argument("--rpe-hidden",     type=int,   default=64,
                   help="Hidden dim of the RPE phi MLP")
    p.add_argument("--embed-dim",      type=int,   default=64,
                   help="Hyena4D backbone embedding dim")
    p.add_argument("--num-blocks",     type=int,   default=4,
                   help="Number of Hyena4DPixelBlocks")
    p.add_argument("--projection-dim", type=int,   default=128,
                   help="Contrastive projection head output dim")
    p.add_argument("--kernel-size",    nargs=4, type=int, default=[3, 3, 3, 3],
                   metavar=("T", "Z", "H", "W"),
                   help="Hyena4D filter kernel size")
    p.add_argument("--voxel-size",     nargs=3, type=float, default=None,
                   metavar=("SZ", "SY", "SX"),
                   help="Voxel side lengths in mm along (z, y, x). "
                        "Auto-detected from the NIfTI header if not specified.")
    p.add_argument("--t-gap",          type=float, default=None,
                   help="TR / time between timepoints in seconds. "
                        "Auto-detected from the NIfTI header if not specified.")
    p.add_argument("--patch-chunk",    type=int,   default=16,
                   help="Patches per RPE forward chunk (lower = less RAM)")
    p.add_argument("--voxel-chunk",    type=int,   default=512,
                   help="Voxels per inner RPE chunk (0 = disabled)")
    # Training
    p.add_argument("--epochs",         type=int,   default=50,
                   help="Total training epochs")
    p.add_argument("--batch-size",     type=int,   default=8,
                   help="Subsequences per batch")
    p.add_argument("--lr",             type=float, default=1e-3,
                   help="Adam learning rate")
    p.add_argument("--temperature",    type=float, default=0.1,
                   help="Contrastive loss temperature")
    p.add_argument("--save-every",     type=int,   default=1,
                   help="Save a numbered checkpoint every N epochs")
    # Resume
    p.add_argument("--resume",         default=None,
                   help="Path to a checkpoint .pt file to resume training from")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # --- Auto-detect physical parameters from first NIfTI file ---
    first_nii = next(
        (os.path.join(args.input_dir, f)
         for f in sorted(os.listdir(args.input_dir)) if _is_nifti(f)),
        None,
    )
    if first_nii is None:
        raise RuntimeError(f"No .nii / .nii.gz files found in '{args.input_dir}'")

    auto_voxel, auto_t_gap = read_nifti_params(first_nii)
    voxel_size = tuple(args.voxel_size) if args.voxel_size is not None else auto_voxel
    t_gap      = args.t_gap             if args.t_gap      is not None else auto_t_gap
    print(f"  Physical params : voxel_size={tuple(f'{v:.4f}' for v in voxel_size)} mm"
          f"  t_gap={t_gap:.4f} s"
          f"{'  (from header)' if args.voxel_size is None else '  (manual)'}")

    # --- Dataset & loader ---
    dataset = FMRISubsequenceDataset(
        input_dir=args.input_dir,
        t_sub=args.t_sub,
        stride=args.stride,
        subject_field=args.subject_field,
    )
    if len(dataset) == 0:
        raise RuntimeError("No valid subsequences found. Check --input-dir and --t-sub.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,          # nibabel + Windows: keep workers=0
        pin_memory=(device.type == "cuda"),
        drop_last=True,         # avoids single-sample batches that break contrastive loss
    )
    print(f"  Batches/epoch : {len(loader)}")

    # --- Model ---
    voxel_chunk = args.voxel_chunk if args.voxel_chunk > 0 else None
    model = Hyena4DFullPipeline(
        patch_size=tuple(args.patch_size),
        rpe_hidden_dim=args.rpe_hidden,
        projection_dim=args.projection_dim,
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
        kernel_size=tuple(args.kernel_size),
        patch_chunk=args.patch_chunk,
        voxel_chunk=voxel_chunk,
        voxel_size=voxel_size,
        t_gap=t_gap,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params : {n_params:,}")

    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Resume from checkpoint if requested ---
    start_epoch = 1
    if args.resume:
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch += 1   # resume at the *next* epoch

    # --- Training loop ---
    best_loss = float("inf")
    epoch_pbar = tqdm(
        range(start_epoch, args.epochs + 1),
        desc="Training",
        unit="epoch",
    )
    for epoch in epoch_pbar:
        loss = train_one_epoch(
            model, loader, optimizer, device,
            epoch, args.epochs, args.temperature,
        )

        epoch_pbar.set_postfix(loss=f"{loss:.4f}", best=f"{best_loss:.4f}")
        print(f"Epoch {epoch:03d}/{args.epochs}  loss={loss:.4f}")

        # Numbered checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch:03d}.pt")
            save_checkpoint(ckpt_path, model, optimizer, epoch, loss)
            print(f"  Saved : {ckpt_path}")

        # Best model
        if loss < best_loss:
            best_loss = loss
            save_checkpoint(
                os.path.join(args.output_dir, "best_model.pt"),
                model, optimizer, epoch, loss,
            )

    # --- Subject-level embeddings ---
    print("\nComputing subject-level embeddings (mean pool over subsequences)...")
    subject_embs = compute_subject_embeddings(model, dataset, device, args.batch_size)
    subject_list = sorted(subject_embs.keys())
    emb_tensor   = torch.stack([subject_embs[s] for s in subject_list])  # [N, proj_dim]

    save_path = os.path.join(args.output_dir, "subject_embeddings.pt")
    torch.save({"embeddings": emb_tensor, "subject_ids": subject_list}, save_path)
    print(f"Saved : {save_path}  ({len(subject_list)} subjects, shape {tuple(emb_tensor.shape)})")


if __name__ == "__main__":
    main()
