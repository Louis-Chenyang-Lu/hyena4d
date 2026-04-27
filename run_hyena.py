"""
run_hyena.py
------------
Inference-only subject embedding pipeline.  Loads a pre-trained Hyena4D
checkpoint (saved by train_hyena.py) and produces subject-level embeddings
for a directory of NIfTI files.

Pipeline per NIfTI file:
    1. Load .nii  →  [H, W, Z, T]  (float32)
    2. RPE        →  [1, out_dim, H_p, W_p, Z_p, T_p]   (CPU, chunked)
       (skipped when --no-rpe is set; raw volume used directly)
    3. Permute    →  [1, C, T_p, Z_p, H_p, W_p]         (Hyena4D layout)
    4. Encoder    →  [1, projection_dim]                 (CPU or CUDA)
    5. Aggregate subsequences per subject (mean pooling)

Typical workflow:
    # 1. Pre-train on your dataset:
    python train_hyena.py --input-dir /data --output-dir /ckpts --voxel-size 0.5 0.5 0.5 --t-gap 1.5

    # 2. Extract embeddings on target data using the best checkpoint:
    python run_hyena.py --input-dir /target --checkpoint /ckpts/best_model.pt --voxel-size 0.5 0.5 0.5 --t-gap 1.5

Device note:
    PyTorch MPS does not support Conv3d. This script uses CUDA when available,
    otherwise falls back to CPU — it never uses MPS.

Usage:
    python run_hyena.py [options]

    --checkpoint       PATH   Path to a checkpoint saved by train_hyena.py
                              (required for meaningful embeddings; omit only for
                               testing with random weights)
    --input-dir        DIR    Directory containing .nii files  (default: input)
    --output-dir       DIR    Directory for output .pt file    (default: output)
    --output-name      NAME   Output filename                  (default: subject_embeddings.pt)
    --no-rpe                  Skip RPE; feed raw voxels (in_channels=1) directly
    --patch-size       H W Z T  Patch size (default: 8 8 4 10)
    --patch-chunk      N     Patches per RPE forward chunk    (default: 16)
    --voxel-chunk      N     Voxels per inner chunk (0=all)   (default: 512)
    --embed-dim        N     Hyena4D embedding dimension      (default: 64)
    --num-blocks       N     Number of Hyena4D blocks         (default: 4)
    --projection-dim   N     Contrastive projection dim       (default: 128)
    --subject-field    N     Token index in filename (split on '_') for subject ID
                              (default 1; use 0 for BIDS-style sub-001_... names)
    --voxel-size SZ SY SX   Voxel side lengths in mm (auto-detected from header if omitted)
    --t-gap            F     TR / time between timepoints in s (auto-detected if omitted)
"""

import argparse
import os
from collections import defaultdict

import nibabel as nib
import numpy as np
import torch

from hyenapixel.hyena4d_model import Hyena4DContrastiveEncoder


# ---------------------------------------------------------------------------
# NIfTI utilities (mirrors train_hyena.py)
# ---------------------------------------------------------------------------

def _is_nifti(fname: str) -> bool:
    return fname.endswith(".nii") or fname.endswith(".nii.gz")


def nifti_basename(fname: str) -> str:
    if fname.endswith(".nii.gz"):
        return fname[:-7]
    if fname.endswith(".nii"):
        return fname[:-4]
    return fname


def read_nifti_params(fpath: str) -> tuple[tuple[float, float, float], float]:
    """Read voxel_size=(sz,sy,sx) in mm and t_gap (TR) in seconds from NIfTI header."""
    hdr = nib.load(fpath).header
    zooms = hdr.get_zooms()
    sz = float(zooms[2]) if len(zooms) > 2 else 1.0
    sy = float(zooms[0]) if len(zooms) > 0 else 1.0
    sx = float(zooms[1]) if len(zooms) > 1 else 1.0
    t_gap = 1.0
    if len(zooms) >= 4 and float(zooms[3]) > 0:
        tr = float(zooms[3])
        time_unit = hdr.get_xyzt_units()[1]
        if time_unit == "msec" or (time_unit == "sec" and tr > 100):
            tr /= 1000.0
        t_gap = tr
    return (sz, sy, sx), t_gap
from run_rpe import rpe_embed_volume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyena4D fMRI subject embedding pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default="input",
                        help="Directory containing .nii files")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for the saved embeddings")
    parser.add_argument("--output-name", default="subject_embeddings.pt",
                        help="Filename for the output .pt file")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a checkpoint .pt saved by train_hyena.py. "
                             "Omitting this uses random weights (useful only for testing).")
    parser.add_argument("--no-rpe", action="store_true",
                        help="Skip RPE; feed raw voxels (in_channels=1) directly into encoder")
    parser.add_argument("--patch-size", nargs=4, type=int, default=[8, 8, 4, 10],
                        metavar=("H", "W", "Z", "T"),
                        help="Patch size — must evenly divide each volume dimension")
    parser.add_argument("--patch-chunk", type=int, default=16,
                        help="Patches processed per RPE forward pass (lower = less RAM)")
    parser.add_argument("--voxel-chunk", type=int, default=512,
                        help="Voxels per inner chunking loop (0 = disabled)")
    parser.add_argument("--embed-dim", type=int, default=64,
                        help="Hyena4D backbone embedding dimension")
    parser.add_argument("--num-blocks", type=int, default=4,
                        help="Number of Hyena4DPixelBlocks in the backbone")
    parser.add_argument("--projection-dim", type=int, default=128,
                        help="Contrastive projection head output dimension")
    parser.add_argument("--subject-field", type=int, default=1,
                        help="Token index in filename (split on '_') for subject ID. "
                             "Use 0 for BIDS-style names (sub-001_...), "
                             "1 for driftRespCardlessBOLD_B16_... style.")
    parser.add_argument("--voxel-size", nargs=3, type=float, default=None,
                        metavar=("SZ", "SY", "SX"),
                        help="Voxel side lengths in mm along (z, y, x). "
                             "Auto-detected from NIfTI header if not specified.")
    parser.add_argument("--t-gap", type=float, default=None,
                        help="TR / time between timepoints in seconds. "
                             "Auto-detected from NIfTI header if not specified.")
    return parser.parse_args()


def pick_device() -> torch.device:
    """Return CUDA if available, else CPU. Never returns MPS (Conv3d unsupported)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()

    device = pick_device()
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    patch_size = tuple(args.patch_size)
    voxel_chunk = args.voxel_chunk if args.voxel_chunk > 0 else None

    nii_files = sorted(f for f in os.listdir(args.input_dir) if _is_nifti(f))
    if not nii_files:
        raise RuntimeError(f"No .nii / .nii.gz files found in '{args.input_dir}'")
    print(f"Found {len(nii_files)} NIfTI file(s)")

    # --- Auto-detect physical parameters from first file ---
    first_fpath = os.path.join(args.input_dir, nii_files[0])
    auto_voxel, auto_t_gap = read_nifti_params(first_fpath)
    voxel_size = tuple(args.voxel_size) if args.voxel_size is not None else auto_voxel
    t_gap      = args.t_gap             if args.t_gap      is not None else auto_t_gap
    print(f"  Physical params : voxel_size={tuple(f'{v:.4f}' for v in voxel_size)} mm"
          f"  t_gap={t_gap:.4f} s"
          f"{'  (from header)' if args.voxel_size is None else '  (manual)'}")

    if args.checkpoint is None:
        print("WARNING: no --checkpoint provided; using random weights.")

    encoder = None
    subj_sum: dict[str, torch.Tensor] = {}
    subj_count: dict[str, int] = defaultdict(int)

    for fname in nii_files:
        parts = nifti_basename(fname).split("_")
        subject_id = parts[args.subject_field] if len(parts) > args.subject_field else parts[0]
        fpath = os.path.join(args.input_dir, fname)
        print(f"  {fname}  (subject={subject_id})")

        vol = nib.load(fpath).get_fdata().astype(np.float32)  # [H, W, Z, T]
        x = torch.from_numpy(vol).unsqueeze(0)                # [1, H, W, Z, T]

        with torch.inference_mode():
            if args.no_rpe:
                # Skip RPE — feed raw volume directly as single-channel input.
                # [1, H, W, Z, T] → [1, 1, H, W, Z, T] → [1, 1, T, Z, H, W]
                emb = x.unsqueeze(1).permute(0, 1, 5, 4, 2, 3).contiguous()
            else:
                # RPE on CPU — raw volumes are large
                emb = rpe_embed_volume(
                    x,
                    patch_size=patch_size,
                    hidden_dim=64,
                    out_dim=args.projection_dim,
                    patch_chunk=args.patch_chunk,
                    voxel_chunk=voxel_chunk,
                )  # [1, projection_dim, H_p, W_p, Z_p, T_p]

                # Reorder axes to match Hyena4D expected layout: [B, C, T_p, Z_p, H_p, W_p]
                emb = emb.permute(0, 1, 5, 4, 2, 3).contiguous()

            if encoder is None:
                in_channels = 1 if args.no_rpe else emb.shape[1]
                encoder = Hyena4DContrastiveEncoder(
                    in_channels=in_channels,
                    embed_dim=args.embed_dim,
                    num_blocks=args.num_blocks,
                    projection_dim=args.projection_dim,
                    kernel_size=(3, 3, 3, 3),
                    voxel_size=voxel_size,
                    t_gap=t_gap,
                ).to(device).eval()

                if args.checkpoint is not None:
                    ckpt = torch.load(args.checkpoint, map_location=device)
                    # train_hyena.py wraps encoder inside Hyena4DFullPipeline,
                    # so strip the "encoder." prefix when loading
                    raw_sd = ckpt["model_state_dict"]
                    enc_sd = {
                        k[len("encoder."):]: v
                        for k, v in raw_sd.items()
                        if k.startswith("encoder.")
                    }
                    encoder.load_state_dict(enc_sd)
                    print(f"  Loaded encoder weights from '{args.checkpoint}' "
                          f"(epoch {ckpt.get('epoch', '?')})")

            z = encoder(emb.to(device)).squeeze(0).detach().cpu()  # [projection_dim]

        subj_sum[subject_id] = subj_sum.get(subject_id, 0) + z
        subj_count[subject_id] += 1

        del x, emb, z
        if device.type == "cuda":
            torch.cuda.empty_cache()

    subject_ids = sorted(subj_sum.keys())
    embeddings = torch.stack(
        [subj_sum[sid] / subj_count[sid] for sid in subject_ids], dim=0
    )  # [N_subjects, projection_dim]

    save_path = os.path.join(args.output_dir, args.output_name)
    torch.save({"embeddings": embeddings, "subject_ids": subject_ids}, save_path)
    print(f"\nSaved: {save_path}")
    print(f"Embeddings shape: {embeddings.shape}  ({len(subject_ids)} subjects)")


if __name__ == "__main__":
    main()
