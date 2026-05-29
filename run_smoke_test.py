"""
run_smoke_test.py
-------------------
Smoke test for run_hyena.py (inference only).

Purpose: verify that the inference pipeline works end-to-end using a
pre-trained checkpoint before running the real embedding extraction.
Checks:

  1. Inference completes without error
  2. subject_embeddings.pt is produced with the expected shape
  3. Embedding L2 norms are in (0, 1]  (mean of unit vectors)
  4. Embeddings are finite (no NaN/Inf)
  5. Peak RAM stays within reason

Prerequisites:
  - A trained checkpoint must exist (run train_smoke_test.py first, or
    point --checkpoint at any valid best_model.pt).
  - The target NIfTI files must be present in the input directory below.

Typical usage:
    # 1. Run training smoke test to produce a checkpoint:
    python train_smoke_test.py

    # 2. Then run this script:
    python run_smoke_test.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import subprocess
import sys
import time
import threading
import psutil
import torch

PROCESS = psutil.Process(os.getpid())

# ---------------------------------------------------------------------------
# Configuration — mirror the flags used in your real inference run
# ---------------------------------------------------------------------------

TRAIN_OUTPUT_DIR = f"output/output_smoke"           # where train_smoke_test.py wrote its checkpoint
INFER_OUTPUT_DIR = TRAIN_OUTPUT_DIR         # where this test writes its output
CHECKPOINT_PATH  = os.path.join(TRAIN_OUTPUT_DIR, "best_model.pt")

# These must match the architecture flags used during training
PATCH_SIZE     = ["9", "4", "5", "10"]
EMBED_DIM      = "64"
NUM_BLOCKS     = "4"
PROJECTION_DIM = "128"
RPE_HIDDEN     = "64"
T_SUB          = "50"

# Target data (inference input) — adjust to your actual target directory
INFER_INPUT_DIR   = "input/target/normalized"
INFER_SUBJECT_FIELD = "1"               # field index when filename is split on '_'

# Expected number of subjects in the inference input directory.
# Set to None to skip the exact-count check and just verify shape is [N, 128].
EXPECTED_N_SUBJECTS = 18


# ---------------------------------------------------------------------------
# Peak RAM tracker (background polling thread)
# ---------------------------------------------------------------------------

class PeakRAMTracker:
    def __init__(self, interval=0.5):
        self._peak = 0.0
        self._stop = threading.Event()
        self._interval = interval

    def _run(self):
        while not self._stop.is_set():
            mem = PROCESS.memory_info().rss / 1024**3  # GiB
            if mem > self._peak:
                self._peak = mem
            time.sleep(self._interval)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._t.join()

    @property
    def peak_gib(self):
        return self._peak


def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    with PeakRAMTracker() as tracker:
        t0 = time.time()
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        result = subprocess.run(cmd, capture_output=False, text=True, env=env)
        elapsed = time.time() - t0
    print(f"\n  [{label}] wall time : {elapsed:.1f} s")
    print(f"  [{label}] peak RAM  : {tracker.peak_gib:.2f} GiB")
    if result.returncode != 0:
        print(f"  [{label}] FAILED (returncode={result.returncode})")
        sys.exit(1)
    return elapsed, tracker.peak_gib


# ---------------------------------------------------------------------------
# Pre-flight: make sure a checkpoint exists to load from
# ---------------------------------------------------------------------------

if not os.path.exists(CHECKPOINT_PATH):
    print(f"ERROR: checkpoint not found at '{CHECKPOINT_PATH}'")
    print("       Run train_smoke_test.py first to produce a checkpoint, or")
    print("       update CHECKPOINT_PATH / TRAIN_OUTPUT_DIR at the top of this file.")
    sys.exit(1)

os.makedirs(INFER_OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"Checkpoint : {CHECKPOINT_PATH}")

# ---------------------------------------------------------------------------
# 1. Inference
# ---------------------------------------------------------------------------

infer_cmd = [
    sys.executable, "run_hyena.py",
    "--input-dir",      INFER_INPUT_DIR,
    "--output-dir",     INFER_OUTPUT_DIR,
    "--output-name",    "subject_embeddings.pt",
    "--checkpoint",     CHECKPOINT_PATH,
    "--subject-field",  INFER_SUBJECT_FIELD,
    "--patch-size",     *PATCH_SIZE,
    "--t-sub",          T_SUB,
    "--embed-dim",      EMBED_DIM,
    "--rpe-hidden",     RPE_HIDDEN,
    "--num-blocks",     NUM_BLOCKS,
    "--projection-dim", PROJECTION_DIM,
]

infer_time, infer_ram = run(infer_cmd, "INFERENCE (smoke test)")


# ---------------------------------------------------------------------------
# 2. Verification
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("  CHECKS")
print(f"{'='*60}")

# --- Output file exists and has the expected keys ---
emb_path = os.path.join(INFER_OUTPUT_DIR, "subject_embeddings.pt")
assert os.path.exists(emb_path), f"Embeddings file missing: {emb_path}"

data = torch.load(emb_path, map_location="cpu")
assert "embeddings" in data and "subject_ids" in data, \
    f"Embedding file missing expected keys: {list(data.keys())}"

embs = data["embeddings"]
ids  = data["subject_ids"]
N_subjects = len(ids)
proj_dim   = int(PROJECTION_DIM)

print(f"  ✓ Output file present at {emb_path}")
print(f"    subject_ids = {ids}")

# --- Shape check ---
assert embs.ndim == 2, \
    f"Expected 2D tensor, got shape {tuple(embs.shape)}"
assert embs.shape[1] == proj_dim, \
    f"Wrong projection dim: got {embs.shape[1]}, expected {proj_dim}"
if EXPECTED_N_SUBJECTS is not None:
    assert embs.shape[0] == EXPECTED_N_SUBJECTS, (
        f"Wrong subject count: got {embs.shape[0]}, expected {EXPECTED_N_SUBJECTS}"
    )
print(f"  ✓ Embeddings shape = {tuple(embs.shape)}  "
      f"({N_subjects} subject(s), dim {proj_dim})")

# --- L2 norm check: mean of unit vectors → norm in (0, 1] ---
norms = embs.norm(dim=-1)
assert (norms > 0).all(), \
    f"Zero-norm embeddings found: {norms.tolist()}"
assert (norms <= 1.0 + 1e-3).all(), \
    f"Subject-embedding norms exceed 1.0: max={norms.max():.4f}"
print(f"  ✓ Embedding norms in (0, 1]: min={norms.min():.4f}, max={norms.max():.4f}")

# --- No NaN / Inf ---
assert torch.isfinite(embs).all(), \
    "Embeddings contain NaN or Inf values"
print(f"  ✓ Embeddings are finite (no NaN/Inf)")

# --- Checkpoint keys loaded correctly (verify rpe.* and encoder.* were used) ---
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
sd_keys = list(ckpt["model_state_dict"].keys())
has_rpe     = any(k.startswith("rpe.")     for k in sd_keys)
has_encoder = any(k.startswith("encoder.") for k in sd_keys)
assert has_rpe and has_encoder, \
    f"Checkpoint appears malformed — missing 'rpe.*' or 'encoder.*' keys"
print(f"  ✓ Checkpoint has 'rpe.*' and 'encoder.*' keys (epoch {ckpt.get('epoch', '?')},"
      f" train loss {ckpt.get('loss', float('nan')):.4f})")

# --- Pairwise cosine similarities (informational, not a hard assert) ---
if N_subjects > 1:
    sims = embs @ embs.T
    print(f"\n  Pairwise cosine similarity matrix ({N_subjects} subjects):")
    for i, si in enumerate(ids):
        row = "  ".join(f"{sims[i, j].item():.3f}" for j in range(N_subjects))
        print(f"    {str(si):20s}  {row}")
    print("  (diagonal=1.0 expected; off-diagonal < 1.0 means subjects are distinguished)")

# --- Peak RAM sanity ---
RAM_LIMIT_GIB = 32.0 if device.type == "cuda" else 10.0
if infer_ram < RAM_LIMIT_GIB:
    print(f"\n  ✓ Peak RAM {infer_ram:.2f} GiB < {RAM_LIMIT_GIB} GiB")
else:
    print(f"\n  ⚠ Peak RAM {infer_ram:.2f} GiB exceeded {RAM_LIMIT_GIB} GiB "
          f"— investigate before the real run")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
print(f"  Wall time     : {infer_time:.1f} s ({infer_time/60:.1f} min)")
print(f"  Peak RAM      : {infer_ram:.2f} GiB")
print(f"  Subjects      : {N_subjects}")
print(f"  Output dir    : {INFER_OUTPUT_DIR}")
print(f"\n  ALL CHECKS PASSED — safe to run the real inference.\n")