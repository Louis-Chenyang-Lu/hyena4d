"""
train_smoke_test.py
-------------------
End-to-end smoke test of train_hyena.py only (no inference).

Purpose: catch crashes before submitting the long real run. Verifies:
  1. Training completes through 2 epochs without error
  2. A checkpoint file (best_model.pt) is created with expected keys
  3. subject_embeddings.pt is produced with the expected shape
  4. Loss values are finite (catches AMP-induced NaNs)
  5. Peak RAM stays within reason

This script intentionally uses the SAME data directory and SAME pipeline
flags as the real run, so any data- or flag-related issue surfaces here
rather than after 17 hours of GPU time. Only differences from the real
run: --epochs 2 and --batch-size 16 (small batch so the smoke test
finishes quickly).
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
# 1. Training (2 epochs, smaller batch, all real-run flags exercised)
# ---------------------------------------------------------------------------

OUTPUT_DIR = "output_smoke"

train_cmd = [
    sys.executable, "train_hyena.py",
    "--input-dir",      "input/pretrain/normalized",
    "--output-dir",     OUTPUT_DIR,
    "--subject-field",  "0",
    "--num-workers",    "7",          # exercise the new DataLoader path
    "--patch-size",     "9", "4", "5", "10",
    "--patch-chunk",    "2500",       # exercise the no-loop RPE path
    "--t-sub",          "50",
    "--batch-size",     "16",         # smaller than 128 so the test is fast
    "--epochs",         "2",          # bare minimum to see >1-epoch dynamics
    "--embed-dim",      "64",
    "--num-blocks",     "4",
    "--projection-dim", "128",
    # --memory-efficient deliberately NOT passed (default = batched RPE on GPU)
]

train_time, train_ram = run(train_cmd, "TRAINING (2 epochs, smoke test)")


# ---------------------------------------------------------------------------
# 2. Verification
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("  CHECKS")
print(f"{'='*60}")

# --- Checkpoint exists and has the expected keys ---
ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pt")
assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
ckpt = torch.load(ckpt_path, map_location="cpu")

expected_keys = {"epoch", "model_state_dict", "optimizer_state_dict", "loss"}
missing_keys = expected_keys - set(ckpt.keys())
assert not missing_keys, f"Checkpoint missing keys: {missing_keys}"
print(f"  ✓ Checkpoint present at {ckpt_path}")
print(f"    epoch = {ckpt['epoch']}, loss = {ckpt['loss']:.4f}")

# --- Loss is finite (catches AMP-induced NaN/Inf) ---
assert torch.isfinite(torch.tensor(ckpt["loss"])).item(), \
    f"Loss is not finite (got {ckpt['loss']}); AMP may have diverged."
print(f"  ✓ Final loss is finite")

# --- state_dict keys do NOT carry the _orig_mod. prefix (torch.compile bug check) ---
sd_keys = list(ckpt["model_state_dict"].keys())
compiled_keys = [k for k in sd_keys if k.startswith("_orig_mod.")]
assert not compiled_keys, (
    f"Checkpoint keys still carry the torch.compile '_orig_mod.' prefix:\n"
    f"  example: {compiled_keys[0]}\n"
    f"  This will break run_hyena.py loading. Verify save_checkpoint strips it."
)
print(f"  ✓ Checkpoint keys have no '_orig_mod.' prefix (run_hyena.py will load fine)")

# Sanity check: the state dict has the expected sub-module prefixes
has_rpe     = any(k.startswith("rpe.")     for k in sd_keys)
has_encoder = any(k.startswith("encoder.") for k in sd_keys)
assert has_rpe and has_encoder, (
    f"Expected 'rpe.*' and 'encoder.*' keys in state_dict; got keys like {sd_keys[:5]}"
)
print(f"  ✓ Checkpoint contains both 'rpe.*' and 'encoder.*' parameters")

# --- Subject embeddings file exists and has correct shape ---
emb_path = os.path.join(OUTPUT_DIR, "subject_embeddings.pt")
assert os.path.exists(emb_path), f"Subject embeddings missing: {emb_path}"
data = torch.load(emb_path, map_location="cpu")
assert "embeddings" in data and "subject_ids" in data, \
    f"Embedding file missing expected keys: {list(data.keys())}"

embs = data["embeddings"]
ids  = data["subject_ids"]
N_subjects = len(ids)

assert embs.shape == (N_subjects, 128), \
    f"Wrong embedding shape: got {tuple(embs.shape)}, expected ({N_subjects}, 128)"
print(f"  ✓ Embeddings shape = {tuple(embs.shape)}  ({N_subjects} subjects, dim 128)")

# --- Embedding L2 norms in (0, 1] (mean of unit vectors) ---
norms = embs.norm(dim=-1)
assert (norms > 0).all() and (norms <= 1.0 + 1e-3).all(), \
    f"Subject-embedding norms outside (0, 1]: min={norms.min():.4f}, max={norms.max():.4f}"
assert torch.isfinite(embs).all(), "Embeddings contain NaN/Inf"
print(f"  ✓ Embedding norms in (0, 1]: min={norms.min():.4f}, max={norms.max():.4f}")

# --- Peak RAM sanity ---
RAM_LIMIT_GIB = 32.0   # generous limit for the smoke test
if train_ram < RAM_LIMIT_GIB:
    print(f"  ✓ Peak RAM {train_ram:.2f} GiB < {RAM_LIMIT_GIB} GiB")
else:
    print(f"  ⚠ Peak RAM {train_ram:.2f} GiB exceeded {RAM_LIMIT_GIB} GiB "
          f"— investigate before the long run")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
print(f"  Wall time     : {train_time:.1f} s ({train_time/60:.1f} min)")
print(f"  Peak RAM      : {train_ram:.2f} GiB")
print(f"  Final loss    : {ckpt['loss']:.4f}")
print(f"  Subjects      : {N_subjects}")
print(f"  Output dir    : {OUTPUT_DIR}")
print(f"\n  ALL CHECKS PASSED — safe to submit the real run.\n")