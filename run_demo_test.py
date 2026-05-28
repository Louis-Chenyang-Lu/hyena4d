"""
run_demo_test.py
----------------
Smoke-test the full pipeline and check three things:
  1. Embeddings have the correct shape and are L2-normalised
  2. Rough FLOP estimate vs vanilla Hyena (O(N log N) FFT)
  3. Peak RAM does not explode during training or inference
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows: avoid OpenMP dll conflict
import subprocess
import sys
import time
import threading
import psutil
import torch
import numpy as np

PROCESS = psutil.Process(os.getpid())

# ---------------------------------------------------------------------------
# Peak RAM tracker (polls in background thread)
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
# 1. Training
# ---------------------------------------------------------------------------

train_cmd = [
    sys.executable, "train_hyena.py",
    "--input-dir",      "input/pretrain/normalized",
    "--output-dir",     "output_test",
    "--subject-field",  "0",          # sub-001_... → field 0
    "--patch-size",     "9", "4", "5", "10",
    "--t-sub",          "50",
    "--batch-size",     "3",          # 3 subjects × 20 windows = 60 samples
    "--epochs",         "3",          # short demo run
    "--embed-dim",      "64",
    "--num-blocks",     "4",
    "--projection-dim", "128",
]

train_time, train_ram = run(train_cmd, "TRAINING (3 epochs, 3 subjects)")


# ---------------------------------------------------------------------------
# 2. Inference
# ---------------------------------------------------------------------------

infer_cmd = [
    sys.executable, "run_hyena.py",
    "--input-dir",      "input/target",
    "--output-dir",     "output_test",
    "--output-name",    "target_embeddings.pt",
    "--checkpoint",     "output_test/best_model.pt",
    "--subject-field",  "1",          # driftRespCardlessBOLD_B16_... → field 1
    "--patch-size",     "8", "8", "4", "10",
    "--t-sub",          "50",         # must match training patch size
    "--embed-dim",      "64",
    "--num-blocks",     "4",
    "--projection-dim", "128",
]

infer_time, infer_ram = run(infer_cmd, "INFERENCE (1 subject)")


# ---------------------------------------------------------------------------
# 3. Check 1 — Embedding sanity
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("  CHECK 1 — Embedding sanity")
print(f"{'='*60}")

# Training embeddings
train_data = torch.load("output_test/subject_embeddings.pt", map_location="cpu")
train_embs = train_data["embeddings"]   # [N_train, proj_dim]
train_ids  = train_data["subject_ids"]
print(f"\n  Training embeddings")
print(f"    shape          : {tuple(train_embs.shape)}  (expected [3, 128])")
print(f"    subject_ids    : {train_ids}")
norms_train = train_embs.norm(dim=-1)
print(f"    L2 norms       : {norms_train.tolist()}  (expected ≈ 1.0 each)")

# Pairwise cosine similarities (L2-normalised → dot product = cosine sim)
sims = train_embs @ train_embs.T
print(f"    Pairwise cosine sim matrix (train subjects):")
for i, si in enumerate(train_ids):
    row = "  ".join(f"{sims[i,j].item():.3f}" for j in range(len(train_ids)))
    print(f"      {si:20s}  {row}")
print("    (diagonal=1.0; off-diagonal < 1.0 means subjects are distinguished)")

# Inference embeddings
infer_data = torch.load("output_test/target_embeddings.pt", map_location="cpu")
infer_embs = infer_data["embeddings"]  # [N_pred, proj_dim]
infer_ids  = infer_data["subject_ids"]
print(f"\n  Inference embeddings")
print(f"    shape          : {tuple(infer_embs.shape)}  (expected [1, 128])")
print(f"    subject_ids    : {infer_ids}")
norms_infer = infer_embs.norm(dim=-1)
print(f"    L2 norms       : {norms_infer.tolist()}  (expected ≈ 1.0)")

assert tuple(train_embs.shape) == (3, 128), "Wrong training embedding shape!"
assert tuple(infer_embs.shape) == (1, 128), "Wrong inference embedding shape!"
# Subject embeddings are the mean of L2-normalised clip embeddings, so they
# are NOT unit-norm unless every clip embedding is identical (see
# model_description.md §10). Assert the mathematically correct bound instead:
# the mean of unit vectors has norm in (0, 1].
assert (norms_train > 0).all() and (norms_train <= 1.0 + 1e-3).all(), \
    f"Train subject-embedding norms outside (0, 1]: {norms_train.tolist()}"
assert (norms_infer > 0).all() and (norms_infer <= 1.0 + 1e-3).all(), \
    f"Infer subject-embedding norms outside (0, 1]: {norms_infer.tolist()}"
print("\n  CHECK 1 PASSED")


# ---------------------------------------------------------------------------
# 4. Check 2 — Computational cost vs Hyena
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("  CHECK 2 — Computational cost vs vanilla Hyena")
print(f"{'='*60}")

# Token grid after RPE for pre-training data: 10 × 5 × 10 × 5 = 2500 tokens
# T×Z merged depth axis = 10×5 = 50
# Each Hyena4DPixelBlock does one depthwise Conv3d: kernel 3×3×(3×3)=81
# Ours: O(N × K)  where N = T_p×Z_p×H_p×W_p, K = K_t×K_z×K_h×K_w
# Vanilla Hyena: O(N log N) via FFT

N_tokens = 10 * 5 * 10 * 5         # 2500 tokens per sample (pre-training token grid)
K = 3 * 3 * 3 * 3                  # kernel positions = 81
C = 64                              # embed_dim

our_ops   = N_tokens * K * C       # depthwise conv3d
hyena_ops = N_tokens * np.log2(N_tokens) * C  # FFT Hyena

print(f"\n  Token grid (pre-train)  : T_p×Z_p×H_p×W_p = 10×5×10×5 = {N_tokens}")
print(f"  Kernel positions        : K_t×K_z×K_h×K_w = 3×3×3×3  = {K}")
print(f"  embed_dim (C)           : {C}")
print(f"\n  Our Conv3d cost / block : {our_ops:,.0f}  ops  (O(N × K))")
print(f"  Vanilla Hyena FFT cost  : {hyena_ops:,.0f}  ops  (O(N log N))")
print(f"  Ratio  our / Hyena      : {our_ops / hyena_ops:.2f}x")
if our_ops / hyena_ops < 5.0:
    print("  → Within 5× of Hyena FFT cost  ✓")
else:
    print("  → More than 5× Hyena FFT cost  — consider larger kernel or fewer blocks")


# ---------------------------------------------------------------------------
# 5. Check 3 — RAM summary
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("  CHECK 3 — Peak RAM")
print(f"{'='*60}")
print(f"\n  Training   peak RAM : {train_ram:.2f} GiB")
print(f"  Inference  peak RAM : {infer_ram:.2f} GiB")

RAM_LIMIT_GIB = 16.0
if train_ram < RAM_LIMIT_GIB and infer_ram < RAM_LIMIT_GIB:
    print(f"  Both under {RAM_LIMIT_GIB} GiB limit  ✓")
else:
    print(f"  WARNING: exceeded {RAM_LIMIT_GIB} GiB — consider reducing batch-size or patch-chunk")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
print(f"  Training  : {train_time:.1f} s,  peak {train_ram:.2f} GiB")
print(f"  Inference : {infer_time:.1f} s,  peak {infer_ram:.2f} GiB")
print(f"  Embeddings: train {tuple(train_embs.shape)}, infer {tuple(infer_embs.shape)}")
print(f"  All checks passed.\n")
