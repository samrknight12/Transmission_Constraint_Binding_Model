"""
GPU availability and XGBoost benchmark check.

Usage:
    py -3 scripts/check_gpu.py
"""
import time

import numpy as np
import xgboost as xgb

N_ROWS   = 10_000
N_COLS   = 60      # matches master dataset feature count
N_ROUNDS = 200

rng = np.random.default_rng(42)
X = rng.standard_normal((N_ROWS, N_COLS)).astype(np.float32)
y = (rng.random(N_ROWS) > 0.85).astype(np.float32)  # ~15% positive, similar to dataset

print(f"XGBoost version : {xgb.__version__}")
print(f"Benchmark data  : {N_ROWS:,} rows x {N_COLS} cols, {N_ROUNDS} rounds\n")

# ── CUDA check via torch (optional) ──────────────────────────────────────────
try:
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"CUDA available  : YES")
        print(f"GPU device      : {device_name}")
        print(f"GPU memory      : {total_mem_gb:.1f} GB")
    else:
        print("CUDA available  : NO  (torch installed but no CUDA device)")
except ImportError:
    cuda_available = False
    print("CUDA available  : unknown (torch not installed; will test via XGBoost directly)")

print()

# ── CPU benchmark ─────────────────────────────────────────────────────────────
params_base = dict(
    n_estimators=N_ROUNDS,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
)

cpu_model = xgb.XGBClassifier(**params_base, device="cpu")
t0 = time.perf_counter()
cpu_model.fit(X, y)
cpu_time = time.perf_counter() - t0
print(f"CPU fit time    : {cpu_time:.3f}s")

# ── GPU benchmark ─────────────────────────────────────────────────────────────
gpu_model = xgb.XGBClassifier(**params_base, device="cuda")
try:
    t0 = time.perf_counter()
    gpu_model.fit(X, y)
    gpu_time = time.perf_counter() - t0
    print(f"GPU fit time    : {gpu_time:.3f}s")
    speedup = cpu_time / gpu_time
    print(f"\nSpeedup         : {speedup:.1f}x  ({'GPU faster' if speedup > 1 else 'CPU faster'})")
    print("\nCUDA is available to XGBoost.")
except xgb.core.XGBoostError as e:
    print(f"GPU fit time    : FAILED")
    print(f"  Error: {e}")
    print("\nXGBoost cannot use CUDA on this machine.")
    print("To enable GPU support: install the CUDA-enabled XGBoost wheel and ensure")
    print("NVIDIA drivers are installed.")
