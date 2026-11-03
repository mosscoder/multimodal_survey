"""Item 7: shared helper so every extraction/training script logs wall time and
peak VRAM for its stage(s) the same way. Appends to results/timing.csv (so
multiple scripts' runs accumulate into one file) -- callers should run each
stage once per fresh-measurement pass to avoid duplicate rows.
"""
import csv
import os
import torch

TIMING_CSV = "results/timing.csv"
_FIELDS = ["stage", "wall_s", "gpu", "peak_vram_gb", "note"]


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def reset_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def log_stage(stage, wall_s, note=""):
    """Call after a stage finishes. Reads+resets the CUDA peak-memory counter,
    so call reset_vram() right before the stage starts."""
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    row = dict(stage=stage, wall_s=round(wall_s, 1), gpu=gpu_name(),
               peak_vram_gb=round(peak, 3), note=note)
    write_header = not os.path.exists(TIMING_CSV)
    os.makedirs(os.path.dirname(TIMING_CSV), exist_ok=True)
    with open(TIMING_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"  [timing] {stage}: {wall_s:.1f}s  peak_vram {peak:.2f}GB"
          + (f"  ({note})" if note else ""), flush=True)
    reset_vram()
