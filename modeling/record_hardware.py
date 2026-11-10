"""Record the hardware (GPU, GPU memory, CPU, RAM, software versions) and the
size on disk of the feature files.

Writes results/hardware.txt and results/feature_cache_sizes.csv.
"""
import os
import platform
import psutil
import torch
import pandas as pd

lines = []
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    lines.append(f"Total VRAM: {props.total_memory/1e9:.1f} GB")
else:
    lines.append("GPU: none (CPU only)")

lines.append(f"CPU: {platform.processor()}  ({psutil.cpu_count(logical=False)} physical / "
             f"{psutil.cpu_count(logical=True)} logical cores)")
lines.append(f"RAM: {psutil.virtual_memory().total/1e9:.1f} GB")
lines.append(f"OS: {platform.platform()}")
lines.append(f"PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}")

text = "\n".join(lines)
print(text)
os.makedirs("results", exist_ok=True)
with open("results/hardware.txt", "w") as f:
    f.write(text + "\n")
print("\nwrote results/hardware.txt")

# ---- size on disk of the feature files the three extract_*.py scripts write ----
CACHE_FILES = {
    "grid_extract_train": "feats/train.ground_image.general.grid1280x720.npy",
    "grid_extract_test": "feats/test.ground_image.general.grid1280x720.npy",
    "cls_extract_train_drone_satellite": "feats/train.drone_image.satellite.npy",
    "cls_extract_test_drone_satellite": "feats/test.drone_image.satellite.npy",
}
rows = []
for stage, path in CACHE_FILES.items():
    if os.path.exists(path):
        rows.append(dict(stage=stage, path=path, size_gb=round(os.path.getsize(path)/1e9, 4)))
    else:
        rows.append(dict(stage=stage, path=path, size_gb=None))

df = pd.DataFrame(rows)
df.to_csv("results/feature_cache_sizes.csv", index=False)
print("\n=== feature cache sizes on disk ===")
print(df.to_string(index=False))
print("\nwrote results/feature_cache_sizes.csv")
