import sys
from datasets import load_dataset

print("checking currently cached dataset ...", flush=True)
ds_old = load_dataset("mpg-ranch/multimodal-survey", split="train")
print("BEFORE force_redownload:", flush=True)
print(f"  n columns: {len(ds_old.column_names)}  has drone_coverage: {'drone_coverage' in ds_old.column_names}", flush=True)
print(f"  drone_image[0] size: {ds_old[0]['drone_image'].size}", flush=True)
print(f"  drone_image[1] size: {ds_old[1]['drone_image'].size}", flush=True)
print(f"  ground_image[0] size: {ds_old[0]['ground_image'].size}", flush=True)

print("\nforcing a fresh redownload ...", flush=True)
ds_new = load_dataset("mpg-ranch/multimodal-survey", split="train", download_mode="force_redownload")
print("AFTER force_redownload:", flush=True)
print(f"  n columns: {len(ds_new.column_names)}  has drone_coverage: {'drone_coverage' in ds_new.column_names}", flush=True)
print(f"  drone_image[0] size: {ds_new[0]['drone_image'].size}", flush=True)
print(f"  drone_image[1] size: {ds_new[1]['drone_image'].size}", flush=True)
print(f"  ground_image[0] size: {ds_new[0]['ground_image'].size}", flush=True)
print(f"  len train: {len(ds_new)}", flush=True)

print("\ndone.", flush=True)
