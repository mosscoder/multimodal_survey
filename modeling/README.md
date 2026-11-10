# Baseline experiments

This folder contains the code and the results behind the baseline table of the
data descriptor. Two classifiers predict the presence of eight plant species in
a ground image, one from the ground image itself and one from its paired drone
image, and a late fusion combines the two. Each classifier is a linear layer on
a frozen DINOv3 model. Paul Salakhutdinov wrote the code and produced the
results.

## Files

| Script | Purpose | Writes |
|---|---|---|
| `make_folds.py` | Assigns the 38 legs of the cross-validation pool to five folds | `folds_leg5.json` |
| `assemble.py` | Builds the pool, its labels and its folds from the dataset. The scripts below import it | |
| `extract_train_drone_features.py` | Whole-image features of the training drone images | `feats/train.drone_image.satellite.npy` |
| `extract_train_ground_features.py` | Patch features of the training ground images at 1280 by 720 | `feats/train.ground_image.general.grid1280x720.npy`, 35.8 GB |
| `extract_test_features.py` | Both feature sets for the test split | `feats/test.*.npy`, 8.9 GB |
| `train_classifiers.py` | Cross-validation, epoch selection, final models, test predictions | `results/selected_epochs.csv`, `results/{ground,drone}_epoch_curve.csv`, `results/{ground,drone}_per_fold.csv`, `results/preds/{ground,drone}_{cv,test}.npz` |
| `fuse_and_score.py` | Late fusion and every score | `results/ap_cv_and_test.csv`, `results/fusion_weights.csv`, `results/fusion_weights_per_fold.csv`, `results/preds/fused_{cv,test}.npz` |
| `confidence_intervals.py` | Leg bootstrap intervals and per-fold scores | `results/ap_confidence_intervals.csv`, `results/ap_per_fold.csv` |
| `record_hardware.py` | Hardware and the size of the feature files | `results/hardware.txt`, `results/feature_cache_sizes.csv` |
| `timing_utils.py` | Stage timing helper used by the scripts above | `results/timing.csv` |
| `check_dataset_version.py` | Confirms that the cached dataset is the current release | |

The repository tracks `folds_leg5.json` and the files in `results/` named
above, except `timing.csv` and `feature_cache_sizes.csv`, which a rerun writes
afresh. Every tracked result is the output of one of these scripts. The feature
files in `feats/` are too large to track.

## Where the numbers in the paper come from

| Entry of the baseline table | File | Columns |
|---|---|---|
| AP under cross-validation and on the test split, per species and as the macro average, for drone, ground and fused | `results/ap_cv_and_test.csv` | `cv_AP`, `test_AP` |
| 95% confidence intervals | `results/ap_confidence_intervals.csv` | `ci_lo`, `ci_hi` |
| Fusion weights | `results/fusion_weights.csv` | `ground_weight`, the drone weight is one minus it |
| Selected epochs, counted from zero | `results/selected_epochs.csv` | `E_star` |

## Reproducing the results

The scripts need the packages in `requirements.txt`, access to the dataset
`mpg-ranch/multimodal-survey` on Hugging Face, and the two DINOv3 models
`facebook/dinov3-vitb16-pretrain-lvd1689m` and
`facebook/dinov3-vitl16-pretrain-sat493m`, whose license may have to be
accepted on Hugging Face first. Feature extraction and training need a CUDA GPU
with 12 GB of memory and about 50 GB of free disk. Run the scripts from this
folder in this order.

```bash
python make_folds.py                       # rewrites folds_leg5.json
python extract_train_drone_features.py     # about 1 min
python extract_train_ground_features.py    # about 12 min
python extract_test_features.py            # about 4 min
python train_classifiers.py                # about 6 h, nearly all of it the ground classifier
python fuse_and_score.py                   # seconds, no GPU needed
python confidence_intervals.py             # about a minute, no GPU needed
python record_hardware.py
```

The durations were measured on the machine described in `results/hardware.txt`.

The scoring does not need a GPU. With only the tracked predictions,
`fuse_and_score.py` and `confidence_intervals.py` recompute every value of the
baseline table.

## Notes

- The cross-validation pool is the 6,089 training ground images on the legs.
  The turns are left out. Folds are grouped by leg, so both capture events of a
  leg fall in one fold.
- Each reported prediction is the average of three runs with different random
  seeds. The three runs of a fold share one seeded batch order and differ in
  their initial weights.
- One epoch is selected per modality, the one with the highest mean validation
  macro AP across the five folds. The final model trains on the whole pool to
  that epoch and predicts the test split.
- The drone image enters its model resized from 308 by 308 to 224 by 224
  pixels. The ground image enters at its full 1280 by 720 pixels.
- Feature extraction runs in reduced precision on the GPU, so a rerun on
  different hardware can differ from the tracked results in the last digits.
