---
license: cc-by-4.0
pretty_name: MPG Ranch Multimodal Survey
task_categories:
- image-classification
tags:
- biology
- ecology
- botany
- multimodal
- remote-sensing
- robotics
size_categories:
- 1K<n<10K
---

# MPG Ranch Multimodal Survey

A quadruped robot walked fixed routes through a grassland under restoration at
MPG Ranch, Montana, and photographed the vegetation every meter. A drone mapped
the same ground within a day of each walk. Each row of this dataset pairs one
ground image with the drone image of the 4 by 4 m of ground ahead of the
robot, and records which of eight plant species are present in the ground
image.

![Animated ground images beside their paired drone images along one leg](images/sample_pairs_animation.gif)

Thirty five consecutive capture positions along one leg. The ground image is
on the left. Its paired 4 m by 4 m drone image is on the right, rotated so the
robot's bearing points up, with the antenna position in yellow and a 1 m scale
bar.

A botanist labeled all 8,287 ground images for eight target species, four
seeded natives (silky lupine, arrowleaf balsamroot, blanketflower, common
yarrow) and four invasive weeds (leafy spurge, tall tumblemustard, cheatgrass,
bulbous bluegrass), and a second annotator graded each image's quality. The
surveys covered nine strips at two sites in June 2026, and every strip was
surveyed twice, from the ground and from the air, in two capture events.

A plant looks different from the ground and from the air, and the two views
differ in scale, since a drone maps a whole site in one flight and a ground
survey advances a meter at a time. The dataset supports comparing what each
view can identify species by species, training a model on drone images from
labels made on the ground, fusing the two views, and learning a representation
they share. Baseline linear classifiers on frozen DINOv3 features reach a
macro average precision on the test split of 0.815 with the two views fused,
0.809 from the ground image and 0.373 from the drone image. The code and
results are in the
[source repository](https://github.com/mosscoder/multimodal_survey/tree/main/modeling).

## Columns

Each of the 33 columns describes one ground image and its paired drone image.

| Column | Type | Description |
|---|---|---|
| `frame_id` | string | Unique identifier, the file name stem of the ground image |
| `ground_image` | Image | The ground image, a 1280 by 720 JPEG |
| `drone_image` | Image | The drone image, 4 by 4 m at 1.30 cm per pixel, 308 by 308 pixels, with the robot's antenna at the middle of the bottom edge and its bearing pointing up |
| `species` | Sequence(ClassLabel) | Class indices of the target species present, empty when none are present. Names are in [`species.csv`](species.csv) |
| `artifact_level` | ClassLabel(4) | Quality level 0 to 3, by the quarter of the ground image degraded by video decoder artifacts |
| `site` | string | `site_1` or `site_2` |
| `strip` | string | `strip_1` to `strip_5` within the site |
| `latitude` | float64 | WGS84 latitude of the robot's GNSS antenna at the shutter instant |
| `longitude` | float64 | WGS84 longitude of the antenna |
| `altitude_ellipsoidal_m` | float64 | Ellipsoidal height of the antenna |
| `accuracy_horizontal_m` | float64 | Receiver-reported horizontal accuracy, 1.4 to 1.7 cm |
| `accuracy_vertical_m` | float64 | Receiver-reported vertical accuracy, 1.0 to 1.7 cm |
| `fix_type` | int8 | RTK solution, 6 (fixed) on 8,285 ground images and 5 (float) on 2 |
| `pdop` | float64 | Position dilution of precision |
| `satellites_used` | int8 | Satellites in the solution |
| `bearing_deg` | float64 | Bearing of the robot in degrees clockwise from true north |
| `footprint_wkt` | string | The 4 by 4 m footprint of the drone image as a WGS84 polygon |
| `ground_captured_at` | string | Shutter instant of the ground image, ISO 8601 |
| `drone_captured_at` | string | Median capture time of the matched drone flight |
| `capture_offset_absolute` | float64 | Hours from the flight's median capture to the ground image, median 1.16 and at most 20.29 |
| `capture_offset_time_of_day` | float64 | The same offset by time of day alone |
| `capture_event` | int8 | First or second capture event of the strip, 1 or 2 |
| `mission_restart_utc` | string, nullable | Resume time of the one interrupted run, set only on the 31 ground images captured after it |
| `pair_frame_id` | string, nullable | `frame_id` of the ground image at the same planned position in the strip's other capture event, null for the 153 without one |
| `pair_distance_m` | float64, nullable | Distance between the two paired antenna positions, median 0.10 m |
| `pair_overlap_fraction` | float64, nullable | Fraction of the footprint covered by the paired row's footprint |
| `pair_offset_hours` | float64, nullable | Hours between the two paired ground images |
| `leg` | string | `leg01` to `leg13`, odd numbers for the straight lines of a strip and even numbers for the turns between them |
| `along_leg_m` | float64 | Nominal distance along the leg, 0 to 150 m |
| `frame_offset_s` | float64 | Shutter instant relative to reaching the planned position |
| `speed_mps` | float64 | Robot speed at the shutter |
| `run_id` | string | The robot run, one per strip and capture event |
| `tier2_overlap_fraction` | float64, nullable | Fraction of the footprint covered by test footprints, `train_tier_2` rows only |

## How to load

```python
# pip install datasets
from datasets import load_dataset

ds = load_dataset("mpg-ranch/multimodal-survey")
train, test = ds["train"], ds["test"]

row = train[69]
row["ground_image"]   # PIL.Image, 1280x720
row["drone_image"]    # PIL.Image, 308x308, bearing up
row["species"]        # [0, 7], class indices

names = train.features["species"].feature.names
[names[i] for i in row["species"]]   # ['Lupinus sericeus', 'Poa bulbosa']
```

The dataset has three splits. `test` (1,613 rows) is the outermost leg of
every strip. `train` (6,480 rows) is the rest, apart from the 194 rows of
`train_tier_2`, whose drone image overlaps a test drone image. Leave tier 2
out for a training split that has no ground in common with the test split in
the drone images, or add it for more training data.

## How to cite

A data descriptor is in preparation. Until it is published, please cite the
dataset.

> Doherty, K. et al. MPG Ranch Multimodal Survey: paired ground-robot and drone imagery of Montana grassland. Hugging Face (2026). https://huggingface.co/datasets/mpg-ranch/multimodal-survey

[`CITATION.cff`](CITATION.cff) gives the same reference in machine-readable
form. The data are released under CC BY 4.0. The
[source repository](https://github.com/mosscoder/multimodal_survey) describes
how the data were collected and contains the code, under the MIT license, that
built this release from commit `SOURCE_COMMIT`.
