<p align="center">
  <img src="assets/multmodal_survey_logo.png" alt="multimodal_survey" width="400">
</p>

# MPG Ranch Multimodal Survey

[![Dataset on Hugging Face](https://img.shields.io/badge/dataset-Hugging%20Face-yellow.svg)](https://huggingface.co/datasets/mpg-ranch/multimodal-survey)
[![Data license CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code license MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

Monitoring vegetation change at the landscape scale is costly in wildland
settings due to factors such as rugged terrain and rarity of botanical
expertise. We release a dataset of paired ground robot and aerial drone imagery
to enable the study of the relative benefit of each platform for automated
vegetation monitoring at fine spatial scales. A Unitree Go2 quadruped robot
walked fixed routes through a grassland in Montana, USA, photographing the
ground ahead every meter, and a DJI Mavic 3M drone photographed the same ground
from above. Both platforms were RTK-enabled and covered every route twice,
capturing the same scenes under different lighting. Each of the 8,287 ground
images is paired with a 4 by 4 m drone image cropped from a 1.30 cm per pixel
orthomosaic at the robot's position. A botanist labeled every ground image for
eight target species, four natives and four invasive weeds. We established
baseline classifier performance with DINOv3 vision encoders, achieving a macro
average precision on the test split of 0.809 (ground), 0.373 (drone), and 0.815
(fusion, a weighted average of both predictions). We hope that others may use
our dataset to advance methods for automating vegetation surveys.

<p align="center">
  <img src="assets/sample_pairs_animation.gif" alt="Animated ground images beside their paired drone images along one leg" width="900">
</p>
<p align="center">
  <sub>A sequence of capture positions along one leg of the Go2 survey. The ground image is on the left. Its paired 4 by 4 m drone image is on the right, rotated so the robot's bearing points up.</sub>
</p>

## The dataset

The dataset is
[`mpg-ranch/multimodal-survey`](https://huggingface.co/datasets/mpg-ranch/multimodal-survey)
on Hugging Face, under CC BY 4.0. Each row pairs one ground image with the
drone image of the 4 by 4 m of ground ahead of the robot, and records which of
eight plant species are present in the ground image.

```python
from datasets import load_dataset

ds = load_dataset("mpg-ranch/multimodal-survey")
```

| | |
|---|---|
| Ground images | 8,287, each 1280 by 720 pixels |
| Drone images | One per ground image, 4 by 4 m at 1.30 cm per pixel, 308 by 308 pixels |
| Labels | Presence of eight species, four seeded natives and four invasive weeds, and an image quality level |
| Extent | Two sites and nine strips, each surveyed in two capture events, over four days in June 2026 |
| Splits | 6,480 training, 1,613 test, 194 tier 2 |
| Columns | 33 |

## Repository layout

```
multimodal_survey/
├── missions/              survey definitions and everything the robot recorded
│   ├── <site>/<strip>/    one folder per strip (configuration, runs, labels)
│   ├── inventory.csv      status of the runs
│   └── planning/          QGIS project and basemap used to lay out the strips
├── multimodal_dataset/    the pipeline that builds, verifies and uploads the dataset
├── modeling/              the baseline experiments
└── dev/                   three early test runs of one strip
```

### Modeling baselines

`modeling/` contains the code and the results of the baseline classifiers
reported with the dataset. One classifier reads the ground image and one reads
the drone image, each a linear layer on a frozen DINOv3 model, and a late
fusion combines the two. The [README of that folder](modeling/README.md) lists
the steps.

| Modality | Macro AP, test split |
|---|---|
| Drone image | 0.373 |
| Ground image | 0.809 |
| Fused | 0.815 |

Scores by species and the cross-validated scores with their confidence
intervals are in [`modeling/results/`](modeling/results).

### Data collection

A Unitree Go2 robot walked each strip under go2-survey, the mission software.
Each strip has a folder, `missions/<site>/<strip>/`. Its `mission.toml`
specifies the robot survey details, and its `runs/` folder contains the ground
images, logs and labels of each walk. See
[go2-survey](https://github.com/mosscoder/go2-survey) for details.

The species and quality labels were made with the app in
[`multimodal_dataset/labeling/`](multimodal_dataset/labeling/README.md).

## Status

The field campaign, the labeling and the dataset build are complete. The
dataset is private while the data descriptor is in preparation. It will become
public, with a DOI, when the descriptor is submitted.

## How to cite

A data descriptor is in preparation. Until it is published, please cite the
dataset.

> Doherty, K., Stam, N., Salakhutdinov, P., Samsoe, E. & Casper, C. MPG Ranch Multimodal Survey: paired ground-robot and drone imagery of Montana grassland. Hugging Face (2026). https://huggingface.co/datasets/mpg-ranch/multimodal-survey

## License

Code in this repository is MIT licensed (see [`LICENSE`](LICENSE)). The dataset
is released under CC BY 4.0.