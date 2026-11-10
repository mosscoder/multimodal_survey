# Labeling app

The local web tool that produced the species and quality labels of the dataset.
Labeling is complete, and the app remains as a record of how the labels were
made. It needs only Python 3 and its standard library.

```bash
python -m multimodal_dataset.labeling label
```

The app opens on the species task. The task and the mission are chosen from the
dropdowns in the header, with no relaunch. The options `--mission` and `--run`
only pick which mission opens first.

## Species

A multilabel task. For each ground image the annotator checks which of the
eight target species are present. The checkboxes are grouped as wildflowers and
weeds, and nothing is checked by default, so the labels are human ground truth
made with no model predictions on screen. Keys `1` to `4` toggle the
wildflowers and `A`, `S`, `D` and `F` the weeds. Labels are saved to
`labels/species.json` in the run folder.

## Image quality

A single choice per ground image, the fraction of the image degraded by video
decoder artifacts, in four levels (0 to 25%, 25 to 50%, 50 to 75% and 75 to
100%). Keys `1` to `4` pick the level. The app stores the level as 1 to 4 in
`labels/quality.json`, and the released `artifact_level` is that integer minus
1, levels 0 to 3.

## Navigation

Each task keeps its own labels file, cursor and progress. Both save
automatically and can be resumed. The right arrow, space or enter confirms and
moves on, the left arrow goes back, `c` copies the labels of the previous
ground image, and `u` jumps to the next unreviewed one.
