# DINOv3 linear-probe results

Frozen backbone, CLS (`pooler_output`) features, standardised, logistic regression (`class_weight=balanced`). Fit on `train`.

## species (multilabel, 8 classes)

| model          | image        | eval   |    n |   classes_scored |   macro_AP |   micro_AP |   macro_AUC |   micro_F1@0.5 |   accuracy |   balanced_acc |   macro_F1 |   qwk |
|:---------------|:-------------|:-------|-----:|-----------------:|-----------:|-----------:|------------:|---------------:|-----------:|---------------:|-----------:|------:|
| vitb_general   | ground_image | test   | 1613 |                8 |   0.488669 |   0.622687 |    0.773445 |       0.603648 |        nan |            nan |        nan |   nan |
| vitb_general   | ground_image | tier2  |  194 |                7 |   0.714104 |   0.687321 |    0.880395 |       0.626667 |        nan |            nan |        nan |   nan |
| vitb_general   | drone_image  | test   | 1613 |                8 |   0.33737  |   0.495089 |    0.726003 |       0.526857 |        nan |            nan |        nan |   nan |
| vitb_general   | drone_image  | tier2  |  194 |                7 |   0.546084 |   0.603674 |    0.86185  |       0.629956 |        nan |            nan |        nan |   nan |
| vitl_satellite | ground_image | test   | 1613 |                8 |   0.395078 |   0.5722   |    0.750555 |       0.58134  |        nan |            nan |        nan |   nan |
| vitl_satellite | ground_image | tier2  |  194 |                7 |   0.577399 |   0.5909   |    0.830576 |       0.592417 |        nan |            nan |        nan |   nan |
| vitl_satellite | drone_image  | test   | 1613 |                8 |   0.395649 |   0.521366 |    0.757896 |       0.551512 |        nan |            nan |        nan |   nan |
| vitl_satellite | drone_image  | tier2  |  194 |                7 |   0.616363 |   0.579855 |    0.856314 |       0.593968 |        nan |            nan |        nan |   nan |

## artifact_level (4 ordinal bands)

| model          | image        | eval   |    n |   classes_scored |   macro_AP |   micro_AP |   macro_AUC |   micro_F1@0.5 |   accuracy |   balanced_acc |   macro_F1 |       qwk |
|:---------------|:-------------|:-------|-----:|-----------------:|-----------:|-----------:|------------:|---------------:|-----------:|---------------:|-----------:|----------:|
| vitb_general   | ground_image | test   | 1613 |                4 |        nan |        nan |         nan |            nan |   0.882207 |       0.574775 |   0.573609 | 0.900637  |
| vitb_general   | ground_image | tier2  |  194 |                4 |        nan |        nan |         nan |            nan |   0.871134 |       0.609858 |   0.598007 | 0.872648  |
| vitb_general   | drone_image  | test   | 1613 |                4 |        nan |        nan |         nan |            nan |   0.477371 |       0.311341 |   0.270706 | 0.0680547 |
| vitb_general   | drone_image  | tier2  |  194 |                4 |        nan |        nan |         nan |            nan |   0.536082 |       0.355443 |   0.324317 | 0.0516392 |
| vitl_satellite | ground_image | test   | 1613 |                4 |        nan |        nan |         nan |            nan |   0.891507 |       0.612174 |   0.604029 | 0.90043   |
| vitl_satellite | ground_image | tier2  |  194 |                4 |        nan |        nan |         nan |            nan |   0.845361 |       0.512636 |   0.519852 | 0.830847  |
| vitl_satellite | drone_image  | test   | 1613 |                4 |        nan |        nan |         nan |            nan |   0.50651  |       0.290819 |   0.260248 | 0.0754669 |
| vitl_satellite | drone_image  | tier2  |  194 |                4 |        nan |        nan |         nan |            nan |   0.603093 |       0.296622 |   0.284656 | 0.0893405 |
