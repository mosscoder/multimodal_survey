# Rebuild plan, October 2026: full coverage at site 1, drone_coverage retired

Written 2026-10-23. Kyle can recover the south-west corner of the site 1 orthomosaics, so the 33 drone images that were clipped at site 1 strip 5's western terminus will have full coverage, and the `drone_coverage` column (34th column, 1.0 on all but those 33 rows) can be dropped. This file is the checklist for the data side, the manuscript, the card and the overview. Local only until Kyle asks to push.

## 1. Data side, before any upload

1. **Re-export both site 1 orthomosaics** with the full extent: `260616-site-1-am-RGB.tif` (event 1) and `260618-site-1-pm-RGB.tif` (event 2). Keep everything else identical, 1.28 cm per pixel, EPSG:6514, the same GCP adjustment, so that Table 1's residuals and every existing drone image's pixels stay the same. Only the extent (and therefore the affine origin) changes; `crops.py` samples by coordinate and tolerates that, and it already rejects a rotated transform.
2. **Where to put them.** The pipeline fetches `https://storage.googleapis.com/mpg-aerial-survey/surveys/multimodal_survey/aerial/processing/drone_deploy/{flight}-RGB.tif` (`crops.py`, `ORTHO_URL`) into `multimodal_dataset/cache/{flight}-RGB.tif` and never re-downloads a file that exists. So: copy the current two files in the bucket to archive names first (for example `{flight}-RGB-clipped-export.tif`) so the published release stays reproducible, upload the full-extent files under the original names, then delete the two cached copies locally so the next build pulls the new ones. The two site 2 files are untouched.
3. **Pipeline commit** (`multimodal_dataset/`), one commit, in this order:
   - `schema.py`: remove `drone_coverage` from `COLUMNS` (34 to 33) and from the features.
   - `extract.py` line 92 and `build.py` line 61: stop writing the column. Keep `Ortho.sample` returning coverage and assert `coverage == 1.0` for every row at build time, raising with the frame id otherwise, so the guarantee is enforced rather than reported.
   - `verify.py`: delete the `clipped_frames`, `coverage_min` and "test coverage all 1.0" gates; add a gate that reads a build-time `out/coverage_check.json` (every row 1.0) or recomputes from the cached orthomosaics. Update the "34 columns in schema order" check to 33.
   - Bundle the other pending fixes so there is one rebuild: `along_leg_m` resume correction for the 31 rows with `mission_restart_utc` (CLAUDE.md) plus the gate that every capture position lies within 3 m of its planned mark; card and overview wording (level not band, ground image, drone image, AP not F1 as the suggested metric); `CITATION.cff` with the six authors in order.
4. **What changes and what does not.** Row count 8,287 and the three splits are unchanged. Keep the test leg of site 1 strip 5 at `leg13`; re-splitting would change every count in the paper and invalidate the fold file Paul's experiments use. The 33 affected drone images gain real pixels where they had fill; nothing else in any image changes. Pair, offset, tier 2 overlap and species counts are unchanged, so verify's other published numbers stand. Parquet shard sizes in Table 3 of the manuscript move slightly.
5. **Build, verify, push**, then re-run the Hugging Face spot check (first row group of each shard: 33 columns, image sizes) and record the new source commit in the card's provenance line.

## 2. Manuscript, while the data process

Deletions and edits that do not wait on numbers. Line numbers are those of main.tex at commit faae7f1.

- **Section 2.3 Drone flights, line 303.** Delete the passage from "orthomosaics, although source photographs covered that ground, for reasons we could not determine" through "(drone\_coverage column)". If the cause is now known (the export extent), one sentence may replace it, or nothing.
- **Fig. 2 caption, line 222.** "The test leg of site 1 strip 5 (S5) lies on its eastern side because the strip's western terminus runs off both orthomosaics." The split stays, so the reason has to become historical. Decision for Kyle: either "lies on its eastern side, a choice made before the strip's western terminus was recovered in the orthomosaics, and the published split is retained" or drop the clause and let the figure show it.
- **Section 2.7 Split construction, line 385.** "to ensure all test data had full aerial coverage (see Section 2.3)" needs the same treatment as the caption.
- **Data Records prose, line 410 area.** "described by 34 columns in all" becomes 33. Line 418: delete "with the drone image's coverage" from the column-group sentence.
- **Table 4.** Caption "The 34 columns" becomes 33. Delete the `drone\_coverage` row. Group title "Run and coverage" becomes "Run".
- **Table 3 (files).** Shard sizes change after the rebuild; mark the three size cells red until the upload.
- **Usage Notes 5.3.** Delete "a model of the drone image only rows whose window lies wholly on the orthomosaic" from the opening sentence and the sentence "drone\_coverage below 1 marks the 33 ground images whose drone image runs off the orthomosaic."
- **Terminology.** "coverage" as the orthomosaic's valid extent (CLAUDE.md table) is retired once the column goes; check that no sentence still uses it in that sense.
- **Annotated bibliography.** Add a dated note under the Data Records and Usage Notes entries that the coverage column and its numbers (33 rows, 0.078 minimum) were retired.

## 3. Card and overview, at the re-upload

- README YAML `features` block: drop `drone_coverage`.
- Column table row for `drone_coverage` (card line 206): delete.
- Section 3 "Splits": "The assignment was made on geometry and imagery coverage" becomes "on geometry". The subsection "Why site_1/strip_5 uses leg13" (card lines 312 to 336) and its figure `images/strip5_terminus.jpg` either shrink to the historical note chosen for the manuscript or go; if the figure goes, Table 3's "Seven figures" becomes six and the images/ size changes.
- Filtering examples in card section 5: remove any `drone_coverage` filter.
- `docs/dataset-overview.html`: lines 160, 184 to 194 (the clipped-frames passage and its figure caption) and the column table row, same edits as the card.
- CLAUDE.md: "34 columns" to 33, the terminology row, and the gates list.

## 4. Order of work

1. Manuscript deletions and the 33-column edits now (section 2 above), with Table 3 sizes and the Fig. 2 rationale decision marked.
2. Orthomosaic re-export and bucket upload, cache cleared.
3. Pipeline commit with all bundled fixes, build, verify.
4. Card and overview edits, push to Hugging Face, spot check.
5. Fill Table 3 sizes, update the provenance commit in the paper's Code Availability comment, clear the red marks.
