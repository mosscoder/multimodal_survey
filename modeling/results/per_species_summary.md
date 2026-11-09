# Per-species results: ground vs. drone vs. fused

Source: `results/item2_ground_drone_fused_cv_vs_test.csv` (AP, test), `results/item6_leg_bootstrap_ci.csv` (CV AP + 95% CI), `results/item2_fusion_weights_final.csv` (fusion weights). Branch `paul_experiments`.

- **CV ± SE**: average precision, 5-fold cross-validation on the 6,089-frame training pool. SE derived from the leg-level 95% bootstrap CI as `(ci_hi - ci_lo) / (2*1.96)` (2000 resamples of the 38 survey legs — legs, not frames, because two photos of the same spot are near-duplicates).
- **Test**: average precision on the 1,613-frame held-out test split. Point estimate only — the test split has just 9 legs, too few to bootstrap.
- **w(ground)**: the fused model's per-species weight on the ground branch (drone gets `1 - w`), refit on the full training pool.

```
AP: CV (±SE) vs Test — Ground / Drone / Fused, plus fusion weight
                  |------ Ground ------|------- Drone -------|------ Fused -------|
Species           CV±SE         Test    CV±SE         Test    CV±SE         Test   w(ground)
Lupinus sericeus  0.946 ±0.018  0.865   0.725 ±0.078  0.592   0.946 ±0.018  0.873    0.95
Balsamorhiza      0.909 ±0.018  0.923   0.768 ±0.047  0.853   0.933 ±0.016  0.964    0.60
Gaillardia        0.878 ±0.033  0.903   0.393 ±0.097  0.103   0.884 ±0.028  0.887    0.75
Achillea          0.727 ±0.048  0.664   0.138 ±0.064  0.051   0.727 ±0.048  0.664    1.00
Euphorbia         0.407 ±0.108  0.773   0.103 ±0.061  0.254   0.398 ±0.125  0.778    0.90
Sisymbrium        0.860 ±0.021  0.643   0.541 ±0.059  0.198   0.864 ±0.019  0.649    0.95
Bromus tectorum   0.808 ±0.037  0.766   0.265 ±0.061  0.190   0.811 ±0.037  0.768    0.95
Poa bulbosa       0.933 ±0.011  0.937   0.720 ±0.033  0.742   0.931 ±0.011  0.935    0.95
---------------------------------------------------------------------------------------------
MACRO             0.809 ±0.015  0.809   0.457 ±0.023  0.373   0.812 ±0.017  0.815     --
```

## Which classes benefit from fusion

Fusion's overall macro gain (CV 0.809→0.812, Test 0.809→0.815) is small and driven almost entirely by one species.

- **Balsamorhiza sagittata — clear winner.** CV 0.909→0.933, Test 0.923→0.964. Drone alone is actually good here (0.77 CV / 0.85 test), so it earns real weight (60% ground / 40% drone).
- **Gaillardia aristata — mixed.** CV ticks up slightly, but Test dips (0.903→0.887) — drone is terrible on test for this species (0.10 AP), so its 25% weight doesn't generalize.
- **Lupinus, Poa, Bromus, Sisymbrium — basically flat** (±0.002–0.006). Weight stays ~95% ground; drone adds little.
- **Achillea millefolium — no change at all.** Weight = 100% ground — drone alone is too weak (0.14 AP) to ever get picked.
- **Euphorbia virgata — too noisy to call either way** (widest CI of any species, ±0.11–0.13).

**tl;dr: fusion only really helps where drone is independently decent (Balsamorhiza). Everywhere else, ground alone is carrying the result.**
