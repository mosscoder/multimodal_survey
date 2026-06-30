# Research log — free-data → robot species detection (MIL)

Dated entries, one per tuning stage (`sweeps/sweep_mil.py --stage ...`). Each stage tunes **one
axis** while reusing the prior stages' winners (carried in `sweeps/sweep_state.json`). The winner
is the highest robot-**val** all-9 macro-F1; we also track the well-supported-only macro and
**flag** when it would have picked a different winner.

**Shared setup** (unless a stage says otherwise): frozen DINOv3 ViT-B/16; MIL head
(`LayerNorm → [hidden] → Linear(9)`) with log-sum-exp bag pooling, one-vs-all BCE, class-balanced
pos-weights; **iNat train images only**; robot frames split 80/20 stratified (**val 426 / test
107**, seed 1312); **any-patch** eval with one frozen threshold per species (argmax val-F1 per
epoch, best epoch frozen). "Well-supported" = **val** support ≥ 10, which is 8/9 species (all but
*Sisymbrium altissimum*); the **test**-time measurable macro (CLAUDE.md headline's companion)
uses test support ≥ 10 = 5/9 (Lupinus, Poa, Gaillardia, Tragopogon, Thinopyrum).

---

## 2026-06-25 — Stage 1: hidden-dim sweep

**Objective.** With LR fixed at 1e-4 and the simplest crop (one native 224 center tile, 256
imgs/species), choose the MIL head width. Candidates: 0 (linear probe), 32, 64, 128, 256.

**Result.** Winner **hidden = 64**, val all-9 macro-F1 **0.608**. All-9 and well-supported
(≥10) rankings agree → no disagreement flag.

| hidden | val all-9 | val meas (≥10) | best epoch |
|-------:|----------:|---------------:|-----------:|
| 0      | 0.505     | 0.558          | 79 (cap)   |
| 32     | 0.542     | 0.574          | 32         |
| **64** | **0.608** | **0.631**      | 35         |
| 128    | 0.602     | 0.606          | 50         |
| 256    | 0.597     | 0.610          | 36         |

**Positive.**
- 64 is the clear best on *both* metrics; reproduces the prior "width 64 settled" finding from
  a clean, isolated sweep.
- The hidden layer earns its keep — the linear probe (hidden 0) trails by ~10 F1 points (0.505),
  so the non-linearity is doing real work, not just adding parameters.

**Negative.**
- No payoff from more capacity: 128 and 256 both land *below* 64 (mild overfit / noise at only
  256 imgs/species).
- hidden 0 never early-stopped (ran to the 80-epoch cap) — underfit; a longer schedule might
  nudge it up, but it's far enough back that the ranking wouldn't change.

**Carried forward:** hidden = 64.

---

## 2026-06-25 — Stage 2: learning-rate sweep

**Objective.** With hidden = 64 (Stage 1) and the same native-224-center crop / 256 imgs/species,
choose the Adam learning rate. Candidates: 1e-5, 5e-5, 1e-4, 5e-4, 1e-3.

**Result.** Winner **LR = 5e-5**, val all-9 macro-F1 **0.613**, edging the prior default 1e-4
(0.608). All-9 and well-supported rankings agree → no flag.

| LR       | val all-9 | val meas (≥10) | best epoch |
|---------:|----------:|---------------:|-----------:|
| 1e-5     | 0.571     | 0.584          | 79 (cap)   |
| **5e-5** | **0.613** | **0.636**      | 37         |
| 1e-4     | 0.608     | 0.631          | 35         |
| 5e-4     | 0.606     | 0.619          | 14         |
| 1e-3     | 0.604     | 0.617          | 9          |

**Positive.**
- Determinism check passed: LR 1e-4 reproduced Stage 1's hidden-64 score to the digit (0.6082),
  so stage-to-stage comparisons are apples-to-apples.
- Broad, forgiving plateau from 5e-5 to 1e-3 (0.604–0.613) — the head is not LR-fragile.

**Negative.**
- The win over 1e-4 is tiny (+0.0045 all-9) — within run-to-run noise; treat "5e-5" as a soft
  preference, not a sharp optimum. (Nudges CLAUDE.md's "LR 1e-4 settled" note — flagging, not
  rewriting the spec.)
- 1e-5 is too slow — underfit, ran to the 80-epoch cap (0.571).
- The two largest LRs early-stop fast (epochs 14, 9) at slightly lower scores — quick convergence
  to a marginally worse basin.

**Carried forward:** hidden = 64, LR = 5e-5.

---

## 2026-06-25 — Stage 3: crop / zoom sweep

**Objective.** With hidden = 64, LR = 5e-5, 256 imgs/species, test the two-output crop: each
image becomes a **square** view (short-side crop → 224, whole plant) **plus** a **native zoom**
view (resize whole image by z, native 224 center). Candidates z ∈ {0.5, 0.75, 1.0}.

**Result.** Winner **square + native z = 1.0**, val all-9 macro-F1 **0.634**. All-9 and
well-supported rankings agree → no flag.

| crop (square + native z) | val all-9 | val meas (≥10) | best epoch |
|:-------------------------|----------:|---------------:|-----------:|
| z = 0.5                  | 0.625     | 0.626          | 16         |
| z = 0.75                 | 0.631     | 0.644          | 25         |
| **z = 1.0**              | **0.634** | **0.651**      | 25         |

**Positive.**
- **The two-view crop is a real win.** Adding the square (whole-plant) view to the native center
  crop lifts val all-9 from **0.613 → 0.634** (+0.021) and well-supported from 0.636 → 0.651,
  vs. the single native view at the same hidden/LR (Stage 2). The whole-plant and native-detail
  views carry complementary signal.
- Clean monotonic ranking (0.5 < 0.75 < 1.0) with fast convergence (ep 16–25).

**Negative.**
- The "distant-plant" intuition did **not** pan out: zooming the native view *out* (z < 1, subject
  smaller / more surrounding context) was *worse* — the model prefers the highest-detail native
  center crop (z = 1.0) beside the square view.
- The spread across zoom levels is small (~0.009 all-9); the lever was *adding* the square view at
  all, not the zoom factor.

**Carried forward:** hidden = 64, LR = 5e-5, crop = square + native(1.0).

---

## 2026-06-25 — Stage 4: training-set size sweep

**Objective.** With hidden = 64, LR = 5e-5, crop = square + native(1.0), vary the number of iNat
images per species: 64, 128, 256, 512 (features extracted once at 512, then subset — nested).

**Result.** Winner by the selection rule (all-9) is **512 imgs/species** (val all-9 **0.636**) —
but this is a **flagged disagreement**: the well-supported (≥10) metric peaks at **128** (0.656).
The all-9 scores for 128/256/512 are within noise.

| imgs/species          | val all-9 | val meas (≥10) | best epoch |
|----------------------:|----------:|---------------:|-----------:|
| 64                    | 0.616     | 0.626          | 28         |
| 128                   | 0.634     | **0.656**      | 59         |
| 256                   | 0.634     | 0.651          | 25         |
| **512** (rule winner) | **0.636** | 0.653          | 13         |

**Positive.**
- Data clearly helps up to ~128/species: 64 → 128 adds +0.018 all-9 and +0.030 well-supported.
- More data → faster convergence (512 best-epoch 13 vs. 128's 59).

**Negative / flag.**
- **Saturation by 128.** Past 128/species the all-9 curve is flat (0.634 → 0.634 → 0.636, noise)
  and well-supported *declines* slightly (0.656 → 0.651 → 0.653). 4× the data (512 vs 128) buys
  essentially nothing.
- **All-9 vs well-supported disagree** — the flag this run was designed to catch. The rule (all-9)
  picks 512; well-supported picks 128. 512's +0.0018 all-9 edge over 128 is within noise, so the
  honest read is "128 ≈ 512", with 128 *better* on the well-supported species and 4× cheaper.

**Carried forward (per rule + "retain best count"):** hidden = 64, LR = 5e-5,
crop = square + native(1.0), **n_per = 512**. *(If optimizing the well-supported species / compute
matters more, 128 is the equal-or-better alternative.)*

---

## 2026-06-25 — Final full training run

**Config (all sweep winners).** hidden = 64 · LR = 5e-5 · crop = square + native(1.0) ·
512 imgs/species · best epoch 13. Saved to `checkpoints/mil.pt` (+ `mil_metrics.json`) with the
frozen per-species thresholds.

**Headline.** Held-out **TEST macro-F1 (all 9) = 0.603**; well-supported (test sup ≥ 10 = 5
species) = **0.628**. Val (n=426): all-9 0.636 / well-supported 0.653. Val→test gap ≈ 0.03.

| species (TEST, n=107)  |   τ  | prec | rec  |  F1  | sup |
|:-----------------------|-----:|-----:|-----:|-----:|----:|
| Thinopyrum intermedium | 0.02 | 1.00 | 1.00 | 1.00 | 107 |
| Lupinus sericeus       | 0.41 | 0.84 | 0.78 | 0.81 |  72 |
| Poa bulbosa            | 0.49 | 0.52 | 0.98 | 0.68 |  56 |
| Balsamorhiza sagittata | 0.67 | 1.00 | 0.50 | 0.67 |   6 |
| Achillea millefolium   | 0.52 | 0.50 | 1.00 | 0.67 |   2 |
| Sisymbrium altissimum  | 0.76 | 1.00 | 0.50 | 0.67 |   2 |
| Gaillardia aristata    | 0.47 | 0.60 | 0.35 | 0.44 |  17 |
| Bromus tectorum        | 0.77 | 0.50 | 0.20 | 0.29 |   5 |
| Tragopogon dubius      | 0.72 | 0.50 | 0.13 | 0.21 |  15 |

**Positive.**
- **Lupinus sericeus** (flagship forb): test F1 **0.81** (P 0.84 / R 0.78) — the demo's headline
  detection is strong.
- **Thinopyrum** perfect (1.00) — present in every frame (matrix grass), τ 0.02 = "fires
  everywhere", which is correct for this site.
- **Balsamorhiza** (third painted forb) holds P 1.00 / R 0.50 → F1 0.67 even at tiny support.
- End-to-end the staged tuning lifted val all-9 0.608 → 0.636; breakdown: LR +0.004, **crop
  (adding the square view) +0.021**, size +0.002. The crop was the decisive lever.

**Negative.**
- **Tragopogon** and **Bromus** weak on test (F1 0.21, 0.29): high frozen thresholds (0.72, 0.77)
  + low recall. Tragopogon's recall collapses val→test (0.40 → 0.13) — its threshold is overfit
  to val; the worst generalization gap.
- **Poa bulbosa** over-fires: recall 0.98 / precision 0.52 — lights up on general grass
  (consistent with the known grass over-prediction).
- Rare species (Achillea, Sisymbrium, Balsamorhiza, Bromus; test sup 2–6) have noise-dominated
  F1s; the all-9 test macro (0.603) is held down mainly by Tragopogon / Bromus / Gaillardia.

**Note.** Used 512 imgs/species per the all-9 rule, but Stage 4 showed saturation by 128
(128 ≈ 512 on all-9, 128 *better* on well-supported) — a 128/species final would be ~equal and
4× cheaper.

---

## 2026-06-25 — Follow-up ablation: super-patch token pooling

**Purpose.** The painted patches cover a tiny fraction of each plant. Test whether scoring larger
units — "super-patches" formed by pooling neighborhoods of DINOv3 tokens — lets the head key on
whole-plant cues instead of 16 px fragments.

**Design & justification.** Inference-only on the deployed head (`mil.pt`) + cached native tiling;
adaptive-average-pool each 14×14 tile token grid to g×g (g ∈ {14,7,4,2,1} → ~16/32/56/112/224 px
units; g=1 = whole-tile mean), any-patch max over super-patches, thresholds re-derived on val. No
retraining — a cheap probe of whether pooling helps *at all* before paying to retrain on pooled
tokens. Deliberate, documented limitation: the head was trained on *individual* tokens, so averaged
tokens are off-distribution for it. g=14 is the control. (`followup_ablations/superpatch_pool.py`)

| g (tokens/tile) | super-patch px | val all-9 | val meas | test all-9 | test meas |
|----------------:|---------------:|----------:|---------:|-----------:|----------:|
| **14** (control)| 16             | **0.636** | 0.653    | **0.603**  | 0.628     |
| 7               | 32             | 0.595     | 0.612    | 0.568      | 0.587     |
| 4               | 56             | 0.491     | 0.522    | 0.483      | 0.602     |
| 2               | 112            | 0.448     | 0.485    | 0.441      | 0.620     |
| 1               | 224            | 0.426     | 0.463    | 0.385      | 0.612     |

**Positive.**
- Control (g=14) reproduces the deployed numbers exactly (val 0.636 / test 0.603) — the threshold
  re-derivation protocol is sound, so the deltas are real, not artifacts.

**Negative (the result).**
- Pooling **monotonically hurts**: val all-9 0.636 → 0.595 → 0.491 → 0.448 → 0.426; test 0.603 →
  0.385. Bigger super-patches = strictly worse.
- Cause is the documented mismatch: averaging blurs the localized, discriminative signal the head
  relies on, and a head trained on single tokens cannot read mean-pooled tokens. (Test "measurable"
  stays ~0.60 throughout — an artifact of tiny test supports + per-config threshold re-derivation;
  the clean val signal is the monotonic drop.)

**Takeaway.** Super-patches as *post-hoc* pooling on the existing head are a dead end — the
diagnostic content is spatially sharp and pooling destroys it. The idea is only viable if the head
is *retrained* on pooled tokens (so they're in-distribution), which is a separate experiment, not
an inference toggle.

---

## 2026-06-25 — Follow-up ablation: multi-scale inference tiling

**Purpose.** Inference tiles robot frames at native resolution only (16 px/patch), while training
also showed the head a whole-plant "square" scale. Test whether adding coarser inference passes
(each patch spanning a bigger chunk of plant) improves detection.

**Design & justification.** Inference-only on the deployed head (`mil.pt`); no retraining — the head
already learned a coarse scale from the square view, so this isolates the *tiling*. Scales {1,2,4}:
s=1 is the canonical native pass (cached); s>1 downsamples the frame by s, reflect-pads to ×224,
tiles, encodes — a 16 px patch then spans s·16 px (s=2→32, s=4→64 ≈ square-view scale). Fuse by the
any-patch rule across scales (per-species max patch-sigmoid over the union). Thresholds re-derived
on val, applied to test; {1} is the control. (`followup_ablations/multiscale_infer.py`)

| scales      | val all-9 | val meas | test all-9 | test meas |
|:------------|----------:|---------:|-----------:|----------:|
| {1} control | 0.636     | 0.653    | 0.603      | 0.628     |
| **{1,2}**   | **0.645** | **0.664**| **0.638**  | 0.628     |
| {1,4}       | 0.635     | 0.652    | 0.603      | 0.628     |
| {1,2,4}     | 0.645     | 0.663    | 0.638      | 0.628     |

**Positive (the result).**
- Adding the 2× pass helps: **val all-9 0.636 → 0.645** (+0.009), **val well-supported 0.653 →
  0.664** (+0.011), and a larger **test all-9 0.603 → 0.638** (+0.035). Control reproduces deployed
  exactly (protocol sound).
- The gain is concentrated in **Bromus tectorum** (cheatgrass): test F1 **0.29 → 0.60** (recall
  0.20 → 0.60), all other species unchanged. A wispy grass benefits from a coarser 32 px patch that
  captures a whole tuft rather than a 16 px sliver — exactly the "bigger chunk of plant" effect we
  were after, and the val-measurable gain confirms it isn't only a test fluke.

**Negative / caveats.**
- **4× is useless**: {1,4} ≈ control and {1,2,4} ≈ {1,2} — 64 px patches add no signal (too coarse /
  redundant). The "match the ~55 px square-view scale" rationale was wrong; the useful coarse scale
  is 2× (32 px), not 4×.
- The **forbs and high-support classes don't move** — multi-scale helps a fine grass, not the
  headline forbs. Test "measurable" is flat (Bromus has test sup 5, so the *test* gain rides on a
  low-support class; the cleaner evidence is val-measurable +0.011, where Bromus has sup 19).
- Genuine pipeline change, not a free toggle: deploying {1,2} means re-freezing thresholds and
  switching infer/render to multi-scale tiling.

**Takeaway.** Multi-scale {1,2} is a real, cheap win (~+0.01 val / +0.035 test all-9), mostly via
Bromus; 4× not worth keeping. Worth deploying if we re-freeze thresholds.

---

# 2026-06-25 (rev 2) — Restructured chain: species-first + r sweep

Chain reordered per request. A **species ablation runs first** — does training on extra HF-dataset
species (as hard-negative / auxiliary one-vs-all classes) help the 9-target robot detection? — then
**hidden → LR → crop → size are re-run** on the winning species set, an **LSE bag-pooling
temperature `r` sweep** is added **last**, and finally the full run. Extra species are scored only on
the 9 targets (kept first, so target indices 0–8 are stable). The entries below **supersede the
first-pass chain above** (targets-only, r=8); the `targets` candidate in the species stage is a
back-compat check and should reproduce the first-pass hidden-stage number (0.608 at hidden 64 /
LR 1e-4 / native / 256).

---

## 2026-06-25 — Stage: species ablation (runs first)

**Purpose.** The deployed head only ever saw the other 8 targets as negatives — never soil, litter,
or non-target vegetation — a likely root of the Poa/grass over-fire. Test whether training on extra
HF-dataset species (as hard-negative / auxiliary one-vs-all classes, scored only on the 9 targets)
improves 9-target robot detection.

**Design & justification.** Three sets at the first-pass defaults (hidden 64 / LR 1e-4 / native-224
crop / 256 imgs/species / r 8): `targets` (9), `grasses` (9 + look-alike grasses Poa secunda /
Pseudoroegneria spicata / Hesperostipa comata), `all` (36 = every HF species). Targets kept first so
the metric slices the 9 target columns; extras add output dims and act as negatives via one-vs-all
BCE. Per-species feature caching lets the sets share extractions.

| species set | classes | val all-9 | val meas (≥10) | best epoch |
|:------------|--------:|----------:|---------------:|-----------:|
| targets     | 9       | 0.608     | 0.631          | 35         |
| grasses     | 12      | 0.616     | 0.639          | 42         |
| **all**     | 36      | **0.628** | **0.654**      | 14         |

**Positive (the result).**
- More species help, monotonically: 0.608 → 0.616 → **0.628** val all-9 (+0.020 over targets;
  well-supported 0.631 → 0.654). Winner **all (36)**; both metrics agree.
- Not just grass look-alikes — the full set beats grass-only by +0.012, so broad negatives
  (forbs/weeds + grasses) shape a better target-vs-everything boundary than grass negatives alone.
  Directly supports "teach it what isn't a target."
- Back-compat check passes: `targets` reproduces the first-pass hidden-stage number to the digit
  (0.6082) — the refactor is sound and rev-2 numbers are comparable to rev-1.
- all-36 converges fastest (ep 14) — richer per-epoch gradient.

**Negative / cost.**
- Modest absolute gain (+0.020 val all-9) for a 4× wider head (36 vs 9 outputs) and ~4× the iNat
  extraction across the rest of the chain (all 36 species now flow through crop/size/r).
- We don't know *which* extras carry the signal — the 24 non-grass species could likely be pruned to
  a cheaper subset; `all` is the safe upper bound, not necessarily the efficient choice (a natural
  follow-up ablation).

**Carried forward:** species = all (36).

---

## 2026-06-25 — Stage: hidden-dim (re-run on all-36 species)

**Objective.** Re-tune head width on the winning species set (all 36); LR 1e-4 / native crop / 256
imgs/species / r 8. Candidates 0, 32, 64, 128, 256.

**Result (FLAGGED disagreement).** All-9 rule → **hidden 256** (0.633); well-supported → **hidden 64**
(0.654). The all-9 curve is noisy/non-monotonic — 0/64/256 ≈ 0.625/0.628/0.633 (a tie); 32 and 128
dip.

| hidden | val all-9 | val meas (≥10) | best epoch |
|-------:|----------:|---------------:|-----------:|
| 0      | 0.625     | 0.648          | 79 (cap)   |
| 32     | 0.588     | 0.620          | 66         |
| 64     | 0.628     | **0.654**      | 14         |
| 128    | 0.607     | 0.632          | 38         |
| **256**| **0.633** | 0.645          | 42         |

**Positive.**
- Adding all-36 species lifted the whole curve vs rev-1 (hidden 0: 0.505 → 0.625; hidden 64: 0.608 →
  0.628) — the auxiliary species help at every width, biggest for the linear probe (the negatives do
  much of the work the hidden layer did before).

**Negative / flag.**
- **All-9 vs well-supported disagree:** the rule picks 256 (+0.004 all-9 over 64 — within noise), but
  64 wins well-supported by +0.009, is 4× smaller, and was the rev-1 winner. The hidden choice is
  effectively a 0/64/256 tie on a noisy curve (32/128 dip; best-epochs swing 14–79 → unstable
  optimization with 36 classes).
- Per the agreed rule (all-9 decides, flag disagreement) I carry **256**; 64 is the robust /
  parsimonious alternative, trivial to re-pin.

**Carried forward (per rule):** hidden = 256.  *(64 is the well-supported / parsimony pick.)*

---

## 2026-06-25 — Stage: learning-rate (re-run; species all, hidden 256)

**Objective.** Re-tune LR with species=all, hidden=256, native crop, 256/species, r 8.

**Result.** Winner **LR = 1e-4** (val all-9 0.633); both metrics agree, no flag. (rev-1 picked 5e-5
at hidden 64 — the optimum shifted back to 1e-4 for the wider 256 head.)

| LR       | val all-9 | val meas | best epoch |
|---------:|----------:|---------:|-----------:|
| 1e-5     | 0.601     | 0.629    | 76         |
| 5e-5     | 0.631     | 0.643    | 48         |
| **1e-4** | **0.633** | **0.645**| 42         |
| 5e-4     | 0.626     | 0.642    | 13         |
| 1e-3     | 0.624     | 0.635    | 4          |

**Positive.** Clean inverted-U, peak at 1e-4; 5e-5 close (0.631). Stable, forgiving plateau.
**Negative.** Only 1e-5 clearly underfits; nothing surprising.

**Carried forward:** species all, hidden 256, LR 1e-4.

---

## 2026-06-25 — Stage: crop (re-run; species all, hidden 256, LR 1e-4)

**Objective.** Re-test square + native-zoom(z) crop, z ∈ {0.5, 0.75, 1.0}, on all-36 / hidden 256 /
LR 1e-4 / 256 imgs / r 8.

**Result.** Winner **square + native z=1.0** (val all-9 0.650); both metrics agree, monotonic.

| crop (square + native z) | val all-9 | val meas | best epoch |
|:-------------------------|----------:|---------:|-----------:|
| z = 0.5                  | 0.635     | 0.647    | 8          |
| z = 0.75                 | 0.642     | 0.651    | 27         |
| **z = 1.0**              | **0.650** | **0.660**| 29         |

**Positive.** Adding the square view lifts native-only 0.633 → **0.650** (+0.017) — the square-view
gain reproduces on all-36 (rev-1 was +0.021). z=1.0 wins again (highest native detail pairs best with
the whole-plant view); clean monotonic 0.5 < 0.75 < 1.0.
**Negative.** Zoom-out (z<1) still doesn't help — same as rev-1.

**Carried forward:** species all, hidden 256, LR 1e-4, crop square+native(1.0).

---

## 2026-06-25 — Stage: training-set size (re-run; species all, hidden 256, LR 1e-4, crop square+native1.0)

**Objective.** Vary imgs/species {64,128,256,512} on the full winning config (features extracted once
at 512, subset down).

**Result.** Winner **512** (val all-9 0.651); both metrics agree — NO disagreement (unlike rev-1).

| imgs/species | val all-9 | val meas | best epoch |
|-------------:|----------:|---------:|-----------:|
| 64           | 0.629     | 0.645    | 8          |
| 128          | 0.644     | 0.666    | 9          |
| 256          | 0.650     | 0.660    | 29         |
| **512**      | **0.651** | **0.674**| 6          |

**Positive.** 512 wins both metrics cleanly. With all-36 species the well-supported macro keeps
climbing with data (0.660 → 0.674 at 512) — unlike rev-1 (targets-only), where 128 ≈ 512 and the two
metrics disagreed. Extra species + more data per species compound.
**Negative.** all-9 still nearly saturates after 256 (256 → 512 only +0.001); the 512 win is carried
by the well-supported classes (meas +0.015) — 4× the data over 128 for a tiny all-9 gain.

**Carried forward:** species all, hidden 256, LR 1e-4, crop square+native(1.0), n_per 512.

---

## 2026-06-25 — Stage: LSE temperature r (last sweep)

**Objective.** Tune bag-pooling sharpness r ∈ {2,4,8,16,32} on the full winning config. Low r =
average-like (broad agreement); high r = max-like (strict any-patch).

**Result.** Winner **r = 32** (val all-9 0.652); both metrics agree, monotonic.

| r     | val all-9 | val meas | best epoch |
|------:|----------:|---------:|-----------:|
| 2     | 0.642     | 0.664    | 9          |
| 4     | 0.649     | 0.672    | 6          |
| 8     | 0.651     | 0.674    | 6          |
| 16    | 0.652     | 0.675    | 6          |
| **32**| **0.652** | **0.676**| 7          |

**Positive.** Peakier pooling is better, monotonically — the model prefers strict "any-patch" (a few
confident patches) over broad agreement. Echoes the super-patch ablation: the discriminative signal
is localized, and diluting/averaging it hurts (r=2 worst at 0.642). The sparse-firing regime is
*preferred* by the metric, not a bug.
**Negative.** Strongly diminishing returns: 8 → 16 → 32 gains ~+0.0006 each; r=32 beats the default
r=8 by only +0.0012. r=32 is the edge of the range but the curve has plateaued — higher r approaches a
hard max with negligible gain and worse trainability, not worth chasing.

**Carried forward:** species all, hidden 256, LR 1e-4, crop square+native(1.0), n_per 512, r 32.

---

## 2026-06-25 — Final run (rev-2 chain)

**Config.** species all (36) · hidden 256 · LR 1e-4 · crop square+native(1.0) · 512 imgs/species ·
r 32 · best epoch 7. Saved `checkpoints/mil.pt` (36-output head, scored on the 9 targets) + metrics.

**Headline — mixed, read carefully.**
- Well-supported (test sup ≥ 10, 5 species): **0.628 → 0.705 (+0.077)** — large, trustworthy gain.
- All-9 test: **0.603 → 0.537 (−0.066)** — but this is **rare-class threshold noise**, not a real
  regression (below). Val: all-9 0.636 → **0.652**, well-supported 0.653 → **0.676**.

Per-species TEST F1 (rev-1 → rev-2):

| species (sup)            | rev-1 | rev-2 |          |
|:-------------------------|------:|------:|:---------|
| Thinopyrum (107)         | 1.00  | 1.00  | =        |
| Lupinus (72)             | 0.81  | 0.83  | +        |
| Poa (56)                 | 0.68  | 0.65  | −        |
| Gaillardia (17)          | 0.44  | 0.71  | **+0.27**|
| Tragopogon (15)          | 0.21  | 0.33  | **+0.12**|
| Balsamorhiza (6)         | 0.67  | 0.50  | − (rare) |
| Bromus (5)               | 0.29  | 0.31  | + (rare) |
| Achillea (2)             | 0.67  | 0.00  | − (rare) |
| Sisymbrium (2)           | 0.67  | 0.50  | − (rare) |

**Positive (the real result).**
- The species ablation + chain substantially improved the *measurable* species — Gaillardia
  0.44 → 0.71, Tragopogon 0.21 → 0.33, Lupinus 0.81 → 0.83 — lifting the test well-supported macro
  0.628 → **0.705**. That is the trustworthy headline.

**Negative / important caveat.**
- All-9 test fell only because the **rare classes collapsed on tiny test sets**: Achillea
  0.67 → 0.00 (test sup **2**; a 0.05 τ shift 0.52 → 0.57 flipped both examples), Sisymbrium
  0.67 → 0.50 (sup 2), Balsamorhiza 0.67 → 0.50 (sup 6). These four classes (test sup 2–6) dominate
  the all-9 macro and are pure noise — the all-9 *test* number is not a trustworthy comparison at this
  support.
- Root cause is **threshold overfitting**: thresholds are tuned on val (sup 9–24 for these) and
  crater on 2–6-example test sets. Directly motivates a threshold-robustness pass (bootstrap /
  shrinkage), the cheap next experiment.

**Verdict.** rev-2 is the better model where we can measure reliably (+0.077 well-supported, big
Gaillardia/Tragopogon gains); the all-9 test drop is rare-class threshold noise on ≤6 examples, not a
capability regression.

---

# 2026-06-25 (rev 3) — full iNat data, 50/50 split, larger sweeps, rare-class-free selection

Per request: (1) train on the FULL iNaturalist dataset per species (train+test, ~2500/species;
82,413 total); (2) robot val/test 80/20 -> 50/50 (val 267 / test 266, seed 1312); (3) size sweep
{64,128,256,512,1024,2048}; (4) r sweep {2,4,8,16,32,64,128}; (5) **sweep SELECTION drops the two
noise-prone rare species** -- Achillea millefolium (yarrow) + Sisymbrium altissimum (tumble mustard).
Every stage selects on **`sel7`** = macro-F1 over the 7 remaining species (Lupinus, Poa, Tragopogon,
Gaillardia, Balsamorhiza, Bromus, Thinopyrum); yarrow + tumble mustard are **reported only on the
final test** (all 9), never used to pick hyperparameters. `all9` is recorded for reference.

Methodology: robot-test untouched during sweeps -- every stage selects on val `sel7`; test is scored
once, in the final run, over all 9. iNat caches (split/selection-independent) are reused. Supersedes
rev-2 and the earlier rev-3 pass (which selected on all-9).

---

## 2026-06-25 — rev-3 species ablation (sel7 selection, full data, 50/50)

**Objective.** Pick the species set selecting on **sel7** (7 species; yarrow + tumble mustard
excluded). Fixed hidden 64 / LR 1e-4 / native crop / 256 imgs / r 8.

**Result.** Winner **all (36)**, val sel7 **0.721** (+0.022 over targets/grasses). No flag —
selection is now the stable 7-species macro.

| species set | classes | val sel7 | all9 (ref) | best epoch |
|:------------|--------:|---------:|-----------:|-----------:|
| targets     | 9       | 0.699    | 0.604      | 23         |
| grasses     | 12      | 0.699    | 0.631      | 46         |
| **all**     | 36      | **0.721**| 0.603      | 10         |

**Positive.**
- **all (36) wins outright on sel7** — the full set of 27 non-target negatives lifts the 7
  well-sampled species by +0.022. The grasses-vs-all fork is resolved: with the 2 noise-prone rare
  classes out of selection, `all` is the clear pick (matching its rev-2 measurable win).
- The grass look-alikes *alone* don't move sel7 (targets ≈ grasses, 0.699) — it's the full negative
  set that helps the well-sampled species.

**Negative.**
- On the *reference* all9, `all` is worst (0.603) — but that's precisely the rare-class noise we
  excluded; all9 here is dragged down by yarrow/tumble (reported at final, not selected on).

**Carried forward:** species = all (36).

---

## 2026-06-25 — rev-3 hidden (sel7; species all, 50/50)

**Result.** Winner **hidden 64** (val sel7 0.721); reproduces the species-stage `all` number. 256
close (0.716). Noisy curve (32/128 dip) but 64 the clear sel7 peak.

| hidden | val sel7 | all9 (ref) | best epoch |
|-------:|---------:|-----------:|-----------:|
| 0      | 0.691    | 0.594      | 52         |
| 32     | 0.662    | 0.559      | 18         |
| **64** | **0.721**| 0.603      | 10         |
| 128    | 0.684    | 0.574      | 30         |
| 256    | 0.716    | 0.628      | 26         |

**Positive.** 64 wins sel7 cleanly; consistency check passes.
**Negative.** Curve noisy (32/128 dip); 256 within 0.005 of 64 on sel7 (and would win the all9
reference — but all9 isn't the selection metric anymore).

**Carried forward:** species all, hidden 64.

---

## 2026-06-25 — rev-3 lr (sel7; species all, hidden 64)

**Result.** Winner **LR 5e-4** (val sel7 0.721) — but a near-flat plateau: 5e-5 / 1e-4 / 5e-4 all
within 0.0012 (0.720–0.721).

| LR       | val sel7 | all9 (ref) | best epoch |
|---------:|---------:|-----------:|-----------:|
| 1e-5     | 0.690    | 0.579      | 79         |
| 5e-5     | 0.720    | 0.600      | 21         |
| 1e-4     | 0.721    | 0.603      | 10         |
| **5e-4** | **0.721**| 0.604      | 2          |
| 1e-3     | 0.710    | 0.594      | 8          |

**Positive.** Very forgiving — anything 5e-5–5e-4 lands ~0.720.
**Negative.** 5e-4 wins by +0.0003 (noise) and peaks at epoch 2 (suspiciously fast); 1e-4 is the
safer equivalent. 1e-5 underfits. Carried the rule winner per "don't relitigate noise."

**Carried forward:** species all, hidden 64, LR 5e-4.

---

## 2026-06-25 — rev-3 crop (sel7; species all, hidden 64, LR 5e-4)

**Result.** Winner **square + native z=0.5** (val sel7 0.756). Monotonic decreasing 0.5 > 0.75 > 1.0
— the zoom-OUT view wins, reversing rev-1/2 (where z=1.0 won).

| crop (square + native z) | val sel7 | all9 (ref) | best epoch |
|:-------------------------|---------:|-----------:|-----------:|
| **z = 0.5**              | **0.756**| 0.655      | 2          |
| z = 0.75                 | 0.747    | 0.651      | 3          |
| z = 1.0                  | 0.742    | 0.634      | 3          |

**Positive.** Big crop gain: native-only 0.721 → square+z0.5 **0.756** (+0.035); all9 climbs too
(0.603 → 0.655). The coarser zoom-out view helps most under full data + all-36 — echoes the
multi-scale ablation (a coarser view of the plant adds signal).
**Negative.** Fast convergence (ep 2–3 at LR 5e-4); z=1.0 (rev-1/2's winner) is now the worst.

**Carried forward:** species all, hidden 64, LR 5e-4, crop square+native(0.5).

---

## 2026-06-25 — rev-3 size (sel7; full data, all-36)

**Result.** Winner **512 imgs/species** (val sel7 0.779). Clean inverted-U — peaks at 512, then
DECLINES with more data (512 > 1024 > 2048).

| imgs/species | val sel7 | all9 (ref) | best epoch |
|-------------:|---------:|-----------:|-----------:|
| 64           | 0.731    | 0.635      | 14         |
| 128          | 0.737    | 0.618      | 7          |
| 256          | 0.756    | 0.655      | 2          |
| **512**      | **0.779**| 0.670      | 1          |
| 1024         | 0.760    | 0.661      | 0          |
| 2048         | 0.753    | 0.624      | 0          |

**Positive.** Strong data scaling to 512 (+0.048 over 64); 512 is the best config in the rev-3 chain
so far (sel7 0.779).
**Negative.** More data past 512 **hurts** (1024/2048 decline), and best-epoch collapses to **0–1** for
n_per ≥ 512 — with LR 5e-4 the head overfits almost immediately on big data, so the 1024/2048 drop is
an **LR×data interaction**, not a pure data effect. A lower LR might let the extra data help; the r
sweep + final-run best-epoch freezing protect the deployed model, but it flags LR 5e-4 as hot for the
larger sets.

**Carried forward:** species all, hidden 64, LR 5e-4, crop square+native(0.5), n_per 512.

---

## 2026-06-25 — rev-3 r (sel7; all, 512)

**Result.** Winner **r = 8** (val sel7 0.779); flat/noisy curve (0.762–0.779), no monotonic trend
(unlike rev-2's "higher is better"). r=8 reproduces the size-stage 512 number exactly (consistency
check).

| r   | val sel7 | all9 (ref) | best epoch |
|----:|---------:|-----------:|-----------:|
| 2   | 0.773    | 0.651      | 2          |
| 4   | 0.771    | 0.659      | 1          |
| **8** | **0.779**| 0.670    | 1          |
| 16  | 0.767    | 0.658      | 3          |
| 32  | 0.762    | 0.648      | 2          |
| 64  | 0.769    | 0.662      | 6          |
| 128 | 0.765    | 0.654      | 2          |

**Positive.** r=8 (the default) is the peak; consistency check passes.
**Negative.** Differences are noise-level (~0.017 spread); r barely matters here — config-dependent
again (rev-2 wanted high r monotonically).

**Carried forward (final config):** species all, hidden 64, LR 5e-4, crop square+native(0.5),
n_per 512, r 8.

---

## 2026-06-25 — rev-3 final run (sel7 chain, 50/50, full data)

**Config.** species all (36) · hidden 64 · LR 5e-4 · crop square+native(0.5) · 512 imgs/species · r 8
· best epoch 1. `checkpoints/mil.pt` (36-output head, scored on 9) + metrics saved.

**Headline (50/50 TEST, n=266).** sel7 (7 well-supported) = **0.696** · all-9 = **0.622**. Val: sel7
0.779 / all-9 0.670. Val→test sel7 gap ≈ 0.08 (some val-overfit via best-epoch-1 + per-species
thresholds).

Per-species TEST F1, with the cross-rev trend on the well-measured species:

| species (rev-3 test sup) | rev-1 | rev-2 | **rev-3** |
|:-------------------------|------:|------:|----------:|
| Thinopyrum (266)         | 1.00  | 1.00  | **1.00**  |
| Lupinus (181)            | 0.81  | 0.83  | **0.87**  |
| Poa (139)                | 0.68  | 0.65  | **0.67**  |
| Gaillardia (43)          | 0.44  | 0.71  | **0.64**  |
| Tragopogon (38)          | 0.21  | 0.33  | **0.50**  |
| Balsamorhiza (15)        | 0.67* | 0.50* | **0.67**  |
| Bromus (12)              | 0.29* | 0.31* | **0.52**  |
| Achillea (6, excl)       | 0.67* | 0.00* | 0.36      |
| Sisymbrium (5, excl)     | 0.67* | 0.50* | 0.36      |

(\* rev-1/2 used 80/20, so those classes had only 2–6 test frames — noisy, not directly comparable.
rev-3's 50/50 gives 5–266 test frames: the first trustworthy per-species read.)

**Positive (the result).**
- rev-3 is the best, most trustworthy model: on a 50/50 test (n=266) the well-supported macro is
  **0.696**, with steady per-species gains across the program — **Lupinus 0.81→0.87, Tragopogon
  0.21→0.50, Bromus 0.29→0.52**, Gaillardia strong (0.64 on n=43 vs rev-1's 0.44 on n=17). Thinopyrum
  perfect.
- The full recipe — all-36 hard negatives + square+native(0.5) crop + 512 imgs + **sel7 selection** +
  50/50 eval — lifted val sel7 to 0.779 and gives a stable, rare-class-robust evaluation.
- The excluded yarrow/tumble now score 0.36/0.36 on real support (5–6 frames) — modest but no longer
  the 0/0.5 noise that swung rev-2's all-9.

**Negative.**
- **Poa still over-fires** (test R 0.88 / P 0.54) — the grass over-prediction persists; hard-negative
  grasses improved detection but not Poa's precision.
- **Val→test gap ≈ 0.08** (sel7) — best-epoch-1 + per-species val thresholds overfit val (the LR-5e-4
  fast-peak again). Threshold-robustness is the obvious next lever.
- Tragopogon / Bromus still recall-limited (R 0.39 / 0.50) at high thresholds.

**Deployed:** `checkpoints/mil.pt` (rev-3, 36-output head scored on the 9 targets).

---

## 2026-06-25 — Follow-up ablation: shift TTA (translation test-time augmentation)

**Purpose.** Inference uses one fixed 16 px patch lattice; a feature straddling a patch boundary is
split/diluted. Test whether re-tiling at a half-patch (8 px) offset (pass-2 centers on pass-1 corners)
+ any-patch max over the union improves detection.

**Design.** Deployed rev-3 head, no retraining. Offsets {(0,0),(8,0),(0,8),(8,8)}. Configs: single
{(0,0)} (control), 2-pass {(0,0),(8,8)}, 4-pass {all}. Per-species any-patch max over the union;
thresholds re-derived on val (sel7); test scored once. (`followup_ablations/tta_shift.py`)

| config           | passes | val sel7 | test sel7 | test all9 | val→test gap |
|:-----------------|-------:|---------:|----------:|----------:|-------------:|
| single (control) | 1      | **0.779**| 0.696     | 0.622     | 0.083        |
| 2-pass           | 2      | 0.765    | **0.708** | 0.632     | 0.057        |
| 4-pass           | 4      | 0.766    | 0.707     | **0.661** | 0.059        |

Per-species TEST F1 (single → 4-pass): Lupinus 0.87→0.88, Poa 0.67→0.70, Bromus 0.52→0.59,
Balsamorhiza 0.67→0.69, Gaillardia 0.64→0.64, Tragopogon 0.50→**0.45**; rare: Achillea 0.36→0.43,
Sisymbrium 0.36→0.57.

**Positive (the result).**
- TTA improves the held-out TEST: sel7 0.696 → 0.708 (+0.012), all9 0.622 → 0.661 (+0.039 at 4-pass),
  and tightens the val→test gap (0.083 → ~0.058) — it regularizes / generalizes better.
- Gains land where predicted: sparse/thin/boundary species — Bromus (+0.07), Poa recall, Balsamorhiza
  recall (0.53→0.60), and the rare Achillea/Sisymbrium (4-pass drives the all9 jump via Sisymbrium
  0.36→0.57). Mechanism: more patch samples = more any-patch firing chances → recall.
- Control reproduces deployed exactly (val 0.779 / test 0.696) — protocol sound. 4-pass beats 2-pass
  only on all9/rare; ~tied on sel7 (so 2-pass gets the cheap well-supported gain, 4-pass adds rare).

**Negative / caveat.**
- **val sel7 marginally PREFERS single** (0.779 vs 0.765) — the strict val-selection rule would
  *reject* TTA. A val/test inversion: the single-pass val was optimistic; TTA's gain shows on test.
- Tragopogon worse (recall 0.39→0.29; threshold pushed to 0.95) — TTA's extra firing raised its FPs.
- Deploy cost: 2×/4× the robot-frame inference (extraction); training unchanged.

**Takeaway.** Translation TTA is a real test-time win (4-pass best on all9/rare, 2-pass for the cheap
well-supported gain) and tightens the generalization gap — but it's invisible-to-slightly-negative on
the val selection metric, so it's an explicit deploy-time choice, not something the sweep would pick.
Worth adopting if the ~2–4× inference cost is acceptable.

---

## 2026-06-26 — Follow-up ablation: relax the any-patch rule (k-of-N)

**Purpose.** Recall is the weak axis; the any-patch (k=1) rule makes a single stray patch a hard FP,
forcing high thresholds (low recall). Test whether a k-of-N rule (present iff ≥k patches ≥ τ) lets τ
drop and recall recover. "≥k patches ≥ τ" == "k-th-largest patch score ≥ τ", so this is a post-hoc
sweep of the order statistic; (k, τ) jointly optimized per species on val F1. (`kpatch_rule.py`)

**Result — it does NOT help (mild negative). k=1 (any-patch) is near-optimal.**

Fixed k (all species), TEST:

| k             | test sel7 | test all9 | macroP(7) | macroR(7) |
|--------------:|----------:|----------:|----------:|----------:|
| 1 (any-patch) | **0.696** | 0.622     | 0.753     | **0.676** |
| 2             | 0.671     | 0.611     | 0.759     | 0.637     |
| 3             | 0.668     | 0.618     | 0.736     | 0.654     |
| 5             | 0.643     | 0.586     | 0.687     | 0.650     |

Per-species k optimized on val: sel7 0.696 → 0.693 (flat), all9 0.622 → 0.638, macroR 0.676 → 0.663.
Chosen k: every well-supported species kept **k=1**; only Poa and the rare Achillea/Sisymbrium chose
k=3.

**Negative (the finding).**
- Requiring more patches **lowers recall monotonically** (macroR 0.676 → 0.637 at k=2) — a sparse true
  plant with few firing patches gets missed; precision barely moves.
- The well-supported species all prefer **k=1**: their FPs are *multi-patch* (Poa fires across whole
  grass regions; Bromus/Tragopogon FPs look real), so k≥2 can't reject them and only costs recall.
  The any-patch rule is well-justified for the deployed species.

**Positive (small).** k=3 helps *rare-class precision* by rejecting strays: Achillea P 0.40→0.67
(F1 0.36→0.44), Sisymbrium P 0.33→0.50 (F1 0.36→0.44) → all9 +0.016 (but these are the excluded noisy
classes, not sel7).

**Takeaway.** Relaxing the any-patch rule does **not** recover recall — recall is limited by the
domain gap + F1-thresholds, not single-stray-patch FPs. k=1 is near-optimal for the deployed species.
The recall lever is the THRESHOLD objective (Fβ, β>1 / lower cutoffs), not the patch count; a per-class
k≥2 is a minor add-on for rare-class precision only.

---

## 2026-06-26 — Follow-up ablation: Fβ thresholds (recall operating point)

**Purpose.** Move the operating point toward recall by tuning per-species thresholds to maximize Fβ
(β>1 weights recall) instead of F1. Post-hoc on the deployed any-patch scores; β=1 = control.
(`followup_ablations/fbeta_thresh.py`)

**Result — Fβ is a clean, smooth recall lever; β=1.5 is a near-free recall gain.**

| β        | test F1 (sel7) | macro-P | macro-R |
|---------:|---------------:|--------:|--------:|
| 1.0 (F1) | 0.696          | 0.753   | 0.676   |
| **1.5**  | 0.690          | 0.712   | **0.713** |
| 2.0      | 0.651          | 0.606   | 0.776   |
| 3.0      | 0.594          | 0.502   | 0.835   |

**Positive.**
- **β=1.5: +0.037 macro recall (0.676→0.713) at ~flat sel7-F1 (0.696→0.690).** Mostly lifts the
  high-scoring species to near-perfect recall — Lupinus R 0.83→**0.97** (F1 0.87→0.86), Poa R 0.88→0.97
  (F1 0.67→**0.69**, up).
- Smooth, monotone trade: β=2 → recall 0.776 (F1 0.651), β=3 → recall 0.835 (F1 0.594). Per-species
  thresholds, so β can be set per class.

**Negative.**
- Net-neutral β=1.5 is a *redistribution*: Lupinus/Poa gain cheaply, but **Gaillardia** dips (F1
  0.64→0.59, P 0.71→0.56). β≥2 cracks precision (0.606 at β=2) and **Tragopogon** craters (P 0.68→0.20
  to buy R 0.39→0.61). all9 falls with β (rare-class precision).
- Tragopogon/Balsamorhiza/Bromus don't move at β=1.5 (keep their F1 threshold) — recall **gap-limited**,
  not threshold-limited; only a better model (domain-gap / TTA) recovers it.

**Takeaway.** Fβ is the recall knob (unlike k-of-N). **β=1.5 = recommended operating point for a
coverage map** (~+0.04 recall for ~free); β=2 if recall is paramount and precision cost is OK.

---

## 2026-06-26 — Follow-up ablation: loss variants (focal, label smoothing)

**Purpose.** Swap the head's BCE+pos_weight for focal loss (γ) or label smoothing (ε). Retrained on
the deployed config; loss on the LSE bag logits; selection = val sel7. (`loss_variants.py`)

**Result — BCE is best on sel7; neither variant beats it.**

| loss                | val sel7 | test sel7 | test all9 |
|:--------------------|---------:|----------:|----------:|
| **BCE (baseline)**  | **0.779**| **0.696** | 0.622     |
| focal γ=0.5         | 0.768    | 0.689     | **0.643** |
| focal γ=1           | 0.764    | 0.673     | 0.610     |
| focal γ=2           | 0.763    | 0.686     | 0.637     |
| label-smooth ε=0.05 | 0.734    | 0.651     | 0.651     |
| label-smooth ε=0.10 | 0.734    | 0.677     | 0.632     |

**Negative.** BCE wins sel7; label smoothing clearly hurts (the small head needs sharp targets;
ε=0.05 peaks at epoch 0). Focal slightly lowers the well-supported.
**Positive (narrow).** Focal γ=0.5 helps the hard *rare* classes — Achillea F1 0.36→0.60 (P 0.40→0.75),
Bromus 0.52→0.55 → test all9 +0.021. Suited to sparse/confusable species only.
**Takeaway.** Loss is not a lever for the deployed (well-supported) metric — BCE+pos_weight stays.
Focal γ=0.5 is a narrow rare-class win; no change to the deployed model.

---

## 2026-06-26 — Lesson: do NOT delete small detections as "clutter"

The alpha-shape overlay briefly used a min-patch-count filter (`MIN_PATCHES=9`) to declutter stray
patches. **This was a mistake that hurt interpretation.** A size filter deleted small *real* detections
(sparse plants that fire on only a few patches) — it **cannot** distinguish a real small detection from
a stray single-patch FP. Consequences:
1. the overlay stopped being faithful to the metric (**painted ≠ any-patch fired**);
2. it **hid real signal** — e.g. the Fβ recall gains on Gaillardia were single-patch and thus invisible
   in the overlay even though the model flagged them;
3. it **actively misled** visual interpretation (areas looked emptier than the model actually marked).

**Fixed:** `MIN_PATCHES=1` — every fired region is drawn, down to a single patch.
**Principle:** the visualization must stay faithful to the firing decision (**painted iff fired**). If
decluttering is ever needed, gate by **confidence, not size**, and make it explicit/optional — never
silently drop small detections.

---

## 2026-06-26 — Macro tile overlap (+ patch offset) — single-frame test

**Idea.** Inference tiles the frame into 24 *disjoint* 224px tiles; a plant at a tile edge — or in the
reflection-contaminated bottom tile (see the bottom-band finding) — is under-sampled. **Macro overlap**
re-tiles at **stride 112** (half a tile), so every tile's *center* lands on its neighbor's *corner* —
each location is seen by a tile where it sits near a center (full natural context) rather than an edge.
It composes with the 8px **patch offset** (shift-TTA): the two are orthogonal (macro = which tile sees
a feature; micro = the patch lattice within), and the passes are unioned via the any-patch max.

**Test** (`tmp/test_overlap.py`, frame leg01_m020, deployed F1 thresholds, qualitative — one frame):

| config | tiles | Lupinus fired-cells |
|:--|--:|--:|
| base (disjoint 224) | 24 | 153 |
| + macro overlap (stride 112) | 77 | 250 |
| + macro + patch offset (8px) | 154 | **478** |

**Positive.** ~3× more Lupinus cells, and visually the region **fills out the *true* lupine** — it
grows *down* over the lower silvery foliage that base missed (exactly the under-sampled bottom band),
**without spreading onto the surrounding grass** (no obvious FP inflation on this frame). Macro and
patch each contribute and stack. This is the "done-right" version of the bottom-tile fix and supersedes
the patch-only shift-TTA by *also* repairing the tile-edge / reflection under-sampling.

**Caveats.**
- Measured at the **deployed F1 thresholds**: more samples → higher any-patch max → more firing, so
  part of the gain is an operating-point shift (same as shift-TTA). A fair precision/recall verdict
  needs thresholds **re-derived on val**, then test — not yet run (full val/test eval pending).
- **Compute:** macro ≈ 3× tiles, macro+patch ≈ 6× the backbone inference per frame.

**Status.** Qualitative single-frame win; the full 533-frame metric eval (macro / macro+patch vs base
vs patch-only shift-TTA, thresholds re-derived on val) is the deciding next step if pursued.

---

## 2026-06-26 — Follow-up ablation: low LR (1e-6) at 2048 imgs/species

**Purpose.** The rev-3 size sweep showed sel7 DECLINES past 512 (512 → 0.779, 1024 → 0.760, 2048 →
0.753) with best-epoch collapsing to 0–1 at LR 5e-4 — flagged as a possible LR×epoch artifact (the head
over-fits 2048 imgs in <1 epoch, so the extra data is never "used"). Test the cleanest version of that
hypothesis: drop LR to **1e-6** so the head takes many gradient steps over the full 2048/species — does
the extra data now help? Deployed config otherwise (species all-36, hidden 64, crop square+native(0.5),
r 8, BCE). Reuses the size sweep's cached 2048 features — zero re-extraction. (`followup_ablations/lr_2048.py`)

**Result — NO. The low-LR hypothesis is REFUTED; 1e-6 under-fits.**

| config (2048 unless noted) | val sel7 | test sel7 | test all9 | best ep  |
|:---------------------------|---------:|----------:|----------:|---------:|
| 512 @ 5e-4 (deployed)      | **0.779**| **0.696** | 0.622     | 1        |
| 2048 @ 5e-4 (size sweep)   | 0.753    | —         | —         | 0        |
| 2048 @ 1e-6 (this run)     | 0.695    | 0.612     | 0.565     | 79 (cap) |

Per-species TEST F1 (1e-6@2048 vs deployed 512@5e-4) — headline forbs in **bold**: **Lupine 0.84**
(vs 0.87), **Arrowleaf/Balsamorhiza 0.70** (0.67), **Blanketflower/Gaillardia 0.38** (0.64); Poa 0.68
(0.67), Tragopogon 0.38 (0.50), Bromus 0.30 (0.52), Thinopyrum 1.00 (=).

**Negative (the finding).**
- 1e-6 @ 2048 lands **val sel7 0.695 — below BOTH 2048@5e-4 (0.753) and 512@5e-4 (0.779)**. Lowering
  the LR did not unlock the extra data; it made training worse.
- best epoch = **79 (the cap)** — at 1e-6 the head never converged in 80 epochs. The failure mode
  flipped from 5e-4's fast over-fit (epoch 0) to 1e-6's slow under-fit (hits the cap), and the underfit
  point is the worse of the two. Blanketflower and Bromus fall hardest (Gaillardia 0.64→0.38, Bromus
  0.52→0.30).

**Takeaway.** The post-512 size decline is **NOT an artifact that low LR repairs** — the 512@5e-4
optimum is robust. 5e-4-at-epoch-0 on 2048 (0.753) still beats 1e-6-to-cap on 2048 (0.695), so the "we
never used the data" story is wrong: more data simply doesn't beat 512 here at any LR tried. Caveat:
1e-6 hit the epoch cap (didn't converge) → this is a lower bound, but it would need +0.084 sel7 from an
under-fit trajectory to match 512@5e-4 (implausible). The only untested cell is a *moderate* LR
(5e-5/1e-4) at 2048 with a longer schedule, but 2048@5e-4 < 512@5e-4 already argues 2048 won't surpass
512. **Deployed config unchanged (512 imgs/species).**

---

## 2026-06-26 — Macro tile overlap — FULL VAL metric at FROZEN thresholds

**Purpose.** The macro-overlap single-frame test (leg01/leg11) was qualitative and measured at deployed
thresholds. Quantify it properly: per-species val F1 for base / macro / macro+patch, **using the frozen
deployed thresholds (NO re-derivation)** — i.e. does denser tiling help *for free*? Monotonicity sets
the prior: frozen thr + any-patch max means a denser pass can only RAISE a score, so predictions only
flip absent→present → recall can only rise, precision can only fall. Net F1 is the question.
(`tmp/val_eval_overlap.py`, n=267 val frames. Compute trick: base tiles ⊂ macro tiles, so only 2 passes
/frame — A=(PAD,112), B=(PAD8,112) — derive base=stride-224 subset of A, macro=all A, mp=A∪B.)

**Control validated.** Accumulate-base == cached pipeline to 4 decimals (sel7 0.7785, all9 0.6702,
every per-species row identical) — protocol sound, deltas real.

**Result — denser tiling MONOTONICALLY HURTS val F1 at frozen thresholds.**

| config        | tiles | val sel7 | val all9 |
|:--------------|------:|---------:|---------:|
| base          | 24    | **0.7785**| **0.6702**|
| macro overlap | 77    | 0.6957   | 0.5926   |
| macro + patch | 154   | 0.6740   | 0.5768   |

Per-species F1 (base → macro → macro+patch), headline forbs in **bold**:

| species          | base | macro | mp   | P (base→mp) | R (base→mp) |
|:-----------------|-----:|------:|-----:|:------------|:------------|
| **Lupine**       | 0.87 | 0.89  | 0.89 | 0.91→0.86   | 0.83→0.93 ↑ |
| **Arrowleaf**    | 0.70 | 0.57  | 0.55 | 1.00→0.57   | 0.53→0.53 = |
| **Blanketflower**| 0.76 | 0.71  | 0.68 | 0.79→0.57   | 0.72→0.84 ↑ |
| Poa              | 0.70 | 0.70  | 0.69 | 0.56→0.53   | 0.92→0.99   |
| Tragopogon       | 0.57 | 0.46  | 0.44 | 0.84→0.37   | 0.43→0.54   |
| Bromus           | 0.86 | 0.54  | 0.47 | 1.00→0.32   | 0.75→0.83   |
| Thinopyrum       | 1.00 | 1.00  | 1.00 | 1.00→1.00   | 1.00→1.00   |

**Negative (the finding).**
- val sel7 −0.083 (macro), −0.105 (macro+patch); all9 likewise. At the frozen operating point the FP
  inflation outweighs the recall gain for everything except lupine.
- **Of the 3 headline forbs, only lupine wins** (F1 0.87→0.89, a real recall gain 0.83→0.93).
  **Blanketflower** degrades (0.76→0.68, P 0.79→0.57) and **arrowleaf** degrades worst (0.70→0.55, P
  1.00→0.57 with recall **dead flat at 0.53**) — for arrowleaf the extra cells fill *already-detected*
  frames (no new TP) and light up *absent* frames (new FP), the pure-FP case. This is the quantified
  version of the leg11 sky-FP.
- Species whose high thresholds were suppressing FPs collapse: Tragopogon P 0.84→0.37, Bromus P
  1.00→0.32. Poa flat (recall already saturated).

**Takeaway.** Macro overlap is **NOT a free win** — at the deployed thresholds it costs ~0.08–0.10 val
sel7 and degrades 2 of 3 headline forbs; only lupine (genuinely recall-limited) benefits. The denser
sampling raises the any-patch max on present AND absent frames alike, so frozen cutoffs calibrated for
base density over-fire. The ONLY way it can pay is thresholds **re-derived on val** for the denser
density (raised to absorb the extra firing) — the complementary eval, still the deciding next step. The
single-frame "fills the true plant" impression was real for lupine but metric-negative for arrowleaf.
**No deploy at frozen thresholds.**

---

## 2026-06-26 — Macro tile overlap — thresholds RE-DERIVED per method (the deciding eval)

**Purpose.** The frozen-threshold eval (above) showed denser tiling HURTS — but that was an
operating-point artifact (cutoffs calibrated for base density over-fire under denser sampling). The fair
test, long flagged: **re-derive each method's per-species thresholds on val (F1-argmax, the deployed
rule), apply to held-out test.** Resolves whether macro overlap genuinely helps once each method runs at
its own best operating point. (`tmp/thresh_by_method.py`; per-frame scores for all 533 frames cached to
`tmp/tiling_scores_533.npz` — threshold experiments now free.)

**Anchor.** base @ frozen deployed thr == base re-derived: val sel7 0.7785 / test 0.6955 / all9 0.6218 —
reproduces the deployed model exactly (the deployed thresholds ARE the val-F1-argmax), so re-derivation
is consistent and the deltas are real.

**Result — REVERSAL. With re-derived thresholds, macro+patch is the BEST model.**

| method (val-derived thr → test)   | val sel7 | val all9 | test sel7 | test all9 |
|:----------------------------------|---------:|---------:|----------:|----------:|
| base (= deployed)                 | 0.7785   | 0.6702   | 0.6955    | 0.6218    |
| macro overlap (77 tiles, 3×)      | 0.7588   | 0.6573   | 0.6990    | 0.6029    |
| **macro + patch (154 tiles, 6×)** | **0.7811**| **0.6948**| **0.7127**| **0.6845**|

Per-species TEST F1 (base → macro → macro+patch), headline forbs in **bold**:

| species           | base | macro | mp   | mp vs base                       |
|:------------------|-----:|------:|-----:|:---------------------------------|
| **Lupine**        | 0.87 | 0.88  | 0.91 | +0.04 (R 0.83→0.91 @ P 0.90)     |
| **Arrowleaf**     | 0.67 | 0.74  | 0.74 | +0.07 (R 0.53→0.67 @ P 0.83)     |
| **Blanketflower** | 0.64 | 0.65  | 0.65 | +0.01 (flat)                     |
| Poa               | 0.67 | 0.68  | 0.65 | −0.02                            |
| Tragopogon        | 0.50 | 0.47  | 0.45 | −0.05 (still the FP magnet)      |
| Bromus            | 0.52 | 0.48  | 0.58 | +0.06                            |
| Thinopyrum        | 1.00 | 1.00  | 1.00 | =                                |
| Achillea (excl)   | 0.36 | 0.11  | 0.60 | +0.24                            |
| Sisymbrium (excl) | 0.36 | 0.43  | 0.57 | +0.21                            |

**Positive (the result).**
- **macro+patch wins on BOTH val (0.781 ≥ base 0.779) and test (sel7 0.696→0.713 +0.017; all9
  0.622→0.685 +0.063)** — not a val-overfit fluke; the val→test gap (0.068) is tighter than base's
  (0.083). The frozen-threshold degradation was 100% operating point: re-deriving lifts macro+patch's val
  from 0.674 (frozen) → 0.781.
- **2 of 3 headline forbs win clearly — lupine 0.87→0.91, arrowleaf 0.67→0.74; blanketflower flat.** The
  two most-favored species gain real recall (lupine R 0.83→0.91, arrowleaf R 0.53→0.67) at held precision
  — the bottom-band / tile-edge under-sampling the macro overlap targets, now confirmed on the full metric.
- Re-derived thresholds rise to absorb the denser firing (Poa 0.49→0.69, Gaillardia 0.39→0.53,
  Balsamorhiza 0.51→0.59, Lupinus 0.27→0.31) — the mechanism that turns the frozen-threshold FP inflation
  back into net recall gains. all9's larger jump is rare-class recovery (Achillea/Sisymbrium).

**Negative / cost.**
- **macro overlap ALONE is a wash** — val sel7 DROPS (0.759 < base 0.779), test ~tied (0.699). The
  **patch offset does the work** (its 8px-shifted 16px lattice samples genuinely new positions); the win
  needs the COMBINATION, not just more tiles. No cheap 3× version — the win is the 6× config.
- Tragopogon still erodes (0.50→0.45; even re-thresholding can't reject its multi-patch FPs) and Poa dips
  slightly. Compute is **6× the backbone** per frame.

**Takeaway.** Re-thresholding REVERSES the verdict: **macro+patch is a real, deployable win** — best on
val and test (+0.017 test sel7 / +0.063 test all9), and crucially **+0.04 / +0.07 F1 on the two favored
headline forbs (lupine, arrowleaf)**, blanketflower flat. Price: 6× inference + re-freezing the (higher)
per-species thresholds. Strongest model-quality lever in the follow-ups (vs shift-TTA's +0.012 test
sel7). Deploy decision = is 6× inference acceptable for +0.04–0.07 on the headline forbs. macro-only not
worth it.

---

## 2026-06-26 — KEY NEXT STEP: sky / cloud false positives (lupine, and blanketflower)

**The issue.** The model fires on **bright sky / clouds**, confusing them with **lupine frequently**,
and **blanketflower to a lesser extent** (arrowleaf rarely — its saturated gold is distinctive). Lupine's
lavender flowers + silvery-hairy foliage read very close to hazy sky / cloud tones; blanketflower's
paler warm tones occasionally do too. This is now flagged as a **key next step**.

**Evidence.** A **long-standing model failure, independent of tiling** — present in the base/deployed
model from the start (observed throughout the project), NOT introduced or caused by macro+patch. Recent
instances simply where it was noticed again:
- leg11_m009: a lupine region in **clear sky**; it dropped out only when the higher re-derived lupine
  threshold (0.27→0.31) was applied — an operating-point **band-aid**, not a fix (the sky patch still
  scored ~0.3 as lupine).
- Over-exposed / bloomed captures (e.g. leg09_m028) light up lupine on washed-out sky/glare regions.

**Why it matters.**
- Sky is a **large region present in ~every frame** (horizon framing). Under the any-patch rule a single
  stray cloud patch is a **hard FP**, so sky confusion directly caps lupine/blanketflower **precision** —
  the two headline forbs we most care about.
- It inflates the per-species **thresholds** at ANY operating point (base included): cutoffs sit higher
  partly to suppress sky firing, which spends real recall. So a real sky fix would let thresholds drop and
  recover recall — a model-quality gain orthogonal to, and independent of, any tiling choice.

**Candidate directions (not yet tried).**
1. **Sky/horizon suppression** — geometric (detect the horizon, zero-out firing in patches above it) or a
   cheap sky/color segmenter; the robot's horizon is stable, so a mask is low-risk and near-free.
2. **Hard-negative sky/cloud imagery** — the current 27 iNat negatives are all plants/ground, never sky;
   add explicit sky/cloud crops as a negative so the head learns "sky ≠ lupine" from the loss, not from a
   threshold. Most aligned with the free-data thesis (no robot labels needed).
3. **A dedicated "sky" auxiliary one-vs-all output** — let the head carve sky out of the feature space
   directly.

**Status.** KEY open issue, **tiling-independent** — it predates and is unrelated to macro+patch. Raising
thresholds only **masks** it (base or any tiling alike), never solves it. A real fix (horizon/sky mask or
sky hard-negatives) is the priority next step.

---

# 2026-06-26 (rev 4) — Sky hard-negative class + license-clean data + tiling as a final sweep

rev-4 attacks the rev-3 **KEY NEXT STEP (sky/cloud false positives)** head-on with a free-data sky
negative, and tightens the data + sweep protocol:

1. **Sky hard-negative class.** A new auxiliary class **Sky** (scored ONLY as a negative, never a
   target; appended LAST so target indices 0–8 are unchanged → `D.N` stays 9) sourced from FREE
   iNaturalist **bald-eagle (Haliaeetus leucocephalus) in-flight** imagery — the bird is centred
   against open sky, so the UPPER corners are sky/cloud. Each eagle image contributes its upper-left +
   upper-right native-224 **corner** crops. A **SegFormer (ADE20K) gate** keeps a corner only if
   **≥10% of its pixels are sky**, dropping perched/ground eagles (pass rate **UL 736/1000, UR
   727/1000**). This teaches the head "bright sky ≠ lupine/blanketflower" from the LOSS — the most
   free-data-thesis-aligned of the rev-3 candidate sky fixes (vs a geometric horizon mask).
2. **License-clean data, 1000/species.** Both datasets re-harvested with an opt-in license keep-list
   (CC0, PUBLIC-DOMAIN, CC-BY, CC-BY-SA, CC-BY-NC, CC-BY-NC-SA — drops all-rights-reserved; iNat's
   default CC-BY-NC is kept, so almost nothing is lost) and capped at **1000 imgs/species** (the size
   sweep only ever uses ≤512). `inat_dataset/` split into `plants/` + `birds/`.
3. **Sweep protocol.** Default LR **5e-5** (gentler than rev-3's "hot" 5e-4, which peaked at epoch
   0–1); **LR swept LAST, after the temperature r** (r/crop/size tuned at a stable LR, then LR
   finalised on the full config); size capped at {64,128,256,512}; **Sky in EVERY species-set
   candidate**.
4. **Tiling as a final sweep.** base / macro (stride-112) / macro+patch (8px) compared on robot frames
   with thresholds re-derived per method (`sweep_tiling.py`); the winner's tiling + thresholds are
   FROZEN into `mil.pt` and drive the final test + inference + visualization refresh.

Methodology otherwise unchanged: frozen DINOv3 ViT-B/16; MIL head (LSE bag pooling, BCE one-vs-all);
robot val/test 50/50 (seed 1312); selection = val **sel7**; iNat feature caches reused (wiped once for
the new license-clean data). Per-stage results appended below as they complete.

---

## 2026-06-26 — rev-4 sweep stages (Sky in every set; 5e-5 default LR; LR swept LAST)

**Data:** license-clean (CC0/PD/CC-BY/-SA/-NC/-NC-SA, drops all-rights-reserved), 1000/species cap.
All 9 targets + 25/27 negatives hit 1000; only Poa secunda (804) + Antennaria (975) supply-limited,
both > the 512 training cap. Sky = bald eagle (1000 imgs), SegFormer-gated upper corners (UL 736 /
UR 727 pass >=10% sky).

| stage | winner | val sel7 | notes |
|:------|:-------|---------:|:------|
| species | **all 36 + Sky** | 0.7345 | targets+Sky 0.719, grasses+Sky 0.703 — broad negatives win, Sky rides along |
| hidden  | **64**          | 0.7345 | 0/256 close (0.716/0.728); 32/128 dip — same shape as rev-3 |
| crop    | **square + native z=0.5** | 0.7763 | monotone 0.5 > 0.75 > 1.0 — zoom-out wins again (rev-3) |
| size    | **512**         | 0.7807 | clean monotone 64→512; best-epoch 79→51→27→**16** (more data converges FASTER) |
| r (LSE) | **2**           | 0.7842 | **REVERSAL** vs rev-3 (r=8 flat): monotone decline to r=128 (0.759) |
| lr      | **1e-5**        | **0.7922** | best of the whole sweep, but at the epoch-budget edge (ep **71**/80); 5e-6 **underfit** (0.519) |

**Key findings.**
- **Structure reproduces rev-3** (all-species, hidden 64, z=0.5 crop, size 512) — robust to the new
  license-clean data + the added Sky class.
- **r reversed (8 → 2).** Lower LSE temperature (training-time bag pooling nearer the *mean* than the
  *max*) now wins, monotone. Read: with the Sky class + broader negatives, max-like training (bet on
  one patch) is punished by spurious single-patch FPs; averaging forces consensus and regularizes.
  The object-centric crops (square + z=0.5) make averaging appropriate (the plant fills the crop).
  Inference is still the any-patch MAX rule — r only shapes training.
- **size scales cleanly to 512** (rev-3 *declined* past 512 at the "hot" 5e-4); the gentle 5e-5
  uses more data well, and convergence epoch *shrinks* with data (79→16) — the LR×size×epoch coupling.
- **lr optimum is 1e-5** — *below* the 5e-5 default — but it only converged at ep 71 (hit the 80-cap
  ~9 flat epochs after its peak, short of the patience-12 confirmation), and **5e-6 underfit hard**
  (0.519, too low to train in budget). The winning LR sits at the convergence edge — the LR×epoch
  confound is real. Deployed at 1e-5 per the selection rule (a raised-cap re-check was offered and
  declined).

### rev-4 final — base tiling (ANCHOR; the deployed result is the tiling sweep below)

Config: species all (37 outs, scored on 9) · hidden 64 · **lr 1e-5** · crop square+native(0.5) ·
n_per 512 · **r 2** · best epoch 71. **Base-tiling TEST (n=266): sel7 0.698 · all9 0.590.**

| species (test sup)   | rev-3 F1 | **rev-4 F1** | P / R (rev-4) |
|:---------------------|---------:|-------------:|:--------------|
| **Lupinus (181)**    | 0.87 | **0.92** | 0.93 / 0.91 |
| **Balsamorhiza (15)**| 0.67 | **0.70** | 1.00 / 0.53 |
| **Gaillardia (43)**  | 0.64 | **0.66** | 0.67 / 0.65 |
| Poa (139)            | 0.67 | 0.68 | 0.54 / 0.89 |
| Tragopogon (38)      | 0.50 | 0.53 | 0.79 / 0.39 |
| Bromus (12)          | 0.52 | 0.41 | 0.35 / 0.50 |
| Thinopyrum (266)     | 1.00 | 1.00 | 1.00 / 1.00 |
| Achillea (6, excl)   | 0.36 | 0.09 | 0.05 / 0.67 |
| Sisymbrium (5, excl) | 0.36 | 0.33 | 0.29 / 0.40 |

**Positive.** All 3 headline forbs up vs rev-3 at base tiling. **Lupine 0.87 → 0.92 with BOTH precision
(0.90 → 0.93) AND recall (0.83 → 0.91) up** — the precision lift is the first sign the **Sky negative is
suppressing sky-FP** (to be confirmed visually); sel7 ~tied (0.696 → 0.698).
**Negative.** all9 fell (0.622 → 0.590): **Achillea over-fires** (P 0.05 — fires nearly everywhere) and
**Bromus regressed** (0.52 → 0.41); both are outside sel7 but drag all9. The Achillea over-fire is a new
rare-class failure to watch (possibly an r=2 broad-firing side effect on low-support classes).

### rev-4 tiling sweep + deployed result

`sweep_tiling.py` on the rev-4 head; per-tiling thresholds re-derived on val (F1-argmax), applied to
held-out test:

| tiling (val-derived thr → test) | val sel7 | val all9 | test sel7 | test all9 |
|:--------------------------------|---------:|---------:|----------:|----------:|
| base (24 tiles)                 | 0.7871 | 0.6789 | 0.6991 | 0.6023 |
| macro (77 tiles, stride-112)    | 0.7869 | 0.7203 | 0.7095 | 0.6401 |
| **macro+patch (154 tiles, 6×)** | **0.7970** | **0.7257** | **0.7323** | **0.6787** |

**WINNER macro+patch** — frozen into `mil.pt` (tiling + re-derived thresholds). **Deployed rev-4 TEST:
sel7 0.732 · all9 0.679.**

Per-species TEST (macro+patch), headline forbs bold, vs rev-3's macro+patch follow-up:

| species (test sup)   | rev-3 mp | **rev-4 mp** | P / R (rev-4) |
|:---------------------|---------:|-------------:|:--------------|
| **Lupinus (181)**    | 0.91 | **0.93** | **0.98 / 0.90** |
| **Balsamorhiza (15)**| 0.74 | **0.75** | 1.00 / 0.60 |
| **Gaillardia (43)**  | 0.65 | **0.72** | 0.72 / 0.72 |
| Poa (139)            | 0.65 | 0.69 | 0.54 / 0.94 |
| Tragopogon (38)      | 0.45 | 0.51 | 0.82 / 0.37 |
| Bromus (12)          | 0.58 | 0.53 | 0.71 / 0.42 |
| Thinopyrum (266)     | 1.00 | 1.00 | 1.00 / 1.00 |
| Achillea (6, excl)   | 0.60 | 0.32 | 0.23 / 0.50 |
| Sisymbrium (5, excl) | 0.57 | 0.67 | 0.75 / 0.60 |

**Result — rev-4 is the best model on the selection metric and on all three headline forbs.** Deployed
macro+patch **test sel7 0.732** beats rev-3's macro+patch (0.713) and rev-3 base (0.696); **all9 0.679 ≈
rev-3 mp 0.685** (a wash — the Achillea over-fire offsets the forb gains). Within rev-4, denser tiling
helps **monotonically** (base 0.699 → macro 0.710 → macro+patch 0.732), and **even macro-alone now helps**
(it was a wash in rev-3) — consistent with the Sky class making denser sampling *safer* (fewer sky-patch
FPs for the extra tiles to amplify).

**The Sky goal shows up where it should — precision on the sky-confusable forbs:**
- **Lupine precision 0.98** at recall 0.90 (rev-3 mp lupine ≈ 0.90 P) — the head almost never fires lupine
  on a non-lupine frame now, exactly the sky/cloud false-positive suppression rev-4 set out to get.
- **Blanketflower 0.65 → 0.72** (P 0.72) — the second sky-confusable forb, also up.
- all9's rare-class recovery (Sisymbrium 0.57→0.67, Bromus recall) rides the denser tiling + re-derived
  thresholds, **except Achillea, which now over-fires** (P 0.23; 0.60→0.32) — a new rare-class failure
  (plausibly the r=2 broad-firing regime on a 6-frame class); outside sel7 but it caps all9.

**Deployed:** `checkpoints/mil.pt` — rev-4, 37-output head scored on the 9 targets, **tiling=macro+patch**,
per-species thresholds re-derived for that tiling. Overlays + `summary.json` + `mission_predictions.mp4`
refreshed from this checkpoint.

### rev-4 open threads (rev-5 candidates)

- **LR×size×epoch (flagged, deferred).** lr 1e-5 won at ep 71/80 (convergence edge) and 5e-6 underfit
  inside the 80-cap. A **raised-cap (~200) joint LR×size grid** over {5e-6,1e-5,5e-5}×{256,512} would
  resolve whether 1e-5 is fully converged, whether 5e-6 wins given enough epochs, and whether the size
  optimum shifts with LR (the staged chain only tunes them along an L). Offered during rev-4, declined.
- **r < 2.** r=2 was the *floor* of the grid and won on a monotone trend; probe r ∈ {0.5, 1} for the
  turnover toward pure mean / global-average pooling (cheap, head-only).
- **Achillea over-fire.** New rev-4 rare-class failure (P 0.23 on 6 test frames) — plausibly the r=2
  broad-firing regime on a tiny-support class. Outside sel7 but it caps all9.
- **Sky gate strictness.** SegFormer threshold is 10%; a stricter gate (30–50%) would purify the Sky
  crops further (≈600 corners still pass) if sky-FP suppression needs more.

---

## 2026-06-26 — Follow-up ablation: Sky class A/B (does the Sky negative drive the gains?)

**Purpose.** rev-4 changed four things at once (license-clean data, the Sky class, r 8→2, lr 5e-4→1e-5).
The headline claim is that the **Sky** negative suppresses the sky/cloud false positives that capped
lupine/blanketflower precision. Isolate it: train the deployed config WITH vs WITHOUT the Sky class,
everything else identical, and read the headline-forb precision.

**Design & justification.** Head-only on cached features (no re-encode); deployed config (all-36 + Sky /
hidden 64 / square+native(0.5) / n_per 512 / r 2 / lr 1e-5) vs the same with the Sky class dropped
(all-36 only, 36 outputs). Scored at BASE any-patch (the tiling the staged sweep selected on; deploy
adds macro+patch on top), thresholds re-derived on val. `followup_ablations/rev4_ablations.py`.

**Result — the Sky negative is what lifts the sky-confusable forbs' precision.**

| config | val sel7 | test sel7 | lupine F (P/R) | blanketflower F (P/R) | arrowleaf F (P/R) |
|:-------|---------:|----------:|:---------------|:----------------------|:------------------|
| **all-36 + Sky** | **0.7922** | **0.698** | **0.92 (0.93/0.91)** | **0.66 (0.67/0.65)** | 0.70 (1.00/0.53) |
| all-36, NO Sky   | 0.7685 | 0.676 | 0.85 (0.84/0.86) | 0.56 (0.57/0.56) | 0.70 (1.00/0.53) |

**Positive (the result).**
- Sky lifts **val sel7 +0.024** and, exactly as predicted, helps **only the two sky-confusable forbs,
  via precision**: lupine **F 0.85→0.92 (P 0.84→0.93)**, blanketflower **F 0.56→0.66 (P 0.57→0.67)**.
- **Arrowleaf is unchanged** (0.70, P 1.00 both) — its saturated gold is not sky-confusable, so the Sky
  class leaves it alone. This *selective* signature (helps lupine+blanketflower, not arrowleaf) is strong
  evidence the mechanism is sky-FP suppression, not generic regularization.

**Negative.**
- all9 is marginally *lower* with Sky (0.590 vs 0.598) — the Achillea rare-class over-fire (see gradient
  clipping below) is unrelated to Sky and drags all9; sel7 and the headline forbs clearly favor Sky.

**Takeaway.** The Sky negative is the rev-4 win — it does exactly what it was added to do (kill the
lupine/blanketflower sky false positives), confirmed by the selective precision gains. Keep it.

---

## 2026-06-26 — Follow-up ablation: raised-cap convergence (LR × epoch)

**Purpose.** The lr stage picked 1e-5 but at best-epoch 71/80 (the cap edge), and 5e-6 scored a dismal
0.519 — flagged as a possible epoch-budget artifact (a low LR starved of epochs), not a true LR effect.
Test by lifting the 80-epoch cap.

**Design & justification.** Head-only, deployed config (all+Sky / 64 / square+native(0.5) / 512 / r 2),
cap 200 with early-stop DISABLED (patience=200) so each LR runs the full budget; base any-patch,
val-rederived thresholds. `followup_ablations/rev4_ablations.py`.

**Result — 1e-5 is already converged; 5e-6 was epoch-starved, not bad.**

| run | val sel7 | best epoch |
|:----|---------:|-----------:|
| 1e-5 @ cap 80 (deployed) | 0.7922 | 71 |
| 1e-5 @ cap 200 (no stop) | 0.7922 | 71 |
| 5e-6 @ cap 200 (no stop) | 0.7921 | 138 |

**Positive.**
- **1e-5 @ cap200 == cap80 to the digit** (best ep 71 both) → the deployed head is fully converged; the
  cap was not truncating it.
- **5e-6 @ cap200 reaches 0.7921 (ep 138) — a dead tie with 1e-5.** Its sweep score (0.519 @ cap80) was
  **pure epoch-starvation**: at 10× lower LR it needs ~2× the epochs to converge to the same place.

**Negative.**
- Confirms the staged sweep's lr stage can mis-rank LRs that share an optimum but converge at different
  speeds (the 80-cap penalizes the slower one). It still picked a correct LR here (1e-5 is converged and
  ties the slow 5e-6), but 5e-6's *ranking* was a cap artifact.

**Takeaway.** The deployed 1e-5 is sound (converged), and there is no hidden low-LR optimum beyond it —
5e-6 only matches, never beats. The LR×epoch interaction is real but does not change the deployment.

---

## 2026-06-26 — Follow-up ablation: r < 2 (LSE temperature turnover)

**Purpose.** r=2 won the rev-4 r stage as the LOWEST value tried, on a monotone trend (r=2 > 4 > … >
128). Does the optimum lie below 2 (toward pure mean pooling), or is r=2 a true peak?

**Design & justification.** Head-only, deployed config, r ∈ {0.5, 1, 2}; base any-patch, val-rederived
thresholds. `followup_ablations/rev4_ablations.py`.

**Result — r=2 is a sharp peak; r<2 COLLAPSES.**

| r | val sel7 | best epoch | blanketflower P | arrowleaf P |
|--:|---------:|-----------:|----------------:|------------:|
| **2** (deployed) | **0.7922** | 71 | 0.67 | 1.00 |
| 1 | 0.4965 | 3 | 0.17 | 0.06 |
| 0.5 | 0.5028 | 2 | 0.16 | 0.07 |

**Negative (the finding).**
- Below r=2 the head **collapses** — val ~0.50 (vs 0.79), peaks instantly (ep 2–3), and precision
  implodes (blanketflower P 0.67→0.17, arrowleaf P 1.00→0.06 — it fires nearly everywhere).
- Mechanism: at low r the LSE bag score → the **mean** patch logit, so BCE pushes *every* patch up on a
  positive bag; under the any-patch **MAX** inference rule that means everything fires → precision dies.
  The training pool (mean) and the inference rule (max) become badly mismatched.

**Takeaway.** r=2 is a genuine optimum bracketed by a **cliff below** (collapse) and a gentle decline
above (r≥4). "Lower toward mean pooling" is firmly closed; the deployed r=2 is well-placed.

---

## 2026-06-26 — Follow-up ablation: gradient clipping

**Purpose.** Two questions: (1) does grad-norm clipping help the deployed optimum (r=2, lr 1e-5)?
(2) does it RESCUE the hot high-LR regime — lr 5e-4 overfits at epoch ~0–1 (best epoch collapses), and
bounding the step might let it train stably (a faster route to the optimum)?

**Design & justification.** Head-only, deployed config (all+Sky / 64 / square+native(0.5) / 512 / r 2);
an opt-in `clip` arg added to `train_head` (default None = unchanged, deployed pipeline untouched).
Grid LR ∈ {1e-5, 5e-4} × clip ∈ {none, 1.0, 0.5}; base any-patch, val-rederived thresholds.
`followup_ablations/rev4_gradclip.py`.

**Result — clip 1.0 at the optimum is a small near-free gain; clip does NOT rescue the hot LR.**

| LR | clip | val sel7 | test sel7 | all9 | best ep |
|---:|:-----|---------:|----------:|-----:|--------:|
| 1e-5 | none (deployed) | 0.7922 | 0.6984 | 0.5904 | 71 |
| 1e-5 | **1.0** | **0.7925** | **0.7034** | **0.6013** | 69 |
| 1e-5 | 0.5 | 0.7885 | 0.6962 | 0.6122 | 76 |
| 5e-4 | none (hot ref) | 0.7772 | 0.6880 | 0.6066 | 1 |
| 5e-4 | 1.0 | 0.7770 | 0.6842 | 0.6111 | 1 |
| 5e-4 | 0.5 | 0.7755 | 0.6987 | 0.6102 | 1 |

**Positive.**
- **clip 1.0 @ 1e-5 (the optimum): a small net gain** — val tied (0.7925), **test sel7 +0.005 (0.703),
  all9 +0.011 (0.601)** — essentially free, headline forbs unchanged. The deployment-relevant cell.
- clip helps **all9 / rare classes** monotonically as it tightens (0.590 → 0.601 → 0.612 at 1e-5) — the
  rare-class pos_weights spike gradients, which clipping tames; relevant to the Achillea over-fire.

**Negative.**
- **Clipping does NOT rescue the hot LR.** 5e-4 still peaks at **epoch 1** at every clip (val ~0.776,
  unchanged) — the ep-0 overfit is fast convergence to a worse basin, not a large-gradient instability,
  so clipping cannot move the optimum upward.
- clip 0.5 @ 1e-5 is too tight — val drops (0.7885) and blanketflower erodes (F 0.66→0.60); its all9
  gain is a rare-class-for-forb trade, not worth it for the headline.

**Takeaway.** Gradient clipping is a minor lever: **clip 1.0 at the deployed 1e-5 is a near-free +0.005
test sel7 / +0.011 all9** (worth folding into the final if re-trained), tighter clipping trades forb
precision for rare-class, and it does NOT unlock a faster high-LR optimum (5e-4 stays ep-1). A candidate
deploy tweak, not a structural change.

---

## 2026-06-26 — Proposed: augmentation sweeps (GPU-enabled; queued for cluster migration)

**Why now / why not yet.** The current pipeline CACHES DINOv3 features — the frozen backbone runs once
per crop, one fixed realization per image (even the 50% flip is baked into the cache) — so true
per-epoch *stochastic* augmentation is infeasible on MPS: it needs a live backbone pass every epoch. The
imminent real-GPU cluster unlocks exactly this. (Cache-compatible fallback if needed sooner: pre-extract
K augmented copies per image — "offline K-fold aug" — at K× extraction/storage.) Separately, per the
gradient-clipping ablation, **clip 1.0 @ 1e-5 is a confirmed near-free tweak** (test sel7 +0.005 / all9
+0.011) — fold it into the GPU re-train rather than redo the MPS pipeline now.

**Framing.** The backbone is frozen, so augmentation's only job is to push iNat training features toward
the ROBOT feature distribution (close the iNat→robot domain gap). **Photometric / domain-shift augs are
therefore prioritized over geometric ones.** Each sweep: rev-4 config (all+Sky, hidden 64, n_per 512,
r 2, lr 1e-5 + clip 1.0), select on val sel7, evaluate at macro+patch; track headline-forb **precision**
+ the Achillea over-fire, not just sel7; and **re-confirm r and the tiling after the winning aug** (aug
changes per-patch signal sharpness, so the r=2 / macro+patch optima may shift). Ranked by expected value:

1. **Photometric / exposure jitter — highest priority.** *Purpose:* lighting is the largest iNat→robot
   gap (robot = harsh outdoor, frequent sky/highlight blowout; iNat = varied / well-exposed). *Design:*
   per-epoch brightness / contrast / saturation / hue + gamma; sweep strength {mild, moderate, strong}
   plus an overexposure-biased variant. *Hypothesis:* the biggest single gap-closer — lighting-robust
   features lift forb precision (fewer lighting-driven FPs, incl. sky) and recall.

2. **RandomResizedCrop / scale-jitter.** *Purpose:* replace the two fixed views (square + native 0.5) with
   stochastic scale + location sampling — the robot sees plants at many scales/offsets within a 224 tile.
   *Design:* per-epoch RandomResizedCrop, sweep scale {(0.5,1.0), (0.3,1.0), (0.2,0.8)} + mild aspect.
   *Hypothesis:* a scale-robust head compounds the macro+patch recall gains (no longer tuned to 2 scales).

3. **Blur / defocus / motion blur.** *Purpose:* the moving quadruped + field depth → motion blur &
   defocus; iNat crops are sharp. *Design:* gaussian + directional motion blur, sweep sigma/kernel.
   *Hypothesis:* recovers recall on blurry robot plants — an FN source the sharp iNat crops never train.

4. **Occlusion (random erasing / cutout).** *Purpose:* robot plants sit in a dense grass matrix (heavily
   occluded); iNat plants are often isolated. *Design:* random-erasing, sweep area-fraction + count.
   *Hypothesis:* recall on partially-hidden plants; synergizes with the any-patch rule (head must fire on
   fragments).

5. **Policy meta-sweep (RandAugment / TrivialAugment).** *Purpose:* avoid hand-tuning the mix — sweep a
   single magnitude knob over a standard, photometric-weighted policy. *Design:* RandAugment N/M sweep.
   *Hypothesis:* a tuned magnitude captures most of 1–4 in one axis; the pragmatic default if hand-tuning
   underperforms.

6. **Synthetic compositing — stretch.** *Purpose:* the deepest gap is "isolated iNat plant" vs "plant
   embedded in grass/field". *Design:* alpha-composite SegFormer/SAM-masked iNat plant crops onto
   robot-like field + sky/horizon backgrounds; sweep paste density. *Hypothesis:* the strongest domain
   bridge, but the most build effort — pursue only if 1–5 plateau.

**Cross-cutting.** If photometric/geometric aug yields little, the GPU also unlocks the bigger lever —
**backbone fine-tuning / LoRA** (a separate axis, out of scope here, but likely larger than any single
augmentation). Suggested cluster sequence: (clip 1.0 + photometric) first, then scale-jitter, then
re-sweep r + tiling on the winner, then decide whether to fine-tune the backbone.

---

## 2026-06-29 — Finding + Proposed: seasonal/temporal sampling bias; month-balanced re-harvest

### Finding — the iNat 1000/species is a recency snapshot, not a seasonal sample

**Purpose.** Audit whether the harvested 1000 imgs/species actually represent each species' available
date range, or carry a seasonal/temporal bias that opens an iNat→robot **phenology** gap (a plant's
appearance changes across the season: bud → flower → fruit → senescence; the robot meets all states).

**Design & justification.** Cross-checked the local artifacts (`metadata.csv` kept set; the
`manifest.raw.jsonl` 1500-candidate pool) against iNaturalist's own **all-time `month_of_year`
histogram**, queried under pixelflora's *exact* filters (research grade, geo, captive=false, license
keep-list) for all 36 species + the eagle Sky class. Filters verified identical: where pixelflora
exhausted the pool, local == iNat (Poa secunda 805=805, Antennaria 975≈976, Thinopyrum 1178≈1187).
N-hemisphere assumption safe (0.9% of obs southern). The 1500→1000 download step is *negligible*
(median month-dist TVD 0.073, peak month preserved on all 36) — the bias is upstream, in the harvest.
[Hesperostipa comata excluded from the iNat comparison: taxon 157658 returns 69 vs 1294 harvested —
a Stipa/Hesperostipa synonymy rollup; its local pool is authoritative.]

**Result — the sample is a snapshot of the weeks *before harvest*, and worse the more abundant the
species.** Mechanism: `inaturalist.py:harvest()` pages `order_by=id desc` (newest-first) to a 1500
buffer (= 1.5 × the 1000 target, `media.overharvest`); at a late-June harvest the newest 1500 of an
abundant taxon all fall inside the current partial year (2026 = Jan–Jun only), so (a) only ~2–5 recent
years are seen and (b) the Jul–Dec tail is amputated. Coverage vs true license-filtered availability:

| species (focal) | available | kept | **% captured** | yrs in kept | %2026 | seasonTVD (kept vs avail) |
|---|--:|--:|--:|--:|--:|--:|
| Achillea millefolium | 189,226 | 1000 | **0.5%** | 5 | 99% | **0.77** |
| Tragopogon dubius | 27,605 | 1000 | 3.6% | 7 | 99% | **0.49** |
| Balsamorhiza sagittata | 15,298 | 1000 | 6.5% | 11 | 98% | 0.29 |
| Bromus tectorum | 14,024 | 1000 | 7.1% | 7 | 99% | 0.26 |
| Poa bulbosa | 11,200 | 1000 | 8.9% | 4 | 100% | 0.18 |
| Gaillardia aristata | 10,752 | 1000 | 9.3% | 23 | 51% | 0.10 |
| Sisymbrium altissimum | 3,636 | 1000 | 27.5% | 15 | 34% | 0.14 |
| Lupinus sericeus | 1,732 | 1000 | 57.7% | 9 | 25% | 0.02 |
| Thinopyrum intermedium | 1,187 | 1000 | 84.2% | 26 | 4% | 0.03 |

Across all 36: **13 faithful** (TVD<0.10), **15 moderate**, **8 severe** (>0.40 — Achillea .77,
Taraxacum .74, Silene .68, H. annuus .66, Chenopodium .62, Tragopogon .49, Verbascum .49, Euphorbia .42).

**Positive.** Spring bloomers whose whole season fits the captured window are sampled faithfully
(Lupinus TVD 0.02, Thinopyrum 0.03, Poa secunda 0.00); scarce species **self-correct** — they must
reach back 20–26 yrs to fill 1500, averaging over all seasons. No action needed for these, and the fix
below provably does not regress them.

**Negative (the finding).** Late-season (Jul–Dec) is systematically erased; severe species collapse
onto a single spring peak: **Achillea** kept 97% June (available spreads Apr–Oct; Δlate −51pp);
**Tragopogon** kept 91% June (Δlate −28pp); **Bromus tectorum** 0% Jul–Dec — we hold **no cured/brown
summer cheatgrass**, the very state it's in as a fire fuel; **Balsamorhiza** lost both the April shoulder
and the July tail. The **Sky class is the worst of all** (TVD 0.77): 84% June / **3% winter** skies vs
**29% Dec–Feb** available — it learned "bright June sky ≠ lupine" and never saw winter / overcast /
low-sun / snow-bright sky. Governing law: corr(log₁₀ availability, seasonTVD) = **+0.83**;
corr(%-from-2026, seasonTVD) = **+0.80** (abundant ⇒ buffer = a few recent weeks ⇒ worst bias).

**Takeaway.** This is a **data-collection** bias, orthogonal to every head hyperparameter swept in rev-4
— no amount of LR/r/tiling tuning fixes a missing phenophase. Photometric augmentation can fake the
missing seasonal *lighting* but **cannot synthesize a fruiting/senescent plant**; only re-harvest can.
Fold into the same cluster re-harvest as the augmentation sweeps.

### Proposed — month-balanced (water-filling) sampling

**Goal.** For each species, strive for **equal proportions per calendar month**, bounded by what iNat
actually has — i.e. uniform across the *observable* season, degrading gracefully where months are empty.

**Why uniform, not representative.** The robot meets a plant across its whole deployment season; lacking
a prior on that season, uniform-over-months is the max-entropy safe target. Model failures concentrate in
the **under-sampled minority phenophases** (senescent Bromus, fruiting Tragopogon, summer-long Achillea);
balancing up-weights exactly those — aligned with the any-patch MIL goal (fire on the plant in *any*
state). Thousands of redundant peak-bloom "Mays" are available; trading some for rare phenophases is a
good deal for a presence/absence model.

**Algorithm (water-filling — the exact "equal per month subject to availability").**
1. Probe the `month_of_year` histogram once → available `a_m`, m=1..12 (cheap; this is what the audit did).
2. Find a common ceiling `L` with `Σ_m min(a_m, L) = B` (buffer = 1.5 × target); per-month candidate
   quota `q_m = min(a_m, L)`. Months with `a_m ≥ L` are **equalized to L** (the "equal" target); scarcer
   months contribute everything they have; the deficit auto-redistributes to richer months. `Σ q_m = B`
   by construction (binary-search `L`; if `Σ a_m ≤ B`, take everything).
3. Harvest per month: page `/observations?…&month=m&order_by=id desc` to `q_m` candidates — iNat's
   `month=` param filters by **observed month-of-year**, exactly the axis we want.
4. Download to target `T`; the 1.5× buffer absorbs dup/fetch failures as today.

**Properties / edge cases.**
- *Achillea* (flat Apr–Oct, 189k): `L≈B/12` → ~uniform across the year — the worst case is fixed.
- *Lomatium / Holosteum* (spring ephemeral, ≈0 in Nov–Feb): quota concentrates on Mar–Jul, capturing
  both shoulders; **no attempt to invent absent winter data**.
- *Thinopyrum / Poa secunda* (`Σ a_m ≤ B`): take everything → **provably no regression** for the
  already-faithful, near-exhausted species.
- *Sky / eagle*: same harvest → winter / overcast skies enter the negative set (highest-value single fix).

**Config surface (pixelflora).**
```toml
[media]
sampling = "month_balanced"   # new; default "recent" keeps today's newest-first behavior
# months = [3,4,5,6,7,8,9]     # optional: restrict to a calendar window (e.g. growing season)
```
`harvest()` branches on `media.sampling`; `recent` is unchanged.

**Success metric flips.** seasonTVD-to-availability is no longer the target (it will be *large* by
design — that's the point). Report instead **month-coverage uniformity** (normalized entropy of kept
months over active months → 1.0) and **active-month coverage** (# months holding ≥5% of the kept set).

**Residual axis (Phase 2, optional).** Within a month it's still newest-first, so an abundant month's
quota can come entirely from 2026. To de-bias **year** too, extend water-filling to month×year cells (or
cap ≤K per (year, month)). The current ask is months; year-balancing is the next layer.

**Cost.** +1 histogram call and up to 12 paginated streams per species; total records fetched ≈ B
(unchanged). Comfortably within iNat rate limits for 36 species + the eagle.

---

## 2026-06-29 — rev-5 sweep stages (month-balanced data; new phenology stage; clip 1.0 locked; 128 tune size)

Re-run of the staged MIL sweep on the **month-balanced re-harvest** (38 plant species + Sky, equal-per-month
sampling — see the month-balanced entries above). Changes vs rev-4: (1) **phenology stage** added after species
(training-image month window: `summer`=JJA / `extended`=May-Sep / `all`=12-mo, each drawn EQUALLY per month);
(2) **`grasses` species level dropped**; (3) **two new hard-negative species** (Helianthella uniflora,
Polygonum douglasii → 38 plants, D.N still 9); (4) **grad-norm clip = 1.0 LOCKED** on every run (gradclip
ablation); (5) **tune size 256 → 128** (keeps the phenology windows comparable — only Holosteum < 128, and only
under JJA); (6) species stage **defaults to the May-Sep window**. Feature caches now carry a dataset
content-fingerprint, so the re-harvest forces a clean rebuild (no stale recency features). Run order:
species → phenology → hidden → crop → size → r → lr → final. Selection = robot-val **sel7**.

### rev-5 stage 1 — species set  (winner: ALL, val sel7 0.7366)

Fixed: pheno `extended` (May-Sep) · hidden 64 · lr 5e-5 · r 8 · crop native(1.0) · 128 imgs/sp · clip 1.0.
Sky-gate on the month-balanced eagles: UL 761/1000, UR 768/1000 corners ≥10% sky (≈ rev-4's 736/727).

| species set | classes | val sel7 | val all9 | best ep |
|---|--:|--:|--:|--:|
| targets | 10 | 0.7036 | 0.6345 | 51 |
| **all** | **39** | **0.7366** | 0.6275 | 63 |

**Result — broad negatives win again.** `all` (38 plants + Sky) beats `targets`-only by **+0.033 sel7**
(0.7366 vs 0.7036), echoing rev-4. all9 is marginally lower for `all` (0.6275 vs 0.6345): the extra negatives
cost a touch on the 2 rare excluded species (Achillea/Sisymbrium) but lift the 7 selection species. Carried
forward: **species = all (39 classes)**. (Phenology stage next, on the `all` set.)

### rev-5 stage 2 — phenology window  (winner: EXTENDED / May-Sep, val sel7 0.7366)

Fixed: species `all` (39) · hidden 64 · lr 5e-5 · r 8 · crop native(1.0) · 128 imgs/sp · clip 1.0. Each window
draws EQUALLY per month (round-robin, capped by per-month availability). The `extended` candidate reproduced
the stage-1 `all` result **exactly** (0.7366, ep 63) — a caching/determinism consistency check that passed.

| window | months | val sel7 | val all9 | best ep |
|---|---|--:|--:|--:|
| summer | Jun–Aug | 0.7299 | **0.6550** | 36 |
| **extended** | **May–Sep** | **0.7366** | 0.6275 | 63 |
| all | 12-mo | 0.7191 | 0.6291 | 59 |

**Result — seasonal matching helps; May–Sep is the sweet spot.** sel7 PEAKS at `extended` (0.7366) and falls
off on BOTH sides: `summer` 0.7299 (too narrow — ~½ the imgs/species, Holosteum down to 25) and `all` 0.7191
(too wide — the off-season half is senescent/dormant/dead-stalk imagery that doesn't match the mid-June robot
scene and adds label noise). Widening to the full 12 months **actively hurts** the selection species
(−0.0175 vs extended). This **validates the phenology thesis**: iNat training imagery should be restricted to
the deployment season — the whole motivation for the month-balanced re-harvest.

**Tradeoff worth noting:** `summer` (JJA) has the BEST all9 (0.6550 vs extended 0.6275) — the tight summer
window most helps the two rare *excluded* species (Achillea/Sisymbrium), even while trailing on the 7
selection species. So the "best" window depends on the objective: **May–Sep for sel7 (the selection metric);
JJA if rare-species recall were weighted in.** On sel7, extended wins.

Carried forward: **phenology = extended (May–Sep)**.

### rev-5 stage 3 — head width  (winner: hidden 64, val sel7 0.7366)

Fixed: species `all` · pheno extended · lr 5e-5 · r 8 · crop native(1.0) · 128 imgs/sp · clip 1.0. Features
cached (no re-extraction). `hidden 64` reproduced the carried baseline exactly (0.7366) — consistency check.

| hidden | val sel7 | val all9 | ep |
|---|--:|--:|--:|
| 0 (linear) | 0.7031 | 0.6033 | 79 |
| 32 | 0.6904 | 0.5767 | 75 |
| **64** | **0.7366** | 0.6275 | 63 |
| 128 | 0.6998 | 0.5870 | 66 |
| 256 | 0.7316 | 0.6196 | 79 |

**Result — hidden 64 wins (non-monotone).** 64 is a clean peak; 256 is runner-up (0.7316) while 0/32/128 dip
to 0.69–0.70. Matches rev-4 (64 won). Carried forward: **hidden = 64**.

### rev-5 stage 4 — crop  (winner: square + native(1.0), val sel7 0.7533)

Fixed: species `all` · pheno extended · hidden 64 · lr 5e-5 · r 8 · 128 imgs/sp · clip 1.0. Each candidate =
square whole-tile view + native(z) zoom view; the square view is fresh-extracted once then cached across zooms.

| crop (square + native z) | val sel7 | val all9 | ep |
|---|--:|--:|--:|
| z=0.5 (zoom-out) | 0.7484 | **0.6706** | 16 |
| z=0.75 | 0.7470 | 0.6574 | 18 |
| **z=1.0 (tight native)** | **0.7533** | 0.6451 | 31 |

**Result — square + native(1.0) wins; all three beat the native-only baseline.** Adding the square whole-tile
view lifts sel7 from 0.7366 (native-1.0 only, stage 3) to 0.747–0.753 regardless of the second view's zoom.
**REVERSAL from rev-4** (zoom-out z=0.5 won there): on the month-balanced data the tightest native(1.0) second
view edges it (0.7533 vs 0.7484). As in earlier stages the **all9 trend inverts sel7** — z=0.5 has the best
all9 (0.6706), i.e. the zoom-out view most helps the 2 rare excluded species, but z=1.0 wins the 7 selection
species. Carried forward: **crop = square + native(1.0)**.

### rev-5 stage 5 — data size  (winner: n_per 512, val sel7 0.7645)

Fixed: species `all` · pheno extended · hidden 64 · lr 5e-5 · r 8 · crop square+native(1.0) · clip 1.0.
Extracted once at 512 (the May-Sep-capped ceiling for data-limited species) and subset to each n_per.

| n_per | val sel7 | val all9 | ep |
|---|--:|--:|--:|
| 64 | 0.7584 | 0.6767 | 60 |
| 128 | 0.7533 | 0.6451 | 31 |
| 256 | 0.7620 | 0.6575 | 41 |
| **512** | **0.7645** | 0.6757 | 14 |

**Result — 512 wins; more data is better at the top.** Non-monotone (128 dips) but the max-data 512 takes it
(0.7645), and the best epoch collapses 60→14 as data grows — same convergence pattern as rev-4. 512 here is the
May-Sep-capped ceiling (abundant species reach 512; spring/scarce species fewer). n_per 128 reproduced the crop
z=1.0 winner exactly (0.7533, consistency ✓). Running val-sel7 trend: 0.7366 → crop 0.7533 → size 0.7645.
Carried forward: **n_per = 512**. (r + lr still to come — these gave rev-4 its biggest lifts.)

### rev-5 stage 6 — r (LSE bag temperature)  (winner: r=2, val sel7 0.7738)

Fixed: species `all` · pheno extended · hidden 64 · lr 5e-5 · crop square+native(1.0) · n_per 512 · clip 1.0.
Cached features.

| r | val sel7 | val all9 | ep |
|---|--:|--:|--:|
| **2** | **0.7738** | 0.6722 | 25 |
| 4 | 0.7685 | 0.6788 | 19 |
| 8 | 0.7645 | 0.6757 | 14 |
| 16 | 0.7600 | 0.6646 | 17 |
| 32 | 0.7648 | 0.6821 | 14 |
| 64 | 0.7711 | 0.6809 | 17 |
| 128 | 0.7666 | 0.6793 | 14 |

**Result — r=2 wins (0.7738), but the curve is U-SHAPED, not monotone.** Unlike rev-4's clean decline above
r=2, here BOTH ends are strong: r=2 (mean-like pooling) peaks, r=64 (max-like) is runner-up (0.7711), and the
middle (r=8–16) sags to ~0.76. So both "average the bag" and "trust the single strongest patch" beat the
middle — the mean end wins sel7, reproducing rev-4's r=2 verdict. r=8 reproduced the size baseline exactly
(0.7645, consistency ✓). Carried forward: **r = 2**. Running val-sel7: 0.7366 → 0.7533 → 0.7645 → **0.7738**.

### rev-5 stage 7 — learning rate  (winner: lr 5e-5, val sel7 0.7738)

Fixed: species `all` · pheno extended · hidden 64 · r 2 · crop square+native(1.0) · n_per 512 · clip 1.0. Cached.

| lr | val sel7 | val all9 | ep |
|---|--:|--:|--:|
| 5e-6 | 0.5227 | 0.4260 | 13 |
| 1e-5 | 0.7546 | 0.6738 | 49 |
| **5e-5** | **0.7738** | 0.6722 | 25 |
| 1e-4 | 0.7653 | 0.6710 | 12 |
| 5e-4 | 0.7567 | 0.6612 | 1 |
| 1e-3 | 0.7655 | 0.6731 | 0 |

**Result — lr 5e-5 wins; REVERSAL from rev-4 (which picked 1e-5).** The standard 5e-5 (carried default) takes
it; gentler 1e-5 trails (0.7546) and 5e-6 underfits hard (0.5227 at the 80-ep cap). Hot rates overfit instantly
(5e-4 best-ep 1, 1e-3 best-ep 0). Likely mechanism: **clip 1.0 (now locked on) stabilizes the higher 5e-5**, so
rev-4's need for a gentler LR disappears. Carried forward: **lr = 5e-5**.

**Final config locked:** species `all` (38 plants + Sky) · **May–Sep** · hidden 64 · crop **square+native(1.0)**
· n_per 512 · **r 2** · lr 5e-5 · clip 1.0. Best val sel7 through the sweep = **0.7738**. Final run (held-out
TEST + checkpoint) below.

### rev-5 stage 8 — FINAL (base-tiling checkpoint)  ·  TEST sel7 0.6964 / all9 0.6860

Config: species `all` (38 plants + Sky, 39 cls) · May–Sep · hidden 64 · crop square+native(1.0) · n_per 512 ·
r 2 · lr 5e-5 · clip 1.0 · best epoch 25. Saved → `checkpoints/mil.pt` (rev-4 preserved as `mil_rev4.pt`).
**BASE tiling** — the tiling sweep (the deployment step that lifted rev-4) is NOT yet run.

VAL (n=267): sel7 **0.7738** · all9 0.6722.  TEST (n=266): sel7 **0.6964** · all9 **0.6860**.

| species (TEST) | τ | P | R | F1 | sup |
|---|--:|--:|--:|--:|--:|
| Lupinus sericeus | 0.09 | 0.93 | 0.91 | **0.92** | 181 |
| Poa bulbosa | 0.49 | 0.59 | 0.86 | 0.70 | 139 |
| Tragopogon dubius | 0.91 | 0.83 | 0.39 | 0.54 | 38 |
| Gaillardia aristata | 0.41 | 0.55 | 0.53 | 0.54 | 43 |
| Balsamorhiza sagittata | 0.55 | 0.90 | 0.60 | 0.72 | 15 |
| Bromus tectorum | 0.64 | 0.43 | 0.50 | 0.46 | 12 |
| Achillea millefolium *(excl)* | 0.29 | 0.80 | 0.67 | **0.73** | 6 |
| Sisymbrium altissimum *(excl)* | 0.82 | 1.00 | 0.40 | 0.57 | 5 |
| Thinopyrum intermedium | 0.02 | 1.00 | 1.00 | 1.00 | 266 |

**Headline — the rev-4 Achillea over-fire is FIXED, and all9 already beats rev-4's *deployed* model at base
tiling.** rev-4's biggest open issue was Achillea firing nearly everywhere (deployed P 0.23, base P 0.05); on
the month-balanced May–Sep data it's now **P 0.80 / F 0.73**. That lifts **base-tiling all9 0.686 vs rev-4's
deployed (macro+patch) all9 0.679** — i.e. the data fix alone, before any tiling gain, recovers the rare-class
collapse. Lupine stays excellent (0.92); arrowleaf 0.72; Thinopyrum perfect.

**Caveats / open items.** (1) **sel7 0.6964 at base trails rev-4's *deployed* 0.732 — but that's not
apples-to-apples**: rev-4's 0.732 includes the macro+patch tiling sweep, which is still pending here; the
fair next step is to run `sweep_tiling.py` on this checkpoint. (2) **Larger val→test gap** (sel7 0.774→0.696,
−0.077 vs rev-4's −0.065), concentrated in the low-support species: blanketflower (val 0.69→test 0.54), Bromus
(0.78→0.46), Tragopogon. Headline **blanketflower is the weak forb at base (0.54)** — watch whether tiling
recovers it. (3) lr/crop reversed vs rev-4 (5e-5 not 1e-5; native-1.0 not 0.5), plausibly because clip 1.0 is
now locked on. (4) The r-stage all9-vs-sel7 split suggests **per-species r heterogeneity** (rare species favor
high r ~32, selection species r=2) — a per-class-r head is a candidate rev-6 lever.

Next: tiling sweep (deployment) → then the apples-to-apples rev-4 comparison.

### rev-5 stage 9 — tiling DEPLOYMENT (macro+patch)  ·  TEST sel7 0.7437 / all9 0.7340  ← DEPLOYED

macro+patch forced (per request); it was ALSO the best val sel7, so the override matched the natural winner.
Per-tiling thresholds re-derived on val, applied to held-out test. **Denser tiling helps MONOTONICALLY** on
both metrics (Sky makes the extra tiles safe, as in rev-4):

| tiling | val sel7 | val all9 | test sel7 | test all9 |
|---|--:|--:|--:|--:|
| base | 0.7731 | 0.6717 | 0.6968 | 0.6802 |
| macro | 0.7774 | 0.7104 | 0.7188 | 0.6844 |
| **macro+patch** | **0.7878** | 0.7134 | **0.7437** | **0.7340** |

macro+patch held-out TEST per-species (n=266):

| species | τ | P | R | F1 | sup |
|---|--:|--:|--:|--:|--:|
| Lupinus sericeus | 0.21 | 0.97 | 0.91 | **0.94** | 181 |
| Poa bulbosa | 0.63 | 0.58 | 0.88 | 0.70 | 139 |
| Tragopogon dubius | 0.94 | 0.74 | 0.45 | 0.56 | 38 |
| Gaillardia aristata | 0.65 | 0.70 | 0.49 | 0.58 | 43 |
| Balsamorhiza sagittata | 0.70 | 0.91 | 0.67 | 0.77 | 15 |
| Bromus tectorum | 0.83 | 0.78 | 0.58 | 0.67 | 12 |
| Achillea millefolium *(excl)* | 0.62 | 1.00 | 0.67 | 0.80 | 6 |
| Sisymbrium altissimum *(excl)* | 0.85 | 0.60 | 0.60 | 0.60 | 5 |
| Thinopyrum intermedium | 0.02 | 1.00 | 1.00 | 1.00 | 266 |

**Result — rev-5 is the best model in the program; beats rev-4 deployed on BOTH metrics.**
rev-5 macro+patch **sel7 0.7437 / all9 0.7340** vs rev-4 deployed **0.732 / 0.679** → **sel7 +0.012, all9
+0.055**. The all9 jump is the Achillea fix (rev-4 over-fired P 0.23 → rev-5 **P 1.00 / F 0.80**). Headline
forbs: **lupine 0.94 (↑ 0.93), arrowleaf 0.77 (↑ 0.75)** — best yet; Bromus jumps (base 0.46 → 0.67).
**One regression: blanketflower 0.58 (↓ from rev-4's 0.72)** — recall-limited (R 0.49); the May-Sep window
or the crop/lr reversals appear to cost Gaillardia recall. Frozen into `mil.pt` (rev-4 → `mil_rev4.pt`).

**Open for rev-6:** blanketflower recall regression; per-class r (the all9-vs-sel7 heterogeneity finding);
the larger val→test gap on low-support species.
