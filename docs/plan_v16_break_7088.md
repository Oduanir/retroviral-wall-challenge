# Plan V16 — Break 0.7088

## Goal

Show that `CLS 0.7088` is not a hard ceiling, then target a higher honest nested LOFO score with minimal drift from the current best pipeline.

Core idea:
- keep the current `v6` backbone as the reference
- protect `W-Spearman = 0.7361`
- focus new work on the remaining classification bottleneck, especially difficult families

---

## Working Hypothesis

`v6` is strong because it already captures most of the ranking signal.
The remaining headroom is likely in:
- better `active/inactive` separation
- better handling of difficult examples (`Retron`, borderline cases, unstable predictions)
- better use of the existing complementarity between `EN`, `L12`, and `L33`

This plan does **not** assume a new representation breakthrough.
It assumes there is still unused value in the current backbone.

---

## Step 1 — Prove Headroom Exists

Before adding new models, run three oracle-style diagnostics on the current `v6` OOF predictions.

### Exp 1.1 — Oracle Classification

Keep the current ranking shape, but replace the `active/inactive` separation with ground truth.

Question:
- how much CLS is left if classification were fixed while ranking stayed mostly intact?

Interpretation:
- large gain => classification is the real remaining bottleneck
- small gain => the current score is much closer to a real ceiling

### Exp 1.2 — Oracle Retron

Replace predictions only for the `Retron` family with ground truth or near-perfect ordering, keep all other families unchanged.

Question:
- how much of the remaining gap comes specifically from `Retron`?

Interpretation:
- large gain => target `Retron` directly
- small gain => the remaining error is broader and not just one-family specific

### Exp 1.3 — Oracle Per-Family Best Base Model

For each outer fold family, retrospectively choose the best among:
- `EN`
- `L12`
- `L33`

Question:
- is there still exploitable complementarity that the current blend does not use well enough?

Interpretation:
- large gain => better meta-combination is worth pursuing
- small gain => the current nested blend is already near-optimal

### Go / No-Go for Step 1

Go if at least one oracle suggests meaningful headroom, e.g.:
- `CLS +0.01` or more from fixing classification
- or clear family-specific upside

No-Go if all three oracles show negligible margin.

---

## Step 2 — Main Bet: One Classification Corrector on Top of v6

Do not replace `v6`.
Add one small correction layer whose only job is to improve `PR-AUC` while preserving ranking.

This step now absorbs the useful parts of the previous contextual stacking and hard-case weighting ideas.
The goal is to avoid building several meta-layers on top of only `57` examples.

### Exp 2.1 — Active Probability Corrector

Train a small classifier to predict `active` using:
- final `v6` score
- base model scores: `EN`, `L12`, `L33`
- pairwise disagreement features
- a very small subset of robust tabular/context features
- missingness indicators already known to matter
- instability features derived from bootstrap / inner-fold variance

Strict budget:
- no large feature set
- target `<= 12` total inputs
- no tree models

Candidate models:
- logistic regression
- Ridge classifier
- very small calibrated linear model

Output:
- `p_active`

### Exp 2.2 — Conservative Score Adjustment

Use `p_active` to adjust the current regression score without breaking rank structure too aggressively.

Candidate forms:
- `score_new = score_v6 + lambda * g(p_active)`
- monotone calibration of `score_v6` by `p_active`
- inactive-penalty only near the decision boundary

Key constraint:
- changes should mostly affect probable false positives / false negatives
- not reorder strong actives heavily

### Exp 2.3 — Optional Weighted Training Inside the Corrector

Do not weight families by hand.
If weighting is used, it should be applied only inside this corrector.

Define hard cases empirically from current model behavior:
- repeatedly misclassified across inner folds
- unstable across bootstrap resamples
- close to the active/inactive boundary
- high-impact errors for CLS

This is intentionally narrower than the previous hard-case weighting experiments from Phase 10:
- no weighting of the full base pipeline
- no rebuilding of the full regression stack
- only a small correction model on top of `v6`

### Success Criterion

Primary:
- improve `PR-AUC` with little `W-Spearman` loss

Concrete target:
- `PR-AUC +0.02` with `W-Spearman` drop <= `0.01`

### Why This Is the Main Bet

Because the current score is imbalanced:
- ranking is already strong
- classification is still the fragile part

This is the highest-probability path to a cleaner gain.

---

## Step 3 — Frontier Bet if Step 2 Fails: ESM2-3B

Only do this if the lower-cost backbone-centered corrector fails.

### Exp 3.1 — Frozen ESM2-3B Mid-Region Layer Sweep

Replicate the successful `650M` recipe:
- no fine-tuning
- mid-region only
- limited layer sweep
- same PCA + Ridge framework

Goal:
- test whether a larger model yields a genuinely new embedding signal

Implementation constraints:
- inference only, frozen model
- expect tight VRAM on `RTX 4070 Super 12GB`
- use conservative batching
- keep a fallback for long sequences if memory becomes unstable

### Minimal Scope

Test only a few promising layers, not the entire stack.

Examples:
- one intermediate layer
- one late layer
- maybe one blend with existing `L12/L33`

### Why This Is Last

It is more expensive and lower-confidence than Step 2, but still more plausible than restarting broad exploration.

---

## What Not To Do

Stop spending time on:
- new global PLMs "just to try"
- new local-only active-site features
- heavy anti-phylogeny methods
- more LoRA variants
- more raw PDB feature blocks
- complex ranking losses
- broad hyperparameter fishing

These have already been explored enough to have low expected value.

---

## Recommended Execution Order

1. Oracle diagnostics
2. One classification corrector on top of `v6`
3. `ESM2-3B` only if Step 2 fails

---

## Decision Rules

### Strong Success

- `CLS > 0.715`
- or stable gain of `>= 0.01` over `0.7088`

### Moderate Success

- `CLS +0.005`
- or meaningful `PR-AUC` gain with preserved ranking

### Failure

- gain only on one fold or one family
- `W-Spearman` loss larger than classification gain
- unstable nested LOFO behavior

---

## Expected Outcome

This plan is not designed to invent a radically new biology signal.
It is designed to answer two practical questions:

1. Is there still measurable headroom above `0.7088`?
2. If yes, can it be captured by fixing classification on top of the current strong ranking backbone?

If the answer to both is no, then `0.7088` becomes much closer to a demonstrated local ceiling.
