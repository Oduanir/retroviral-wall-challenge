# Plan V20 — Dual-Objective Blending

## Observation

We have two signals that are individually strong but anti-correlated:

| Pipeline | PR-AUC | W-Spearman | CLS |
|----------|--------|------------|-----|
| v6 backbone | 0.683 | **0.736** | **0.709** |
| Complex features (top3 novel) | **0.786** | 0.485 | 0.600 |

CLS = harmonic mean of PR-AUC and W-Spearman. The v6 backbone is balanced. The complex features are lopsided toward classification.

## Hypothesis

If we blend the **final prediction vectors** (not the features) from a ranking-optimized pipeline and a classification-optimized pipeline, there may exist a mixing coefficient α where:
- classification improves enough from the complex signal
- ranking doesn't collapse because the v6 backbone dominates

This is different from everything we've tried because:
- Previous attempts added features to ONE model (overfitting, feature interference)
- This blends TWO complete, independent prediction vectors post-hoc
- The blend happens at the score level, not the feature level

## Method

### Step 1 — Generate two independent OOF prediction vectors

**Vector A (ranking-optimized)**: the current v6 pipeline (CLS 0.7088).
Already available from the standard nested LOFO.

**Vector B (classification-optimized)**: a pipeline designed to maximize PR-AUC.
Options for Vector B:
- Ridge on top3 complex novel features (PR-AUC 0.786)
- Logistic regression on complex features (probability of active)
- EN on original features trained on binary target
- Any combination that maximizes PR-AUC in inner LOFO

Both vectors must be generated via proper nested LOFO (no data leak).

### Step 2 — Blend at the score level

For each outer fold:
1. Generate Vector A (v6 OOF predictions)
2. Generate Vector B (classification OOF predictions)
3. Normalize both to [0,1]
4. Blend: `score = α × A + (1-α) × B`
5. Optimize α on inner folds to maximize CLS

### Step 3 — Evaluate

- Compare blended CLS to v6 baseline (0.7088)
- Check that W-Spearman doesn't collapse
- Check per-family breakdown, especially Retron

## Key Design Constraints

- α must be optimized in the **inner** LOFO loop (not ex-post)
- Vector B must be generated independently of Vector A (no shared training)
- α sweep: 0.0, 0.05, 0.10, ..., 1.0 (21 values)
- If optimal α = 1.0 on most folds → Vector B adds nothing → NO-GO

## Variants to Test

1. **Vector B = Ridge on top3 complex features** (alignment_rmsd, alignment_strain, alignment_strain_max)
2. **Vector B = Logistic on complex features** (predict P(active), use as score)
3. **Vector B = EN on original features with binary target** (different objective, same features)
4. **Vector B = Ridge on ESM2 L12 PCA3 with binary target** (L12 is the classification layer)

## Success Criterion

- CLS > 0.7088 with stable α across folds
- Ideally: PR-AUC improves ≥ 0.01 with W-Spearman drop ≤ 0.01

## Failure Criterion

- Optimal α = 1.0 on all folds (Vector B useless)
- Or optimal α = 0.0 on all folds (Vector B dominates but CLS drops)
- Or α is unstable across folds (no robust blend)

## Compute

~15 min on CPU (ESM2 extraction + nested LOFO with α sweep). No GPU needed.
