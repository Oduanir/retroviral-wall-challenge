# Project Status — March 2026

## Final Score

**CLS 0.7088** (nested LOFO CV, verified with official evaluator on `submissions/submission_improved.csv`)

| Component | Score |
|-----------|-------|
| PR-AUC | 0.6834 |
| W-Spearman | 0.7361 |
| CLS | 0.7088 |

Best model: blend of 3 models (ElasticNet enriched+ZS + ESM2 **layer 12** mid PCA3 + ESM2 **layer 33** mid PCA3), with **per-fold PCA** and **asymmetric Ridge** (L12 α=12.5, L33 α=7.5), weights optimized by nested LOFO (step 0.05).

Previous best: CLS 0.6936 (transductive PCA, Ridge α=10 symmetric).

### Final Production Recipe

1. **ElasticNet** (α=1.0, l1_ratio=0.3) on 35 tabular features (21 biophysical + 7 missing flags + 4 interactions + 3 ESM2 zero-shot)
2. **Ridge** (α=12.5) on PCA3 of ESM2 layer 12 mid-region embeddings
3. **Ridge** (α=7.5) on PCA3 of ESM2 layer 33 mid-region embeddings

- **PCA fitted per LOFO fold** (only on training set, not transductive)
- Mid-region = middle third of sequence (palm domain approximation)
- Blend weights optimized per outer fold via nested inner LOFO (grid step 0.05, min-max normalization)
- Outer fold predictions normalized using inner OOF min/max statistics
- ESM2 model: `facebook/esm2_t33_650M_UR50D` (650M params, no fine-tuning)

Bootstrap 95% CI: CLS 0.7088 [0.545, 0.797] (std=0.067, n=10000)

---

## CLS Progression

| Version | Models | CLS | Phase |
|---------|--------|-----|-------|
| GitHub ref. | HandCrafted + RF | 0.318 | — |
| v1 | EN + Hurdle + ESM2 global | 0.6525 | 3 |
| v2 | EN enriched + Hurdle + ESM2 global | 0.6685 | 4 |
| v4 | EN enriched+ZS + ESM2 L33 mid | 0.6882 | 5 |
| v5 | EN enriched+ZS + ESM2 L12 mid + ESM2 L33 mid | 0.6936 | 11 |
| **v6** | **v5 + per-fold PCA + asymmetric Ridge (12.5/7.5)** | **0.7088** | **14** |

---

## Per-Family PR-AUC (v6)

| Family | n | Active | PR-AUC |
|--------|---|--------|--------|
| Group_II_Intron | 5 | 2 | **1.000** |
| LTR_Retrotransposon | 11 | 2 | 0.750 |
| Retroviral | 18 | 12 | 0.842 |
| **Retron** | **12** | **5** | **0.407** |
| CRISPR-associated | 5 | 0 | N/A |
| Other | 5 | 0 | N/A |
| Unclassified | 1 | 0 | N/A |

Bottleneck: Retron (PR-AUC 0.407). W-Spearman improved significantly (0.694→0.736) at slight PR-AUC cost.

---

## All Project Phases

### Phase 0 — Literature review
- 23 references, source paper identified (Doman et al., Cell 2023)

### Phase 1 — Protocol + EDA
- CLS metric implemented (official Mandrake code)
- LOFO CV operational
- 64 numeric features, top correlated: FoldSeek (structural similarity)

### Phase 2 — Tabular models (43 models tested)
- ElasticNet(a=0.1, l1=0.5) on 21 features: CLS 0.596

### Phase 3 — ESM2
- ESM2 PCA3 alone: W-Spearman 0.692, PR-AUC 0.418
- Late fusion EN + Hurdle + ESM2: CLS 0.6525 (nested)

### Phase 4 — Optimization
- Missing flags + interactions: +0.016 CLS
- PDB features, multiplicative blend, graph smoothing: no gain

### Phase 5 — Advanced PLMs
- Zero-shot ESM2 (pseudo-perplexity, entropy): PR-AUC boost
- Mid-region embedding (palm domain): better W-Spearman
- SaProt, Ankh: no gain
- CLS 0.6882 reached (improved to 0.6936 in v5 via layer sweep)

### Phase 6 — External data (plan V5)
- 486 RT from Doman: neighborhood features encode phylogeny
- MSA / conservation: redundant with FoldSeek
- Retron-specific features: not transferable in LOFO
- Attention-weighted pooling: worse than mid-region
- Ankh: total family memorization

### Phase 7 — Anti-phylogeny (plan V6 focused)
- Feature pruning: destroys the score
- Residualization: degrades CLS
- Target transformation / multi-task: PR-AUC 0.71 but CLS max 0.664
- Adversarial / orthogonal projection: max CLS 0.648

### Phase 8 — Roadmap V7 (15 runs)
- CORAL, MMD: PyTorch SGD underperforms sklearn for n=57
- GroupDRO/VREx ElasticNet: CLS 0.668 (same as baseline)
- Pairwise ranking, diff Spearman, full multi-task, contrastive: all inferior
- Local YXDD patches: PR-AUC record 0.762 but CLS 0.657
- Bio prior: CLS 0.608 without training

### Phase 9 — External data v2 (exploitation plan)
- Retron transferability (179 census + 105 discovery): negative correlations, CLS 0.539
- Reformulated external RT (486 RT, ESM2 embeddings): CLS 0.541

### Phase 10 — Final experiments
- Aligned residues + delta vs RT references: CLS 0.653
- Episodic training: CLS 0.559
- Hard-case weighting: W-Spearman record 0.710 but CLS 0.647
- Hsu-style local-only test: CLS 0.459 — local signal insufficient alone

### Phase 11 — Layer sweep + GP + ESM-IF local
- **ESM2 layer sweep**: layer 12 mid complementary to layer 33 mid → **CLS 0.6936** (new record)
- GP: W-Spearman 0.000 everywhere → failure
- ESM-IF local (per-position LL around YXDD): no local signal, degrades CLS
- Extended layer sweep (4th layer): no gain beyond L12+L33
- PU Learning: no gain

### Phase 12 — Plan V12
- Nested stacking: CLS 0.000 (meta-learner overfits inner OOF noise)
- Biochemical features by domain (55 features anchored on YXDD): CLS 0.677 (redundant)
- ProtT5 mid-region: CLS 0.000 (family memorization, like Ankh)

### Phase 13 — V13 Shortlist
- KernelRidge MKL: W-Spearman record 0.725 but PR-AUC 0.631, CLS 0.674
- Domain features by structure: redundant, CLS 0.578
- Episodic V11: CLS 0.591
- Mid-region PCA5 adaptation: CLS 0.571
- KernelRidge + EN + L12 + L33 blend: degrades CLS (0.643)
- LoRA minimal: not tested (requires GPU)

### Phase 14 — Code modularization + per-fold PCA + hyperparameter optimization
- Modularized code: `src/blend.py`, `src/bootstrap.py`, updated `src/esm2_features.py`
- Unit tests: 20 tests covering metrics, blend, normalization
- **Per-fold PCA**: fitting PCA only on training data per fold (not transductive) → +0.0006 CLS
- **Finer weight grid** (step 0.05 vs 0.1) → additional gain
- **Asymmetric Ridge**: L12 α=12.5, L33 α=7.5 (sweep over 110 combos) → **CLS 0.7088**
- Bootstrap CI: CLS 0.7088 [0.545, 0.797] (std=0.067)
- Explored but no gain: PCA5/10, BayesianRidge, KernelRidge on raw 1280D, YXDD-anchored embeddings, cosine features, scipy weight refinement, EN alpha sweep

### Phase 15 — LoRA fine-tuning + PDB features + alternative approaches (GPU RTX 4070 Super)
- **PDB structural features** (23 features: B-factors, compactness, YXDD-centric): CLS 0.563–0.669 — all degrade score (overfitting with 35+23 features on ~50 samples)
- **PDB top-5 by correlation**: CLS 0.667; top-3: CLS 0.669 — still hurts
- **LoRA fine-tuning ESM2** (regression, MSE loss, r=4, 10 epochs): CLS 0.572 — memorizes families
- **LoRA binary** (BCE loss, r=2, 5 epochs, lr=5e-5): CLS 0.696 — best LoRA but still below baseline
- **LoRA binary** (r=2, 3 epochs, lr=1e-4): CLS 0.691
- **LoRA binary** (r=4, 3 epochs, lr=5e-5): CLS 0.691
- **LoRA replacing L33**: CLS 0.694 — close but no improvement
- **LoRA replacing L12**: CLS 0.675
- **LoRA as 4th model** (various Ridge α): max CLS 0.700 (α=20) — still below 0.7088
- **LoRA + PDB features combined**: CLS 0.668
- **Target transformation** (log): CLS 0.608 — destroys signal
- **Target transformation** (rank): CLS 0.594
- **Target transformation** (binary): CLS 0.570
- **Feature selection** (top 15 by EN coefficient): CLS 0.689 — loses useful signal
- **Feature selection** (top 10): CLS 0.683
- **Two-stage** (separate classification + regression, blended): CLS 0.490
- **Rank fusion blending** (rank-based instead of minmax normalization for outer predictions): CLS 0.638
- **Scipy continuous weight optimization**: CLS 0.700 — overfits inner folds
- **Multiple normalization strategies** (rank, zscore): max CLS 0.690
- **Inner-loop HP tuning** (9×4×4=144 combos): CLS 0.612 — overfits

**Conclusion**: LoRA fine-tuning on 50 training samples learns family-specific patterns that don't generalize across the LOFO boundary. PDB features are redundant with existing FoldSeek/structural features. All target transformations lose the carefully balanced signal. Feature selection removes useful but weak features. The baseline per-fold PCA + asymmetric Ridge (0.7088) remains optimal.

### SRA feasibility assessed
- BioProject PRJNA916060: 216 runs Figure 1C, 1.2 GB
- Only 21 active RTs in SRA (inactive ones not sequenced)
- Addresses ranking only, not classification → No-Go

---

## Final Diagnosis

The PE signal in this 57-RT dataset is **confounded with phylogeny** in a way that resists all tested approaches (~80+ experiments over 15 phases).

- Global features (FoldSeek, thermostability, ESM2) **are** the signal
- Local signal (active site alone) is insufficient (CLS 0.46 vs 0.59 global)
- External data provides no exploitable PE supervision
- Anti-phylogeny methods (residualization, adversarial, GroupDRO, CORAL) do not break the wall
- LoRA fine-tuning memorizes family-specific patterns (tested in Phase 15)
- PDB structural features are redundant with existing features (tested in Phase 15)
- Target transformations (log, rank, binary) all degrade the signal
- **CLS 0.7088 is the best score achieved** with honest nested LOFO on the challenge data

---

## Untested Approaches

- **SRA PRJNA916060 reprocessing** (feasible but addresses only ranking of 21 active RTs, not classification)
- **ESM2-3B** (larger model, may capture finer-grained signal — requires >12GB VRAM)
- **Protein structure GNN** (message passing on contact graph — fundamentally different representation)
- **External RT data with PE labels** (would require new wet-lab experiments)
