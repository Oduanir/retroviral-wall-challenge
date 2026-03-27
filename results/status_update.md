# Project Status — March 2026

## Final Score

**CLS 0.6936** (nested LOFO CV, verified with official evaluator on `submissions/submission_layer_sweep.csv`)

| Component | Score |
|-----------|-------|
| PR-AUC | 0.6930 |
| W-Spearman | 0.6942 |
| CLS | 0.6936 |

Best model: blend of 3 models (ElasticNet enriched+ZS + ESM2 **layer 12** mid PCA3 + ESM2 **layer 33** mid PCA3), weights optimized by nested LOFO.

---

## CLS Progression

| Version | Models | CLS | Phase |
|---------|--------|-----|-------|
| GitHub ref. | HandCrafted + RF | 0.318 | — |
| v1 | EN + Hurdle + ESM2 global | 0.6525 | 3 |
| v2 | EN enriched + Hurdle + ESM2 global | 0.6685 | 4 |
| v4 | EN enriched+ZS + ESM2 L33 mid | 0.6882 | 5 |
| **v5** | **EN enriched+ZS + ESM2 L12 mid + ESM2 L33 mid** | **0.6936** | **11** |

---

## Per-Family PR-AUC (v5)

| Family | n | Active | PR-AUC |
|--------|---|--------|--------|
| Group_II_Intron | 5 | 2 | **1.000** |
| LTR_Retrotransposon | 11 | 2 | **1.000** |
| Retroviral | 18 | 12 | 0.847 |
| **Retron** | **12** | **5** | **0.423** |
| CRISPR-associated | 5 | 0 | N/A |
| Other | 5 | 0 | N/A |
| Unclassified | 1 | 0 | N/A |

Bottleneck: Retron (PR-AUC 0.42).

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

### SRA feasibility assessed
- BioProject PRJNA916060: 216 runs Figure 1C, 1.2 GB
- Only 21 active RTs in SRA (inactive ones not sequenced)
- Addresses ranking only, not classification → No-Go

---

## Final Diagnosis

The PE signal in this 57-RT dataset is **confounded with phylogeny** in a way that resists all tested approaches (~60 experiments over 13 phases).

- Global features (FoldSeek, thermostability, ESM2) **are** the signal
- Local signal (active site alone) is insufficient (CLS 0.46 vs 0.59 global)
- External data provides no exploitable PE supervision
- Anti-phylogeny methods (residualization, adversarial, GroupDRO, CORAL) do not break the wall
- CLS 0.6936 is the best score achieved with honest LOFO on the challenge data

---

## Untested Approaches

- **LoRA fine-tuning** of ESM2 (requires GPU)
- **SRA PRJNA916060 reprocessing** (feasible but addresses only ranking of 21 active RTs, not classification)
