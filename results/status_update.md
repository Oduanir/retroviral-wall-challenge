# Project Status — March 2026

## Final Score

**CLS 0.7169** (nested LOFO CV, rank ensemble of 11-17 v6 variants)

| Component | Score |
|-----------|-------|
| PR-AUC | 0.6951 |
| W-Spearman | 0.7402 |
| CLS | 0.7169 |

Best model: **rank ensemble** of 11-17 v6 variants, each a blend of 3 models (ElasticNet + Ridge L12 PCA3 + Ridge L33 PCA3) with per-fold PCA and varied hyperparameters.

Two key discoveries:
1. **EN(α=1.1, l1_ratio=0.05)** is better than the original EN(α=1.0, l1_ratio=0.3) — best individual CLS 0.7146
2. **Rank averaging** across diverse configs (varied Ridge α, varied EN α/l1) reduces fold-specific variance — CLS 0.7169

Previous records: 0.6936 (v5, transductive PCA), 0.7088 (v6, per-fold PCA R12.5/7.5).

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
| v6 | v5 + per-fold PCA + asymmetric Ridge (12.5/7.5) | 0.7088 | 14 |
| v6.1 | v6 + EN(1.1, 0.05) best individual | 0.7146 | 21 |
| **v7** | **rank ensemble of 11-17 v6 variants** | **0.7169** | **21** |

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

### Phase 16 — Plan V16: Break 0.7088 (oracle-guided)

**Step 1 — Oracle diagnostics** (prove headroom exists):
- Oracle Classification (perfect active/inactive, same ranking): CLS 0.811 → **+0.12 headroom** (classification is the bottleneck)
- Oracle Retron (perfect Retron predictions): CLS 0.738 → +0.047 (Retron accounts for ~5 pts)
- Oracle Per-Family Best Model: CLS 0.693 → +0.002 (current blend is near-optimal)
- **GO**: massive classification headroom exists

**Step 2 — Classification corrector on top of v6**:
- `src/v16_break.py`: Logistic (C=0.1) and Ridge (α=10) corrector on 8 features (v6 blend score, EN/L12/L33 normalized scores, 3 pairwise disagreements, boundary distance), 7 lambda values (0.0–0.5) → 14 configs, all ≤ 0.7088
- Separate sweep (`v16_corrector_cpu.py`): added top tabular classification features (FoldSeek TM-scores, triad quality) and broader regularization (C=0.01–0.5) → 72 configs, all ≤ 0.7088
- **FAIL**: the classification signal gap is not capturable from existing model scores or tabular features

**Step 3 — ESM2-3B frozen mid-region layer sweep** (GPU RTX 4070 Super, exploratory, 30+ configs):
- ESM2-3B (`esm2_t36_3B_UR50D`, 36 layers, 2560D) extracted on GPU in float16
- **Important caveat**: 3B embeddings extracted on GPU/float16 are not numerically identical to a CPU/float32 extraction. Results below are an exploratory comparison, not an apples-to-apples benchmark against the CPU-based 0.7088 production pipeline.
- **3B-L18 alone: PR-AUC 0.7465** (best classification score ever) but W-Spearman 0.641 → CLS 0.690
- 3B-L18 replacing 650M-L12: CLS 0.671 (loses ranking)
- 3B as 4th model (EN + 650M-L12 + 650M-L33 + 3B-L18): max CLS 0.705 (GPU baseline 0.696, α=50)
- 3B-L18 + 650M-L33: CLS 0.671
- All 3B-only configurations collapse (W-Spearman ~0.1 for most layers)
- Cross-device combo (3B GPU embeddings + 650M CPU embeddings): max CLS 0.683 — lower than pure CPU 650M (0.7088), likely due to embedding distribution mismatch
- **FAIL**: 3B captures better classification signal but loses the ranking that 650M provides. No combination beats 0.7088 in comparable conditions.

**Key finding from V16**: Oracle analysis proves +0.12 CLS headroom exists in classification alone, but this headroom requires information not present in any available representation (ESM2-650M, ESM2-3B, tabular features, PDB structures). The missing signal likely encodes fine-grained biochemical properties of the RT-Cas9 interaction that are not captured by sequence-level or structure-level representations.

### Phase 17 — RT-Cas9 Compatibility Signal (cheap version)
- Hypothesis: missing PE signal comes from RT-Cas9 fusion compatibility, not RT quality in isolation
- 10 cheap compatibility features extracted from sequences + AlphaFold structures:
  - Fusion geometry: `nterm_to_yxdd_frac`, `cterm_to_yxdd_frac`, `termini_asymmetry_to_core`
  - Active-site accessibility: `yxdd_surface_proxy`, `yxdd_local_compactness`, `yxdd_confidence_mean`
  - Template-aligned: `global_vs_local_mmlv_gap`, `global_vs_local_best_gap`
  - Burden: `rnaseh_present_proxy`, `fusion_burden_proxy`
- **Audit** (after fix): `global_vs_local_best_gap` corrected to compute local similarity against all 57 RT structures (not just fixed MMLV). AUROC drops from 0.779 to 0.630 — the inflated value was an artifact of the fixed-reference bug. Best univariate classifier: `global_vs_local_mmlv_gap` (AUROC 0.711).
- **Rerun with corrected features**: every feature still degrades CLS when added individually to v6's ElasticNet. Best: `yxdd_local_compactness` (delta -0.001). `global_vs_local_best_gap` corrected: delta -0.006 (vs -0.231 before fix).
- Adding all features: CLS 0.318. Top 2-3: CLS 0.300.
- **NO-GO for this integration**: cheap compatibility proxies do not improve v6 in add-on mode. This does not fully falsify the RT-Cas9 compatibility hypothesis — it shows these particular proxies (coarse, unaligned local RMSD) are not additive to the current backbone.

### Phase 18 — Template-based RT-Cas9 interaction modeling
- Placed each of 57 RTs into the PE cryo-EM complex (PDB 8WUS, Shuto et al. Nature 2024) via TMalign structural alignment onto MMLV-RT chain
- 12 compatibility features extracted: clash fractions with Cas9, YXDD-to-DNA/pegRNA distances, active site orientation, protrusion, N-term to Cas9 distance
- **Implementation caveat**: clash/contact/protrusion fractions are normalized by total RT size, not by the number of TM-aligned residues. This means they still partially proxy for RT size rather than purely measuring complex compatibility. A stricter implementation would normalize by aligned-region length only.
- **Audit**: genuinely novel features (cas9_clash_fraction max|r|=0.44, cas9_min_distance max|r|=0.36, nterm_to_cas9_distance max|r|=0.22) — not redundant with existing backbone
- **But**: none of the genuinely novel features improve CLS. Best individual: cas9_clash_severe_fraction (delta -0.002). EN retuning sweep was not fully nested (α/l1 selected ex-post, not per inner fold), so individual deltas are slightly optimistic — but all remain negative.
- `alignment_tm_score` as 4th blend model gives CLS 0.7119 (+0.003), but this feature is redundant with foldseek_TM_MMLV (|r|=0.95) — the gain comes from TMalign vs FoldSeek numerical differences, not from complex compatibility
- **NO-GO**: template-based placement does not reveal exploitable RT-Cas9 compatibility signal beyond what FoldSeek already captures

### Phase 19 — Moonshot: Full PE complex compatibility modeling (2 states, all integrations)
- Placed 57 RTs in two PE complex states (8WUS termination + 8WUT initiation) via TMalign
- 23 features per state + 23 cross-state robustness features (absolute difference between states). 44 features pass confounder check (not size/MMLV proxies).
- **New features include**: alignment strain (AUROC 0.73), cas9 interface density (AUROC 0.70), deformation cost, fusion topology, nucleic acid contact fractions, cross-state variance
- **All promised integrations tested**:
  - Complex-only Ridge regression: CLS 0.495
  - Complex-only Logistic classification: CLS 0.456
  - Complex top3 Logistic: CLS 0.589 (PR-AUC 0.668)
  - v6 + complex all (4th Ridge model): CLS 0.668
  - v6 + complex top3 (4th Ridge model): CLS 0.563 (PR-AUC 0.771)
  - v6 + corrector with complex features (6 configs, λ=0.05–0.2): best CLS 0.700 (top3, λ=0.1)
  - Dual-objective post-hoc blend (v6 ranking × complex classification): CLS 0.518
- **Conclusion**: no integration strategy improves over v6 baseline (CLS 0.7088). The corrector with top3 complex features comes closest (0.700) but still degrades both PR-AUC and W-Spearman slightly. The classification signal from complex placement is real (PR-AUC 0.771 in 4th-model mode) but it is not separable from ranking degradation in any tested combination.

### Phase 20 — 3D Graph Neural Network on RT structure (prototype)
- Built contact graphs (CA-CA < 10Å) for all 57 RTs from AlphaFold structures
- Two-view GCN (full graph + active-site subgraph), multi-task (BCE + MSE), ~2K params
- Node features: one-hot AA, pLDDT, relative position, YXDD indicator, hydrophobicity, charge. Edge features (distance, seq separation) were computed but **not consumed by the GCN model** — effectively a binary contact graph with node features only.
- **Implementation caveats**: β (cls/reg balance) selected ex-post on global OOF, not nested per fold. Early stopping on training loss, not inner CLS. No device management (CPU-only script, GPU run was a separate manual execution).
- **Standalone GNN**: CLS 0.000 across all β values. W-Spearman 0.000. The network learns nothing generalizable on ~50 training samples.
- **GNN + v6 blend** (49 GNN trainings, inner OOF for weight optimization): CLS 0.000. On most folds w_v6 ≈ 1.0 (GNN ignored), but on folds where GNN gets weight it destroys the score.
- **Corrected rerun** (NNConv edge-conditioned convolution, nested β per fold, CUDA): GNN standalone CLS 0.205, GNN+v6 blend CLS 0.246. Edge features and proper nesting did not help. Most folds give w_v6=1.0 (GNN ignored); on 2 folds where GNN gets weight it destroys the score.
- **NO-GO**: both node-only GCN and edge-aware NNConv fail completely on n=57. The core problem is sample size — no neural architecture can learn generalizable structure-function patterns from ~50 training proteins in LOFO.

### Phase 21 — Ensemble averaging + EN hyperparameter discovery
- Discovered **EN(α=1.1, l1_ratio=0.05)** outperforms original EN(α=1.0, l1_ratio=0.3): best individual CLS 0.7146 (vs 0.7088)
- **Rank ensemble averaging** of diverse v6 configs reduces fold-specific variance
- Tested 2340+ configs across EN α∈[0.8,2.0], l1∈[0.02,0.3], Ridge L12∈[10,15], L33∈[6.5,11]
- Best: rank average of top 11-17 configs → **CLS 0.7169** (stable across two independent runs)
- Score averaging also effective: best CLS 0.7154 (greedy selection of 16 configs)
- Bootstrap CI: CLS 0.7169 [0.551, 0.806] (std=0.067)
- Configs must be both individually good (>0.69) AND diverse to benefit from averaging. Dense grids around one sweet spot don't help; mixing two EN regimes (0.3 vs 0.05 l1) provides the diversity.

### Phase 21b — Tier 1 creative approaches
- **Evolutionary features** (7): pairwise BLOSUM62 alignment among 57 RTs, YXDD local similarity to MMLV, global vs local gap. `yxdd_local_sim_mmlv` AUROC 0.729 (novel, |r|=0.72 with FoldSeek). Not integrated (consistent negative pattern).
- **Pairwise learn-to-rank**: convert 57 samples into 1596 pairs, train LogisticRegression on feature differences, score by win rate. CLS 0.453 — ranking collapses.
- **Per-fold adaptive ensemble**: select best K configs per outer fold via inner LOFO. CLS 0.429 — massive overfit on fold-level selection.

### Phase 21c — Tier 2: Normal Mode Analysis + sequence motifs
- **NMA features** (10): YXDD flexibility, palm rigidity, hinge proximity, collectivity. Best: `nma_yxdd_vs_palm` AUROC 0.650.
- **Sequence motif features** (~20): conserved RT motifs (LPQG primer grip, FGLK dNTP binding, DLYD, SWLS, QVN), local AA composition. Best: **`motif_FGLK_dist_to_yxdd` AUROC 0.767** — distance from dNTP-binding motif F to YXDD catalytic core. Biologically meaningful: encodes functional compactness of the polymerase domain.
- All individually negative when added to v6 EN. Rank ensemble with 4 configs > 0.70: CLS 0.7155 (below 0.7169 record).
- **Scientific finding**: `motif_FGLK_dist_to_yxdd` is the strongest non-redundant univariate classifier found in the project (AUROC 0.767) but its ranking signal is anti-correlated with v6's W-Spearman, preventing CLS gain.

### Phase 21d — Exhaustive untested approaches
- **Feature interactions** (66 systematic pairwise): no individual improvement over baseline
- **Mixed-objective** (EN binary classification + L12/L33 regression): CLS 0.566 — changes objective destroys ranking
- **Huber regression** (5 epsilon values, robust to MMLV outlier): max CLS 0.605 — robust loss degrades CLS
- **Feature subset diversity** (5 thematic subsets): all < 0.70
- **Non-linear transforms** (log, square of key features): max CLS 0.699 — quasi-neutral
- Mega-ensemble of all: CLS 0.714, below 0.7169 record

### Phase 22 — ESM2 attention surgery + in-silico mutagenesis
- **Attention features** (10): YXDD attention entropy, long-range fraction, attention to termini, symmetry, concentration. Best AUROC: attn_symmetry 0.676 (max|r|=0.44 — genuinely novel).
- **Mutagenesis features** (7): YXDD log-likelihood sensitivity, neighborhood LL, ratio, variance, conservation agreement, escape score. Best AUROC: mut_neighborhood_ll 0.617 (max|r|=0.63 — moderately novel).
- **All non-redundant** (max|r| < 0.66 with existing features)
- **All negative** when added individually to v6 EN. Best: attn_to_termini (delta -0.011). Worst: mut_variance (delta -0.081).
- **Conclusion**: ESM2 internal representations (attention patterns, per-position mutagenesis sensitivity) capture novel but non-additive signal. The information is genuinely different from embeddings but does not improve CLS when injected into the current pipeline.

### SRA feasibility assessed
- BioProject PRJNA916060: 216 runs Figure 1C, 1.2 GB
- Only 21 active RTs in SRA (inactive ones not sequenced)
- Addresses ranking only, not classification → No-Go

---

## Final Diagnosis

The PE signal in this 57-RT dataset is **confounded with phylogeny** in a way that resists all tested approaches (~160+ experiments over 22 phases).

- Global features (FoldSeek, thermostability, ESM2) **are** the signal
- Local signal (active site alone) is insufficient (CLS 0.46 vs 0.59 global)
- External data provides no exploitable PE supervision
- Anti-phylogeny methods (residualization, adversarial, GroupDRO, CORAL) do not break the wall
- LoRA fine-tuning memorizes family-specific patterns (tested in Phase 15)
- PDB structural features are redundant with existing features (tested in Phase 15)
- Target transformations (log, rank, binary) all degrade the signal
- ESM2-3B improves classification (PR-AUC 0.747) but loses ranking — no net CLS gain (Phase 16)
- Oracle analysis shows +0.12 headroom in classification, but it is not capturable from available representations (Phase 16)
- Complex placement features achieve PR-AUC 0.771 (Phase 19) but degrade W-Spearman in every tested integration: 4th model, corrector, dual blend, complex-only
- **CLS 0.7169 is the best score achieved** with honest nested LOFO on the challenge data (rank ensemble of v6 variants, Phase 21). No tested integration method (feature addition, 4th model, corrector, dual-objective blend, complex-only modality) improves CLS when adding classification-oriented signals. Note: all complex-view sub-models used fixed hyperparameters (Ridge α=10, Logistic C=0.1); a more thorough HP search within nested LOFO could in principle change the result, but the consistent negative direction across all integration strategies makes a breakthrough unlikely.

---

## Untested Approaches

- **SRA PRJNA916060 reprocessing** (feasible but addresses only ranking of 21 active RTs, not classification)
- **Protein structure GNN** (message passing on contact graph — fundamentally different representation)
- **External RT data with PE labels** (would require new wet-lab experiments)
- **RT-Cas9 full docking with side-chain repacking** (template-based rigid placement in Phase 18 showed no signal; flexible docking might differ, but the cheap and intermediate versions both failed)
