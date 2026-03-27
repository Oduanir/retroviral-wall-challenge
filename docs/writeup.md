# Retroviral Wall Challenge — Technical Writeup

## Summary

We predict prime editing (PE) activity of 57 reverse transcriptases (RTs) using a blend of three models evaluated by Leave-One-Family-Out (LOFO) cross-validation. Our approach combines tabular biophysical features with protein language model (PLM) embeddings extracted from two complementary layers of ESM2-650M. The final model achieves **CLS 0.6936** on nested LOFO evaluation.

## Approach

### Feature Engineering

We use 35 tabular features organized in three groups:

**Handcrafted biophysical features (21)**: FoldSeek structural similarity scores (TM-score to MMLV, HIV-1, best overall), catalytic triad geometry (YXDD motif distances, RMSD), thermostability predictions (t40-t60), and physicochemical properties (instability index, GRAVY, CamSol, net charge).

**Missing value indicators (7)**: Binary flags for structurally informative missing values. The absence of `connection_mean_pot` (33/57 NaN) indicates RTs lacking the connection domain. Missing `yxdd_*` and `triad_*` values indicate RTs without a detectable catalytic motif — both are predictive of inactivity.

**Engineered features (7)**: Interaction terms (`foldseek_gap_MMLV`, `t40 × foldseek_MMLV`, `triad_quality`, `seq_struct_compat`) and ESM2 zero-shot features (pseudo-perplexity, mean entropy, mean log-likelihood) extracted from ESM2-650M in a single forward pass per sequence.

### ESM2 Layer-Specific Embeddings

A key finding of our work is that **different ESM2 layers capture complementary signals** for PE prediction:

- **Layer 12** (intermediate): better PR-AUC (0.503), captures classification signal
- **Layer 33** (final): better W-Spearman (0.615), captures ranking signal

For each layer, we extract per-residue embeddings from ESM2-650M, pool over the middle third of the sequence (approximating the palm domain containing the catalytic core), and reduce to 3 dimensions via PCA. The PCA is fit on all 57 RTs (unsupervised, transductive).

### Model Architecture

The final model is a blend of three components:

1. **ElasticNet** (α=1.0, l1_ratio=0.3) on 35 tabular features — provides the primary classification and ranking signal from biophysical properties
2. **Ridge** (α=10) on PCA3 of ESM2 layer 12 mid-region — provides complementary classification signal
3. **Ridge** (α=10) on PCA3 of ESM2 layer 33 mid-region — provides complementary ranking signal

Blend weights are determined by **nested LOFO**: for each outer fold (1 family held out), an inner LOFO on the 6 remaining families optimizes blend weights on a grid (step 0.1). Models are then trained on all 6 inner families and applied to the outer fold with the inner-optimized weights.

### Evaluation Protocol

All results are reported on **nested LOFO** out-of-fold predictions, pooled across 7 folds. CLS is computed once on the full 57-prediction array using the same formula as the official evaluation script. Predictions are ordered to match `rt_sequences.csv` for deterministic ranking.

## Results

| Component | PR-AUC | W-Spearman | CLS |
|-----------|--------|------------|-----|
| ElasticNet alone | 0.706 | 0.633 | 0.667 |
| + ESM2 L33 mid | 0.685 | 0.692 | 0.688 |
| **+ ESM2 L12 mid** | **0.693** | **0.694** | **0.694** |

Per-family PR-AUC:

| Family | n | Active | PR-AUC |
|--------|---|--------|--------|
| Group II Intron | 5 | 2 | 1.000 |
| LTR Retrotransposon | 11 | 2 | 1.000 |
| Retroviral | 18 | 12 | 0.847 |
| Retron | 12 | 5 | 0.423 |

The main bottleneck is the Retron family (PR-AUC 0.42) where the model struggles to distinguish 5 active retrons from 7 inactive ones.

## Reproducibility

The solution notebook (`notebooks/solution.ipynb`) is self-contained and implements the full pipeline:

1. Load training data and features
2. Extract ESM2-650M layer 12 and layer 33 mid-region embeddings (CPU, ~10 min)
3. Compute zero-shot features (pseudo-perplexity, entropy)
4. Run nested LOFO CV with 3-model blend (7 outer folds × 6 inner folds)
5. Pool 57 out-of-fold predictions, compute CLS, and output `submission.csv`

The notebook produces **57 LOFO out-of-fold predictions** — each RT is predicted by a model that never saw its evolutionary family during training.

Dependencies: numpy, pandas, scikit-learn, scipy, torch, transformers.

## External Data and Tools

- **ESM2-650M** (`facebook/esm2_t33_650M_UR50D`): pre-trained protein language model (HuggingFace), used for embedding extraction and zero-shot features. No fine-tuning.
