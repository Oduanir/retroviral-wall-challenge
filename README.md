# Retroviral Wall Challenge — Solution

Predict which reverse transcriptases (RTs) enable prime editing, scored by CLS (Cross-Lineage Score) on Leave-One-Family-Out cross-validation.

- **Competition**: [Kaggle](https://www.kaggle.com/competitions/retroviral-challenge-predict) | [GitHub](https://github.com/Mandrake-Bioworks/Retroviral-Wall-Challenge)
- **Score**: **CLS 0.7176** (nested LOFO CV, official evaluator)

## Approach

**Rank ensemble** of diverse v6 pipeline variants, each a blend of 3 models:

1. **ElasticNet** on 35 tabular features (biophysical + zero-shot ESM2)
2. **Ridge** on PCA3 of ESM2 **layer 12** mid-region embeddings (per-fold PCA)
3. **Ridge** on PCA3 of ESM2 **layer 33** mid-region embeddings (per-fold PCA)

Key insights:
- **Different ESM2 layers capture complementary signals**. Layer 12 (intermediate) is better for classification (PR-AUC), layer 33 (final) is better for ranking (W-Spearman). Pooling over the middle third of the sequence (palm domain) reduces phylogenetic bias.
- **Per-fold PCA** (fit only on training set of each LOFO fold) avoids transductive data leak.
- **Asymmetric regularization** (L12 α=12.5-14.5, L33 α=7.5-8.5) reflects different noise levels per layer.
- **EN(α=1.1, l1_ratio=0.05)** outperforms the standard EN(α=1.0, l1_ratio=0.3) — discovered via large-scale hyperparameter sweep.
- **Rank ensemble averaging** of diverse hyperparameter configurations reduces fold-specific variance. Diversity (mixing two EN regimes) is more important than density.
- **Sliced-Wasserstein Embedding** (optimal transport on per-residue ESM2 tokens) provides orthogonal diversity for ensemble averaging.

Blend weights are optimized per fold via nested LOFO (inner LOFO on 6 families, applied to the 7th).

## Repository Structure

```
├── notebooks/
│   └── solution.ipynb          # Self-contained solution notebook
├── src/
│   ├── data.py                 # Data loading
│   ├── metrics.py              # CLS metric (from official evaluate.py)
│   ├── validation.py           # LOFO CV
│   ├── esm2_features.py        # ESM2 feature extraction
│   ├── blend.py                # Nested LOFO blend engine
│   └── bootstrap.py            # Bootstrap confidence intervals
├── docs/
│   ├── writeup.md              # Technical writeup
│   └── literature_review.md    # Background literature (23 refs)
├── submissions/
│   └── submission_ensemble.csv # Final submission (57 LOFO OOF predictions)
├── results/
│   └── status_update.md        # Full project history (22 phases, 175+ experiments)
├── tests/                      # Unit tests
├── archive/                    # Exploration history (experiments, plans, old submissions)
├── data/raw/                   # Competition data
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook solution.ipynb
```

The notebook is self-contained: loads data, extracts ESM2 embeddings (~10 min CPU), runs nested LOFO CV, outputs `submission.csv`.

## Results

| Stage | PR-AUC | W-Spearman | CLS |
|-------|--------|------------|-----|
| ElasticNet alone | 0.706 | 0.633 | 0.667 |
| + ESM2 L33 mid | 0.685 | 0.692 | 0.688 |
| + ESM2 L12 mid | 0.693 | 0.694 | 0.694 |
| + per-fold PCA + asymmetric Ridge | 0.683 | 0.736 | 0.709 |
| + EN sweet spot (α=1.1, l1=0.05) | 0.693 | 0.738 | 0.715 |
| **+ rank ensemble + SWE diversity** | **0.726** | **0.710** | **0.718** |

Bootstrap 95% CI: CLS 0.7176 [0.529, 0.809]

## External Tools

- **ESM2-650M** (`facebook/esm2_t33_650M_UR50D`): pre-trained protein language model (HuggingFace). No fine-tuning.

## References

- Doman et al. (2023). *Phage-assisted evolution and protein engineering yield compact, efficient prime editors.* **Cell**.
- Shuto et al. (2024). *Structural basis for pegRNA-guided reverse transcription by a prime editor.* **Nature**.
- Naderi et al. (2025). *Sliced-Wasserstein embedding for protein language model aggregation.* **Bioinformatics Advances**.
