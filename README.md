# Retroviral Wall Challenge — Solution

Predict which reverse transcriptases (RTs) enable prime editing, scored by CLS (Cross-Lineage Score) on Leave-One-Family-Out cross-validation.

- **Competition**: [Kaggle](https://www.kaggle.com/competitions/retroviral-challenge-predict) | [GitHub](https://github.com/Mandrake-Bioworks/Retroviral-Wall-Challenge)
- **Score**: **CLS 0.7088** (nested LOFO CV, official evaluator)

## Approach

Blend of 3 models:

1. **ElasticNet** (α=1.0, l1=0.3) on 35 tabular features (biophysical + zero-shot ESM2)
2. **Ridge** (α=12.5) on PCA3 of ESM2 **layer 12** mid-region embeddings (per-fold PCA)
3. **Ridge** (α=7.5) on PCA3 of ESM2 **layer 33** mid-region embeddings (per-fold PCA)

Key insights:
- **Different ESM2 layers capture complementary signals**. Layer 12 (intermediate) is better for classification (PR-AUC), layer 33 (final) is better for ranking (W-Spearman). Pooling over the middle third of the sequence (palm domain) reduces phylogenetic bias compared to global mean pooling.
- **Per-fold PCA** (fit only on training set of each LOFO fold) avoids transductive data leak and improves generalization.
- **Asymmetric regularization** (L12 α=12.5, L33 α=7.5) reflects the different noise levels of intermediate vs. final layer representations.

Blend weights are optimized per fold via nested LOFO (inner LOFO on 6 families, applied to the 7th, weight step 0.05).

## Repository Structure

```
├── notebooks/
│   └── solution.ipynb          # Self-contained solution notebook
├── src/
│   ├── data.py                 # Data loading
│   ├── metrics.py              # CLS metric (from official evaluate.py)
│   ├── validation.py           # LOFO CV
│   └── esm2_features.py        # ESM2 feature extraction
├── docs/
│   ├── writeup.md              # Technical writeup (1-2 pages)
│   └── literature_review.md    # Background literature (23 refs)
├── submissions/
│   └── submission_layer_sweep.csv  # 57 LOFO out-of-fold predictions (each RT predicted by a model that never saw its family)
├── results/
│   └── status_update.md        # Full project history
├── archive/                    # Exploration history (~60 experiments, 13 phases)
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

| Component | PR-AUC | W-Spearman | CLS |
|-----------|--------|------------|-----|
| ElasticNet alone | 0.706 | 0.633 | 0.667 |
| + ESM2 L33 mid (transductive PCA) | 0.685 | 0.692 | 0.688 |
| + ESM2 L12 mid (transductive PCA) | 0.693 | 0.694 | 0.694 |
| **+ per-fold PCA + asymmetric Ridge** | **0.683** | **0.736** | **0.709** |

## External Tools

- **ESM2-650M** (`facebook/esm2_t33_650M_UR50D`): pre-trained protein language model (HuggingFace). No fine-tuning.

## References

- Doman et al. (2023). *Phage-assisted evolution and protein engineering yield compact, efficient prime editors.* **Cell**.
