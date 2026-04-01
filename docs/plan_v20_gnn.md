# Plan V20 — 3D Graph Neural Network on RT Structure

## Hypothesis

The missing signal is encoded in the **spatial topology** of residue interactions — patterns that PCA, FoldSeek TM-scores, and hand-crafted features compress away. A GNN operating directly on the 3D contact graph can learn non-linear residue-residue interactions relevant to PE activity.

## Architecture

### Graph Construction

For each RT:
- **Nodes** = residues (N = sequence length)
- **Node features** (~25D):
  - One-hot amino acid type (20D)
  - pLDDT / B-factor from AlphaFold (1D)
  - Relative position in sequence (1D, normalized 0-1)
  - YXDD indicator (1D, binary: is this residue in the YXDD motif?)
  - Hydrophobicity (1D, Kyte-Doolittle scale)
  - Charge at pH 7 (1D)
- **Edges** = CA-CA distance < 10Å (contact graph)
- **Edge features** (2D):
  - 3D distance (1D, in Å)
  - Sequence separation (1D, |i-j| normalized)

### Two Views

1. **Full graph**: entire RT structure → captures global topology, compactness, domain arrangement
2. **Active-site subgraph**: residues within 15Å of YXDD centroid → captures catalytic geometry

Each view produces a graph-level embedding via mean pooling of node features after message passing.

### Model

```
GNN_full(graph_full) → embedding_global (32D)
GNN_site(graph_site) → embedding_local (32D)
concat → [64D]
  → head_cls: Linear(64, 1) → sigmoid → P(active)
  → head_reg: Linear(64, 1) → PE efficiency
```

- GNN layers: 2× GAT (Graph Attention Network) or GCN
- Hidden dim: 32 (very small — n=57 demands brutal simplicity)
- Dropout: 0.3
- No batch norm (too few samples)

### Multi-Task Loss

```
L = α * BCE(head_cls, active) + (1-α) * MSE(head_reg, pe_efficiency)
```

α optimized to balance classification and ranking. Start with α=0.5.

### Score Construction (Critical)

The final `predicted_score` must capture both classification and ranking from the GNN. Using `head_reg` alone discards the classification signal. Using `head_cls` alone discards the ranking signal.

**Primary score formula (additive, robust):**
```
predicted_score = β * norm(head_reg) + (1-β) * norm(sigmoid(head_cls))
```
with β optimized on inner LOFO (sweep 0.0 to 1.0, step 0.1).

This additive form is preferred because:
- Both components are normalized to [0,1] before combination → stable across folds
- β directly controls classification vs ranking trade-off
- No sensitivity to `head_reg` sign or scale

**Secondary score formula (multiplicative, tested for comparison):**
```
predicted_score = norm(head_reg) * sigmoid(head_cls)
```
Only tested if the additive form shows signal. The multiplicative form is more aggressive (zeros out inactive RTs) but less stable across folds.

### Training Protocol

- Nested LOFO (strict, same as all previous phases)
- Outer loop: 7 folds
- Inner loop: for each outer fold, train on 6 families, validate on inner held-out family
- Epochs: 50-100 with early stopping on inner CLS
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-2
- **No data augmentation** (sequences are fixed)

### Output Integration

**Option 1 — GNN standalone**: use the combined score (see Score Construction above) directly as `predicted_score`. Evaluates what the graph representation captures on its own.

**Option 2 — GNN + v6 blend (leak-free protocol)**:

```
For each outer fold (held-out family F):
  
  # --- Inner LOFO to generate OOF scores for weight optimization ---
  inner_families = 6 families (excluding F)
  For each inner fold (held-out inner family G):
    Train GNN on 5 remaining families
    Predict on held-out family G → inner OOF GNN scores for G
  
  # Now we have inner OOF GNN scores for all 6 train families
  # (each RT's GNN score comes from a model that never saw its family)
  
  # v6 inner LOFO produces inner OOF v6 scores (same protocol as v6 baseline)
  
  # Optimize blend weights on inner OOF:
  #   score = w_v6 * norm(v6_inner_oof) + w_gnn * norm(gnn_inner_oof)
  #   pick w that maximizes inner CLS
  
  # --- Outer prediction ---
  Train GNN on all 6 inner families
  Predict on held-out family F → outer GNN score
  Train v6 on all 6 inner families → outer v6 score
  
  Blend: outer_score = w_v6 * norm(v6_outer) + w_gnn * norm(gnn_outer)
```

This requires training the GNN **6 times per outer fold** (inner LOFO) + 1 time for the outer prediction = 7 × 7 = **49 GNN trainings total**. This is the honest cost of nested LOFO with a neural network.

Cost: 49 GNN trainings × ~1-2 min each ≈ **50-100 min CPU**. GPU cuts this to ~15-30 min.

## Regularization Strategy (Critical)

n=57 is extremely small for any neural network. Overfitting is the primary risk.

Defenses:
- **Model size**: 32 hidden dim, 2 layers, ~2K trainable parameters total
- **Dropout**: 0.3 on node features and embeddings
- **Weight decay**: 1e-2 (aggressive)
- **Early stopping**: on inner LOFO CLS
- **No fancy architecture**: plain GCN or GAT, no residual connections, no edge convolutions
- **Evaluation**: only nested LOFO CLS counts, never train-set performance

## Compute Requirements

- **PyTorch Geometric** needed (pip install torch-geometric)
- CPU is sufficient but slow. GPU recommended if available.
- Estimated cost: 7 outer folds × (GNN training ~2-5 min + v6 LOFO ~2 min + weight search ~1 min) ≈ **30-60 min CPU**, faster on GPU
- Full pipeline including ESM2 extraction: **40-70 min CPU**
- Model has ~2-5K trainable parameters but nested LOFO multiplies the training runs

## Go/No-Go

### GO
- GNN standalone CLS > 0.60 (shows the graph captures *some* signal)
- GNN + v6 blend CLS > 0.7088 (actual improvement)

### NO-GO
- GNN standalone CLS < 0.50 (graph representation is useless)
- GNN + v6 CLS ≤ 0.7088 (no additive value)

## What Makes This Different

| Previous | GNN |
|----------|-----|
| Features are human-designed summaries | Network learns which residue patterns matter |
| PCA compresses 1280D → 3D | Graph preserves full spatial topology |
| One score per protein | Per-residue computation, then pooling |
| Linear models | Non-linear message passing |
| Same features for all RTs | Graph structure adapts to each RT's fold |
