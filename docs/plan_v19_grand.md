# Plan V19 — The Grand Assault

## Thesis

Every approach so far treats RT fitness as a **static property** of the protein: its sequence, its structure, its similarity to MMLV. But PE activity is a **dynamic, functional** property: can this enzyme perform reverse transcription in a specific molecular context?

The oracle shows +0.12 headroom in classification. That signal exists — we just haven't looked in the right representation. This plan attacks the problem from **five independent angles**, each capturing a fundamentally different aspect of RT biology that has never been tested in this repo.

---

## The Five Angles

### Angle 1 — Evolutionary Fitness Landscape

**What**: Score each RT by how "natural" it looks in the context of the RT protein family evolutionary landscape — not just ESM2 pseudo-perplexity (which is generic), but an RT-specific fitness model.

**How**:
1. Download all known RT sequences from UniProt/Pfam (RT_pol domain, PF00078 — thousands of sequences)
2. Build a curated MSA of diverse RT sequences
3. Compute **position-specific conservation scores** around YXDD and key functional residues
4. Compute **coevolution scores** (mutual information or DCA) between YXDD and surrounding residues — functional RTs should have coevolved residue pairs
5. Score each of the 57 challenge RTs against this evolutionary model

**Why it's new**: ESM2 pseudo-perplexity is trained on ALL proteins. This is RT-specific evolutionary intelligence. An RT that is "natural" in the RT family but "weird" in the YXDD region might be non-functional.

**Features** (~8):
- `rt_conservation_yxdd`: mean conservation at YXDD ± 10 positions
- `rt_conservation_global`: mean conservation across aligned positions
- `rt_conservation_ratio`: local/global conservation ratio
- `rt_coevolution_yxdd`: mean coevolution score between YXDD and other conserved positions
- `rt_msa_depth`: how many MSA sequences match this RT (evolutionary isolation)
- `rt_insertion_score`: does this RT have unusual insertions/deletions vs the MSA consensus?
- `rt_subfamily_density`: density of close homologs in sequence space
- `rt_evolutionary_rate`: estimated evolutionary rate at functional positions

---

### Angle 2 — Structural Dynamics (Normal Mode Analysis)

**What**: Predict the **flexibility and collective motions** of each RT. A functional RT for PE needs specific dynamic properties: the palm domain must be rigid enough for catalysis but flexible enough to accommodate the pegRNA template.

**How**:
1. Build elastic network models (ANM/GNM) from AlphaFold structures using ProDy
2. Extract normal modes (lowest-frequency collective motions)
3. Compute dynamic features focused on the catalytic region

**Why it's new**: Every structural feature so far is static. NMA captures functional dynamics — which parts move together, which are rigid. This is biologically relevant for enzyme function.

**Features** (~8):
- `yxdd_flexibility`: mean B-factor predicted from GNM at YXDD region
- `yxdd_collectivity`: do YXDD residues move collectively or independently?
- `palm_rigidity`: mean stiffness of the palm domain
- `hinge_proximity`: distance from YXDD to the nearest predicted hinge region
- `mode1_yxdd_displacement`: how much YXDD moves in the dominant collective mode
- `domain_coupling`: dynamic coupling between palm and fingers/thumb domains
- `open_close_mode`: does the RT have an open-close motion compatible with template binding?
- `dynamic_accessibility`: predicted fluctuation of active-site accessibility

---

### Angle 3 — ESM2 Attention Surgery

**What**: Instead of using ESM2 embeddings (which average over all information), surgically extract the **attention patterns** that encode functional relationships between residues.

**How**:
1. For each RT, extract all 33×20 = 660 attention heads from ESM2
2. Focus on attention FROM the YXDD motif TO other residues
3. Identify which heads attend to catalytically important relationships
4. Extract attention-derived features that capture "functional wiring"

**Why it's new**: Embeddings are a lossy summary. Attention patterns preserve the relational structure — which residues "talk to" the catalytic site. This has been used successfully in protein contact prediction and function annotation.

**Features** (~6):
- `yxdd_attention_entropy`: diversity of attention from YXDD (focused = structured site, diffuse = disordered)
- `yxdd_long_range_attention`: fraction of YXDD attention going to residues >30aa away
- `yxdd_attention_to_termini`: does YXDD "see" the N/C-termini? (fusion-relevant)
- `top_head_yxdd_concentration`: maximum attention concentration on YXDD from any head
- `attention_symmetry`: is the attention pattern around YXDD symmetric? (structural regularity)
- `catalytic_wiring_score`: attention between YXDD and the 3 other conserved RT motifs (if identifiable)

---

### Angle 4 — In-Silico Mutagenesis Landscape

**What**: Use ESM2 to predict the **mutational sensitivity** of each RT at key positions. Active RTs should have a specific pattern of sensitivity — tolerant at non-critical positions, highly sensitive at catalytic residues.

**How**:
1. For each RT, mask each position in the YXDD ± 20 region one at a time
2. Score the log-likelihood of the original residue vs alternatives
3. Build a "mutation sensitivity profile" around the catalytic site
4. Compare profiles between active and inactive RTs

**Why it's new**: ESM2 zero-shot (pseudo-perplexity) averages over the whole sequence. This focuses specifically on the functional tolerance landscape around the catalytic core.

**Features** (~6):
- `yxdd_sensitivity`: mean delta log-likelihood when masking YXDD residues
- `yxdd_neighborhood_sensitivity`: sensitivity of the ±20 region
- `sensitivity_ratio`: YXDD sensitivity / global sensitivity
- `sensitivity_variance`: how variable is sensitivity around the active site?
- `conservation_agreement`: does ESM2's sensitivity agree with MSA conservation? (indicates model "understands" this protein)
- `escape_score`: are there positions near YXDD that ESM2 thinks could be mutated without penalty? (flexibility for engineering)

---

### Angle 5 — Graph Neural Network on Protein Structure

**What**: Represent each RT as a **residue contact graph** and learn structural motifs that distinguish active from inactive RTs.

**How**:
1. Build a graph for each RT: nodes = residues (with ESM2 per-residue features), edges = contacts (<8Å)
2. Use a simple GNN (GCN or GAT, 2-3 layers) with graph-level readout
3. Train in the nested LOFO framework
4. Key: use per-residue ESM2 embeddings as node features (not just one-hot or coordinate-based)

**Why it's new**: All current models treat the RT as a bag of features or a sequence. A GNN captures the **spatial topology** — which residues are near which in 3D. Combined with ESM2 per-residue features, this is a rich structural-functional representation.

**Implementation**: PyTorch Geometric, simple architecture to avoid overfitting on n=57.

**Output**: Graph-level embedding → Ridge regression → blend score.

---

## Integration Strategy

Do NOT dump all features into ElasticNet. Use a **layered ensemble**:

### Layer 1: Independent Models (5 angles + existing backbone)
- v6 backbone (EN + L12 + L33): existing, CLS 0.7088
- Angle 1 (evolutionary): Ridge on top features
- Angle 2 (dynamics): Ridge on top features
- Angle 3 (attention): Ridge on top features
- Angle 4 (mutagenesis): Ridge on top features
- Angle 5 (GNN): graph-level Ridge

### Layer 2: Diversity-Aware Blending
- Each angle gets ONE blend weight (not per-feature)
- v6 backbone always gets weight ≥ 0.5 (protect the base)
- New angles can only ADD information, not override
- Weight optimization via nested LOFO with constraint: `w_v6 >= 0.5`

### Layer 3: Classification Correction
- If the ensemble improves PR-AUC, apply a conservative post-hoc calibration
- Only adjust predictions near the active/inactive boundary
- Protect strong ranking (W-Spearman ≥ 0.73)

---

## Execution Order

### Phase A — Quick Wins (1-2 hours each, CPU)
1. **Angle 3 (Attention Surgery)**: extract during ESM2 forward pass, pure Python
2. **Angle 4 (Mutagenesis)**: masked language model scoring, pure Python
3. **Angle 1 partial (Conservation)**: download Pfam MSA, compute conservation scores

### Phase B — Medium Effort (2-4 hours, CPU)
4. **Angle 2 (Dynamics)**: install ProDy, compute NMA on AlphaFold structures
5. **Angle 1 full (Coevolution)**: DCA or MI from MSA

### Phase C — Heavy (4-8 hours, GPU)
6. **Angle 5 (GNN)**: PyTorch Geometric, train in LOFO

### Phase D — Integration
7. Audit all new features (redundancy, AUROC, confounders)
8. Layered ensemble with constrained blending
9. Final evaluation + bootstrap CI

---

## Go/No-Go Gates

### After Phase A (Angles 3+4)
- If at least 2 features have AUROC > 0.70 and max|r| < 0.7 with existing: GO to Phase B
- Otherwise: evaluate whether to continue

### After Phase B (Angles 1+2)
- If any angle individually improves CLS: GO to Phase C
- If at least 3 features are genuinely non-redundant: GO to integration

### After Phase C (GNN)
- If GNN alone beats EN alone: promising
- If GNN adds to v6 blend: very promising

### After Integration
- CLS > 0.72: **strong success**
- CLS > 0.715: **moderate success**
- CLS ≤ 0.7088: **the wall is real**

---

## What Makes This Different from Previous Attempts

| Previous | This plan |
|----------|-----------|
| Static features | Dynamic properties (NMA) |
| Mean-pooled embeddings | Surgical attention extraction |
| Generic ESM2 pseudo-PPL | RT-family-specific evolutionary model |
| Whole-protein features | Per-residue mutational landscape |
| Tabular/linear models | Graph neural network on 3D structure |
| Add features to same model | Layered ensemble with diversity constraint |
| One idea at a time | Five independent angles simultaneously |

---

## Compute Requirements

- **Phase A**: CPU only, ~2 hours total
- **Phase B**: CPU only, ~4 hours total (MSA download may take time)
- **Phase C**: GPU recommended for GNN training (~2 hours)
- **Phase D**: CPU, ~30 min

Total: **~8 hours**, parallelizable across phases.

---

## Risk Assessment

**Overall probability of beating 0.7088**: Medium (40-60%)

**Best angle by expected value**:
1. Angle 4 (mutagenesis) — most targeted, biologically grounded
2. Angle 3 (attention) — cheapest to extract, novel signal type
3. Angle 1 (evolution) — strongest biological prior
4. Angle 2 (dynamics) — biologically relevant but coarse
5. Angle 5 (GNN) — highest ceiling but highest overfitting risk

**Most likely failure mode**: All angles individually fail because n=57 is simply too small to learn from any representation. In that case, the layered ensemble is the last chance — diversity of weak signals might combine into a detectable one.
