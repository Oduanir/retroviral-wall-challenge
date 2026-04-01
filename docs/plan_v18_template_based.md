# Plan V18 — Template-Based RT-Cas9 Interaction Modeling

## Goal

Extract features that encode **how well each candidate RT fits into the prime editor complex**, by structurally placing each RT into a real PE cryo-EM structure and measuring compatibility.

This is the only remaining direction that adds genuinely new structural information not available from RT-alone analysis.

---

## Template Structure

**PDB 8WUS** — SpCas9-MMLV RT-pegRNA-target DNA complex (termination state)
- Source: Shuto et al., Nature 2024 (DOI: 10.1038/s41586-024-07497-8)
- Resolution: 2.90 Å (cryo-EM)
- Contains: SpCas9 nickase (chain A), MMLV-RT (chain B), pegRNA (chain C), target DNA (chains D/E/F)
- RT domain is well-resolved in catalytic position

Secondary template (optional): **8WUT** (initiation state, 3.00 Å) for comparison.

---

## Scientific Rationale

The V17 cheap proxies failed because they approximated compatibility without actually placing the RT in the complex. This plan does the placement:

1. **Align** each candidate RT structure onto the MMLV-RT chain in 8WUS
2. After alignment, the candidate RT now occupies the same 3D position as MMLV-RT in the PE complex
3. **Measure** how well it fits: clashes with Cas9, orientation of the active site relative to the DNA path, steric burden, etc.

This is not full docking. It is template-based homology placement — cheap, deterministic, and well-defined.

---

## Data Requirements

### Already available
- 57 AlphaFold-predicted RT structures (`data/raw/structures/`)
- YXDD motif positions (from `find_yxdd()`)
- Existing FoldSeek TM-scores to MMLV

### To download
- **8WUS.pdb** from RCSB (https://files.rcsb.org/download/8WUS.pdb)
- Optionally 8WUT.pdb for a second reference state

### Tools needed
- BioPython PDB module (already installed)
- **TMalign** for structural alignment (or BioPython Superimposer for CA-based alignment)

---

## Implementation

### Step 0 — Prepare template

1. Download 8WUS.pdb
2. Parse and identify chains:
   - Chain A: SpCas9 (keep as context)
   - Chain B: MMLV-RT (alignment target)
   - Chains C-F: pegRNA + DNA (keep as context for distance features)
3. Extract MMLV-RT CA coordinates as alignment reference
4. Extract Cas9 surface atoms (for clash detection)
5. Extract DNA/pegRNA backbone atoms (for active-site-to-substrate distance)

### Step 1 — Align each RT onto MMLV-RT in the complex

For each of the 57 candidate RTs:

1. Load candidate RT AlphaFold structure
2. **Structural alignment** of candidate RT onto chain B (MMLV-RT) of 8WUS:
   - Option A (preferred): TMalign if available — gives TM-score + rotation/translation matrix
   - Option B: BioPython `Superimposer` on matched CA atoms (sequence alignment → structure alignment)
   - Option C: YXDD-anchored local alignment (align YXDD regions only, then apply to full structure)
3. Apply the rotation+translation to the **full candidate RT structure**
4. The candidate RT is now placed in the PE complex context

### Step 2 — Extract compatibility features

After placement, compute these features for each RT:

**Important design constraint — avoid size/MMLV-likeness confounders:**

Many candidate RTs are much shorter than MMLV-RT (250 vs 677 residues). Raw features like "total clash count" or "volume ratio" will trivially correlate with size. Every feature must be either:
- **Normalized** by the aligned region size (not total RT size)
- **Computed only on the aligned/overlapping region**
- **Inherently size-independent** (angles, distances, ratios)

The audit (Step 3) must explicitly check each feature against `seq_length` and `foldseek_TM_MMLV`. If |r| > 0.8 with either, the feature is rejected as a phylogeny/size proxy.

---

**Clash/burden features (normalized by aligned region):**
- `cas9_clash_fraction`: fraction of **aligned** candidate RT CA atoms within 3Å of any Cas9 CA atom (normalized by number of aligned residues, not total)
- `cas9_clash_severe_fraction`: same with 2Å cutoff
- `cas9_min_distance`: minimum distance from any candidate RT atom to Cas9 surface (size-independent)
- `cas9_contact_fraction`: fraction of aligned CA atoms within 5Å of Cas9 (interface density, not area)

**Active-site orientation features (inherently size-independent):**
- `yxdd_to_dna_distance`: distance from YXDD centroid (after alignment) to the nearest DNA backbone atom
- `yxdd_to_pegrna_distance`: distance from YXDD centroid to nearest pegRNA backbone atom
- `yxdd_orientation_angle`: angle between YXDD local normal vector and the DNA axis direction (is the active site pointing toward the substrate?)

**Fit quality features (alignment-region only):**
- `alignment_rmsd`: RMSD of the structural alignment on matched residues only
- `alignment_coverage`: fraction of candidate RT residues that match within 5Å (alignment quality)
- `protrusion_fraction`: fraction of aligned residues extending beyond MMLV-RT local envelope (excess bulk per aligned residue)

**Context accessibility features:**
- `yxdd_occlusion_by_cas9`: fraction of YXDD solvent-accessible surface occluded by Cas9 after placement (does Cas9 block the active site?)
- `nterm_to_cas9_distance`: distance from candidate N-terminus to nearest Cas9 atom (fusion point geometry — inherently size-independent)

Total: ~12 features, all either normalized by aligned region or inherently size-independent.

### Step 3 — Audit

Same protocol as V17:
- Missingness
- Correlation with existing features
- AUROC for active
- Max |r| with existing backbone features
- Reject obviously redundant features

Key audit questions:
- Are the post-placement features genuinely different from the pre-placement RT-alone features?
- Does each feature correlate with `seq_length` or `foldseek_TM_MMLV` at |r| > 0.8? If yes, reject — it's a size/phylogeny proxy, not a compatibility signal.
- Do normalized features (fractions, distances) still show the same pattern as raw counts? If normalization kills the signal, the signal was size-driven.

### Step 4 — Integration test (controlled, not raw add-on)

V17 showed that raw feature addition to v6's fixed ElasticNet(α=1.0, l1=0.3) is not a fair test — the regularization was tuned for 35 features, not 36-37. The integration must account for this.

**Protocol:**

1. **Individual feature addition with retuning**: for each candidate feature, run nested LOFO with a small ElasticNet α sweep (0.5, 1.0, 2.0) × l1_ratio (0.2, 0.3, 0.4) to give the model a chance to absorb the new feature. Pick inner-best. This is 9 configs per feature — cheap enough.

2. **Feature-as-separate-model**: instead of adding to EN, use the top 1-2 features as a 4th model in the blend (e.g., Ridge on the feature alone). This avoids perturbing EN entirely and lets the blend weights decide relevance per fold.

3. **Corrector integration**: feed top features into a classification corrector on top of v6 scores (as attempted in V16, but now with genuinely new features that V16 didn't have).

**Comparison must be fair:**
- Always compare against v6 baseline run in the same script/environment
- Never compare across different ESM2 extraction runs (CPU vs GPU numerical differences)

---

## Alignment Strategy Detail

The quality of the alignment is critical. Three approaches in order of preference:

### Option A: TMalign (best)

TMalign produces a proper structural alignment with TM-score, RMSD, and a rotation matrix.

```bash
TMalign candidate.pdb reference_RT_chain.pdb -o alignment_output
```

- Pros: gold-standard structural alignment, handles insertions/deletions
- Cons: external binary, needs to be installed
- Installation: `conda install -c bioconda tmalign` or compile from source

### Option B: BioPython Superimposer on YXDD-anchored region

If TMalign is not available:
1. Find YXDD in both candidate and reference
2. Extract ±30 residues around YXDD
3. Sequence-align these local regions
4. Use matched positions for `Superimposer.set_atoms()` + `apply()`

- Pros: pure Python, no external tool
- Cons: less robust than TMalign, sensitive to local structural differences

### Option C: FoldSeek alignment matrix (if available)

If FoldSeek was run with output alignment matrices, these could be reused directly.

---

## Success Criteria

### GO (continue to deeper modeling)
- At least 2-3 post-placement features are genuinely non-redundant (max|r| < 0.7 with existing features)
- At least 1 feature improves CLS individually (delta > 0)
- OR: clear Retron-specific improvement even without global CLS gain

### NO-GO
- All features are proxies for RT size/similarity
- All features degrade CLS
- Post-placement features correlate >0.85 with pre-placement analogs

---

## What This Plan Can and Cannot Show

**Can show:**
- Whether placing RTs in the PE complex context reveals compatibility features not visible from RT alone
- Whether clash/orientation features are genuinely orthogonal to existing features
- Whether the template-based approach is worth escalating

**Cannot show:**
- Whether the RT actually functions in the complex (that requires dynamics/docking)
- Whether pegRNA/DNA interactions matter (we treat them as static context only)

---

## Compute Requirements

- **CPU only** — no GPU needed
- TMalign: ~1 sec per alignment × 57 = ~1 min
- Feature extraction: ~5 min total
- LOFO evaluation: ~2 min (no ESM2 needed if run on existing v6 pipeline)

Total: **~20 min** including ESM2 extraction for the LOFO test.

---

## Risk Assessment

**Scientific risk: Medium**
- The hypothesis (compatibility = fit in complex) is well-motivated
- But template-based placement is crude (static, no side-chain repacking, no flexibility)
- MMLV-RT has 677 residues; compact RTs have ~250 — the alignment will be partial

**Technical risk: Low**
- All tools available (BioPython, optionally TMalign)
- Geometry computations are straightforward
- No new models or training needed

**Probability of meaningful CLS gain: Low-to-Medium**
- Higher than V17 cheap proxies (which had no spatial context)
- Lower than full docking (which accounts for conformational flexibility)
- Best realistic outcome: +0.005–0.015 from improved Retron classification

---

## Recommended Execution Order

1. Download 8WUS.pdb, parse chains
2. Install/verify TMalign (or fall back to BioPython Superimposer)
3. Align all 57 RTs onto MMLV-RT in complex
4. Extract ~14 compatibility features
5. Audit (redundancy, correlation, AUROC)
6. Integration test (individual addition to v6)
7. Go/No-Go decision
