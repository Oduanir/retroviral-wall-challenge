# Plan — RT-Cas9 Compatibility Signal (Cheap Version)

## Goal

Test whether the missing signal behind PE activity comes from **RT compatibility with the prime editor complex** rather than from RT sequence/structure alone.

This plan is intentionally **cheap-first**:
- first test whether a compatibility signal exists at all
- only escalate to heavier structural modeling if the cheap signal is real

Primary question:

> Can simple RT-Cas9 compatibility features improve `PR-AUC` without destroying the current strong `W-Spearman` backbone?

---

## Scientific Hypothesis

The current repo models mostly:
- RT sequence
- RT isolated structure
- RT phylogeny

But PE efficiency likely depends on:
- how the RT sits in the Cas9 fusion
- whether the catalytic core is accessible in the fused architecture
- whether the RT shape/orientation is compatible with the DNA/pegRNA path

So the missing signal may be:
- **not “is this RT a good enzyme in isolation?”**
- but **“is this RT geometrically compatible with the prime editor complex?”**

---

## Why Start Cheap

The main risk is not just implementation cost.
The main risk is that this entire direction may be too noisy or wrong.

So the right sequence is:

1. test a few coarse but biologically plausible compatibility features
2. check whether they add anything on top of the current best model
3. escalate only if they show signal

If the cheap version fails, there is no reason to spend weeks on docking or complex GNNs.

---

## Cheap Version — Scope

The cheap version does **not** build a full RT-Cas9-pegRNA-DNA complex.

It only builds features that approximate:
- fusion compatibility
- active-site accessibility
- gross steric burden
- architectural compatibility with a prime editor template

These features should be added **on top of the current best backbone** (`v6`), not used as a standalone model.

---

## Data Inputs

### Required

- challenge RT sequences
- current tabular + FoldSeek + ESM2 backbone
- predicted or available RT structures already used in the repo

### Strongly recommended

- one or more reference prime editor structures / templates
- at minimum: a reference RT known to work in PE (`MMLV`-like template)

### Optional

- domain boundaries if already inferred
- residue-level confidence scores from structure prediction

---

## Cheap Feature Families

## Priority Cheap Feature Set (Implement First)

Do **not** start by engineering every possible proxy.
Start with a very small high-priority set that is easy to audit and hard to confuse with the current backbone.

Recommended first batch:

1. `nterm_to_yxdd_frac`
- normalized sequence distance from N-terminus to `YXDD`
- proxy for where the catalytic core sits relative to the likely fusion geometry

2. `cterm_to_yxdd_frac`
- normalized sequence distance from C-terminus to `YXDD`
- complementary proxy for fusion topology / tail burden

3. `yxdd_surface_proxy`
- coarse proxy for whether the catalytic patch is surface-accessible
- can be approximated from existing structure-derived descriptors if full SASA is unavailable

4. `yxdd_local_compactness`
- local packing / compactness around the catalytic patch
- proxy for whether the active region is sterically tight or open

5. `global_vs_local_mmlv_gap`
- difference between global structural similarity to `MMLV` and local catalytic-patch similarity to `MMLV`
- captures the “locally MMLV-like, globally different” hypothesis

6. `global_vs_local_best_gap`
- same idea as above, but relative to the best available PE-compatible template rather than MMLV only

7. `rnaseh_present_proxy`
- binary or coarse continuous feature indicating the likely presence / absence of RNase H or a long C-terminal extension
- fusion burden proxy

8. `fusion_burden_proxy`
- simple size / elongation burden, combining sequence length and gross compactness

9. `yxdd_confidence_mean`
- mean local confidence around the catalytic patch
- measures whether the structure around the active site is well-defined

10. `termini_asymmetry_to_core`
- asymmetry between N-term and C-term distance to the catalytic core
- cheap proxy for whether one end is unusually long relative to where catalysis happens

Implementation rule:
- code these first
- audit these first
- only add a second wave if at least 2–3 of them show non-redundant signal

---

## Block A — Fusion Geometry Proxies

These are simple descriptors of how plausible the RT is as a fusion partner.

Candidate features:
- RT sequence length
- RT structure compactness / radius proxy
- aspect ratio / elongation proxy
- distance from N-terminus to catalytic core
- distance from C-terminus to catalytic core
- whether the catalytic core is buried far from the likely fusion-exposed side
- RNase H present / absent proxy

Rationale:
- a PE RT is not only required to be active
- it must also fit physically and topologically into the fused editor architecture

---

## Block B — Active-Site Accessibility Proxies

These features estimate whether the catalytic core is sterically reachable.

Candidate features:
- solvent exposure proxy around `YXDD`
- local packing density around catalytic residues
- mean confidence / disorder around the catalytic patch
- local compactness of the palm region
- distance from catalytic patch to protein surface
- whether the catalytic pocket opens toward the same side as the reference RT

Rationale:
- an enzymatically valid RT may still be poorly usable in PE if the active site is badly oriented or inaccessible in fusion context

---

## Block C — Template-Aligned Compatibility Proxies

Use a known PE-compatible RT as a template and compare each candidate RT to it.

Candidate features:
- global structural alignment score to the PE reference RT
- local alignment score around palm / catalytic patch
- distance between candidate and reference catalytic patch centroids
- candidate-vs-reference N/C-terminal placement relative to the catalytic core
- mismatch between global shape and local catalytic similarity
- “locally MMLV-like, globally different” score

Rationale:
- the missing signal may be specifically “compatible with the PE scaffold”, not merely “globally RT-like”

---

## Block D — Burden / Clash Proxies

These are cheap substitutes for full docking.

Candidate features:
- total volume or mass proxy
- protrusion / irregularity score
- number of long flexible appendages
- local confidence drop near putative interaction-facing surfaces
- simple clash-risk proxy after coarse placement onto a reference RT frame

Rationale:
- some RTs may fail because they physically burden the PE architecture even if their catalytic core is fine

---

## Minimal Experimental Plan

## Exp 1 — Feature Audit

Build all cheap compatibility features and answer:
- are they numerically stable?
- are they strongly redundant with existing features?
- do any correlate with `active` or `pe_efficiency_pct`?

Deliverable:
- one compact feature table
- one summary of missingness / redundancy / raw correlation

Go if:
- at least a few features are not trivial copies of length / FoldSeek / thermostability

---

## Exp 2 — Additive Value on Top of v6

Train:
- current `v6` backbone
- `v6 + cheap compatibility block`

Use strict nested LOFO.

Readout:
- does `PR-AUC` improve?
- does `CLS` improve?
- does `W-Spearman` stay close to baseline?

Success criterion:
- any stable `CLS` gain
- or `PR-AUC +0.02` with little ranking loss

This is the main cheap test.

---

## Exp 3 — Retron-Focused Readout

Even if the global gain is small, check whether compatibility features help specifically on:
- `Retron`
- other compact/non-canonical RTs

Readout:
- family PR-AUC deltas
- false-positive / false-negative changes

Go if:
- the compatibility block clearly improves the difficult families

---

## Exp 4 — Cheap Corrector

If Exp 2 shows signal but not enough global gain:
- feed only a tiny subset of compatibility features into the existing classification corrector logic
- do not rebuild the whole model

Goal:
- use compatibility signal only where `v6` is uncertain

---

## Implementation Plan

## Step 0 — Decide Template(s)

Pick one or two PE-compatible RT references:
- minimum: `MMLV`
- optional second reference if already trusted in the repo

Output:
- fixed list of reference RTs
- explicit justification

---

## Step 1 — Build Cheap Structural Descriptors

Create one script, for example:
- `src/rt_cas9_cheap_features.py`

Inputs:
- RT names
- sequences
- existing structures / derived coordinates

Outputs:
- CSV with one row per RT
- columns grouped by `fusion`, `accessibility`, `template`, `burden`

Expected implementation tasks:
- compute catalytic-core anchor position
- compute termini-to-core distances
- compute simple compactness/surface proxies
- compute template-aligned local/global mismatch scores

Priority implementation order:
1. termini-to-core distances
2. local catalytic compactness / confidence
3. local-vs-global template mismatch
4. RNaseH / tail-burden proxies

---

## Step 2 — Audit the New Features

Create one evaluation script, for example:
- `src/rt_cas9_cheap_audit.py`

Checks:
- missingness
- pairwise correlation with existing features
- raw univariate association with `active`
- raw univariate association with `pe_efficiency_pct`

Deliverable:
- short markdown results note

Purpose:
- reject obviously redundant or noisy features before model integration

Priority audit questions:
- are `global_vs_local_*` features saying anything not already contained in FoldSeek TM-scores?
- are termini-to-core distances doing more than raw sequence length?
- do local catalytic accessibility features correlate with `active` independently of current structural similarity features?

---

## Step 3 — Integrate into Nested LOFO

Create one experiment script, for example:
- `src/v17_rt_cas9_cheap.py`

Runs:
- baseline `v6`
- `v6 + all cheap features`
- `v6 + top cheap subset`
- optional `corrector + cheap subset`

Evaluation:
- strict nested LOFO
- same official evaluator
- family-level breakdown

---

## Step 4 — Decide Go / No-Go

### Go

Continue to the intermediate version if:
- `CLS` improves cleanly
- or `PR-AUC` improves clearly on difficult families
- or compatibility features consistently survive regularization

### No-Go

Stop this direction if:
- features collapse into length/FoldSeek proxies
- no gain over `v6`
- retron bottleneck remains unchanged

---

## Feasibility Assessment

## Scientific Feasibility

Rating: **Medium**

Why:
- the hypothesis is biologically plausible
- it directly targets a signal missing from the current repo
- it is more orthogonal than “another PLM” or “another local embedding”

Main uncertainty:
- the cheap proxies may still be too crude to reflect the true PE complex geometry

---

## Technical Feasibility

Rating: **High** for the cheap version

Why:
- no new heavy model required
- can reuse existing structures / structural summaries
- no fine-tuning
- no docking required
- fits naturally into the current tabular + nested LOFO framework

Expected effort:
- low-to-moderate scripting effort
- mostly feature engineering and audit

---

## Compute Feasibility

Rating: **High**

Why:
- feature extraction should be light
- nested LOFO on a few additional tabular features is cheap
- no GPU dependency required for the cheap version

---

## Probability of Meaningful Gain

Rating: **Low-to-Medium**, but higher than most remaining “small tweaks”

Why:
- this direction may contain genuinely new information
- but cheap proxies may be too weak
- therefore it is a good **falsification experiment**

Expected realistic outcomes:
- most likely: small or no gain
- best case: measurable `PR-AUC` lift, especially on difficult families
- low probability but valuable outcome: first evidence that fusion compatibility is the missing signal

Most likely source of gain if this works:
- improved classification on borderline inactive/active RTs
- especially compact or non-canonical families where gross RT quality is not enough

---

## What This Plan Can and Cannot Prove

It **can** show:
- whether coarse RT-Cas9 compatibility signal exists
- whether this signal helps the current backbone
- whether this direction deserves escalation

It **cannot** prove:
- that full docking / complex modeling will work
- that lack of gain means the complex hypothesis is false

Failure of the cheap version means:
- either the hypothesis is wrong
- or the approximation is too crude

That is still valuable information.

---

## Recommended Decision

This direction is worth trying **only because it is one of the few remaining ideas that may add genuinely new information**.

Do it as a cheap falsification test.

Do **not** jump directly to:
- full docking
- AF-Multimer pipelines
- complex GNNs

Only escalate if the cheap compatibility block shows real additive value over `v6`.
