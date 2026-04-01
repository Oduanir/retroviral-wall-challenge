# Plan V19 — Moonshot to Break the Ceiling

## Intent

This is the **last serious plan**.

It assumes:
- `CLS 0.7088` is a real local ceiling for the current repo
- the missing signal is **not** in better regularization, more blending, or another RT-only embedding
- the missing signal is most likely in the **functional geometry of the prime editor complex**

So the moonshot is:

> stop modeling the RT in isolation and start modeling the **RT inside the prime editor machine**

This is not a small extension of the current pipeline.
It is a change of object:

- from `RT -> PE score`
- to `RT placed in PE complex -> compatibility features -> PE score`

---

## Central Hypothesis

The challenge is not asking:
- which RT is globally “good”

It is implicitly asking:
- which RT works best **when fused to Cas9 and operating on pegRNA-guided DNA in the PE architecture**

Therefore the missing signal may be encoded in:
- steric compatibility with Cas9
- accessibility of the RT active site to the DNA/pegRNA path
- fusion topology
- local geometry of the catalytic region in complex context
- interaction surfaces not visible in the isolated RT

---

## Moonshot Architecture

The grand plan has **4 layers**.

### Layer 1 — Build a Structural “PE Complex Atlas”

For each of the 57 RTs:
- start from a real PE structural template (`8WUS`, optionally `8WUT`)
- align the RT structurally onto the template RT domain
- place the full RT into the prime editor complex frame
- build several versions of the placed complex:
  - rigid placement
  - local side-chain repacking
  - light relaxation / minimization
  - optional alternative PE state (`8WUT`)

Output:
- one small structural atlas of candidate RTs in the PE complex frame

This becomes the core dataset for everything else.

### Layer 2 — Extract Multi-Scale Compatibility Features

Not just one feature family.
Extract a full hierarchy:

- **Global fit**
  - clash burden
  - compactness in complex
  - protrusion beyond template envelope
  - fusion-topology plausibility

- **Catalytic fit**
  - active-site to DNA distance
  - active-site to pegRNA distance
  - orientation of catalytic patch
  - occlusion by Cas9 / nucleic acids

- **Interface fit**
  - candidate RT surface near Cas9
  - candidate RT surface near nucleic acids
  - local contact density
  - buried/exposed interface area proxies

- **Deformation cost**
  - how much local distortion is needed to place the RT
  - alignment strain
  - local mismatch around catalytic region

- **State robustness**
  - same features under multiple template states / relaxations
  - variance across placements = confidence / instability proxy

### Layer 3 — Learn Compatibility as a New View

Do not replace the current best model.

Build a new **complex-view model** and compare:
- `v6` alone
- `complex-only`
- `v6 + complex-view`
- `v6 + complex-view corrector`

Possible model families:
- regularized tabular model on complex features
- graph model on local contact neighborhoods
- two-head model:
  - one head for `active`
  - one head for ranking among actives

The key is that the complex-view must be treated as a **new modality**, not just a few extra features thrown into the old EN.

### Layer 4 — Generate Counterfactuals

Once the complex representation exists, use it to ask:
- why does a candidate fail?
- what would make it more MMLV-like in the relevant local geometry without making it globally MMLV-like?

Counterfactual analyses:
- mutate or swap local catalytic surroundings in silico
- quantify which local structural deviations most strongly hurt compatibility
- identify whether the failure mode is:
  - clash
  - active-site misorientation
  - excessive burden
  - missing interface accessibility

Even if this does not improve the score much, it can produce a much more interesting scientific result.

---

## Workstreams

## Workstream A — Structural Atlas

### A1. Template selection

Use:
- `8WUS` as primary PE complex template
- `8WUT` as optional second conformational state

### A2. RT placement

For each RT:
- structural alignment to template RT
- save rigidly placed complex coordinates
- compute aligned residue mapping
- keep coverage and strain metadata

### A3. Relaxed placement

Optional but strongly recommended:
- local side-chain repacking
- lightweight minimization
- maybe restricted relaxation around the RT / interface only

Purpose:
- reduce “obvious rigid-body artifacts”

Deliverables:
- placed structures
- alignment metadata
- per-state quality report

## Workstream B — Complex Feature Bank

### B1. Global complex fit

Features:
- clash fractions
- minimum distances
- contact density
- protrusion fraction
- envelope mismatch

### B2. Catalytic-route geometry

Features:
- YXDD centroid to DNA / pegRNA
- orientation vectors
- active-site occlusion
- catalytic patch exposure in complex

### B3. Fusion topology

Features:
- N/C terminal distances to Cas9
- path plausibility from fusion anchor to active site
- tail burden near Cas9

### B4. Robustness features

Features:
- variance across template states
- variance across relaxed vs rigid placement
- confidence score per RT

Deliverables:
- `v18`-style CSV, but now as a serious complex-view dataset

## Workstream C — Learning and Integration

### C1. Complex-only benchmark

Goal:
- can complex-view features alone classify active/inactive better than RT-only local features?

### C2. Additive benchmark against v6

Goal:
- do complex features improve `PR-AUC` while preserving `W-Spearman`?

### C3. Corrector benchmark

Goal:
- use complex features only to correct uncertain `v6` predictions

### C4. Retron-special benchmark

Goal:
- determine whether the complex-view is especially useful on the Retron bottleneck

## Workstream D — Scientific Interpretation

### D1. Failure mode atlas

For each difficult RT:
- dominant compatibility failure
- comparison to MMLV / strongest known actives

### D2. Counterfactual local analysis

Estimate:
- which local geometric differences matter most
- whether active retrons succeed through a different geometry regime than retrovirals

---

## Implementation Plan

## Stage 1 — Infrastructure

Create:
- `src/v19_build_complex_atlas.py`
- `src/v19_extract_complex_features.py`
- `src/v19_evaluate.py`
- optional `src/v19_relax_complex.py`

Outputs:
- placed complex structures
- per-RT alignment metadata
- complex feature CSV
- evaluation tables

## Stage 2 — Rigid Template Placement

Implement first:
- template parsing
- RT alignment to template RT
- rigid placement in complex frame
- aligned residue coverage
- base clash/distance/orientation features

This is the minimum moonshot core.

## Stage 3 — Local Relaxation

If Stage 2 features are promising:
- add side-chain repacking or local minimization
- recompute feature bank
- test whether relaxed placement improves signal quality

## Stage 4 — Model Integration

Run:
- `complex-only`
- `v6 + complex`
- `v6 + complex corrector`

Always compare to `v6` in the same script/environment.

## Stage 5 — Counterfactual Analysis

If there is any signal:
- quantify which complex features explain success/failure
- inspect Retron vs Retroviral differences
- generate a short scientific note

---

## Evaluation Protocol

Must remain strict:
- nested LOFO
- same official evaluator
- same output ordering
- same `v6` baseline run in the same script

Additional readouts:
- per-family PR-AUC delta
- Retron-only impact
- classification / ranking trade-off
- bootstrap CI for any claimed gain

---

## Go / No-Go Gates

## Gate 1 — Structural Validity

Go if:
- most RTs can be placed cleanly
- alignment coverage is not catastrophically low
- features are numerically stable

No-Go if:
- placements are mostly meaningless
- compact RTs cannot be mapped in a coherent way

## Gate 2 — Feature Novelty

Go if:
- at least several complex features are not just size/FoldSeek proxies
- they survive redundancy audit

No-Go if:
- everything collapses into length / MMLV similarity

## Gate 3 — Additive Value

Go if:
- any complex feature or feature block improves `CLS`
- or gives clear Retron-specific improvement

No-Go if:
- complex view adds no signal over `v6`

## Gate 4 — Scientific Value

Even if score gain is weak, continue only if:
- the complex representation reveals meaningful failure modes
- or gives a compelling explanation for the current ceiling

Otherwise stop.

---

## Feasibility

## Scientific feasibility

Rating: **Medium**

Why:
- strong biological motivation
- directly targets the likely missing signal
- far more relevant than another RT-only embedding

Main risk:
- the true missing signal may depend on dynamics or chemistry beyond rigid or lightly relaxed geometry

## Technical feasibility

Rating: **Medium**

Why:
- feasible with current structures and templates
- but significantly more involved than any previous phase
- alignment, placement, relaxation, and geometric feature extraction all need careful engineering

## Compute feasibility

Rating: **Medium**

Why:
- rigid placement is cheap
- relaxation can be moderate
- full docking or multistate modeling can become expensive quickly

## Probability of meaningful score gain

Rating: **Low-to-Medium**

Why:
- this is one of the very few remaining directions that may add genuinely new information
- but it is also a hard hypothesis to cash out in a static dataset of 57 RTs

Best realistic outcome:
- `+0.005` to `+0.02` if the missing classification signal is really geometric and template-accessible

Moonshot outcome:
- a defensible new structural explanation of PE compatibility, with or without a large score jump

---

## Why This Plan Is Grandiose

Because it does not ask:
- “which hyperparameter did we miss?”

It asks:
- “have we been modeling the wrong object all along?”

If the answer is yes, then this is the first plan in the repo that actually attacks that mistake directly.

---

## Recommendation

If you want one last serious shot, this is it.

Not because it is safe.
Because it is the only remaining plan that could plausibly:
- improve the score
- explain the ceiling
- and produce a genuinely interesting scientific result even if the score gain is modest
