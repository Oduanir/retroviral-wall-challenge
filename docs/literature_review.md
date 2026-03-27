# Retroviral Wall Challenge — Literature Review

> **Goal**: Predict the efficiency of diverse reverse transcriptases (RTs) as prime editors.
> **Date**: March 2026

---

## Table of Contents

1. [Background: prime editing](#1-background-prime-editing)
2. [Screening diverse RTs for prime editing](#2-screening-diverse-rts-for-prime-editing)
3. [Structural and biochemical properties of RTs](#3-structural-and-biochemical-properties-of-rts)
4. [ML approaches for prime editing](#4-ml-approaches-for-prime-editing)
5. [Protein language models and RT engineering](#5-protein-language-models-and-rt-engineering)
6. [Strategic synthesis for the challenge](#6-strategic-synthesis-for-the-challenge)
7. [References](#7-references)

---

## 1. Background: prime editing

Prime editing is a CRISPR-derived genome editing technique that enables substitutions, insertions, and deletions without double-strand breaks or donor templates. The system relies on fusing a Cas9 nickase (H840A) with a **reverse transcriptase** (RT), guided by a **pegRNA** (prime editing guide RNA) encoding both the guide sequence and the editing template.

The standard RT used is the Moloney murine leukemia virus (**MMLV-RT**), as an optimized pentamutant (D200N/L603W/T306K/W313F/T330P). However, this RT is large (~2 kb), limiting AAV packaging for in vivo therapeutic applications.

The foundational framework was established by **Anzalone et al. (Nature 2019)**, who introduced the **PE1, PE2, and PE3** architectures. This paper is important for the challenge as it highlights two key points: (i) the reference editor relies on an **MMLV-RT truncated of the RNase H domain**, and (ii) prime editing efficiency is not an intrinsic property of the RT alone, but the result of a tradeoff between catalysis, processivity, Cas9 compatibility, and pegRNA dynamics.

---

## 2. Screening diverse RTs for prime editing

### 2.1 The foundational PE6 study — Liu et al., Cell 2023

This is the study most directly linked to the challenge. The Liu laboratory (Broad Institute) screened **59 reverse transcriptases from 14 phylogenetic classes** as prime editors:

- **Families tested**: retroviruses (MMLV, ASLV, PERV, etc.), LTR retrotransposons (Tf1, Ty3, Gypsy), bacterial retrons (Ec48, Ec67), group II introns (LtrA, Er), CRISPR-associated, telomerases, and others
- **Key results**:
  - Different RTs specialize in different edit types — some are better for long insertions, others for substitutions
  - Retroviral RTs show the highest overall prime editing efficiency
  - Some compact RTs (retrons, group II introns) show detectable but weak activity
- **Directed evolution (PACE)**: compact RTs were improved up to **22x** via phage-assisted evolution
- **PE6a-g variants**: 516-810 bp smaller than PEmax, AAV-compatible
  - **PE6c** (evolved Tf1 RT, retrotransposon): best overall efficiency among compact editors
  - **PE6d** (evolved MMLV): 2-3.5x more efficient than PE3
- **In vivo** (mouse brain): PE6 dual-AAV systems achieve **12 to 183x** higher efficiency than previous systems

> **Direct relevance**: The 57 RTs in the challenge very likely correspond to a subset of the 59 RTs tested in this study. The families (Retroviral, Retron, LTR_Retrotransposon, Group_II_Intron, CRISPR-associated) are identical.

### 2.2 Porcine RT (PERV) — pvPE, 2025

An RT derived from porcine endogenous retrovirus (PERV, Bama minipig) was developed as a prime editor:

- After iterative optimization (RT engineering, structural modifications, fusion with La protein), **pvPE-V4** achieves **24-102x** higher efficiency than the initial version
- Up to **2.39x** more efficient than PE7
- Significantly fewer unwanted edits

> **Relevance**: Demonstrates that non-MMLV RTs can be superior, and that key properties are processivity, heteroduplex stability, and interaction with accessory proteins.

### 2.3 PE7 — Stabilization by La protein, Nature 2024

- **La** is an endogenous eukaryotic protein that binds and stabilizes 3' polyU RNA ends
- Fused to the C-terminus of PEmax, it produces **PE7**, significantly improving efficiency
- The N-terminal domain of La (aa 1-194) is sufficient

> **Relevance**: pegRNA stabilization by accessory proteins is a critical factor, but beyond the intrinsic properties of the RT (out of scope for the challenge).

### 2.4 Implications of the PE6 screen for this challenge

Beyond the simple observation that some families perform better than others, the PE6 study suggests three directly useful lessons:

- **Compactness does not equal inefficiency**: some shorter RTs remain editors after optimization, so length alone should not be used as a naive proxy for performance
- **Phylogeny explains part, not all**: within a single family, activity varies strongly, justifying the exploitation of fine motif, structure, and electrostatic features
- **Small local modifications can have large effects**: gains from directed evolution indicate that subtle signatures around the active site and nucleic acid binding surfaces can tip an RT from inactive to useful

---

## 3. Structural and biochemical properties of RTs

### 3.1 Cryo-EM structures of the prime editor complex — Nature 2024

Two major structural studies resolved the **SpCas9-MMLV RT(ΔRNaseH)-pegRNA-target DNA** complex:

- MMLV RT extends reverse transcription beyond the intended site, causing scaffold-derived incorporations (unwanted edits)
- The RT remains in the same relative position to SpCas9 during reverse transcription
- The synthesized pegRNA/DNA heteroduplex accumulates along the SpCas9 surface

**Functional RT subdomains** (directly related to challenge features):
- **Fingers**: template and incoming dNTP binding
- **Palm**: contains the catalytic site (YXDD motif) and motifs 1-2
- **Thumb**: double-stranded nucleic acid binding
- **Connection**: links palm to RNaseH domain

### 3.2 RT contribution to DNA repair — Nature Biotechnology 2025

- The RT of prime editors is rapidly recruited to DNA damage sites and contributes to endogenous repair
- RT/PE overexpression increases short insertions and decreases HDR
- A compact PE without RNaseH domain shows less repair activity — potentially more suitable for clinical use

### 3.3 Domain architecture and compactness/processivity tradeoff

The literature suggests that the **overall RT architecture** is nearly as important as the catalytic site itself:

- The **original prime editor** relies on an **MMLV-RT ΔRNaseH**, showing that the RNase H domain is not required for basic prime editing activity
- PE6 work shows that a more compact RT can become highly competitive after optimization, but potentially at the cost of processivity, complex stability, or editing window
- Cryo-EM structures indicate that the relative geometry between **Cas9, pegRNA, and RT** strongly constrains catalysis, making plausible major effects from **length**, **subdomain positioning**, and **heteroduplex interaction surface**

> **Inference for the challenge**: if the dataset contains proxies for compactness, length, or domain annotation, they should be treated as first-order variables, not mere descriptive covariates.

### 3.4 Structural features relevant for prediction

Based on the literature, the following properties correlate with PE activity:

| Property | Challenge feature(s) | Rationale |
|----------|---------------------|-----------|
| Catalytic site geometry | `triad_found_bin`, `D1_D2_dist`, `D2_D3_dist`, `triad_best_rmsd` | The catalytic Asp triad must adopt a precise geometry |
| YXDD motif | `yxdd_seq`, `yxdd_*` features | Catalytic core of the RT, YMDD (MMLV) is the gold standard |
| Structural similarity to MMLV | `foldseek_TM_MMLV`, `foldseek_TM_MMLVPE` | MMLV is the most efficient known RT for PE |
| Thermostability | `t40_raw` to `t80_raw`, `thermophilicity_num` | Moderate stability needed (37°C), hyperthermostability potentially deleterious |
| Sequence-structure compatibility | `perplexity`, `log_likelihood` (ESM-IF) | Good compatibility = correct folding |
| Structural quality | `rama_fav`, `rama_out` | Indicators of predicted structure quality |
| Active site properties | `pocket_*` features | Accessibility and environment of the catalytic site |
| Electrostatic potential | `*_mean_pot` features | Charge distribution across each subdomain |
| Beta-hairpin | `hairpin_pass`, `hairpin_confidence` | Conserved structural element near the active site |
| Size / domain architecture | Length or compactness proxies if available | PE1/PE2/PE6 literature shows a direct tradeoff between compactness, structural positioning, and performance |

---

## 4. ML approaches for prime editing

### 4.1 DeepPE — Nature Biotechnology 2021

First widely used model for pegRNA design:

- **Data**: 54,836 pegRNA-target pairs for PE2
- **Output**: pegRNA efficiency prediction based on PBS, RTT, edit type, and position
- **Historical interest**: shows as early as 2021 that large-scale systematic screening enables learning robust determinants of prime editing

> **Note**: published online September 2020, appears in Nature Biotechnology 39 (2021); DOI `s41587-020-0677-y` is consistent.

> **Limitation for us**: the target is pegRNA/site design, not the RT itself.

### 4.2 PRIDICT — Nature Biotechnology 2023

First high-performance model for PE on large screens:

- **Architecture**: Bidirectional recurrent neural network with attention (AttnBiRNN)
- **Data**: 92,423 pegRNAs, 13,349 pathogenic human mutations
- **Performance**: Spearman R = 0.85 for desired edits
- **Key contribution**: also predicts byproducts, not just raw efficiency

### 4.3 PRIDICT2.0 / ePRIDICT — Schwank Lab, Nature Biotechnology 2024

Reference model for PE efficiency prediction:

- **Architecture**: Bidirectional recurrent neural network with attention (AttnBiRNN)
- **Data**: >400,000 pegRNAs, 13,349 pathogenic human mutations
- **Performance**: Spearman ρ = 0.91 in HEK293T, ρ = 0.81 in K562

**ePRIDICT** (epigenetic version): XGBoost + 6 ENCODE datasets, evaluates chromatin influence.

> **Limitation**: Predicts efficiency as a function of **pegRNA and target site**, not the RT. The challenge poses the inverse problem.

### 4.4 DeepPrime — Cell 2023

- **Data**: 338,996 pegRNA-target pairs, 3,979 epegRNAs
- **Scope**: multiple PE systems and cell types
- **Interest**: shows that cross-system generalization is possible with sufficiently large training space

### 4.5 DTMP-Prime — Molecular Therapy Nucleic Acids, 2024

- **Architecture**: Bidirectional transformer (BERT) with multi-head attention
- **Innovation**: Novel pegRNA-DNA pair encoding, DNABERT integration
- **43 features** extracted for PE2 and PE3

### 4.6 PrimeNet — Briefings in Bioinformatics, 2025

- **Architecture**: Multi-scale convolutions + attention
- **Innovation**: Integration of epigenetic factors
- **Performance**: Spearman ρ = 0.94 (best published)

### 4.7 Summary

All these models predict efficiency **as a function of pegRNA/target site, for a fixed RT (MMLV)**. **No published model predicts efficiency as a function of the RT's intrinsic properties** — this is exactly the gap this challenge fills.

---

## 5. Protein language models and RT engineering

### 5.1 ESM2 — Meta/FAIR

- Protein language model pre-trained on billions of sequences
- Generates rich embeddings capturing structure, function, and evolution
- Used in zero-shot mode to predict mutation effects on protein fitness
- The challenge provides **pre-computed ESM2 embeddings** (`esm2_embeddings.npz`)

### 5.2 AI-driven RT engineering for Taq polymerase — Frontiers 2024

The most directly transferable approach to our problem:

- **Pipeline**: PLM embeddings of Taq polymerase variants → **Ridge regression** → RT activity prediction
- In silico screening of **>18 million potential mutations**, reduced to 16 candidates
- 18 variants identified with significantly improved RTase activity
- Validation by real-time RT-qPCR

> **Major relevance**: Directly applicable pipeline — ESM2 embeddings of the RT + lightweight supervised model (Ridge, XGBoost) = RT activity prediction.

### 5.3 rPE with AI-driven optimization — Nature Communications 2025

- Reverse prime editing (rPE): SpCas9 variant enabling 3' editing from the HNH nick
- **The RT was optimized using protein language models (ESM2-type)** combined with La
- Variants **erPEmax** and **erPE7max** reaching **44.41% efficiency**

> **Relevance**: Direct demonstration that PLMs can guide RT engineering for prime editing.

### 5.4 AI-generated MLH1 small binder — Cell 2025

- Use of **RFdiffusion and AlphaFold 3** to design a mini-binder (MLH1-SB, 82 aa) inhibiting mismatch repair
- PE7-SB2 system: **18.8x** more efficient than PEmax

---

## 6. Strategic synthesis for the challenge

### 6.1 What is unique about this challenge

- **Inverse problem** compared to the literature: we predict PE efficiency as a function of the **RT**, not the pegRNA
- **Very small dataset** (57 samples, 21 active) → deep learning unsuitable, focus on classical models
- **Rich features** (~65 tabular features + ESM2 embeddings + PDB structures)
- **Transductive**: the same RTs appear in train and test

### 6.2 Hypotheses from the literature

1. **Structural similarity to MMLV is the best predictor** — `foldseek_TM_MMLV` and `foldseek_TM_MMLVPE` should be strongly correlated with efficiency
2. **Phylogenetic family is informative** — retroviral RTs are systematically the most active
3. **Catalytic site geometry is necessary but not sufficient** — `triad_found_bin` and `triad_best_rmsd` are prerequisites
4. **ESM2 embeddings capture complementary signals** — sequence-structure compatibility and evolution
5. **Two-stage approach** (classify active/inactive then regress efficiency) could be relevant given the bimodal target distribution

### 6.3 Recommended modeling approaches

| Approach | Literature justification |
|----------|------------------------|
| Ridge regression on ESM2 embeddings | Validated for RTase activity prediction (Taq study, Frontiers 2024) |
| XGBoost/LightGBM on tabular features | ePRIDICT shows gradient boosting effectiveness for PE |
| Aggressive feature selection | Feature/sample ratio ~65/57, high overfitting risk |
| Leave-One-Family-Out cross-validation | `family_splits.csv` provided, avoids data leakage between related families |
| Ensembling (Ridge + XGBoost + SVR) | Standard approach for small datasets, increased robustness |

### 6.4 Essential evaluation discipline for this benchmark

- **All variable selection, PCA, hyperparameter tuning, and stacking must be done within each fold**; otherwise, information leakage risk is very high
- **Family must be the primary evaluation unit**: good performance on random splits would have little scientific value here
- The **transductive** nature of the benchmark allows certain **unsupervised** steps on all available sequences/structures, but never using test targets to calibrate a model
- With only 57 RTs, prioritize **ranking stability**, bootstrap confidence intervals, and per-family error analysis over a single aggregate score

---

## 7. References

### Prime editing — Fundamentals and RT

1. Anzalone et al. (2019). *Search-and-replace genome editing without double-strand breaks or donor DNA.* **Nature**. [Link](https://www.nature.com/articles/s41586-019-1711-4)

2. Liu et al. (2023). *Phage-assisted evolution and protein engineering yield compact, efficient prime editors.* **Cell**. [Link](https://www.cell.com/cell/fulltext/S0092-8674(23)00854-1)

3. Chen & Liu (2023). *Prime editing for precise and highly versatile genome manipulation.* **Nature Reviews Genetics**, 24, 161-177.

4. Zhang et al. (2025). *Highly efficient prime editors based on porcine endogenous retrovirus RT.* **Trends in Biotechnology**. [Link](https://www.cell.com/trends/biotechnology/fulltext/S0167-7799(25)00314-2)

5. Dolan et al. (2024). *Improving prime editing with an endogenous small RNA-binding protein (PE7).* **Nature**. [Link](https://www.nature.com/articles/s41586-024-07259-6)

### Cryo-EM structures

6. Yan et al. (2024). *Structural basis for pegRNA-guided reverse transcription by a prime editor.* **Nature**. [Link](https://www.nature.com/articles/s41586-024-07497-8)

7. Bock et al. (2024). *Cryo-EM structure of a prime editor complex in multiple states.* **Nature Biotechnology**. [Link](https://www.nature.com/articles/s41587-024-02325-w)

### RT and DNA repair

8. Dolan et al. (2025). *The reverse transcriptase domain of prime editors contributes to DNA repair.* **Nature Biotechnology**. [Link](https://www.nature.com/articles/s41587-025-02568-1)

### ML for prime editing

9. Kim et al. (2021). *Predicting the efficiency of prime editing guide RNAs in human cells* (**DeepPE model**; online 2020). **Nature Biotechnology**. [Link](https://www.nature.com/articles/s41587-020-0677-y)

10. Mathis et al. (2023). *Predicting prime editing efficiency and product purity by deep learning (PRIDICT).* **Nature Biotechnology**. [Link](https://www.nature.com/articles/s41587-022-01613-7)

11. Mathis et al. (2024). *Machine learning prediction of prime editing efficiency across diverse chromatin contexts (PRIDICT2.0 / ePRIDICT).* **Nature Biotechnology**. [Link](https://www.nature.com/articles/s41587-024-02268-2)

12. Mathis et al. (2025). *PRIDICT2.0 protocol.* **Nature Protocols**. [Link](https://www.nature.com/articles/s41596-025-01244-7)

13. Yu et al. (2023). *Prediction of efficiencies for diverse prime editing systems in multiple cell types.* **Cell**. [Link](https://www.sciencedirect.com/science/article/pii/S0092867423003318)

14. Panahi et al. (2024). *DTMP-Prime: Bidirectional transformer for prime editing efficiency prediction.* **Molecular Therapy Nucleic Acids**. [Link](https://www.sciencedirect.com/science/article/pii/S2162253124002579)

15. PrimeNet (2025). *Multi-scale convolution and attention for prime editing prediction.* **Briefings in Bioinformatics**. [Link](https://academic.oup.com/bib/article/26/3/bbaf293/8169298)

16. Koeppel et al. (2023). *Prediction of prime editing insertion efficiencies.* **Nature Biotechnology**. [Link](https://www.nature.com/articles/s41587-023-01678-y)

### PLM and RT engineering

17. Lin et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model (ESM2).* **Science**, 379(6637).

18. Sun et al. (2024). *Enhancing reverse transcriptase function in Taq polymerase via AI-driven protein design.* **Frontiers in Bioengineering and Biotechnology**. [Link](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2024.1495267/full)

19. Yang et al. (2025). *Prime editor with rational design and AI-driven optimization (rPE).* **Nature Communications**. [Link](https://www.nature.com/articles/s41467-025-60495-w)

20. Guo et al. (2025). *Integrating protein language models and automatic biofoundry for protein evolution.* **Nature Communications**. [Link](https://www.nature.com/articles/s41467-025-56751-8)

21. Schmirler et al. (2024). *Expert-guided protein language models for fitness prediction.* **Bioinformatics**. [Link](https://academic.oup.com/bioinformatics/article/40/11/btae621/7907184)

### Computational tools

22. van Kempen et al. (2024). *Fast and accurate protein structure search with Foldseek.* **Nature Biotechnology**.

23. Hsu et al. (2022). *Learning inverse folding from millions of predicted structures (ESM-IF).* **ICML**.

### GitHub resources

- [PRIDICT](https://github.com/uzh-dqbm-cmi/PRIDICT)
- [ePRIDICT](https://github.com/Schwank-Lab/epridict)
- [DTMP-Prime](https://github.com/alipanahiCRISPR/DTMP-Prime)
- [ESM](https://github.com/facebookresearch/esm)
