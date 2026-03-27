# Retroviral Wall Challenge — Revue bibliographique

> **Objectif** : Prédire l'efficacité de reverse transcriptases (RT) diverses en tant que prime editors.
> **Date** : Mars 2026

---

## Table des matières

1. [Contexte : le prime editing](#1-contexte--le-prime-editing)
2. [Criblage de RT diverses pour le prime editing](#2-criblage-de-rt-diverses-pour-le-prime-editing)
3. [Propriétés structurales et biochimiques des RT](#3-propriétés-structurales-et-biochimiques-des-rt)
4. [Approches ML pour le prime editing](#4-approches-ml-pour-le-prime-editing)
5. [Modèles de langage protéique et ingénierie de RT](#5-modèles-de-langage-protéique-et-ingénierie-de-rt)
6. [Synthèse stratégique pour le challenge](#6-synthèse-stratégique-pour-le-challenge)
7. [Références](#7-références)

---

## 1. Contexte : le prime editing

Le prime editing est une technique d'édition du génome dérivée de CRISPR qui permet d'introduire des substitutions, insertions et délétions sans nécessiter de cassure double-brin ni de template donneur. Le système repose sur la fusion d'une nickase Cas9 (H840A) avec une **reverse transcriptase** (RT), guidée par un **pegRNA** (prime editing guide RNA) qui encode à la fois la séquence guide et le template d'édition.

La RT standard utilisée est celle du virus de la leucémie murine de Moloney (**MMLV-RT**), sous forme d'un pentamutant optimisé (D200N/L603W/T306K/W313F/T330P). Cependant, cette RT est volumineuse (~2 kb), ce qui limite le packaging en AAV pour les applications thérapeutiques in vivo.

Le cadre fondateur a été établi par **Anzalone et al. (Nature 2019)**, qui ont introduit les architectures **PE1, PE2 et PE3**. Ce papier est important pour le challenge car il rappelle deux points structurants : (i) l'éditeur de référence repose sur une **MMLV-RT tronquée du domaine RNase H** et (ii) l'efficacité du prime editing n'est pas une propriété intrinsèque de la RT seule, mais le résultat d'un compromis entre catalyse, processivité, compatibilité avec Cas9 et dynamique du pegRNA.

---

## 2. Criblage de RT diverses pour le prime editing

### 2.1 L'étude fondatrice PE6 — Liu et al., Cell 2023

C'est l'étude la plus directement liée au challenge. Le laboratoire Liu (Broad Institute) a criblé **59 reverse transcriptases de 14 classes phylogénétiques** comme prime editors :

- **Familles testées** : rétrovirus (MMLV, ASLV, PERV, etc.), rétrotransposons LTR (Tf1, Ty3, Gypsy), retrons bactériens (Ec48, Ec67), group II introns (LtrA, Er), CRISPR-associated, télomérases, et autres
- **Résultats clés** :
  - Différentes RT se spécialisent dans différents types d'éditions — certaines sont meilleures pour les insertions longues, d'autres pour les substitutions
  - Les RT rétrovirales montrent globalement la meilleure efficacité de prime editing
  - Certaines RT compactes (retrons, group II introns) montrent une activité détectable mais faible
- **Évolution dirigée (PACE)** : les RT compactes ont été améliorées jusqu'à **22x** via l'évolution assistée par phage
- **Variants PE6a-g** : 516-810 pb plus petits que PEmax, compatibles AAV
  - **PE6c** (RT évoluée de Tf1, rétrotransposon) : meilleure efficacité globale parmi les compacts
  - **PE6d** (MMLV évolué) : 2-3.5x plus efficace que PE3
- **In vivo** (cerveau de souris) : les systèmes PE6 en dual-AAV atteignent **12 à 183x** plus d'efficacité que les systèmes précédents

> **Pertinence directe** : Les 57 RT du challenge correspondent très probablement à un sous-ensemble des 59 RT testées dans cette étude. Les familles (Retroviral, Retron, LTR_Retrotransposon, Group_II_Intron, CRISPR-associated) sont identiques.

### 2.2 RT porcine (PERV) — pvPE, 2025

Une RT dérivée d'un rétrovirus endogène porcin (PERV, miniporc Bama) a été développée en prime editor :

- Après optimisation itérative (ingénierie RT, modifications structurales, fusion avec la protéine La), **pvPE-V4** atteint **24-102x** plus d'efficacité que la version initiale
- Jusqu'à **2.39x** plus efficace que PE7
- Significativement moins d'éditions non souhaitées

> **Pertinence** : Démontre que des RT non-MMLV peuvent être supérieures, et que les propriétés clés sont la processivité, la stabilité de l'hétéroduplex et l'interaction avec des protéines accessoires.

### 2.3 PE7 — Stabilisation par la protéine La, Nature 2024

- **La** est une protéine endogène eucaryote qui se lie et stabilise les extrémités 3' polyU des ARN
- Fusionnée au C-terminal de PEmax, elle produit **PE7**, améliorant significativement l'efficacité
- Le domaine N-terminal de La (aa 1-194) est suffisant

> **Pertinence** : La stabilisation du pegRNA par des protéines accessoires est un facteur critique, mais au-delà des propriétés intrinsèques de la RT (hors scope du challenge).

### 2.4 Ce que le criblage PE6 implique pour ce challenge

Au-delà du simple constat que certaines familles performent mieux que d'autres, l'étude PE6 suggère trois leçons directement utiles :

- **La compacité n'est pas synonyme d'inefficacité** : certaines RT plus courtes restent éditrices après optimisation, donc la longueur seule ne doit pas être utilisée comme proxy naïf de performance
- **La phylogénie explique une partie, pas tout** : au sein d'une même famille, l'activité varie fortement, ce qui justifie d'exploiter les features fines de motif, de structure et d'électrostatique
- **De petites modifications locales peuvent avoir de grands effets** : les gains obtenus par évolution dirigée indiquent que des signatures subtiles autour du site actif et des surfaces de liaison aux acides nucléiques peuvent suffire à faire basculer une RT d'inactive à utile

---

## 3. Propriétés structurales et biochimiques des RT

### 3.1 Structures cryo-EM du complexe prime editor — Nature 2024

Deux études structurales majeures ont résolu le complexe **SpCas9-MMLV RT(ΔRNaseH)-pegRNA-ADN cible** :

- La MMLV RT étend la transcription inverse au-delà du site prévu, causant des incorporations dérivées du scaffold (éditions indésirables)
- La RT reste dans la même position relative à SpCas9 pendant la transcription inverse
- L'hétéroduplex pegRNA/ADN synthétisé s'accumule le long de la surface de SpCas9

**Sous-domaines fonctionnels de la RT** (directement liés aux features du challenge) :
- **Fingers** : liaison au template et au dNTP entrant
- **Palm** : contient le site catalytique (motif YXDD) et les motifs 1-2
- **Thumb** : liaison à l'acide nucléique double-brin
- **Connection** : lien entre palm et domaine RNaseH

### 3.2 Contribution de la RT à la réparation de l'ADN — Nature Biotechnology 2025

- La RT des prime editors est recrutée rapidement aux sites de dommages ADN et contribue à la réparation endogène
- La surexpression de RT/PE augmente les insertions courtes et diminue le HDR
- Un PE compact sans domaine RNaseH montre moins d'activité de réparation — potentiellement plus adapté en clinique

### 3.3 Architecture de domaine et compromis compacité/processivité

La littérature suggère que l'**architecture globale de la RT** est presque aussi importante que le site catalytique lui-même :

- Le **prime editor original** repose sur une **MMLV-RT ΔRNaseH**, ce qui montre que le domaine RNase H n'est pas requis pour l'activité de prime editing de base
- Les travaux PE6 montrent qu'une RT plus compacte peut devenir très compétitive après optimisation, mais au prix possible d'un compromis sur la processivité, la stabilité du complexe ou la fenêtre d'édition
- Les structures cryo-EM indiquent que la géométrie relative entre **Cas9, pegRNA et RT** contraint fortement la catalyse, ce qui rend plausibles des effets majeurs de la **longueur**, de la **position des sous-domaines** et de la **surface d'interaction** avec l'hétéroduplex

> **Inférence pour le challenge** : si le jeu de données contient des proxies de compacité, de longueur ou d'annotation de domaines, ils doivent être traités comme des variables de premier ordre, et pas comme de simples covariables descriptives.

### 3.4 Features structurales pertinentes pour la prédiction

D'après la littérature, les propriétés suivantes corrèlent avec l'activité de PE :

| Propriété | Feature(s) du challenge | Rationnel |
|-----------|------------------------|-----------|
| Géométrie du site catalytique | `triad_found_bin`, `D1_D2_dist`, `D2_D3_dist`, `triad_best_rmsd` | La triade Asp catalytique doit adopter une géométrie précise |
| Motif YXDD | `yxdd_seq`, `yxdd_*` features | Cœur catalytique de la RT, YMDD (MMLV) est le gold standard |
| Similarité structurale avec MMLV | `foldseek_TM_MMLV`, `foldseek_TM_MMLVPE` | MMLV est la RT la plus efficace connue pour le PE |
| Thermostabilité | `t40_raw` à `t80_raw`, `thermophilicity_num` | Stabilité modérée nécessaire (37°C), hyperthermostabilité potentiellement délétère |
| Compatibilité séquence-structure | `perplexity`, `log_likelihood` (ESM-IF) | Bonne compatibilité = repliement correct |
| Qualité structurale | `rama_fav`, `rama_out` | Indicateurs de qualité de la structure prédite |
| Propriétés du site actif | `pocket_*` features | Accessibilité et environnement du site catalytique |
| Potentiel électrostatique | `*_mean_pot` features | Distribution de charges sur chaque sous-domaine |
| Beta-hairpin | `hairpin_pass`, `hairpin_confidence` | Élément structural conservé près du site actif |
| Taille / architecture de domaine | Proxies de longueur ou de compacité si disponibles | La littérature PE1/PE2/PE6 montre un compromis direct entre compacité, positionnement structural et performance |

---

## 4. Approches ML pour le prime editing

### 4.1 Kim et al. / modèle DeepPE — Nature Biotechnology 2021

Premier modèle largement utilisé pour la conception de pegRNAs :

- **Données** : 54 836 paires pegRNA-cible pour PE2
- **Papier** : *Predicting the efficiency of prime editing guide RNAs in human cells*
- **Nom du modèle** : **DeepPE** (mentionné par les auteurs dans la disponibilité du code)
- **Sortie** : prédiction de l'efficacité de pegRNAs selon PBS, RTT, type d'édition et position
- **Intérêt historique** : montre dès 2021 qu'un grand criblage systématique permet d'apprendre des déterminants robustes du prime editing

> **Note bibliographique** : l'article a été publié en ligne en **septembre 2020**, mais paraît dans **Nature Biotechnology 39 (2021)** ; le DOI `s41587-020-0677-y` est donc cohérent.

> **Limitation pour nous** : la cible reste le design du pegRNA et du site cible, pas la RT elle-même.

### 4.2 PRIDICT — Nature Biotechnology 2023

Le premier modèle de très haut niveau de performance pour le PE sur grands écrans :

- **Architecture** : Réseau neuronal récurrent bidirectionnel avec attention (AttnBiRNN)
- **Données** : 92 423 pegRNAs, 13 349 mutations pathogènes humaines
- **Performance** : Spearman R = 0.85 pour les éditions souhaitées
- **Apport important** : prédit aussi une partie des sous-produits, pas seulement l'efficacité brute

### 4.3 PRIDICT2.0 / ePRIDICT — Schwank Lab, Nature Biotechnology 2024

Le modèle de référence pour la prédiction d'efficacité PE :

- **Architecture** : Réseau neuronal récurrent bidirectionnel avec attention (AttnBiRNN)
- **Données** : >400 000 pegRNAs, 13 349 mutations pathogènes humaines
- **Performance** : Spearman ρ = 0.91 dans HEK293T, ρ = 0.81 dans K562
- **Features** : longueur RTT, contenu GC du PBS, features de séquence

**ePRIDICT** (version épigénétique) : XGBoost + 6 datasets ENCODE, évalue l'influence de la chromatine.

> **Limitation** : Prédit l'efficacité en fonction du **pegRNA et du site cible**, pas de la RT. Le challenge pose le problème inverse.

### 4.4 DeepPrime — Cell 2023

- **Données** : 338 996 paires pegRNA-cible, 3 979 epegRNAs
- **Portée** : plusieurs systèmes de prime editing et plusieurs types cellulaires
- **Intérêt** : montre que la généralisation inter-systèmes est possible quand l'espace d'entraînement est suffisamment large

> **Pertinence indirecte** : le challenge n'a pas assez d'exemples pour reproduire ce paradigme en deep learning, mais l'idée de transférer des représentations apprises plutôt que d'apprendre end-to-end reste très pertinente.

### 4.5 DTMP-Prime — Molecular Therapy Nucleic Acids, 2024

- **Architecture** : Transformeur bidirectionnel (BERT) avec attention multi-têtes
- **Innovation** : Encodage novel des paires pegRNA-ADN, intégration de DNABERT
- **43 features** extraites pour PE2 et PE3
- Performance supérieure à l'état de l'art

> **Pertinence** : L'utilisation de DNABERT pour l'embedding est transférable à l'embedding de séquences RT via ESM2.

### 4.6 PrimeNet — Briefings in Bioinformatics, 2025

- **Architecture** : Convolutions multi-échelles + attention
- **Innovation** : Intégration de facteurs épigénétiques
- **Performance** : Spearman ρ = 0.94 (meilleure publiée)

### 4.7 Bilan

Tous ces modèles prédisent l'efficacité **en fonction du pegRNA/site cible, pour une RT fixe (MMLV)**. **Aucun modèle publié ne prédit l'efficacité en fonction des propriétés intrinsèques de la RT** — c'est exactement le gap que ce challenge comble.

---

## 5. Modèles de langage protéique et ingénierie de RT

### 5.1 ESM2 — Meta/FAIR

- Modèle de langage protéique pré-entraîné sur des milliards de séquences
- Génère des embeddings riches capturant structure, fonction et évolution
- Utilisé en zero-shot pour prédire l'effet de mutations sur la fitness protéique
- Le challenge fournit des **embeddings ESM2 pré-calculés** (`esm2_embeddings.npz`)

### 5.2 AI-driven RT engineering pour Taq polymérase — Frontiers 2024

L'approche la plus directement transférable à notre problème :

- **Pipeline** : Embeddings PLM de variants de Taq polymérase → **Ridge regression** → prédiction d'activité RT
- Criblage in silico de **>18 millions de mutations potentielles**, réduites à 16 candidats
- 18 variants identifiés avec activité RTase significativement améliorée
- Validation en RT-qPCR temps réel

> **Pertinence majeure** : Pipeline directement applicable — embeddings ESM2 de la RT + modèle supervisé léger (Ridge, XGBoost) = prédiction d'activité RT.

### 5.3 rPE avec optimisation AI-driven — Nature Communications 2025

- Reverse prime editing (rPE) : variant SpCas9 permettant l'édition en direction 3' du nick HNH
- **La RT a été optimisée en utilisant des protein language models (type ESM2)** combinés avec La
- Variants **erPEmax** et **erPE7max** atteignant **44.41% d'efficacité**

> **Pertinence** : Démonstration directe que les PLMs peuvent guider l'ingénierie de la RT pour le prime editing.

### 5.4 AI-generated MLH1 small binder — Cell 2025

- Utilisation de **RFdiffusion et AlphaFold 3** pour concevoir un mini-binder (MLH1-SB, 82 aa) inhibant le mismatch repair
- Système PE7-SB2 : **18.8x** plus efficace que PEmax

---

## 6. Synthèse stratégique pour le challenge

### 6.1 Ce qui est unique dans ce challenge

- **Problème inverse** par rapport à la littérature : on prédit l'efficacité PE en fonction de la **RT**, pas du pegRNA
- **Très petit dataset** (57 échantillons, 21 actifs) → deep learning inadapté, focus sur modèles classiques
- **Features riches** (~65 features tabulaires + embeddings ESM2 + structures PDB)
- **Transductif** : les mêmes RT apparaissent dans train et test

### 6.2 Hypothèses issues de la littérature

1. **La similarité structurale avec MMLV est le meilleur prédicteur** — les features `foldseek_TM_MMLV` et `foldseek_TM_MMLVPE` devraient être fortement corrélées avec l'efficacité
2. **La famille phylogénétique est informative** — les RT rétrovirales sont systématiquement les plus actives
3. **La géométrie du site catalytique est nécessaire mais pas suffisante** — `triad_found_bin` et `triad_best_rmsd` sont des pré-requis
4. **Les embeddings ESM2 capturent des signaux complémentaires** — compatibilité séquence-structure et évolution
5. **Approche 2 étapes** (classif actif/inactif puis régression de l'efficacité) pourrait être pertinente vu la distribution bimodale de la cible

### 6.3 Approches de modélisation recommandées

| Approche | Justification bibliographique |
|----------|-------------------------------|
| Ridge regression sur embeddings ESM2 | Validée pour prédiction d'activité RTase (étude Taq, Frontiers 2024) |
| XGBoost/LightGBM sur features tabulaires | ePRIDICT montre l'efficacité de gradient boosting pour le PE |
| Sélection de features agressive | Ratio features/échantillons ~65/57, risque de surapprentissage élevé |
| Cross-validation Leave-One-Family-Out | `family_splits.csv` fourni, évite le data leakage entre familles proches |
| Ensembling (Ridge + XGBoost + SVR) | Approche standard pour petits datasets, robustesse accrue |

### 6.4 Discipline d'évaluation indispensable sur ce benchmark

Cette partie relève en partie d'une **inférence méthodologique** à partir du design du challenge plutôt que d'un papier unique, mais elle est importante :

- **Toute sélection de variables, PCA, tuning d'hyperparamètres et stacking doit être faite à l'intérieur de chaque fold** ; sinon, le risque de fuite d'information est très élevé
- **La famille doit être l'unité d'évaluation principale** : une bonne performance en split aléatoire aurait peu de valeur scientifique ici
- Le caractère **transductif** du benchmark autorise potentiellement certaines étapes **non supervisées** sur l'ensemble des séquences/structures disponibles, mais jamais l'utilisation des cibles de test pour calibrer un modèle
- Avec seulement 57 RT, il faut privilégier la **stabilité des classements**, les intervalles de confiance par bootstrap et l'analyse des erreurs par famille plutôt qu'un unique score agrégé

---

## 7. Références

### Prime editing — Fondamentaux et RT

1. Anzalone et al. (2019). *Search-and-replace genome editing without double-strand breaks or donor DNA.* **Nature**. [Lien](https://www.nature.com/articles/s41586-019-1711-4)

2. Liu et al. (2023). *Phage-assisted evolution and protein engineering yield compact, efficient prime editors.* **Cell**. [Lien](https://www.cell.com/cell/fulltext/S0092-8674(23)00854-1)

3. Chen & Liu (2023). *Prime editing for precise and highly versatile genome manipulation.* **Nature Reviews Genetics**, 24, 161-177.

4. Zhang et al. (2025). *Highly efficient prime editors based on porcine endogenous retrovirus RT.* **Trends in Biotechnology**. [Lien](https://www.cell.com/trends/biotechnology/fulltext/S0167-7799(25)00314-2)

5. Dolan et al. (2024). *Improving prime editing with an endogenous small RNA-binding protein (PE7).* **Nature**. [Lien](https://www.nature.com/articles/s41586-024-07259-6)

### Structures cryo-EM

6. Yan et al. (2024). *Structural basis for pegRNA-guided reverse transcription by a prime editor.* **Nature**. [Lien](https://www.nature.com/articles/s41586-024-07497-8)

7. Bock et al. (2024). *Cryo-EM structure of a prime editor complex in multiple states.* **Nature Biotechnology**. [Lien](https://www.nature.com/articles/s41587-024-02325-w)

### RT et réparation ADN

8. Dolan et al. (2025). *The reverse transcriptase domain of prime editors contributes to DNA repair.* **Nature Biotechnology**. [Lien](https://www.nature.com/articles/s41587-025-02568-1)

### ML pour le prime editing

9. Kim et al. (2021). *Predicting the efficiency of prime editing guide RNAs in human cells* (**modèle DeepPE**; publication en ligne en 2020). **Nature Biotechnology**. [Lien](https://www.nature.com/articles/s41587-020-0677-y)

10. Mathis et al. (2023). *Predicting prime editing efficiency and product purity by deep learning (PRIDICT).* **Nature Biotechnology**. [Lien](https://www.nature.com/articles/s41587-022-01613-7)

11. Mathis et al. (2024). *Machine learning prediction of prime editing efficiency across diverse chromatin contexts (PRIDICT2.0 / ePRIDICT).* **Nature Biotechnology**. [Lien](https://www.nature.com/articles/s41587-024-02268-2)

12. Mathis et al. (2025). *PRIDICT2.0 protocol.* **Nature Protocols**. [Lien](https://www.nature.com/articles/s41596-025-01244-7)

13. Yu et al. (2023). *Prediction of efficiencies for diverse prime editing systems in multiple cell types.* **Cell**. [Lien](https://www.sciencedirect.com/science/article/pii/S0092867423003318)

14. Panahi et al. (2024). *DTMP-Prime: Bidirectional transformer for prime editing efficiency prediction.* **Molecular Therapy Nucleic Acids**. [Lien](https://www.sciencedirect.com/science/article/pii/S2162253124002579)

15. PrimeNet (2025). *Multi-scale convolution and attention for prime editing prediction.* **Briefings in Bioinformatics**. [Lien](https://academic.oup.com/bib/article/26/3/bbaf293/8169298)

16. Koeppel et al. (2023). *Prediction of prime editing insertion efficiencies.* **Nature Biotechnology**. [Lien](https://www.nature.com/articles/s41587-023-01678-y)

### PLM et ingénierie de RT

17. Lin et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model (ESM2).* **Science**, 379(6637).

18. Sun et al. (2024). *Enhancing reverse transcriptase function in Taq polymerase via AI-driven protein design.* **Frontiers in Bioengineering and Biotechnology**. [Lien](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2024.1495267/full)

19. Yang et al. (2025). *Prime editor with rational design and AI-driven optimization (rPE).* **Nature Communications**. [Lien](https://www.nature.com/articles/s41467-025-60495-w)

20. Guo et al. (2025). *Integrating protein language models and automatic biofoundry for protein evolution.* **Nature Communications**. [Lien](https://www.nature.com/articles/s41467-025-56751-8)

21. Schmirler et al. (2024). *Expert-guided protein language models for fitness prediction.* **Bioinformatics**. [Lien](https://academic.oup.com/bioinformatics/article/40/11/btae621/7907184)

### Outils computationnels

22. van Kempen et al. (2024). *Fast and accurate protein structure search with Foldseek.* **Nature Biotechnology**.

23. Hsu et al. (2022). *Learning inverse folding from millions of predicted structures (ESM-IF).* **ICML**.

### Ressources GitHub

- [PRIDICT](https://github.com/uzh-dqbm-cmi/PRIDICT)
- [ePRIDICT](https://github.com/Schwank-Lab/epridict)
- [DTMP-Prime](https://github.com/alipanahiCRISPR/DTMP-Prime)
- [ESM](https://github.com/facebookresearch/esm)
