# Status update — Mars 2026

## Score final

**CLS 0.6936** (nested LOFO CV, vérifié via évaluateur officiel sur `submissions/submission_layer_sweep.csv`)

| Composante | Score |
|------------|-------|
| PR-AUC | 0.6930 |
| W-Spearman | 0.6942 |
| CLS | 0.6936 |

Meilleur modèle : blend 3 modèles (EN enriched+ZS + ESM2 **layer 12** mid PCA3 + ESM2 **layer 33** mid PCA3), poids nested LOFO.

---

## Architecture du meilleur modèle (v4)

**Modèle 1 — ElasticNet (tabulaire)** :
- ElasticNet(alpha=1.0, l1_ratio=0.3)
- 35 features : P123 (21) + missing flags (7) + interactions (4) + zero-shot ESM2 (3)
- Force : PR-AUC (classification)

**Modèle 2 — Ridge sur ESM2 mid-region** :
- Ridge(alpha=10.0) sur PCA3 du tiers central (palm domain) des embeddings ESM2-650M
- Force : W-Spearman (ranking)

**Blend** : poids optimisés par nested LOFO (inner LOFO sur 6 familles, grille 0.1).

---

## Progression du CLS

| Version | Modèles | CLS | Phase |
|---------|---------|-----|-------|
| Réf. GitHub | HandCrafted + RF | 0.318 | — |
| v1 | EN + Hurdle + ESM2 global | 0.6525 | 3 |
| v2 | EN enriched + Hurdle + ESM2 global | 0.6685 | 4 |
| v4 | EN enriched+ZS + ESM2 L33 mid | 0.6882 | 5 |
| **v5** | **EN enriched+ZS + ESM2 L12 mid + ESM2 L33 mid** | **0.6936** | **11** |

---

## Détail par famille (v5)

| Famille | n | Actives | PR-AUC |
|---------|---|---------|--------|
| Group_II_Intron | 5 | 2 | **1.000** |
| LTR_Retrotransposon | 11 | 2 | **1.000** |
| Retroviral | 18 | 12 | 0.847 |
| **Retron** | **12** | **5** | **0.423** |
| CRISPR-associated | 5 | 0 | N/A |
| Other | 5 | 0 | N/A |
| Unclassified | 1 | 0 | N/A |

Bottleneck : Retron (PR-AUC 0.39).

---

## Toutes les phases du projet

### Phase 0 — Revue bibliographique
- 23 références, papier source identifié (Doman et al., Cell 2023)

### Phase 1 — Protocole + EDA
- Métrique CLS implémentée (code officiel Mandrake)
- LOFO CV opérationnelle
- 64 features numériques, top corrélées : FoldSeek (similarité structurale)

### Phase 2 — Modèles tabulaires (43 modèles testés)
- ElasticNet(a=0.1, l1=0.5) sur 21 features : CLS 0.596

### Phase 3 — ESM2
- ESM2 PCA3 seul : W-Spearman 0.692, PR-AUC 0.418
- Late fusion EN + Hurdle + ESM2 : CLS 0.6525 (nested)

### Phase 4 — Optimisation
- Missing flags + interactions : +0.016 CLS
- Features PDB, blend multiplicatif, graph smoothing : aucun gain

### Phase 5 — PLM avancés
- Zero-shot ESM2 (pseudo-perplexité, entropie) : boost PR-AUC
- Mid-region embedding (palm domain) : meilleur W-Spearman
- SaProt, Ankh : aucun gain
- CLS 0.6882 atteint (amélioré en v5 à 0.6936 via layer sweep)

### Phase 6 — Données externes (plan V5)
- 486 RT Doman : features de voisinage encodent la phylogénie
- MSA / conservation : redondant avec FoldSeek
- Features retron-spécifiques : non transférables en LOFO
- Attention-weighted pooling : pire que mid-region
- Ankh : family memorization totale

### Phase 7 — Anti-phylogénie (plan V6 focalisé)
- Pruning features pro-famille : détruit le score
- Résidualisation : dégrade le CLS
- Target transformation / multi-task : PR-AUC 0.71 mais CLS max 0.664
- Adversarial / projection orthogonale : max CLS 0.648

### Phase 8 — Roadmap V7 (15 runs)
- CORAL, MMD : sous-performance PyTorch vs sklearn
- GroupDRO/VREx ElasticNet : CLS 0.668 (identique au baseline)
- Pairwise ranking, diff Spearman, multi-task complet, contrastive : tous inférieurs
- Patches locaux YXDD : PR-AUC record 0.762 mais CLS 0.657
- Bio prior : CLS 0.608 sans entraînement

### Phase 9 — Données externes v2 (plan exploitation)
- Retron transferability (179 census + 105 discovery) : corrélations négatives, CLS 0.539
- RT externe reformulé (486 RT, ESM2 embeddings) : CLS 0.541

### Phase 10 — Expériences finales
- Résidus alignés + delta vs RT de référence : CLS 0.653
- Episodic training : CLS 0.559
- Hard-case weighting : W-Spearman record 0.710 mais CLS 0.647
- Test Hsu-style local-only : CLS 0.459 — le signal local ne tient pas seul

### Phase 11 — Layer sweep + GP + ESM-IF local
- **ESM2 layer sweep** : layer 12 mid complémentaire à layer 33 mid → **CLS 0.6936** (nouveau record)
- GP : W-Spearman 0.000 partout → échec
- ESM-IF local (per-position LL autour YXDD) : ρ=0.02 → aucun signal local, dégrade le CLS

### Phase 12 — Plan V12
- Stacking nested : CLS 0.000 (méta-learner surfit le bruit des OOF internes)
- Features biochimiques par domaine (55 features ancrées YXDD) : CLS 0.677 (redondant)
- ProtT5 mid-region : CLS 0.000 (family memorization, comme Ankh)

### Phase 13 — V13 Shortlist
- KernelRidge MKL : W-Spearman record 0.725 mais PR-AUC 0.631, CLS 0.674
- Domaines par structure : redondant, CLS 0.578
- Episodic V11 : CLS 0.591
- Adaptation mid PCA5 : CLS 0.571
- Blend KernelRidge + EN + L12 + L33 : W-Spearman record 0.725 en solo mais dégrade le blend (CLS 0.643)
- LoRA minimal : non testé (nécessite GPU)

### Faisabilité SRA évaluée
- BioProject PRJNA916060 : 216 runs Figure 1C, 1.2 Go
- Seulement 21 RT actives dans le SRA (les inactives non séquencées)
- N'adresse que le ranking, pas la classification → No-Go

---

## Diagnostic final

Le signal PE dans ce dataset de 57 RT est **confounded avec la phylogénie** de manière qui résiste à toutes les approches testées (~60 expériences sur 13 phases).

- Les features globales (FoldSeek, thermostabilité, ESM2) **sont** le signal
- Le signal local (site actif seul) ne suffit pas (CLS 0.46 vs 0.59 global)
- Les données externes n'apportent pas de supervision PE exploitable
- Les méthodes anti-phylogénie (résidualisation, adversarial, GroupDRO, CORAL) ne cassent pas le mur
- Le CLS 0.6936 est le meilleur score atteint en LOFO honnête avec les données du challenge

---

## Pistes non tentées

- **Fine-tuning LoRA** d'ESM2 (nécessite GPU)
- **Reprocessing SRA PRJNA916060** (faisable mais n'adresse que le ranking des 21 actives, pas la classification)
