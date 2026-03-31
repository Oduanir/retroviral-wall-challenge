#!/usr/bin/env python3
"""
V7 pipeline: LoRA fine-tuned ESM2 + PDB structural features + per-fold PCA blend.

Target: CLS >= 0.75

Architecture:
  Model 1: ElasticNet on tabular features (35 original + ~23 PDB structural)
  Model 2: Ridge on PCA3 of LoRA-finetuned ESM2 L33 mid-region (per-fold)
  Model 3: Ridge on PCA3 of pretrained ESM2 L12 mid-region (per-fold)
  Model 4: Ridge on PCA3 of pretrained ESM2 L33 mid-region (per-fold)

Blend: nested LOFO with weight step 0.05
"""

import os, sys, warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from itertools import product as iprod
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import average_precision_score
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_cls
from src.bootstrap import bootstrap_cls, print_bootstrap_results


def load_and_prepare_data():
    """Load data and engineer all features."""
    DATA_DIR = PROJECT_ROOT / 'data' / 'raw'

    train = pd.read_csv(DATA_DIR / 'train.csv')
    splits_df = pd.read_csv(DATA_DIR / 'family_splits.csv')
    splits = {}
    for _, row in splits_df.iterrows():
        splits[row['family']] = row['rt_names'].split('|')

    gt_order = pd.read_csv(DATA_DIR / 'rt_sequences.csv', usecols=['rt_name'])['rt_name'].tolist()
    sequences = dict(zip(train['rt_name'], train['sequence']))

    # Missing flags
    for col in ['connection_mean_pot', 'triad_best_rmsd', 'D1_D2_dist', 'D2_D3_dist',
                'yxdd_hydrophobic_fraction', 'yxdd_mean_hydrophobicity', 'yxdd_5A_mean_pot']:
        train[f'{col}_missing'] = train[col].isna().astype(int)

    # Interactions
    train['foldseek_gap_MMLV'] = train['foldseek_best_TM'] - train['foldseek_TM_MMLV']
    train['t40_x_foldseek_MMLV'] = train['t40_raw'] * train['foldseek_TM_MMLV']
    train['triad_quality'] = train['triad_found_bin'] * (1 / (train['triad_best_rmsd'].fillna(99) + 1))
    train['seq_struct_compat'] = -train['perplexity'] * train['instability_index']

    # PDB structural features
    pdb_features_path = PROJECT_ROOT / 'data' / 'processed' / 'pdb_structural_features.csv'
    if pdb_features_path.exists():
        pdb_feats = pd.read_csv(pdb_features_path)
        train = train.merge(pdb_feats, on='rt_name', how='left')
        pdb_cols = [c for c in pdb_feats.columns if c != 'rt_name']
        print(f'  PDB features loaded: {len(pdb_cols)} features')
    else:
        pdb_cols = []
        print('  PDB features not found, extracting...')
        from src.pdb_features import extract_all_pdb_features
        pdb_feats = extract_all_pdb_features()
        train = train.merge(pdb_feats, on='rt_name', how='left')
        pdb_cols = [c for c in pdb_feats.columns if c != 'rt_name']
        print(f'  PDB features extracted: {len(pdb_cols)} features')

    return train, splits, gt_order, sequences, pdb_cols


def extract_esm2_embeddings(sequences):
    """Extract pretrained ESM2 embeddings (L12 + L33 mid-region)."""
    from transformers import EsmTokenizer, EsmModel, EsmForMaskedLM

    print('Loading ESM2-650M...')
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
    model = EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D')
    model.eval()

    l12_raw, l33_raw = {}, {}
    for rt, seq in sequences.items():
        inp = tokenizer(seq, return_tensors='pt', truncation=True, max_length=1024)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
            n3 = len(seq) // 3
            l12_raw[rt] = out.hidden_states[12][0, 1:-1, :][n3:2*n3].mean(0).numpy()
            l33_raw[rt] = out.hidden_states[33][0, 1:-1, :][n3:2*n3].mean(0).numpy()
    del model
    print('  Pretrained embeddings extracted.')

    # Zero-shot features
    model_mlm = EsmForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D')
    model_mlm.eval()
    zs_features = {}
    for rt, seq in sequences.items():
        inp = tokenizer(seq, return_tensors='pt', truncation=True, max_length=1024)
        with torch.no_grad():
            logits = model_mlm(**inp).logits
            probs = torch.softmax(logits[0, 1:-1, :], dim=-1)
            ids = inp['input_ids'][0, 1:-1]
            ll = torch.log(probs[range(len(ids)), ids])
            ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        zs_features[rt] = {
            'esm2_pseudo_ppl': torch.exp(-ll.mean()).item(),
            'esm2_mean_entropy': ent.mean().item(),
            'esm2_mean_ll': ll.mean().item(),
        }
    del model_mlm
    print('  Zero-shot features extracted.')

    return l12_raw, l33_raw, zs_features


def run_nested_lofo(train, splits, gt_order, l12_raw, l33_raw,
                    features_base, lora_embeddings=None,
                    ra12=12.5, ra33=7.5, weight_step=0.05):
    """
    Run nested LOFO with per-fold PCA and optional LoRA embeddings.
    """
    families = list(splits.keys())
    all_oof = []
    model_names = ['EN', 'L12', 'L33']
    if lora_embeddings is not None:
        model_names.append('LORA')

    def norm(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn) if mx - mn > 1e-12 else np.zeros_like(a)

    for outer_family in families:
        outer_rts = splits[outer_family]
        outer_mask = train['rt_name'].isin(outer_rts)
        inner_mask = ~outer_mask
        inner_splits = {f: rts for f, rts in splits.items() if f != outer_family}
        inner_df = train[inner_mask].reset_index(drop=True)

        # Per-fold PCA on pretrained embeddings
        inner_l12 = np.array([l12_raw[n] for n in inner_df['rt_name']])
        inner_l33 = np.array([l33_raw[n] for n in inner_df['rt_name']])
        pca_l12 = PCA(n_components=3).fit(inner_l12)
        pca_l33 = PCA(n_components=3).fit(inner_l33)
        inner_l12_pca = pca_l12.transform(inner_l12)
        inner_l33_pca = pca_l33.transform(inner_l33)
        outer_l12 = np.array([l12_raw[n] for n in train.loc[outer_mask, 'rt_name']])
        outer_l33 = np.array([l33_raw[n] for n in train.loc[outer_mask, 'rt_name']])
        outer_l12_pca = pca_l12.transform(outer_l12)
        outer_l33_pca = pca_l33.transform(outer_l33)

        # Per-fold PCA on LoRA embeddings (if available)
        if lora_embeddings is not None:
            lora_embs = lora_embeddings[outer_family]
            inner_lora = np.array([lora_embs[n] for n in inner_df['rt_name']])
            pca_lora = PCA(n_components=3).fit(inner_lora)
            inner_lora_pca = pca_lora.transform(inner_lora)
            outer_lora = np.array([lora_embs[n] for n in train.loc[outer_mask, 'rt_name']])
            outer_lora_pca = pca_lora.transform(outer_lora)

        # Inner LOFO
        n_inner = len(inner_df)
        inner_oof = {m: np.full(n_inner, np.nan) for m in model_names}

        for ifam, irts in inner_splits.items():
            tm = inner_df['rt_name'].isin(irts)
            trm = ~tm
            y = inner_df.loc[trm, 'pe_efficiency_pct'].values

            inner_oof['EN'][tm.values] = make_pipeline(
                SimpleImputer(strategy='median'), StandardScaler(),
                ElasticNet(alpha=1.0, l1_ratio=0.3, max_iter=10000)
            ).fit(inner_df.loc[trm, features_base].values, y).predict(
                inner_df.loc[tm, features_base].values)

            inner_oof['L12'][tm.values] = make_pipeline(
                StandardScaler(), Ridge(alpha=ra12)
            ).fit(inner_l12_pca[trm.values], y).predict(inner_l12_pca[tm.values])

            inner_oof['L33'][tm.values] = make_pipeline(
                StandardScaler(), Ridge(alpha=ra33)
            ).fit(inner_l33_pca[trm.values], y).predict(inner_l33_pca[tm.values])

            if lora_embeddings is not None:
                inner_oof['LORA'][tm.values] = make_pipeline(
                    StandardScaler(), Ridge(alpha=10.0)
                ).fit(inner_lora_pca[trm.values], y).predict(inner_lora_pca[tm.values])

        # Weight optimization
        n_models = len(model_names)
        best_cls, best_w = -1, tuple([1.0/n_models] * n_models)
        for weights in iprod(np.arange(0, 1.0 + weight_step/2, weight_step), repeat=n_models):
            if abs(sum(weights) - 1.0) > weight_step * 0.5:
                continue
            b = sum(weights[i] * norm(inner_oof[m]) for i, m in enumerate(model_names))
            y_true = inner_df['active'].values
            pe_eff = inner_df['pe_efficiency_pct'].values
            result = compute_cls(y_true, b, pe_eff)
            if result['cls'] > best_cls:
                best_cls = result['cls']
                best_w = weights

        # Predict outer fold
        y_inner = train.loc[inner_mask, 'pe_efficiency_pct'].values

        en_pred = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
            ElasticNet(alpha=1.0, l1_ratio=0.3, max_iter=10000)
        ).fit(train.loc[inner_mask, features_base].values, y_inner).predict(
            train.loc[outer_mask, features_base].values)

        l12_pred = make_pipeline(StandardScaler(), Ridge(alpha=ra12)
        ).fit(inner_l12_pca, y_inner).predict(outer_l12_pca)

        l33_pred = make_pipeline(StandardScaler(), Ridge(alpha=ra33)
        ).fit(inner_l33_pca, y_inner).predict(outer_l33_pca)

        preds = {'EN': en_pred, 'L12': l12_pred, 'L33': l33_pred}

        if lora_embeddings is not None:
            lora_pred = make_pipeline(StandardScaler(), Ridge(alpha=10.0)
            ).fit(inner_lora_pca, y_inner).predict(outer_lora_pca)
            preds['LORA'] = lora_pred

        # Normalize and blend
        blended = np.zeros(outer_mask.sum())
        for i, m in enumerate(model_names):
            inner_arr = inner_oof[m]
            mn, mx = inner_arr.min(), inner_arr.max()
            normed = (preds[m] - mn) / max(mx - mn, 1e-12)
            blended += best_w[i] * normed

        fold_df = train.loc[outer_mask, ['rt_name', 'active', 'pe_efficiency_pct', 'rt_family']].copy()
        fold_df['predicted_score'] = blended
        all_oof.append(fold_df)

        w_str = ' '.join(f'{m}={best_w[i]:.2f}' for i, m in enumerate(model_names))
        print(f'  {outer_family:<25s} w=[{w_str}]  inner_cls={best_cls:.4f}')

    oof = pd.concat(all_oof).set_index('rt_name').loc[gt_order].reset_index()
    result = compute_cls(oof['active'].values, oof['predicted_score'].values,
                         oof['pe_efficiency_pct'].values)
    return result, oof


def main():
    print('='*70)
    print('V7 PIPELINE: LoRA + PDB features + per-fold PCA')
    print('='*70)

    # 1. Load data
    print('\n[1] Loading data...')
    train, splits, gt_order, sequences, pdb_cols = load_and_prepare_data()

    # 2. Extract pretrained ESM2 embeddings
    print('\n[2] Extracting pretrained ESM2 embeddings...')
    l12_raw, l33_raw, zs_features = extract_esm2_embeddings(sequences)

    # Add zero-shot features to train
    for rt, feats in zs_features.items():
        for k, v in feats.items():
            train.loc[train['rt_name'] == rt, k] = v

    # 3. Feature lists
    FEATURES_ORIG = [c for c in [
        'foldseek_TM_MMLV', 'foldseek_TM_MMLVPE', 'foldseek_best_TM',
        'foldseek_best_LDDT', 'foldseek_best_fident', 'foldseek_TM_HIV1',
        'triad_found_bin', 'triad_best_rmsd', 'perplexity', 'log_likelihood',
        'D1_D2_dist', 'D2_D3_dist',
        't40_raw', 't45_raw', 't50_raw', 't55_raw', 't60_raw',
        'instability_index', 'gravy', 'camsol', 'net_charge',
    ] if c in train.columns] + [
        f'{c}_missing' for c in ['connection_mean_pot', 'triad_best_rmsd', 'D1_D2_dist',
        'D2_D3_dist', 'yxdd_hydrophobic_fraction', 'yxdd_mean_hydrophobicity', 'yxdd_5A_mean_pot']
    ] + ['foldseek_gap_MMLV', 't40_x_foldseek_MMLV', 'triad_quality', 'seq_struct_compat',
         'esm2_pseudo_ppl', 'esm2_mean_entropy', 'esm2_mean_ll']

    FEATURES_ENRICHED = FEATURES_ORIG + [c for c in pdb_cols if c in train.columns]

    # 4. Run experiments
    print(f'\n[3] Running experiments...')
    print(f'  Original features: {len(FEATURES_ORIG)}')
    print(f'  Enriched features: {len(FEATURES_ENRICHED)}')

    # Experiment A: baseline (original features, per-fold PCA R12.5/7.5)
    print(f'\n--- Experiment A: Baseline (original features) ---')
    result_a, oof_a = run_nested_lofo(train, splits, gt_order, l12_raw, l33_raw,
                                       FEATURES_ORIG)
    print(f'  CLS = {result_a["cls"]:.4f}  PR={result_a["pr_auc"]:.4f}  WSp={result_a["w_spearman"]:.4f}')

    # Experiment B: enriched features (original + PDB)
    if pdb_cols:
        print(f'\n--- Experiment B: Enriched features (+{len(pdb_cols)} PDB) ---')
        result_b, oof_b = run_nested_lofo(train, splits, gt_order, l12_raw, l33_raw,
                                           FEATURES_ENRICHED)
        print(f'  CLS = {result_b["cls"]:.4f}  PR={result_b["pr_auc"]:.4f}  WSp={result_b["w_spearman"]:.4f}')

    # Experiment C: LoRA fine-tuned embeddings
    try:
        from src.lora_finetune import lora_lofo_embeddings
        print(f'\n--- Experiment C: LoRA fine-tuned ESM2 ---')
        lora_embs = lora_lofo_embeddings(train, splits, sequences)

        # C1: original features + LoRA
        print(f'\n  C1: Original features + LoRA')
        result_c1, oof_c1 = run_nested_lofo(train, splits, gt_order, l12_raw, l33_raw,
                                              FEATURES_ORIG, lora_embeddings=lora_embs)
        print(f'  CLS = {result_c1["cls"]:.4f}  PR={result_c1["pr_auc"]:.4f}  WSp={result_c1["w_spearman"]:.4f}')

        # C2: enriched features + LoRA
        if pdb_cols:
            print(f'\n  C2: Enriched features + LoRA')
            result_c2, oof_c2 = run_nested_lofo(train, splits, gt_order, l12_raw, l33_raw,
                                                  FEATURES_ENRICHED, lora_embeddings=lora_embs)
            print(f'  CLS = {result_c2["cls"]:.4f}  PR={result_c2["pr_auc"]:.4f}  WSp={result_c2["w_spearman"]:.4f}')
    except Exception as e:
        print(f'\n  LoRA failed: {e}')
        result_c1 = result_c2 = None

    # Summary
    print(f'\n{"="*70}')
    print('SUMMARY')
    print('='*70)
    all_results = {'A_baseline': result_a}
    all_oofs = {'A_baseline': oof_a}
    if pdb_cols:
        all_results['B_pdb'] = result_b
        all_oofs['B_pdb'] = oof_b
    if result_c1:
        all_results['C1_lora'] = result_c1
        all_oofs['C1_lora'] = oof_c1
    if result_c2:
        all_results['C2_lora_pdb'] = result_c2
        all_oofs['C2_lora_pdb'] = oof_c2

    for name, res in sorted(all_results.items(), key=lambda x: -x[1]['cls']):
        print(f'  {name:<20s}  CLS={res["cls"]:.4f}  PR={res["pr_auc"]:.4f}  WSp={res["w_spearman"]:.4f}')

    # Save best
    best_name = max(all_results, key=lambda k: all_results[k]['cls'])
    best_oof = all_oofs[best_name]
    best_cls = all_results[best_name]['cls']

    print(f'\nBest: {best_name}  CLS={best_cls:.4f}')

    best_oof[['rt_name', 'predicted_score']].to_csv(
        PROJECT_ROOT / 'submissions' / 'submission_v7.csv', index=False)
    print(f'Submission saved.')

    # Bootstrap
    boot = bootstrap_cls(best_oof, n_bootstrap=10000)
    print()
    print_bootstrap_results(boot)


if __name__ == '__main__':
    main()
