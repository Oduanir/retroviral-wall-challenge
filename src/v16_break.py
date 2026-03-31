#!/usr/bin/env python3
"""
Plan V16 — Break 0.7088
Step 1: Oracle diagnostics
Step 2: Classification corrector on top of v6
"""
import numpy as np, pandas as pd, torch, warnings, os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression, RidgeClassifier
from sklearn.metrics import average_precision_score
from itertools import product as iprod
from pathlib import Path

ROOT = Path(__file__).parent.parent
import sys; sys.path.insert(0, str(ROOT))
from src.metrics import compute_cls
from src.bootstrap import bootstrap_cls, print_bootstrap_results


# ===========================================================================
# DATA + ESM2
# ===========================================================================

def load_all():
    train = pd.read_csv(ROOT / 'data/raw/train.csv')
    splits_df = pd.read_csv(ROOT / 'data/raw/family_splits.csv')
    splits = {}
    for _, row in splits_df.iterrows():
        splits[row['family']] = row['rt_names'].split('|')
    gt_order = pd.read_csv(ROOT / 'data/raw/rt_sequences.csv', usecols=['rt_name'])['rt_name'].tolist()
    sequences = dict(zip(train['rt_name'], train['sequence']))

    for col in ['connection_mean_pot','triad_best_rmsd','D1_D2_dist','D2_D3_dist',
                'yxdd_hydrophobic_fraction','yxdd_mean_hydrophobicity','yxdd_5A_mean_pot']:
        train[f'{col}_missing'] = train[col].isna().astype(int)
    train['foldseek_gap_MMLV'] = train['foldseek_best_TM'] - train['foldseek_TM_MMLV']
    train['t40_x_foldseek_MMLV'] = train['t40_raw'] * train['foldseek_TM_MMLV']
    train['triad_quality'] = train['triad_found_bin'] * (1 / (train['triad_best_rmsd'].fillna(99) + 1))
    train['seq_struct_compat'] = -train['perplexity'] * train['instability_index']
    return train, splits, gt_order, sequences


def extract_esm2(sequences, train):
    from transformers import EsmTokenizer, EsmModel, EsmForMaskedLM
    print('[ESM2] Loading...')
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
    model = EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D'); model.eval()
    l12_raw, l33_raw = {}, {}
    for rt, seq in sequences.items():
        inp = tokenizer(seq, return_tensors='pt', truncation=True, max_length=1024)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
            n3 = len(seq)//3
            l12_raw[rt] = out.hidden_states[12][0,1:-1,:][n3:2*n3].mean(0).numpy()
            l33_raw[rt] = out.hidden_states[33][0,1:-1,:][n3:2*n3].mean(0).numpy()
    del model

    model_mlm = EsmForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D'); model_mlm.eval()
    for rt, seq in sequences.items():
        inp = tokenizer(seq, return_tensors='pt', truncation=True, max_length=1024)
        with torch.no_grad():
            logits = model_mlm(**inp).logits
            probs = torch.softmax(logits[0,1:-1,:], dim=-1)
            ids = inp['input_ids'][0,1:-1]
            ll = torch.log(probs[range(len(ids)), ids])
            ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        train.loc[train['rt_name']==rt, 'esm2_pseudo_ppl'] = torch.exp(-ll.mean()).item()
        train.loc[train['rt_name']==rt, 'esm2_mean_entropy'] = ent.mean().item()
        train.loc[train['rt_name']==rt, 'esm2_mean_ll'] = ll.mean().item()
    del model_mlm
    print('[ESM2] Done.')
    return l12_raw, l33_raw


FEATURES = None  # set after ESM2 extraction


def get_features(train):
    return [c for c in [
        'foldseek_TM_MMLV','foldseek_TM_MMLVPE','foldseek_best_TM',
        'foldseek_best_LDDT','foldseek_best_fident','foldseek_TM_HIV1',
        'triad_found_bin','triad_best_rmsd','perplexity','log_likelihood',
        'D1_D2_dist','D2_D3_dist','t40_raw','t45_raw','t50_raw','t55_raw','t60_raw',
        'instability_index','gravy','camsol','net_charge',
    ] if c in train.columns] + [
        f'{c}_missing' for c in ['connection_mean_pot','triad_best_rmsd','D1_D2_dist',
        'D2_D3_dist','yxdd_hydrophobic_fraction','yxdd_mean_hydrophobicity','yxdd_5A_mean_pot']
    ] + ['foldseek_gap_MMLV','t40_x_foldseek_MMLV','triad_quality','seq_struct_compat',
         'esm2_pseudo_ppl','esm2_mean_entropy','esm2_mean_ll']


def norm(a):
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn) if mx - mn > 1e-12 else np.zeros_like(a)


# ===========================================================================
# V6 PIPELINE — returns per-model OOF scores + blended score
# ===========================================================================

def run_v6_full(train, splits, gt_order, l12_raw, l33_raw, FEATURES,
                ra12=12.5, ra33=7.5):
    """Run v6 pipeline, return per-model and blended OOF predictions."""
    all_oof = []

    for outer_family in splits:
        outer_rts = splits[outer_family]
        om = train['rt_name'].isin(outer_rts); im = ~om
        isp = {f: rts for f, rts in splits.items() if f != outer_family}
        idf = train[im].reset_index(drop=True)

        il12 = np.array([l12_raw[n] for n in idf['rt_name']])
        il33 = np.array([l33_raw[n] for n in idf['rt_name']])
        p12 = PCA(3).fit(il12); p33 = PCA(3).fit(il33)
        il12p = p12.transform(il12); il33p = p33.transform(il33)
        ol12p = p12.transform(np.array([l12_raw[n] for n in train.loc[om,'rt_name']]))
        ol33p = p33.transform(np.array([l33_raw[n] for n in train.loc[om,'rt_name']]))

        n = len(idf)
        ioof = {'EN': np.full(n, np.nan), 'L12': np.full(n, np.nan), 'L33': np.full(n, np.nan)}

        for ifam, irts in isp.items():
            tm = idf['rt_name'].isin(irts); trm = ~tm
            y = idf.loc[trm, 'pe_efficiency_pct'].values
            ioof['EN'][tm.values] = make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                ElasticNet(alpha=1.0,l1_ratio=0.3,max_iter=10000)).fit(
                idf.loc[trm,FEATURES].values,y).predict(idf.loc[tm,FEATURES].values)
            ioof['L12'][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=ra12)).fit(
                il12p[trm.values],y).predict(il12p[tm.values])
            ioof['L33'][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=ra33)).fit(
                il33p[trm.values],y).predict(il33p[tm.values])

        # Weight optimization
        best_c, best_w = -1, (0.33,0.33,0.34)
        for w in iprod(np.arange(0,1.05,0.05), repeat=3):
            if abs(sum(w)-1.0) > 0.025: continue
            b = w[0]*norm(ioof['EN']) + w[1]*norm(ioof['L12']) + w[2]*norm(ioof['L33'])
            cc = compute_cls(idf['active'].values, b, idf['pe_efficiency_pct'].values)['cls']
            if cc > best_c: best_c, best_w = cc, w

        # Outer predictions
        yi = train.loc[im,'pe_efficiency_pct'].values
        pr = {}
        pr['EN'] = make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
            ElasticNet(alpha=1.0,l1_ratio=0.3,max_iter=10000)).fit(
            train.loc[im,FEATURES].values,yi).predict(train.loc[om,FEATURES].values)
        pr['L12'] = make_pipeline(StandardScaler(),Ridge(alpha=ra12)).fit(il12p,yi).predict(ol12p)
        pr['L33'] = make_pipeline(StandardScaler(),Ridge(alpha=ra33)).fit(il33p,yi).predict(ol33p)

        # Normalize and blend
        normed = {}
        blended = np.zeros(om.sum())
        for i, m in enumerate(['EN','L12','L33']):
            mn, mx = ioof[m].min(), ioof[m].max()
            normed[m] = (pr[m]-mn)/max(mx-mn,1e-12)
            blended += best_w[i] * normed[m]

        fd = train.loc[om,['rt_name','active','pe_efficiency_pct','rt_family']].copy()
        fd['predicted_score'] = blended
        fd['score_EN'] = normed['EN']
        fd['score_L12'] = normed['L12']
        fd['score_L33'] = normed['L33']
        fd['w_EN'] = best_w[0]
        fd['w_L12'] = best_w[1]
        fd['w_L33'] = best_w[2]
        all_oof.append(fd)

    oof = pd.concat(all_oof).set_index('rt_name').loc[gt_order].reset_index()
    r = compute_cls(oof['active'].values, oof['predicted_score'].values, oof['pe_efficiency_pct'].values)
    return r, oof


# ===========================================================================
# STEP 1 — ORACLE DIAGNOSTICS
# ===========================================================================

def step1_oracles(oof):
    print('\n' + '='*70)
    print('STEP 1 — ORACLE DIAGNOSTICS')
    print('='*70)

    y_true = oof['active'].values
    y_score = oof['predicted_score'].values
    pe_eff = oof['pe_efficiency_pct'].values
    baseline = compute_cls(y_true, y_score, pe_eff)
    print(f'\nBaseline v6: CLS={baseline["cls"]:.4f}  PR={baseline["pr_auc"]:.4f}  WSp={baseline["w_spearman"]:.4f}')

    # --- Exp 1.1: Oracle Classification ---
    # Give a big boost to all actives and penalty to inactives, preserving relative order within each group
    score_oracle_cls = y_score.copy()
    active_mask = y_true == 1
    inactive_mask = y_true == 0
    # Shift all actives above all inactives while preserving internal ranking
    if active_mask.any() and inactive_mask.any():
        max_inactive = score_oracle_cls[inactive_mask].max()
        min_active = score_oracle_cls[active_mask].min()
        if min_active <= max_inactive:
            gap = max_inactive - min_active + 0.1
            score_oracle_cls[active_mask] += gap

    r11 = compute_cls(y_true, score_oracle_cls, pe_eff)
    print(f'\nExp 1.1 — Oracle Classification (perfect active/inactive separation, same within-group ranking):')
    print(f'  CLS={r11["cls"]:.4f}  PR={r11["pr_auc"]:.4f}  WSp={r11["w_spearman"]:.4f}  (delta CLS: {r11["cls"]-baseline["cls"]:+.4f})')

    # --- Exp 1.2: Oracle Retron ---
    score_oracle_retron = y_score.copy()
    retron_mask = oof['rt_family'] == 'Retron'
    # For Retron: set scores to ground truth efficiency (perfect ranking + classification)
    score_oracle_retron[retron_mask.values] = pe_eff[retron_mask.values] + (y_true[retron_mask.values] * 50)

    r12 = compute_cls(y_true, score_oracle_retron, pe_eff)
    print(f'\nExp 1.2 — Oracle Retron (perfect Retron predictions, rest unchanged):')
    print(f'  CLS={r12["cls"]:.4f}  PR={r12["pr_auc"]:.4f}  WSp={r12["w_spearman"]:.4f}  (delta CLS: {r12["cls"]-baseline["cls"]:+.4f})')

    # --- Exp 1.3: Oracle Per-Family Best Base Model ---
    # For each family, pick the single model (EN/L12/L33) that gives best CLS on that family
    score_oracle_model = y_score.copy()
    total_cls_gain = 0
    for fam in sorted(oof['rt_family'].unique()):
        fam_mask = oof['rt_family'] == fam
        fam_oof = oof[fam_mask]
        if fam_oof['active'].sum() == 0 or fam_oof['active'].sum() == len(fam_oof):
            continue
        best_model_cls = -1
        best_model = None
        for m in ['score_EN','score_L12','score_L33']:
            # Substitute this model's score for this family
            test_score = y_score.copy()
            test_score[fam_mask.values] = fam_oof[m].values
            test_r = compute_cls(y_true, test_score, pe_eff)
            if test_r['cls'] > best_model_cls:
                best_model_cls = test_r['cls']
                best_model = m
        score_oracle_model[fam_mask.values] = fam_oof[best_model].values
        print(f'  {fam:<25s} best={best_model:<10s} family-oracle CLS={best_model_cls:.4f}')

    r13 = compute_cls(y_true, score_oracle_model, pe_eff)
    print(f'\nExp 1.3 — Oracle Per-Family Best Model:')
    print(f'  CLS={r13["cls"]:.4f}  PR={r13["pr_auc"]:.4f}  WSp={r13["w_spearman"]:.4f}  (delta CLS: {r13["cls"]-baseline["cls"]:+.4f})')

    # --- Go/No-Go ---
    max_oracle_gain = max(r11['cls'], r12['cls'], r13['cls']) - baseline['cls']
    print(f'\n--- GO/NO-GO ---')
    print(f'Max oracle headroom: +{max_oracle_gain:.4f}')
    if max_oracle_gain >= 0.01:
        print('=> GO: meaningful headroom exists')
        go = True
    else:
        print('=> NO-GO: oracles show negligible margin')
        go = False

    return go, baseline, {'oracle_cls': r11, 'oracle_retron': r12, 'oracle_model': r13}


# ===========================================================================
# STEP 2 — CLASSIFICATION CORRECTOR
# ===========================================================================

def step2_corrector(train, splits, gt_order, l12_raw, l33_raw, FEATURES,
                    ra12=12.5, ra33=7.5):
    """
    Run v6 with a classification corrector on top.
    The corrector is trained in the nested inner loop.
    """
    print('\n' + '='*70)
    print('STEP 2 — CLASSIFICATION CORRECTOR')
    print('='*70)

    results = {}

    for lambda_val in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        for corrector_type in ['logistic', 'ridge_cls']:
            all_oof = []

            for outer_family in splits:
                outer_rts = splits[outer_family]
                om = train['rt_name'].isin(outer_rts); im = ~om
                isp = {f: rts for f, rts in splits.items() if f != outer_family}
                idf = train[im].reset_index(drop=True)

                il12 = np.array([l12_raw[n] for n in idf['rt_name']])
                il33 = np.array([l33_raw[n] for n in idf['rt_name']])
                p12 = PCA(3).fit(il12); p33 = PCA(3).fit(il33)
                il12p = p12.transform(il12); il33p = p33.transform(il33)
                ol12p = p12.transform(np.array([l12_raw[n] for n in train.loc[om,'rt_name']]))
                ol33p = p33.transform(np.array([l33_raw[n] for n in train.loc[om,'rt_name']]))

                n = len(idf)
                ioof = {'EN': np.full(n, np.nan), 'L12': np.full(n, np.nan), 'L33': np.full(n, np.nan)}

                for ifam, irts in isp.items():
                    tm = idf['rt_name'].isin(irts); trm = ~tm
                    y = idf.loc[trm, 'pe_efficiency_pct'].values
                    ioof['EN'][tm.values] = make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                        ElasticNet(alpha=1.0,l1_ratio=0.3,max_iter=10000)).fit(
                        idf.loc[trm,FEATURES].values,y).predict(idf.loc[tm,FEATURES].values)
                    ioof['L12'][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=ra12)).fit(
                        il12p[trm.values],y).predict(il12p[tm.values])
                    ioof['L33'][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=ra33)).fit(
                        il33p[trm.values],y).predict(il33p[tm.values])

                # Weight optimization (same as v6)
                best_c, best_w = -1, (0.33,0.33,0.34)
                for w in iprod(np.arange(0,1.05,0.05), repeat=3):
                    if abs(sum(w)-1.0) > 0.025: continue
                    b = w[0]*norm(ioof['EN']) + w[1]*norm(ioof['L12']) + w[2]*norm(ioof['L33'])
                    cc = compute_cls(idf['active'].values, b, idf['pe_efficiency_pct'].values)['cls']
                    if cc > best_c: best_c, best_w = cc, w

                # Build inner blended score
                inner_blend = best_w[0]*norm(ioof['EN']) + best_w[1]*norm(ioof['L12']) + best_w[2]*norm(ioof['L33'])

                # Build corrector features on inner data
                inner_en_n = norm(ioof['EN'])
                inner_l12_n = norm(ioof['L12'])
                inner_l33_n = norm(ioof['L33'])

                corr_features_inner = np.column_stack([
                    inner_blend,
                    inner_en_n,
                    inner_l12_n,
                    inner_l33_n,
                    inner_en_n - inner_l12_n,   # disagreement EN vs L12
                    inner_en_n - inner_l33_n,   # disagreement EN vs L33
                    inner_l12_n - inner_l33_n,  # disagreement L12 vs L33
                    np.abs(inner_blend - 0.5),  # distance to decision boundary
                ])

                # Train corrector on inner data (predicting active)
                inner_active = idf['active'].values
                if corrector_type == 'logistic':
                    corr_model = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.1, max_iter=10000))
                else:
                    corr_model = make_pipeline(StandardScaler(),
                        RidgeClassifier(alpha=10.0))

                corr_model.fit(corr_features_inner, inner_active)

                # Outer predictions (v6 backbone)
                yi = train.loc[im,'pe_efficiency_pct'].values
                pr = {}
                pr['EN'] = make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                    ElasticNet(alpha=1.0,l1_ratio=0.3,max_iter=10000)).fit(
                    train.loc[im,FEATURES].values,yi).predict(train.loc[om,FEATURES].values)
                pr['L12'] = make_pipeline(StandardScaler(),Ridge(alpha=ra12)).fit(il12p,yi).predict(ol12p)
                pr['L33'] = make_pipeline(StandardScaler(),Ridge(alpha=ra33)).fit(il33p,yi).predict(ol33p)

                outer_en_n = (pr['EN']-ioof['EN'].min())/max(ioof['EN'].max()-ioof['EN'].min(),1e-12)
                outer_l12_n = (pr['L12']-ioof['L12'].min())/max(ioof['L12'].max()-ioof['L12'].min(),1e-12)
                outer_l33_n = (pr['L33']-ioof['L33'].min())/max(ioof['L33'].max()-ioof['L33'].min(),1e-12)
                outer_blend = best_w[0]*outer_en_n + best_w[1]*outer_l12_n + best_w[2]*outer_l33_n

                # Corrector features for outer fold
                corr_features_outer = np.column_stack([
                    outer_blend,
                    outer_en_n,
                    outer_l12_n,
                    outer_l33_n,
                    outer_en_n - outer_l12_n,
                    outer_en_n - outer_l33_n,
                    outer_l12_n - outer_l33_n,
                    np.abs(outer_blend - 0.5),
                ])

                # Get p_active
                if corrector_type == 'logistic':
                    p_active = corr_model.predict_proba(corr_features_outer)[:, 1]
                else:
                    p_active = corr_model.decision_function(corr_features_outer)
                    p_active = (p_active - p_active.min()) / max(p_active.max() - p_active.min(), 1e-12)

                # Adjusted score: score_new = score_v6 + lambda * (p_active - 0.5)
                adjusted = outer_blend + lambda_val * (p_active - 0.5)

                fd = train.loc[om,['rt_name','active','pe_efficiency_pct','rt_family']].copy()
                fd['predicted_score'] = adjusted
                all_oof.append(fd)

            oof = pd.concat(all_oof).set_index('rt_name').loc[gt_order].reset_index()
            r = compute_cls(oof['active'].values, oof['predicted_score'].values, oof['pe_efficiency_pct'].values)

            key = f'{corrector_type}_lambda{lambda_val}'
            results[key] = (r, oof)

            if lambda_val == 0.0 and corrector_type == 'logistic':
                print(f'  Baseline (lambda=0):  CLS={r["cls"]:.4f}  PR={r["pr_auc"]:.4f}  WSp={r["w_spearman"]:.4f}')
            elif r['cls'] > 0.7088:
                print(f'  {key:<30s} CLS={r["cls"]:.4f}  PR={r["pr_auc"]:.4f}  WSp={r["w_spearman"]:.4f}  ***')
            else:
                print(f'  {key:<30s} CLS={r["cls"]:.4f}  PR={r["pr_auc"]:.4f}  WSp={r["w_spearman"]:.4f}')

    # Summary
    print(f'\n  {"Config":<30s} {"CLS":>6s} {"PR-AUC":>8s} {"W-Sp":>8s}')
    print(f'  {"-"*55}')
    for key, (r, _) in sorted(results.items(), key=lambda x: -x[1][0]['cls']):
        tag = ' ***' if r['cls'] > 0.7088 else ''
        print(f'  {key:<30s} {r["cls"]:6.4f} {r["pr_auc"]:8.4f} {r["w_spearman"]:8.4f}{tag}')

    best_key = max(results, key=lambda k: results[k][0]['cls'])
    best_r, best_oof = results[best_key]
    print(f'\n  Best: {best_key}  CLS={best_r["cls"]:.4f}')

    return best_r, best_oof, results


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print('='*70)
    print('PLAN V16 — BREAK 0.7088')
    print('='*70)

    train, splits, gt_order, sequences = load_all()
    l12_raw, l33_raw = extract_esm2(sequences, train)
    FEATURES = get_features(train)

    # --- STEP 1: Oracle diagnostics ---
    print('\nRunning v6 baseline with full per-model scores...')
    baseline_r, baseline_oof = run_v6_full(train, splits, gt_order, l12_raw, l33_raw, FEATURES)
    print(f'v6 baseline: CLS={baseline_r["cls"]:.4f}  PR={baseline_r["pr_auc"]:.4f}  WSp={baseline_r["w_spearman"]:.4f}')

    go, _, oracles = step1_oracles(baseline_oof)

    if not go:
        print('\nOracles show no headroom. Stopping.')
        return

    # --- STEP 2: Classification corrector ---
    best_r, best_oof, all_results = step2_corrector(
        train, splits, gt_order, l12_raw, l33_raw, FEATURES)

    if best_r['cls'] > 0.7088:
        print(f'\n*** NEW RECORD: CLS={best_r["cls"]:.4f} ***')
        best_oof[['rt_name','predicted_score']].to_csv(
            ROOT / 'submissions/submission_v16.csv', index=False)
        print('Submission saved.')
        boot = bootstrap_cls(best_oof, n_bootstrap=10000)
        print()
        print_bootstrap_results(boot)
    else:
        print(f'\nNo improvement over 0.7088. Best was {best_r["cls"]:.4f}.')
        print('Step 2 failed. Consider Step 3 (ESM2-3B) if desired.')


if __name__ == '__main__':
    main()
