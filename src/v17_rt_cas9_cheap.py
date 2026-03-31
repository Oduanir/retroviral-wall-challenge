#!/usr/bin/env python3
"""
V17 — RT-Cas9 Compatibility Signal (Cheap Version)
Exp 1: Feature audit (redundancy, correlation)
Exp 2: Additive value on top of v6
Exp 3: Retron-focused readout
Exp 4: Corrector with compatibility features
"""
import numpy as np, pandas as pd, torch, warnings, os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from itertools import product as iprod
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.metrics import compute_cls
from src.bootstrap import bootstrap_cls, print_bootstrap_results


# ===========================================================================
# DATA
# ===========================================================================

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

# ESM2
print('[ESM2] Loading...')
from transformers import EsmTokenizer, EsmModel, EsmForMaskedLM
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
        logits = model_mlm(**inp).logits; probs = torch.softmax(logits[0,1:-1,:], dim=-1)
        ids = inp['input_ids'][0,1:-1]
        ll = torch.log(probs[range(len(ids)), ids])
        ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    train.loc[train['rt_name']==rt, 'esm2_pseudo_ppl'] = torch.exp(-ll.mean()).item()
    train.loc[train['rt_name']==rt, 'esm2_mean_entropy'] = ent.mean().item()
    train.loc[train['rt_name']==rt, 'esm2_mean_ll'] = ll.mean().item()
del model_mlm
print('[ESM2] Done.\n')

FEATURES_BASE = [c for c in [
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


# ===========================================================================
# EXP 1 — FEATURE AUDIT
# ===========================================================================

print('='*70)
print('EXP 1 — FEATURE AUDIT')
print('='*70)

# Extract or load compatibility features
compat_path = ROOT / 'data/processed/rt_cas9_cheap_features.csv'
if compat_path.exists():
    compat = pd.read_csv(compat_path)
    print(f'Loaded {len(compat.columns)-1} compatibility features.')
else:
    from src.rt_cas9_cheap_features import extract_rt_cas9_features
    compat = extract_rt_cas9_features()

compat_cols = [c for c in compat.columns if c != 'rt_name']
train = train.merge(compat, on='rt_name', how='left')

# Missingness
print(f'\nMissingness:')
for c in compat_cols:
    n_miss = train[c].isna().sum()
    if n_miss > 0:
        print(f'  {c:<35s} {n_miss}/57 missing')

# Correlation with target
print(f'\nCorrelation with pe_efficiency_pct:')
for c in compat_cols:
    vals = train[c].dropna()
    if len(vals) < 30:
        print(f'  {c:<35s} too many NaN')
        continue
    corr = train.loc[vals.index, ['pe_efficiency_pct', c]].corr().iloc[0,1]
    print(f'  {c:<35s} r={corr:+.3f}')

# AUROC for active classification
print(f'\nAUROC for active:')
for c in compat_cols:
    vals = train[c].dropna()
    idx = vals.index
    if len(idx) < 30:
        continue
    try:
        auc = roc_auc_score(train.loc[idx, 'active'], vals)
        auc_dir = max(auc, 1-auc)
        print(f'  {c:<35s} AUROC={auc_dir:.3f} {"*" if auc_dir > 0.65 else ""}')
    except:
        pass

# Redundancy with existing features
print(f'\nMax |corr| with existing features:')
existing_numeric = [c for c in FEATURES_BASE if c in train.columns and train[c].dtype in ['float64','int64','float32']]
for c in compat_cols:
    if train[c].isna().sum() > 20:
        continue
    max_corr = 0
    max_feat = ''
    for ef in existing_numeric:
        both = train[[c, ef]].dropna()
        if len(both) < 30:
            continue
        r = abs(both.corr().iloc[0,1])
        if r > max_corr:
            max_corr = r
            max_feat = ef
    redundant = '  REDUNDANT' if max_corr > 0.85 else ''
    print(f'  {c:<35s} max|r|={max_corr:.3f} with {max_feat}{redundant}')


# ===========================================================================
# LOFO ENGINE
# ===========================================================================

def norm(a):
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn) if mx - mn > 1e-12 else np.zeros_like(a)

def run_lofo(features, ra12=12.5, ra33=7.5):
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
                idf.loc[trm,features].values,y).predict(idf.loc[tm,features].values)
            ioof['L12'][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=ra12)).fit(
                il12p[trm.values],y).predict(il12p[tm.values])
            ioof['L33'][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=ra33)).fit(
                il33p[trm.values],y).predict(il33p[tm.values])

        best_c, best_w = -1, (0.33,0.33,0.34)
        for w in iprod(np.arange(0,1.05,0.05), repeat=3):
            if abs(sum(w)-1.0) > 0.025: continue
            b = w[0]*norm(ioof['EN']) + w[1]*norm(ioof['L12']) + w[2]*norm(ioof['L33'])
            cc = compute_cls(idf['active'].values, b, idf['pe_efficiency_pct'].values)['cls']
            if cc > best_c: best_c, best_w = cc, w

        yi = train.loc[im,'pe_efficiency_pct'].values
        pr = {}
        pr['EN'] = make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
            ElasticNet(alpha=1.0,l1_ratio=0.3,max_iter=10000)).fit(
            train.loc[im,features].values,yi).predict(train.loc[om,features].values)
        pr['L12'] = make_pipeline(StandardScaler(),Ridge(alpha=ra12)).fit(il12p,yi).predict(ol12p)
        pr['L33'] = make_pipeline(StandardScaler(),Ridge(alpha=ra33)).fit(il33p,yi).predict(ol33p)

        bl = np.zeros(om.sum())
        for i,m in enumerate(['EN','L12','L33']):
            mn,mx = ioof[m].min(), ioof[m].max()
            bl += best_w[i] * ((pr[m]-mn)/max(mx-mn,1e-12))

        fd = train.loc[om,['rt_name','active','pe_efficiency_pct','rt_family']].copy()
        fd['predicted_score'] = bl
        all_oof.append(fd)

    oof = pd.concat(all_oof).set_index('rt_name').loc[gt_order].reset_index()
    r = compute_cls(oof['active'].values, oof['predicted_score'].values, oof['pe_efficiency_pct'].values)
    return r, oof


# ===========================================================================
# EXP 2 — ADDITIVE VALUE ON TOP OF V6
# ===========================================================================

print('\n' + '='*70)
print('EXP 2 — ADDITIVE VALUE ON TOP OF V6')
print('='*70)

# Baseline
r_base, oof_base = run_lofo(FEATURES_BASE)
print(f'\nBaseline v6:          CLS={r_base["cls"]:.4f}  PR={r_base["pr_auc"]:.4f}  WSp={r_base["w_spearman"]:.4f}')

# Filter to non-redundant features (max|r| < 0.85 with existing)
non_redundant = []
for c in compat_cols:
    if train[c].isna().sum() > 20:
        continue
    max_corr = 0
    for ef in [f for f in FEATURES_BASE if f in train.columns and train[f].dtype in ['float64','int64','float32']]:
        both = train[[c, ef]].dropna()
        if len(both) < 30:
            continue
        r = abs(both.corr().iloc[0,1])
        if r > max_corr:
            max_corr = r
    if max_corr < 0.85:
        non_redundant.append(c)

print(f'\nNon-redundant compat features: {non_redundant}')

# All compat features
r_all, _ = run_lofo(FEATURES_BASE + [c for c in compat_cols if train[c].isna().sum() <= 20])
print(f'v6 + all compat:      CLS={r_all["cls"]:.4f}  PR={r_all["pr_auc"]:.4f}  WSp={r_all["w_spearman"]:.4f}')

# Non-redundant only
if non_redundant:
    r_nr, _ = run_lofo(FEATURES_BASE + non_redundant)
    print(f'v6 + non-redundant:   CLS={r_nr["cls"]:.4f}  PR={r_nr["pr_auc"]:.4f}  WSp={r_nr["w_spearman"]:.4f}')

# Top 2 by AUROC
auroc_scores = {}
for c in compat_cols:
    vals = train[c].dropna()
    if len(vals) < 30:
        continue
    try:
        auc = roc_auc_score(train.loc[vals.index, 'active'], vals)
        auroc_scores[c] = max(auc, 1-auc)
    except:
        pass
top2 = sorted(auroc_scores, key=auroc_scores.get, reverse=True)[:2]
top3 = sorted(auroc_scores, key=auroc_scores.get, reverse=True)[:3]

if top2:
    r_t2, _ = run_lofo(FEATURES_BASE + top2)
    print(f'v6 + top2 AUROC:      CLS={r_t2["cls"]:.4f}  PR={r_t2["pr_auc"]:.4f}  WSp={r_t2["w_spearman"]:.4f}  ({top2})')

if top3:
    r_t3, _ = run_lofo(FEATURES_BASE + top3)
    print(f'v6 + top3 AUROC:      CLS={r_t3["cls"]:.4f}  PR={r_t3["pr_auc"]:.4f}  WSp={r_t3["w_spearman"]:.4f}  ({top3})')

# Each feature individually
print('\nIndividual feature addition:')
for c in compat_cols:
    if train[c].isna().sum() > 20:
        continue
    r_i, _ = run_lofo(FEATURES_BASE + [c])
    delta = r_i['cls'] - r_base['cls']
    tag = ' ***' if delta > 0.002 else ''
    print(f'  +{c:<35s} CLS={r_i["cls"]:.4f}  delta={delta:+.4f}{tag}')


# ===========================================================================
# EXP 3 — RETRON-FOCUSED READOUT
# ===========================================================================

print('\n' + '='*70)
print('EXP 3 — RETRON-FOCUSED READOUT')
print('='*70)

# Best config from Exp 2
all_configs = {'baseline': r_base}
if non_redundant:
    all_configs['non_redundant'] = r_nr
if top2:
    all_configs['top2'] = r_t2
if top3:
    all_configs['top3'] = r_t3

best_name = max(all_configs, key=lambda k: all_configs[k]['cls'])
print(f'Best config: {best_name} (CLS={all_configs[best_name]["cls"]:.4f})')

# Run best config and show per-family breakdown
if best_name != 'baseline':
    if best_name == 'non_redundant':
        feats = FEATURES_BASE + non_redundant
    elif best_name == 'top2':
        feats = FEATURES_BASE + top2
    elif best_name == 'top3':
        feats = FEATURES_BASE + top3
    else:
        feats = FEATURES_BASE
    _, oof_best = run_lofo(feats)
else:
    oof_best = oof_base

print('\nPer-family PR-AUC comparison:')
print(f'  {"Family":<25s} {"Baseline":>8s} {"Best":>8s} {"Delta":>8s}')
for fam in sorted(oof_base['rt_family'].unique()):
    mask_base = oof_base['rt_family'] == fam
    mask_best = oof_best['rt_family'] == fam
    n = mask_base.sum()
    na = int(oof_base.loc[mask_base, 'active'].sum())
    if 0 < na < n:
        pr_base = average_precision_score(oof_base.loc[mask_base,'active'], oof_base.loc[mask_base,'predicted_score'])
        pr_best = average_precision_score(oof_best.loc[mask_best,'active'], oof_best.loc[mask_best,'predicted_score'])
        delta = pr_best - pr_base
        print(f'  {fam:<25s} {pr_base:8.3f} {pr_best:8.3f} {delta:+8.3f}')
    else:
        print(f'  {fam:<25s}      N/A      N/A')


# ===========================================================================
# SUMMARY + GO/NO-GO
# ===========================================================================

print('\n' + '='*70)
print('SUMMARY + GO/NO-GO')
print('='*70)

best_cls = max(v['cls'] for v in all_configs.values())
baseline_cls = r_base['cls']
delta = best_cls - baseline_cls

print(f'  Baseline CLS:  {baseline_cls:.4f}')
print(f'  Best CLS:      {best_cls:.4f}')
print(f'  Delta:         {delta:+.4f}')

if delta >= 0.005:
    print('\n  => GO: compatibility features show additive value')
elif delta > 0:
    print('\n  => MARGINAL: small positive signal, may not be robust')
else:
    print('\n  => NO-GO: compatibility features do not improve v6')

if best_cls > 0.7088:
    print(f'\n  *** NEW RECORD: CLS={best_cls:.4f} ***')
    oof_best[['rt_name','predicted_score']].to_csv(ROOT / 'submissions/submission_v17.csv', index=False)
    print('  Submission saved.')
    boot = bootstrap_cls(oof_best, n_bootstrap=10000)
    print()
    print_bootstrap_results(boot)

print('\nDone.')
