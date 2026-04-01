#!/usr/bin/env python3
"""
V18 — Audit + integration test for template-based features.
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

# Merge template features
v18_path = ROOT / 'data/processed/v18_template_features.csv'
if not v18_path.exists():
    print('Extracting template features...')
    from src.v18_template_features import extract_template_features
    v18_feats = extract_template_features()
else:
    v18_feats = pd.read_csv(v18_path)
    print(f'Template features loaded: {len(v18_feats.columns)-1} features')

v18_cols = [c for c in v18_feats.columns if c != 'rt_name']
train = train.merge(v18_feats, on='rt_name', how='left')


def norm(a):
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn) if mx - mn > 1e-12 else np.zeros_like(a)


# ===========================================================================
# AUDIT
# ===========================================================================

print('\n' + '='*70)
print('AUDIT — Template-based features')
print('='*70)

# Missingness
print('\nMissingness:')
for c in v18_cols:
    n_miss = train[c].isna().sum()
    if n_miss > 0:
        print(f'  {c:<30s} {n_miss}/57')

# AUROC
print('\nAUROC for active:')
for c in v18_cols:
    vals = train[c].dropna()
    if len(vals) < 30: continue
    try:
        auc = roc_auc_score(train.loc[vals.index,'active'], vals)
        auc_dir = max(auc, 1-auc)
        print(f'  {c:<30s} AUROC={auc_dir:.3f} {"*" if auc_dir > 0.65 else ""}')
    except: pass

# Correlation with pe_efficiency_pct
print('\nCorrelation with pe_efficiency_pct:')
for c in v18_cols:
    vals = train[c].dropna()
    if len(vals) < 30: continue
    corr = train.loc[vals.index, ['pe_efficiency_pct', c]].corr().iloc[0,1]
    print(f'  {c:<30s} r={corr:+.3f}')

# Redundancy checks (size + phylogeny confounders)
print('\nRedundancy check (confounders):')
confounders = {'seq_length': train['seq_length'] if 'seq_length' in train.columns else train['protein_length_aa'],
               'foldseek_TM_MMLV': train['foldseek_TM_MMLV']}
for c in v18_cols:
    if train[c].isna().sum() > 20: continue
    for conf_name, conf_vals in confounders.items():
        both = train[[c]].join(conf_vals.rename('conf')).dropna()
        if len(both) < 30: continue
        r = abs(both.corr().iloc[0,1])
        if r > 0.7:
            print(f'  {c:<30s} |r|={r:.3f} with {conf_name}  {"REJECT" if r > 0.8 else "WARNING"}')

# Max |r| with any existing feature
print('\nMax |r| with existing backbone:')
for c in v18_cols:
    if train[c].isna().sum() > 20: continue
    max_r, max_f = 0, ''
    for ef in FEATURES_BASE:
        if ef not in train.columns: continue
        both = train[[c, ef]].dropna()
        if len(both) < 30: continue
        r = abs(both.corr().iloc[0,1])
        if r > max_r: max_r, max_f = r, ef
    tag = '  REDUNDANT' if max_r > 0.85 else ''
    print(f'  {c:<30s} max|r|={max_r:.3f} with {max_f}{tag}')


# ===========================================================================
# INTEGRATION — Strategy 1: Individual addition with EN retuning
# ===========================================================================

def run_lofo(features, ea=1.0, el=0.3, ra12=12.5, ra33=7.5):
    all_oof = []
    for outer_family in splits:
        outer_rts = splits[outer_family]; om = train['rt_name'].isin(outer_rts); im = ~om
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
                ElasticNet(alpha=ea,l1_ratio=el,max_iter=10000)).fit(idf.loc[trm,features].values,y).predict(idf.loc[tm,features].values)
            ioof['L12'][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=ra12)).fit(il12p[trm.values],y).predict(il12p[tm.values])
            ioof['L33'][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=ra33)).fit(il33p[trm.values],y).predict(il33p[tm.values])
        best_c, best_w = -1, (0.33,0.33,0.34)
        for w in iprod(np.arange(0,1.05,0.05), repeat=3):
            if abs(sum(w)-1.0) > 0.025: continue
            b = w[0]*norm(ioof['EN']) + w[1]*norm(ioof['L12']) + w[2]*norm(ioof['L33'])
            cc = compute_cls(idf['active'].values, b, idf['pe_efficiency_pct'].values)['cls']
            if cc > best_c: best_c, best_w = cc, w
        yi = train.loc[im,'pe_efficiency_pct'].values
        pr = {}
        pr['EN'] = make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
            ElasticNet(alpha=ea,l1_ratio=el,max_iter=10000)).fit(train.loc[im,features].values,yi).predict(train.loc[om,features].values)
        pr['L12'] = make_pipeline(StandardScaler(),Ridge(alpha=ra12)).fit(il12p,yi).predict(ol12p)
        pr['L33'] = make_pipeline(StandardScaler(),Ridge(alpha=ra33)).fit(il33p,yi).predict(ol33p)
        bl = np.zeros(om.sum())
        for i,m in enumerate(['EN','L12','L33']):
            mn,mx = ioof[m].min(), ioof[m].max()
            bl += best_w[i] * ((pr[m]-mn)/max(mx-mn,1e-12))
        fd = train.loc[om,['rt_name','active','pe_efficiency_pct','rt_family']].copy()
        fd['predicted_score'] = bl; all_oof.append(fd)
    oof = pd.concat(all_oof).set_index('rt_name').loc[gt_order].reset_index()
    return compute_cls(oof['active'].values, oof['predicted_score'].values, oof['pe_efficiency_pct'].values), oof


print('\n' + '='*70)
print('INTEGRATION — Individual feature addition with EN retuning')
print('='*70)

r_base, oof_base = run_lofo(FEATURES_BASE)
print(f'\nBaseline v6:  CLS={r_base["cls"]:.4f}  PR={r_base["pr_auc"]:.4f}  WSp={r_base["w_spearman"]:.4f}')

# Filter to usable features
usable = [c for c in v18_cols if train[c].isna().sum() <= 20]

print('\nStrategy 1: Add to EN with retuning (α×l1 sweep):')
for c in usable:
    best_cls_feat = 0
    best_cfg = ''
    for ea, el in [(0.5,0.2),(0.5,0.3),(1.0,0.2),(1.0,0.3),(1.0,0.4),(2.0,0.3)]:
        r, _ = run_lofo(FEATURES_BASE + [c], ea=ea, el=el)
        if r['cls'] > best_cls_feat:
            best_cls_feat = r['cls']
            best_cfg = f'a={ea},l1={el}'
    delta = best_cls_feat - r_base['cls']
    tag = ' ***' if delta > 0 else ''
    print(f'  +{c:<28s} best CLS={best_cls_feat:.4f}  delta={delta:+.4f}  ({best_cfg}){tag}')

# Strategy 2: top features as 4th model
print('\nStrategy 2: Feature as 4th Ridge model in blend:')
for c in usable:
    all_oof2 = []
    for outer_family in splits:
        outer_rts = splits[outer_family]; om = train['rt_name'].isin(outer_rts); im = ~om
        isp = {f: rts for f, rts in splits.items() if f != outer_family}
        idf = train[im].reset_index(drop=True)
        il12 = np.array([l12_raw[n] for n in idf['rt_name']])
        il33 = np.array([l33_raw[n] for n in idf['rt_name']])
        p12 = PCA(3).fit(il12); p33 = PCA(3).fit(il33)
        il12p = p12.transform(il12); il33p = p33.transform(il33)
        ol12p = p12.transform(np.array([l12_raw[n] for n in train.loc[om,'rt_name']]))
        ol33p = p33.transform(np.array([l33_raw[n] for n in train.loc[om,'rt_name']]))
        n = len(idf)
        # 4th model: just the single feature
        feat_inner = idf[c].fillna(idf[c].median()).values.reshape(-1,1)
        feat_outer = train.loc[om,c].fillna(idf[c].median()).values.reshape(-1,1)
        ioof = {'EN': np.full(n, np.nan), 'L12': np.full(n, np.nan), 'L33': np.full(n, np.nan), 'F': np.full(n, np.nan)}
        for ifam, irts in isp.items():
            tm = idf['rt_name'].isin(irts); trm = ~tm
            y = idf.loc[trm, 'pe_efficiency_pct'].values
            ioof['EN'][tm.values] = make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                ElasticNet(alpha=1.0,l1_ratio=0.3,max_iter=10000)).fit(idf.loc[trm,FEATURES_BASE].values,y).predict(idf.loc[tm,FEATURES_BASE].values)
            ioof['L12'][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=12.5)).fit(il12p[trm.values],y).predict(il12p[tm.values])
            ioof['L33'][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=7.5)).fit(il33p[trm.values],y).predict(il33p[tm.values])
            ioof['F'][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=10.0)).fit(feat_inner[trm.values],y).predict(feat_inner[tm.values])
        best_c2, best_w2 = -1, (0.25,0.25,0.25,0.25)
        for w in iprod(np.arange(0,1.05,0.1), repeat=4):
            if abs(sum(w)-1.0) > 0.05: continue
            b = w[0]*norm(ioof['EN']) + w[1]*norm(ioof['L12']) + w[2]*norm(ioof['L33']) + w[3]*norm(ioof['F'])
            cc = compute_cls(idf['active'].values, b, idf['pe_efficiency_pct'].values)['cls']
            if cc > best_c2: best_c2, best_w2 = cc, w
        yi = train.loc[im,'pe_efficiency_pct'].values
        pr = {}
        pr['EN'] = make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
            ElasticNet(alpha=1.0,l1_ratio=0.3,max_iter=10000)).fit(train.loc[im,FEATURES_BASE].values,yi).predict(train.loc[om,FEATURES_BASE].values)
        pr['L12'] = make_pipeline(StandardScaler(),Ridge(alpha=12.5)).fit(il12p,yi).predict(ol12p)
        pr['L33'] = make_pipeline(StandardScaler(),Ridge(alpha=7.5)).fit(il33p,yi).predict(ol33p)
        pr['F'] = make_pipeline(StandardScaler(),Ridge(alpha=10.0)).fit(feat_inner,yi).predict(feat_outer)
        bl = np.zeros(om.sum())
        for i,m in enumerate(['EN','L12','L33','F']):
            mn,mx = ioof[m].min(), ioof[m].max()
            bl += best_w2[i] * ((pr[m]-mn)/max(mx-mn,1e-12))
        fd = train.loc[om,['rt_name','active','pe_efficiency_pct','rt_family']].copy()
        fd['predicted_score'] = bl; all_oof2.append(fd)
    oof2 = pd.concat(all_oof2).set_index('rt_name').loc[gt_order].reset_index()
    r2 = compute_cls(oof2['active'].values, oof2['predicted_score'].values, oof2['pe_efficiency_pct'].values)
    delta = r2['cls'] - r_base['cls']
    tag = ' ***' if delta > 0 else ''
    print(f'  +{c:<28s} CLS={r2["cls"]:.4f}  delta={delta:+.4f}{tag}')

print('\nDone.')
