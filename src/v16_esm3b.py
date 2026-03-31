#!/usr/bin/env python3
"""
V16 Step 3 — ESM2-3B frozen mid-region layer sweep.
Replicate the successful 650M recipe with the larger model.
"""
import numpy as np, pandas as pd, torch, warnings, gc, os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import average_precision_score
from itertools import product as iprod
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.metrics import compute_cls

# --- Data ---
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

FEATURES = [c for c in [
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

# --- ESM2-650M zero-shot features (needed for EN) ---
print('[1] ESM2-650M zero-shot features...')
from transformers import EsmTokenizer, EsmForMaskedLM
tok_650 = EsmTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
mlm_650 = EsmForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D').cuda().half().eval()
for rt, seq in sequences.items():
    inp = tok_650(seq, return_tensors='pt', truncation=True, max_length=1024).to('cuda')
    with torch.no_grad():
        logits = mlm_650(**inp).logits.float()
        probs = torch.softmax(logits[0,1:-1,:], dim=-1)
        ids = inp['input_ids'][0,1:-1]
        ll = torch.log(probs[range(len(ids)), ids])
        ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    train.loc[train['rt_name']==rt, 'esm2_pseudo_ppl'] = torch.exp(-ll.mean()).item()
    train.loc[train['rt_name']==rt, 'esm2_mean_entropy'] = ent.mean().item()
    train.loc[train['rt_name']==rt, 'esm2_mean_ll'] = ll.mean().item()
del mlm_650; torch.cuda.empty_cache(); gc.collect()
print('  Done.')

# --- ESM2-3B embeddings ---
print('\n[2] ESM2-3B mid-region layer sweep...')
MODEL_3B = 'facebook/esm2_t36_3B_UR50D'  # 36 layers, 2560D
from transformers import EsmModel

tok_3b = EsmTokenizer.from_pretrained(MODEL_3B)
# Load in float16 to fit in 12GB VRAM
model_3b = EsmModel.from_pretrained(MODEL_3B, torch_dtype=torch.float16).cuda().eval()
print('  ESM2-3B loaded.')

# Extract mid-region embeddings for selected layers
# 36 layers total. Test: early (12), mid (18, 24), late (30, 36)
LAYERS_3B = [12, 18, 24, 30, 36]
layer_embs = {l: {} for l in LAYERS_3B}

for i, (rt, seq) in enumerate(sequences.items()):
    # Truncate long sequences more aggressively if needed for memory
    inp = tok_3b(seq, return_tensors='pt', truncation=True, max_length=1024).to('cuda')
    try:
        with torch.no_grad():
            out = model_3b(**inp, output_hidden_states=True)
            n3 = len(seq) // 3
            for l in LAYERS_3B:
                layer_embs[l][rt] = out.hidden_states[l][0, 1:-1, :][n3:2*n3].mean(0).float().cpu().numpy()
    except RuntimeError as e:
        if 'out of memory' in str(e):
            torch.cuda.empty_cache()
            # Retry with shorter truncation
            inp = tok_3b(seq, return_tensors='pt', truncation=True, max_length=512).to('cuda')
            with torch.no_grad():
                out = model_3b(**inp, output_hidden_states=True)
                n3 = min(len(seq), 512) // 3
                for l in LAYERS_3B:
                    layer_embs[l][rt] = out.hidden_states[l][0, 1:-1, :][n3:2*n3].mean(0).float().cpu().numpy()
        else:
            raise
    if (i+1) % 10 == 0:
        print(f'  {i+1}/57 sequences processed')

del model_3b; torch.cuda.empty_cache(); gc.collect()
print(f'  ESM2-3B embeddings extracted for layers {LAYERS_3B}.')

# Save for reuse
save_dict = {'names': np.array(list(sequences.keys()))}
for l in LAYERS_3B:
    save_dict[f'layer_{l}'] = np.array([layer_embs[l][n] for n in sequences.keys()])
np.savez(ROOT / 'data/processed/esm2_3b_multilayer_mid.npz', **save_dict)
print('  Saved.')

# --- Also extract ESM2-650M L12/L33 for baseline comparison ---
print('\n[3] ESM2-650M L12/L33 embeddings...')
model_650 = EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D').cuda().eval()
l12_650, l33_650 = {}, {}
for rt, seq in sequences.items():
    inp = tok_650(seq, return_tensors='pt', truncation=True, max_length=1024).to('cuda')
    with torch.no_grad():
        out = model_650(**inp, output_hidden_states=True)
        n3 = len(seq) // 3
        l12_650[rt] = out.hidden_states[12][0,1:-1,:][n3:2*n3].mean(0).cpu().numpy()
        l33_650[rt] = out.hidden_states[33][0,1:-1,:][n3:2*n3].mean(0).cpu().numpy()
del model_650; torch.cuda.empty_cache(); gc.collect()
print('  Done.')

# --- LOFO function ---
def norm(a):
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn) if mx - mn > 1e-12 else np.zeros_like(a)

def run_lofo(model_configs, ra_dict=None):
    """
    model_configs: list of (name, embeddings_dict) for Ridge models
    ra_dict: {name: ridge_alpha}
    """
    if ra_dict is None:
        ra_dict = {}
    all_oof = []
    model_names = ['EN'] + [mc[0] for mc in model_configs]

    for outer_family in splits:
        outer_rts = splits[outer_family]
        om = train['rt_name'].isin(outer_rts); im = ~om
        isp = {f: rts for f, rts in splits.items() if f != outer_family}
        idf = train[im].reset_index(drop=True)
        n = len(idf)

        # Per-fold PCA for each embedding model
        pca_data = {}
        for mname, emb_dict in model_configs:
            inner_arr = np.array([emb_dict[r] for r in idf['rt_name']])
            outer_arr = np.array([emb_dict[r] for r in train.loc[om,'rt_name']])
            pca = PCA(3).fit(inner_arr)
            pca_data[mname] = {
                'inner': pca.transform(inner_arr),
                'outer': pca.transform(outer_arr),
            }

        ioof = {m: np.full(n, np.nan) for m in model_names}
        for ifam, irts in isp.items():
            tm = idf['rt_name'].isin(irts); trm = ~tm
            y = idf.loc[trm, 'pe_efficiency_pct'].values
            ioof['EN'][tm.values] = make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                ElasticNet(alpha=1.0,l1_ratio=0.3,max_iter=10000)).fit(
                idf.loc[trm,FEATURES].values,y).predict(idf.loc[tm,FEATURES].values)
            for mname, _ in model_configs:
                ra = ra_dict.get(mname, 10.0)
                ioof[mname][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=ra)).fit(
                    pca_data[mname]['inner'][trm.values],y).predict(pca_data[mname]['inner'][tm.values])

        nm = len(model_names)
        best_c, best_w = -1, tuple([1.0/nm]*nm)
        for w in iprod(np.arange(0,1.05,0.05), repeat=nm):
            if abs(sum(w)-1.0) > 0.025: continue
            b = sum(w[i]*norm(ioof[m]) for i,m in enumerate(model_names))
            cc = compute_cls(idf['active'].values, b, idf['pe_efficiency_pct'].values)['cls']
            if cc > best_c: best_c, best_w = cc, w

        yi = train.loc[im,'pe_efficiency_pct'].values
        pr = {}
        pr['EN'] = make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
            ElasticNet(alpha=1.0,l1_ratio=0.3,max_iter=10000)).fit(
            train.loc[im,FEATURES].values,yi).predict(train.loc[om,FEATURES].values)
        for mname, _ in model_configs:
            ra = ra_dict.get(mname, 10.0)
            pr[mname] = make_pipeline(StandardScaler(),Ridge(alpha=ra)).fit(
                pca_data[mname]['inner'],yi).predict(pca_data[mname]['outer'])

        bl = np.zeros(om.sum())
        for i,m in enumerate(model_names):
            mn,mx = ioof[m].min(), ioof[m].max()
            bl += best_w[i] * ((pr[m]-mn)/max(mx-mn,1e-12))

        fd = train.loc[om,['rt_name','active','pe_efficiency_pct','rt_family']].copy()
        fd['predicted_score'] = bl; all_oof.append(fd)

    oof = pd.concat(all_oof).set_index('rt_name').loc[gt_order].reset_index()
    r = compute_cls(oof['active'].values, oof['predicted_score'].values, oof['pe_efficiency_pct'].values)
    return r, oof


# --- EXPERIMENTS ---
print('\n' + '='*70)
print('ESM2-3B EXPERIMENTS')
print('='*70)

# Baseline: 650M L12 + L33 (GPU version)
r_base, _ = run_lofo([('L12', l12_650), ('L33', l33_650)],
                      ra_dict={'L12': 12.5, 'L33': 7.5})
print(f'Baseline 650M L12+L33:           CLS={r_base["cls"]:.4f}  PR={r_base["pr_auc"]:.4f}  WSp={r_base["w_spearman"]:.4f}')

# Test each 3B layer alone (replacing both L12 and L33)
for l in LAYERS_3B:
    r, _ = run_lofo([('3B', layer_embs[l])], ra_dict={'3B': 10.0})
    print(f'3B L{l} alone:                    CLS={r["cls"]:.4f}  PR={r["pr_auc"]:.4f}  WSp={r["w_spearman"]:.4f}')

# Test each 3B layer as replacement for L33 (keep 650M L12)
print()
for l in LAYERS_3B:
    r, _ = run_lofo([('L12', l12_650), ('3B_L'+str(l), layer_embs[l])],
                     ra_dict={'L12': 12.5, '3B_L'+str(l): 7.5})
    print(f'650M-L12 + 3B-L{l}:               CLS={r["cls"]:.4f}  PR={r["pr_auc"]:.4f}  WSp={r["w_spearman"]:.4f}')

# Test each 3B layer as replacement for L12 (keep 650M L33)
print()
for l in LAYERS_3B:
    r, _ = run_lofo([('3B_L'+str(l), layer_embs[l]), ('L33', l33_650)],
                     ra_dict={'3B_L'+str(l): 12.5, 'L33': 7.5})
    print(f'3B-L{l} + 650M-L33:               CLS={r["cls"]:.4f}  PR={r["pr_auc"]:.4f}  WSp={r["w_spearman"]:.4f}')

# Best 3B layer + 650M L12 + 650M L33 (4-model)
print()
for l in LAYERS_3B:
    r, _ = run_lofo([('L12', l12_650), ('L33', l33_650), ('3B', layer_embs[l])],
                     ra_dict={'L12': 12.5, 'L33': 7.5, '3B': 10.0})
    print(f'650M-L12 + 650M-L33 + 3B-L{l}:    CLS={r["cls"]:.4f}  PR={r["pr_auc"]:.4f}  WSp={r["w_spearman"]:.4f}')

# Best 2x3B layers (replace both 650M)
print()
for l1, l2 in [(12,36), (18,36), (24,36), (12,30), (18,30)]:
    r, _ = run_lofo([('3B_a', layer_embs[l1]), ('3B_b', layer_embs[l2])],
                     ra_dict={'3B_a': 12.5, '3B_b': 7.5})
    print(f'3B-L{l1} + 3B-L{l2}:                  CLS={r["cls"]:.4f}  PR={r["pr_auc"]:.4f}  WSp={r["w_spearman"]:.4f}')

print('\nDone.')
