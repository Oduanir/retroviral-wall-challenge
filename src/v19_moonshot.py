#!/usr/bin/env python3
"""
V19 Moonshot — Full PE complex compatibility modeling.

Stage 1: Place all 57 RTs in two PE complex states (8WUS + 8WUT)
Stage 2: Extract comprehensive complex features (~30 features × 2 states + robustness)
Stage 3: Audit + integration as separate modality
"""
import numpy as np, pandas as pd, torch, warnings, os, gc
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from itertools import product as iprod
from scipy.spatial.distance import cdist
from pathlib import Path
from Bio.PDB import PDBParser, MMCIFParser
import tmtools
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.metrics import compute_cls
from src.esm2_features import find_yxdd

DATA_DIR = ROOT / "data" / "raw"
STRUCT_DIR = DATA_DIR / "structures"
TEMPLATE_DIR = DATA_DIR / "templates"
PROCESSED_DIR = ROOT / "data" / "processed"

# Chain mapping for both templates (verified)
CHAIN_RT = "E"
CHAIN_CAS9 = "A"
CHAIN_PEGRNA = "B"
CHAIN_DNA = ["C", "D", "F"]


# ===========================================================================
# STRUCTURE UTILITIES
# ===========================================================================

def load_structure(path):
    if str(path).endswith('.cif'):
        return MMCIFParser(QUIET=True).get_structure("s", path)
    return PDBParser(QUIET=True).get_structure("s", path)


def get_ca_coords(structure, chain_id):
    coords = []
    for model in structure:
        if chain_id not in model:
            continue
        for res in model[chain_id]:
            if "CA" in res:
                coords.append(res["CA"].get_vector().get_array())
    return np.array(coords, dtype=np.float64) if coords else None


def get_all_atom_coords(structure, chain_id):
    coords = []
    for model in structure:
        if chain_id not in model:
            continue
        for res in model[chain_id]:
            for atom in res:
                coords.append(atom.get_vector().get_array())
    return np.array(coords, dtype=np.float64) if coords else None


def get_backbone_coords(structure, chain_ids):
    coords = []
    for model in structure:
        for cid in chain_ids:
            if cid not in model:
                continue
            for res in model[cid]:
                for aname in ["P", "C3'", "C4'"]:
                    if aname in res:
                        coords.append(res[aname].get_vector().get_array())
                        break
    return np.array(coords, dtype=np.float64) if coords else None


# ===========================================================================
# STAGE 1: PLACE ALL RTs IN COMPLEX
# ===========================================================================

def place_rt_in_complex(cand_ca, cand_seq, ref_ca, ref_seq):
    """Align candidate RT onto reference RT via TMalign, return transformed coords + metadata."""
    result = tmtools.tm_align(
        cand_ca.astype(np.float32), ref_ca.astype(np.float32),
        cand_seq, ref_seq
    )
    # Apply transformation
    aligned_ca = (result.u @ cand_ca.T).T + result.t

    # Count aligned residues
    n_aligned = sum(1 for a, b in zip(result.seqxA, result.seqyA) if a != '-' and b != '-')

    # Compute per-residue alignment distances for aligned pairs
    aligned_dists = []
    qi, ri = 0, 0
    for a, b in zip(result.seqxA, result.seqyA):
        if a != '-' and b != '-':
            d = np.linalg.norm(aligned_ca[qi] - ref_ca[ri])
            aligned_dists.append(d)
        if a != '-':
            qi += 1
        if b != '-':
            ri += 1

    return {
        'aligned_ca': aligned_ca,
        'tm_score': float(result.tm_norm_chain2),
        'rmsd': float(result.rmsd),
        'n_aligned': n_aligned,
        'n_total': len(cand_ca),
        'coverage': n_aligned / len(cand_ca),
        'aligned_dists': np.array(aligned_dists),
        'seqxA': result.seqxA,
        'seqyA': result.seqyA,
    }


# ===========================================================================
# STAGE 2: EXTRACT COMPREHENSIVE FEATURES
# ===========================================================================

def extract_complex_features(placement, cas9_ca, cas9_all, dna_bb, pegrna_bb,
                              ref_ca, ref_yxdd_pos, yxdd_pos):
    """Extract ~25 features from a placed RT in the PE complex."""
    ca = placement['aligned_ca']
    n_aligned = max(placement['n_aligned'], 1)
    n_total = len(ca)
    feats = {}

    # --- Alignment quality ---
    feats['tm_score'] = placement['tm_score']
    feats['alignment_rmsd'] = placement['rmsd']
    feats['alignment_coverage'] = placement['coverage']
    feats['alignment_strain'] = float(np.mean(placement['aligned_dists'])) if len(placement['aligned_dists']) > 0 else np.nan
    feats['alignment_strain_max'] = float(np.max(placement['aligned_dists'])) if len(placement['aligned_dists']) > 0 else np.nan

    # --- Global fit (clashes with Cas9) ---
    dist_ca_cas9 = cdist(ca, cas9_ca)
    min_to_cas9 = dist_ca_cas9.min(axis=1)

    feats['cas9_clash_frac'] = float((min_to_cas9 < 3.0).sum() / n_aligned)
    feats['cas9_clash_severe_frac'] = float((min_to_cas9 < 2.0).sum() / n_aligned)
    feats['cas9_min_dist'] = float(min_to_cas9.min())
    feats['cas9_contact_frac'] = float((min_to_cas9 < 5.0).sum() / n_aligned)
    feats['cas9_interface_density'] = float((min_to_cas9 < 8.0).sum() / n_aligned)

    # --- Global fit (protrusion beyond ref envelope) ---
    ref_centroid = ref_ca.mean(axis=0)
    ref_dists = np.linalg.norm(ref_ca - ref_centroid, axis=1)
    ref_max_dist = ref_dists.max()
    ref_p95_dist = np.percentile(ref_dists, 95)
    cand_dists = np.linalg.norm(ca - ref_centroid, axis=1)
    feats['protrusion_frac'] = float((cand_dists > ref_max_dist).sum() / n_aligned)
    feats['protrusion_frac_p95'] = float((cand_dists > ref_p95_dist).sum() / n_aligned)

    # --- Catalytic fit ---
    if yxdd_pos is not None and yxdd_pos + 4 <= n_total:
        yxdd_ca = ca[yxdd_pos:yxdd_pos+4]
        yxdd_centroid = yxdd_ca.mean(axis=0)

        # Distance to DNA
        if dna_bb is not None and len(dna_bb) > 0:
            feats['yxdd_to_dna'] = float(cdist([yxdd_centroid], dna_bb).min())
        else:
            feats['yxdd_to_dna'] = np.nan

        # Distance to pegRNA
        if pegrna_bb is not None and len(pegrna_bb) > 0:
            feats['yxdd_to_pegrna'] = float(cdist([yxdd_centroid], pegrna_bb).min())
        else:
            feats['yxdd_to_pegrna'] = np.nan

        # Orientation: compare YXDD→DNA vector to reference
        if ref_yxdd_pos is not None and dna_bb is not None and len(dna_bb) > 0:
            ref_yxdd_centroid = ref_ca[ref_yxdd_pos:ref_yxdd_pos+4].mean(axis=0)
            # Candidate vector
            nearest_dna = dna_bb[cdist([yxdd_centroid], dna_bb).argmin()]
            vec_c = nearest_dna - yxdd_centroid
            vec_c = vec_c / max(np.linalg.norm(vec_c), 1e-10)
            # Reference vector
            nearest_dna_ref = dna_bb[cdist([ref_yxdd_centroid], dna_bb).argmin()]
            vec_r = nearest_dna_ref - ref_yxdd_centroid
            vec_r = vec_r / max(np.linalg.norm(vec_r), 1e-10)
            feats['yxdd_orient_angle'] = float(np.degrees(np.arccos(np.clip(np.dot(vec_c, vec_r), -1, 1))))
        else:
            feats['yxdd_orient_angle'] = np.nan

        # Occlusion by Cas9
        yxdd_region = ca[max(0,yxdd_pos-8):yxdd_pos+12]
        dist_yxdd_cas9 = cdist(yxdd_region, cas9_ca)
        n_yr = len(yxdd_region)
        feats['yxdd_cas9_occlusion'] = float((dist_yxdd_cas9.min(axis=1) < 5.0).sum() / max(n_yr, 1))
        feats['yxdd_cas9_min_dist'] = float(dist_yxdd_cas9.min())

        # Catalytic patch exposure: how many directions from YXDD are NOT blocked by Cas9
        # Sample directions, check if Cas9 is in the way
        feats['yxdd_displacement_from_ref'] = float(np.linalg.norm(yxdd_centroid - ref_yxdd_centroid)) if ref_yxdd_pos is not None else np.nan
    else:
        for k in ['yxdd_to_dna','yxdd_to_pegrna','yxdd_orient_angle',
                   'yxdd_cas9_occlusion','yxdd_cas9_min_dist','yxdd_displacement_from_ref']:
            feats[k] = np.nan

    # --- Fusion topology ---
    nterm = ca[0]
    cterm = ca[-1]
    feats['nterm_to_cas9'] = float(cdist([nterm], cas9_ca).min())
    feats['cterm_to_cas9'] = float(cdist([cterm], cas9_ca).min())
    feats['termini_asymmetry_cas9'] = abs(feats['nterm_to_cas9'] - feats['cterm_to_cas9'])

    # --- Nucleic acid interface ---
    if dna_bb is not None and len(dna_bb) > 0:
        dist_to_dna = cdist(ca, dna_bb).min(axis=1)
        feats['dna_contact_frac'] = float((dist_to_dna < 5.0).sum() / n_aligned)
    else:
        feats['dna_contact_frac'] = np.nan

    if pegrna_bb is not None and len(pegrna_bb) > 0:
        dist_to_rna = cdist(ca, pegrna_bb).min(axis=1)
        feats['pegrna_contact_frac'] = float((dist_to_rna < 5.0).sum() / n_aligned)
    else:
        feats['pegrna_contact_frac'] = np.nan

    return feats


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main():
    print('='*70)
    print('V19 MOONSHOT — PE Complex Compatibility Modeling')
    print('='*70)

    # Load sequences
    seqs_df = pd.read_csv(DATA_DIR / "rt_sequences.csv")
    sequences = dict(zip(seqs_df["rt_name"], seqs_df["sequence"]))
    mmlv_seq = sequences.get("MMLV-RT", "")
    ref_yxdd = find_yxdd(mmlv_seq)

    # Load templates
    templates = {}
    for name, path, fmt in [('8WUS', TEMPLATE_DIR/'8WUS.pdb', 'pdb'),
                             ('8WUT', TEMPLATE_DIR/'8WUT.cif', 'cif')]:
        if not path.exists():
            print(f'  Template {name} not found, skipping')
            continue
        s = load_structure(path)
        ref_ca = get_ca_coords(s, CHAIN_RT)
        cas9_ca = get_ca_coords(s, CHAIN_CAS9)
        cas9_all = get_all_atom_coords(s, CHAIN_CAS9)
        dna_bb = get_backbone_coords(s, CHAIN_DNA)
        pegrna_bb = get_backbone_coords(s, [CHAIN_PEGRNA])
        ref_seq = mmlv_seq[:len(ref_ca)]
        templates[name] = {
            'ref_ca': ref_ca, 'cas9_ca': cas9_ca, 'cas9_all': cas9_all,
            'dna_bb': dna_bb, 'pegrna_bb': pegrna_bb, 'ref_seq': ref_seq,
        }
        print(f'  Template {name}: RT={len(ref_ca)} CA, Cas9={len(cas9_ca)} CA, '
              f'DNA={len(dna_bb) if dna_bb is not None else 0}, '
              f'pegRNA={len(pegrna_bb) if pegrna_bb is not None else 0}')

    if not templates:
        print('No templates available!')
        return

    # --- STAGE 1+2: Place and extract features ---
    print(f'\n[Stage 1+2] Placing 57 RTs and extracting features...')

    all_features = {tname: [] for tname in templates}

    for rt_name, seq in sequences.items():
        yxdd_pos = find_yxdd(seq)
        pdb_path = STRUCT_DIR / f"{rt_name}.pdb"
        if not pdb_path.exists():
            for tname in templates:
                feats = {'rt_name': rt_name}
                all_features[tname].append(feats)
            continue

        cand_s = load_structure(pdb_path)
        cand_ca = get_ca_coords(cand_s, list(cand_s[0].get_chains())[0].id)
        if cand_ca is None or len(cand_ca) < 10:
            for tname in templates:
                all_features[tname].append({'rt_name': rt_name})
            continue

        cand_seq = seq[:len(cand_ca)]

        for tname, tdata in templates.items():
            try:
                placement = place_rt_in_complex(cand_ca, cand_seq, tdata['ref_ca'], tdata['ref_seq'])
                feats = extract_complex_features(
                    placement, tdata['cas9_ca'], tdata['cas9_all'],
                    tdata['dna_bb'], tdata['pegrna_bb'],
                    tdata['ref_ca'], ref_yxdd, yxdd_pos
                )
                feats['rt_name'] = rt_name
                all_features[tname].append(feats)
            except Exception as e:
                print(f'  {rt_name} / {tname}: failed ({e})')
                all_features[tname].append({'rt_name': rt_name})

    # Build DataFrames
    dfs = {}
    for tname in templates:
        df = pd.DataFrame(all_features[tname])
        feat_cols = [c for c in df.columns if c != 'rt_name']
        dfs[tname] = df
        n_ok = df[feat_cols[0]].notna().sum() if feat_cols else 0
        print(f'  {tname}: {n_ok}/57 RTs placed, {len(feat_cols)} features')

    # --- Robustness features (variance across states) ---
    if len(dfs) >= 2:
        print('\n[Robustness] Computing cross-state variance...')
        df1 = dfs[list(dfs.keys())[0]].set_index('rt_name')
        df2 = dfs[list(dfs.keys())[1]].set_index('rt_name')
        common_cols = [c for c in df1.columns if c in df2.columns and df1[c].dtype == 'float64']
        robustness = pd.DataFrame(index=df1.index)
        robustness['rt_name'] = robustness.index
        for c in common_cols:
            robustness[f'{c}_var'] = (df1[c] - df2[c]).abs()
        robust_cols = [c for c in robustness.columns if c.endswith('_var')]
        print(f'  {len(robust_cols)} robustness features computed')

    # Use primary template (8WUS) as main feature set
    primary = dfs[list(dfs.keys())[0]].copy()
    feat_cols = [c for c in primary.columns if c != 'rt_name']

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    primary.to_csv(PROCESSED_DIR / 'v19_moonshot_features.csv', index=False)
    print(f'\nSaved {len(primary)} RTs × {len(feat_cols)} features')

    # --- STAGE 3: AUDIT ---
    print('\n' + '='*70)
    print('AUDIT')
    print('='*70)

    train = pd.read_csv(DATA_DIR / 'train.csv')
    train = train.merge(primary, on='rt_name', how='left')

    # AUROC
    print('\nAUROC for active:')
    good_feats = []
    for c in feat_cols:
        vals = train[c].dropna()
        if len(vals) < 30: continue
        try:
            auc = roc_auc_score(train.loc[vals.index,'active'], vals)
            auc_dir = max(auc, 1-auc)
            tag = ' *' if auc_dir > 0.65 else ''
            print(f'  {c:<30s} AUROC={auc_dir:.3f}{tag}')
            if auc_dir > 0.60:
                good_feats.append((c, auc_dir))
        except: pass

    # Confounder check
    print('\nConfounder check (|r| with seq_length and foldseek_TM_MMLV):')
    confounders = {}
    if 'seq_length' in train.columns:
        confounders['seq_length'] = train['seq_length']
    elif 'protein_length_aa' in train.columns:
        confounders['seq_length'] = train['protein_length_aa']
    confounders['foldseek_TM_MMLV'] = train['foldseek_TM_MMLV']

    genuinely_novel = []
    for c in feat_cols:
        if train[c].isna().sum() > 20: continue
        is_confounder = False
        for cn, cv in confounders.items():
            both = train[[c]].join(cv.rename('conf')).dropna()
            if len(both) < 30: continue
            r = abs(both.corr().iloc[0,1])
            if r > 0.8:
                print(f'  {c:<30s} |r|={r:.3f} with {cn}  REJECT')
                is_confounder = True
        if not is_confounder:
            genuinely_novel.append(c)

    print(f'\nGenuinely novel features (not size/MMLV proxies): {len(genuinely_novel)}')
    for c in genuinely_novel[:15]:
        print(f'  {c}')

    # --- STAGE 3: INTEGRATION ---
    print('\n' + '='*70)
    print('INTEGRATION')
    print('='*70)

    # Load full v6 pipeline data
    for col in ['connection_mean_pot','triad_best_rmsd','D1_D2_dist','D2_D3_dist',
                'yxdd_hydrophobic_fraction','yxdd_mean_hydrophobicity','yxdd_5A_mean_pot']:
        train[f'{col}_missing'] = train[col].isna().astype(int)
    train['foldseek_gap_MMLV'] = train['foldseek_best_TM'] - train['foldseek_TM_MMLV']
    train['t40_x_foldseek_MMLV'] = train['t40_raw'] * train['foldseek_TM_MMLV']
    train['triad_quality'] = train['triad_found_bin'] * (1 / (train['triad_best_rmsd'].fillna(99) + 1))
    train['seq_struct_compat'] = -train['perplexity'] * train['instability_index']

    # ESM2
    print('\n[ESM2] Loading...')
    from transformers import EsmTokenizer, EsmModel, EsmForMaskedLM
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
    esm = EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D'); esm.eval()
    l12_raw, l33_raw = {}, {}
    for rt, seq in sequences.items():
        inp = tokenizer(seq, return_tensors='pt', truncation=True, max_length=1024)
        with torch.no_grad():
            out = esm(**inp, output_hidden_states=True)
            n3 = len(seq)//3
            l12_raw[rt] = out.hidden_states[12][0,1:-1,:][n3:2*n3].mean(0).numpy()
            l33_raw[rt] = out.hidden_states[33][0,1:-1,:][n3:2*n3].mean(0).numpy()
    del esm
    mlm = EsmForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D'); mlm.eval()
    for rt, seq in sequences.items():
        inp = tokenizer(seq, return_tensors='pt', truncation=True, max_length=1024)
        with torch.no_grad():
            logits = mlm(**inp).logits; probs = torch.softmax(logits[0,1:-1,:], dim=-1)
            ids = inp['input_ids'][0,1:-1]
            ll = torch.log(probs[range(len(ids)), ids])
            ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        train.loc[train['rt_name']==rt, 'esm2_pseudo_ppl'] = torch.exp(-ll.mean()).item()
        train.loc[train['rt_name']==rt, 'esm2_mean_entropy'] = ent.mean().item()
        train.loc[train['rt_name']==rt, 'esm2_mean_ll'] = ll.mean().item()
    del mlm
    print('[ESM2] Done.')

    splits_df = pd.read_csv(DATA_DIR / 'family_splits.csv')
    splits = {}
    for _, row in splits_df.iterrows():
        splits[row['family']] = row['rt_names'].split('|')
    gt_order = pd.read_csv(DATA_DIR / 'rt_sequences.csv', usecols=['rt_name'])['rt_name'].tolist()

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

    def norm(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn) if mx - mn > 1e-12 else np.zeros_like(a)

    def run_lofo_blend(features_en, extra_ridge_features=None, ra12=12.5, ra33=7.5):
        """Run v6 LOFO with optional extra Ridge model on complex features."""
        all_oof = []
        models = ['EN','L12','L33']
        if extra_ridge_features is not None:
            models.append('COMPLEX')

        for outer_family in splits:
            outer_rts = splits[outer_family]; om = train['rt_name'].isin(outer_rts); im = ~om
            isp = {f: rts for f, rts in splits.items() if f != outer_family}
            idf = train[im].reset_index(drop=True)
            n = len(idf)

            il12 = np.array([l12_raw[r] for r in idf['rt_name']])
            il33 = np.array([l33_raw[r] for r in idf['rt_name']])
            p12 = PCA(3).fit(il12); p33 = PCA(3).fit(il33)
            il12p = p12.transform(il12); il33p = p33.transform(il33)
            ol12p = p12.transform(np.array([l12_raw[r] for r in train.loc[om,'rt_name']]))
            ol33p = p33.transform(np.array([l33_raw[r] for r in train.loc[om,'rt_name']]))

            ioof = {m: np.full(n, np.nan) for m in models}
            for ifam, irts in isp.items():
                tm = idf['rt_name'].isin(irts); trm = ~tm
                y = idf.loc[trm, 'pe_efficiency_pct'].values
                ioof['EN'][tm.values] = make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                    ElasticNet(alpha=1.0,l1_ratio=0.3,max_iter=10000)).fit(
                    idf.loc[trm,features_en].values,y).predict(idf.loc[tm,features_en].values)
                ioof['L12'][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=ra12)).fit(
                    il12p[trm.values],y).predict(il12p[tm.values])
                ioof['L33'][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=ra33)).fit(
                    il33p[trm.values],y).predict(il33p[tm.values])
                if extra_ridge_features is not None:
                    X_c_tr = idf.loc[trm, extra_ridge_features].fillna(0).values
                    X_c_te = idf.loc[tm, extra_ridge_features].fillna(0).values
                    ioof['COMPLEX'][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=10.0)).fit(X_c_tr,y).predict(X_c_te)

            nm = len(models)
            best_c, best_w = -1, tuple([1.0/nm]*nm)
            step = 0.05 if nm <= 3 else 0.1
            for w in iprod(np.arange(0, 1.0+step/2, step), repeat=nm):
                if abs(sum(w)-1.0) > step*0.5: continue
                b = sum(w[i]*norm(ioof[m]) for i,m in enumerate(models))
                cc = compute_cls(idf['active'].values, b, idf['pe_efficiency_pct'].values)['cls']
                if cc > best_c: best_c, best_w = cc, w

            yi = train.loc[im,'pe_efficiency_pct'].values
            pr = {}
            pr['EN'] = make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                ElasticNet(alpha=1.0,l1_ratio=0.3,max_iter=10000)).fit(
                train.loc[im,features_en].values,yi).predict(train.loc[om,features_en].values)
            pr['L12'] = make_pipeline(StandardScaler(),Ridge(alpha=ra12)).fit(il12p,yi).predict(ol12p)
            pr['L33'] = make_pipeline(StandardScaler(),Ridge(alpha=ra33)).fit(il33p,yi).predict(ol33p)
            if extra_ridge_features is not None:
                X_c_full = train.loc[im, extra_ridge_features].fillna(0).values
                X_c_out = train.loc[om, extra_ridge_features].fillna(0).values
                pr['COMPLEX'] = make_pipeline(StandardScaler(),Ridge(alpha=10.0)).fit(X_c_full,yi).predict(X_c_out)

            bl = np.zeros(om.sum())
            for i,m in enumerate(models):
                mn,mx = ioof[m].min(), ioof[m].max()
                bl += best_w[i] * ((pr[m]-mn)/max(mx-mn,1e-12))

            fd = train.loc[om,['rt_name','active','pe_efficiency_pct','rt_family']].copy()
            fd['predicted_score'] = bl; all_oof.append(fd)

        oof = pd.concat(all_oof).set_index('rt_name').loc[gt_order].reset_index()
        r = compute_cls(oof['active'].values, oof['predicted_score'].values, oof['pe_efficiency_pct'].values)
        return r, oof

    # --- Experiments ---
    print('\n--- Experiments ---')

    # Baseline
    r_base, _ = run_lofo_blend(FEATURES_BASE)
    print(f'Baseline v6:                      CLS={r_base["cls"]:.4f}  PR={r_base["pr_auc"]:.4f}  WSp={r_base["w_spearman"]:.4f}')

    # Complex-only (Ridge on all novel features)
    if genuinely_novel:
        r_cx, _ = run_lofo_blend(FEATURES_BASE, extra_ridge_features=genuinely_novel)
        print(f'v6 + complex (all novel):         CLS={r_cx["cls"]:.4f}  PR={r_cx["pr_auc"]:.4f}  WSp={r_cx["w_spearman"]:.4f}  delta={r_cx["cls"]-r_base["cls"]:+.4f}')

    # Top features by AUROC (genuinely novel only)
    novel_by_auroc = [(c, a) for c, a in good_feats if c in genuinely_novel]
    novel_by_auroc.sort(key=lambda x: -x[1])

    if len(novel_by_auroc) >= 3:
        top3 = [c for c, _ in novel_by_auroc[:3]]
        r_t3, _ = run_lofo_blend(FEATURES_BASE, extra_ridge_features=top3)
        print(f'v6 + top3 novel:                  CLS={r_t3["cls"]:.4f}  PR={r_t3["pr_auc"]:.4f}  WSp={r_t3["w_spearman"]:.4f}  delta={r_t3["cls"]-r_base["cls"]:+.4f}  {top3}')

    if len(novel_by_auroc) >= 5:
        top5 = [c for c, _ in novel_by_auroc[:5]]
        r_t5, _ = run_lofo_blend(FEATURES_BASE, extra_ridge_features=top5)
        print(f'v6 + top5 novel:                  CLS={r_t5["cls"]:.4f}  PR={r_t5["pr_auc"]:.4f}  WSp={r_t5["w_spearman"]:.4f}  delta={r_t5["cls"]-r_base["cls"]:+.4f}')

    # Individual features as 4th model
    print('\nIndividual novel features as 4th Ridge model:')
    for c in genuinely_novel:
        if train[c].isna().sum() > 20: continue
        r_i, _ = run_lofo_blend(FEATURES_BASE, extra_ridge_features=[c])
        delta = r_i['cls'] - r_base['cls']
        tag = ' ***' if delta > 0 else ''
        print(f'  +{c:<28s} CLS={r_i["cls"]:.4f}  delta={delta:+.4f}{tag}')

    # Per-family breakdown for best result
    all_results = {'baseline': r_base}
    if genuinely_novel:
        all_results['complex_all'] = r_cx
    best_name = max(all_results, key=lambda k: all_results[k]['cls'])

    print(f'\n--- SUMMARY ---')
    for n, r in sorted(all_results.items(), key=lambda x: -x[1]['cls']):
        print(f'  {n:<30s} CLS={r["cls"]:.4f}')
    print(f'\nBest: {best_name}  CLS={all_results[best_name]["cls"]:.4f}')

    if all_results[best_name]['cls'] > 0.7088:
        print('\n*** NEW RECORD ***')
    else:
        print(f'\nNo improvement over 0.7088.')

    print('\nDone.')


if __name__ == '__main__':
    main()
