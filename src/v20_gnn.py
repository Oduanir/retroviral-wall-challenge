#!/usr/bin/env python3
"""
V20 — 3D Graph Neural Network on RT Structure.

Two views (full graph + active-site subgraph), multi-task (cls + reg),
strict nested LOFO, leak-free blend with v6.
"""
import numpy as np, pandas as pd, torch, warnings, os
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
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.metrics import compute_cls
from src.esm2_features import find_yxdd

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import NNConv, global_mean_pool

DATA_DIR = ROOT / "data" / "raw"
STRUCT_DIR = DATA_DIR / "structures"

# Kyte-Doolittle hydrophobicity scale
HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
         'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
         'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
CHARGE = {'R':1,'K':1,'D':-1,'E':-1,'H':0.5}
AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY'


# ===========================================================================
# GRAPH CONSTRUCTION
# ===========================================================================

def build_graph(pdb_path, sequence, contact_threshold=10.0):
    """Build a PyG Data object from a PDB structure."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    chain = list(structure[0].get_chains())[0]

    # Extract CA coords and B-factors
    cas, bfactors, resnames = [], [], []
    for res in chain:
        if "CA" in res:
            cas.append(res["CA"].get_vector().get_array())
            bfactors.append(res["CA"].get_bfactor())
            resnames.append(res.get_resname())
    cas = np.array(cas)
    n_res = len(cas)
    if n_res < 5:
        return None, None

    # Node features (~25D)
    yxdd_pos = find_yxdd(sequence)
    node_feats = []
    for i in range(n_res):
        # One-hot AA (20D)
        aa_char = sequence[i] if i < len(sequence) else 'A'
        onehot = [1.0 if aa_char == a else 0.0 for a in AA_ORDER]
        # pLDDT (1D)
        plddt = bfactors[i] if i < len(bfactors) else 0.5
        # Relative position (1D)
        rel_pos = i / max(n_res - 1, 1)
        # YXDD indicator (1D)
        is_yxdd = 1.0 if (yxdd_pos is not None and yxdd_pos <= i < yxdd_pos + 4) else 0.0
        # Hydrophobicity (1D)
        hydro = HYDRO.get(aa_char, 0.0) / 4.5  # normalize
        # Charge (1D)
        charge = CHARGE.get(aa_char, 0.0)

        node_feats.append(onehot + [plddt, rel_pos, is_yxdd, hydro, charge])

    x = torch.tensor(node_feats, dtype=torch.float32)  # [n_res, 25]

    # Edges: CA-CA distance < threshold
    dists = cdist(cas, cas)
    src, dst, edge_attr_list = [], [], []
    for i in range(n_res):
        for j in range(i+1, n_res):
            if dists[i, j] < contact_threshold:
                src.extend([i, j])
                dst.extend([j, i])
                d = dists[i, j] / contact_threshold  # normalized distance
                seq_sep = abs(i - j) / max(n_res - 1, 1)  # normalized seq separation
                edge_attr_list.extend([[d, seq_sep], [d, seq_sep]])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

    # Full graph
    full_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                      num_nodes=n_res)

    # Active-site subgraph: residues within 15Å of YXDD centroid
    if yxdd_pos is not None and yxdd_pos + 4 <= n_res:
        yxdd_centroid = cas[yxdd_pos:yxdd_pos+4].mean(axis=0)
        dists_to_yxdd = np.linalg.norm(cas - yxdd_centroid, axis=1)
        site_mask = dists_to_yxdd < 15.0
        site_indices = np.where(site_mask)[0]

        if len(site_indices) > 5:
            # Remap indices
            idx_map = {old: new for new, old in enumerate(site_indices)}
            site_x = x[site_indices]
            site_src, site_dst, site_ea = [], [], []
            for k in range(0, len(src), 2):  # iterate original edges
                i, j = src[k], dst[k]
                if i in idx_map and j in idx_map:
                    ni, nj = idx_map[i], idx_map[j]
                    site_src.extend([ni, nj])
                    site_dst.extend([nj, ni])
                    site_ea.extend([edge_attr_list[k], edge_attr_list[k]])

            if site_src:
                site_graph = Data(
                    x=site_x,
                    edge_index=torch.tensor([site_src, site_dst], dtype=torch.long),
                    edge_attr=torch.tensor(site_ea, dtype=torch.float32),
                    num_nodes=len(site_indices)
                )
            else:
                site_graph = None
        else:
            site_graph = None
    else:
        site_graph = None

    return full_graph, site_graph


# ===========================================================================
# GNN MODEL
# ===========================================================================

class RTGraphNet(nn.Module):
    def __init__(self, node_dim=25, edge_dim=2, hidden=32, dropout=0.3):
        super().__init__()
        # NNConv: edge-conditioned convolution that consumes edge_attr
        # edge network maps edge_dim -> node_dim * hidden (weight matrix per edge)
        self.edge_nn1_full = nn.Sequential(nn.Linear(edge_dim, node_dim * hidden))
        self.conv1_full = NNConv(node_dim, hidden, self.edge_nn1_full, aggr='mean')
        self.edge_nn2_full = nn.Sequential(nn.Linear(edge_dim, hidden * hidden))
        self.conv2_full = NNConv(hidden, hidden, self.edge_nn2_full, aggr='mean')

        self.edge_nn1_site = nn.Sequential(nn.Linear(edge_dim, node_dim * hidden))
        self.conv1_site = NNConv(node_dim, hidden, self.edge_nn1_site, aggr='mean')
        self.edge_nn2_site = nn.Sequential(nn.Linear(edge_dim, hidden * hidden))
        self.conv2_site = NNConv(hidden, hidden, self.edge_nn2_site, aggr='mean')

        # Heads
        self.head_cls = nn.Linear(hidden * 2, 1)
        self.head_reg = nn.Linear(hidden * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self._hidden = hidden

    def _conv_branch(self, x, edge_index, edge_attr, conv1, conv2):
        x = F.relu(conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = F.relu(conv2(x, edge_index, edge_attr))
        x = self.dropout(x)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(x, batch)

    def forward(self, full_graph, site_graph=None):
        emb_full = self._conv_branch(
            full_graph.x, full_graph.edge_index, full_graph.edge_attr,
            self.conv1_full, self.conv2_full)

        if site_graph is not None and site_graph.num_nodes > 0 and site_graph.edge_index.numel() > 0:
            emb_site = self._conv_branch(
                site_graph.x, site_graph.edge_index, site_graph.edge_attr,
                self.conv1_site, self.conv2_site)
        else:
            emb_site = torch.zeros(1, self._hidden, device=emb_full.device)

        emb = torch.cat([emb_full, emb_site], dim=1)
        logit_cls = self.head_cls(emb)
        pred_reg = self.head_reg(emb)
        return logit_cls.squeeze(), pred_reg.squeeze(), emb.squeeze()


# ===========================================================================
# DEVICE
# ===========================================================================

def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def _graph_to_device(g, device):
    """Move a PyG Data object to device."""
    if g is None:
        return None
    return Data(x=g.x.to(device), edge_index=g.edge_index.to(device),
                edge_attr=g.edge_attr.to(device), num_nodes=g.num_nodes)


# ===========================================================================
# TRAINING
# ===========================================================================

def train_gnn(graphs_train, labels_active, labels_eff, n_epochs=80, lr=1e-3,
              wd=1e-2, alpha=0.5, patience=15, device=None):
    """Train GNN on training set, return trained model."""
    if device is None:
        device = _get_device()
    model = RTGraphNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    best_loss = float('inf')
    patience_count = 0

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        indices = list(range(len(graphs_train)))
        np.random.shuffle(indices)
        optimizer.zero_grad()

        for step, idx in enumerate(indices):
            full_g, site_g = graphs_train[idx]
            if full_g is None:
                continue
            full_g_d = _graph_to_device(full_g, device)
            site_g_d = _graph_to_device(site_g, device)
            logit_cls, pred_reg, _ = model(full_g_d, site_g_d)
            target_cls = torch.tensor(float(labels_active[idx]), device=device)
            target_reg = torch.tensor(float(labels_eff[idx]), device=device)
            loss = alpha * bce(logit_cls, target_cls) + (1-alpha) * mse(pred_reg, target_reg)
            loss = loss / 4
            loss.backward()
            total_loss += loss.item() * 4

            if (step + 1) % 4 == 0 or step == len(indices) - 1:
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = total_loss / max(len(indices), 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    return model


def predict_gnn(model, graphs, beta=0.5, device=None):
    """Predict scores for a list of graphs. Returns combined score."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    scores = []
    p_actives = []
    for full_g, site_g in graphs:
        if full_g is None:
            scores.append(0.0)
            p_actives.append(0.0)
            continue
        full_g_d = _graph_to_device(full_g, device)
        site_g_d = _graph_to_device(site_g, device)
        with torch.no_grad():
            logit_cls, pred_reg, _ = model(full_g_d, site_g_d)
            scores.append(pred_reg.item())
            p_actives.append(torch.sigmoid(logit_cls).item())
    scores = np.array(scores)
    p_actives = np.array(p_actives)

    # Additive score: β * norm(reg) + (1-β) * norm(p_active)
    def norm(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn) if mx - mn > 1e-12 else np.zeros_like(a)

    return beta * norm(scores) + (1-beta) * norm(p_actives)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print('='*70)
    print('V20 — 3D Graph Neural Network on RT Structure')
    print('='*70)

    # Load data
    train = pd.read_csv(DATA_DIR / 'train.csv')
    splits_df = pd.read_csv(DATA_DIR / 'family_splits.csv')
    splits = {}
    for _, row in splits_df.iterrows():
        splits[row['family']] = row['rt_names'].split('|')
    gt_order = pd.read_csv(DATA_DIR / 'rt_sequences.csv', usecols=['rt_name'])['rt_name'].tolist()
    sequences = dict(zip(train['rt_name'], train['sequence']))

    # Build all graphs
    print('\n[1] Building graphs from PDB structures...')
    all_graphs = {}
    for rt_name, seq in sequences.items():
        pdb_path = STRUCT_DIR / f"{rt_name}.pdb"
        if pdb_path.exists():
            full_g, site_g = build_graph(pdb_path, seq)
            all_graphs[rt_name] = (full_g, site_g)
        else:
            all_graphs[rt_name] = (None, None)
    n_ok = sum(1 for g in all_graphs.values() if g[0] is not None)
    print(f'  {n_ok}/57 graphs built successfully')

    # Labels
    active_dict = dict(zip(train['rt_name'], train['active']))
    eff_dict = dict(zip(train['rt_name'], train['pe_efficiency_pct']))
    rt_family = dict(zip(train['rt_name'], train['rt_family']))

    def norm(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn) if mx - mn > 1e-12 else np.zeros_like(a)

    # ===================================================================
    # GNN STANDALONE (nested LOFO with β optimized per outer fold)
    # ===================================================================
    print('\n[2] GNN Standalone (nested LOFO, β nested)...')

    all_oof_standalone = []
    for outer_family in splits:
        outer_rts = splits[outer_family]
        inner_splits = {f: rts for f, rts in splits.items() if f != outer_family}
        train_rts = [rt for rts in inner_splits.values() for rt in rts]

        # Inner LOFO to find best β
        best_beta_inner, best_cls_inner = 0.5, -1
        for beta_candidate in [0.3, 0.5, 0.7]:
            inner_oof_scores = np.full(len(train_rts), np.nan)
            rt_to_idx = {rt: i for i, rt in enumerate(train_rts)}
            for inner_held_fam in inner_splits:
                inner_held_rts = inner_splits[inner_held_fam]
                inner_train_rts = [rt for f, rts in inner_splits.items() if f != inner_held_fam for rt in rts]
                g_tr = [all_graphs[rt] for rt in inner_train_rts]
                la = [active_dict[rt] for rt in inner_train_rts]
                le = [eff_dict[rt] for rt in inner_train_rts]
                m = train_gnn(g_tr, la, le, alpha=0.5)
                g_held = [all_graphs[rt] for rt in inner_held_rts]
                s_held = predict_gnn(m, g_held, beta=beta_candidate)
                for j, rt in enumerate(inner_held_rts):
                    inner_oof_scores[rt_to_idx[rt]] = s_held[j]
                del m
            inner_active = np.array([active_dict[rt] for rt in train_rts])
            inner_eff = np.array([eff_dict[rt] for rt in train_rts])
            cc = compute_cls(inner_active, inner_oof_scores, inner_eff)['cls']
            if cc > best_cls_inner:
                best_cls_inner, best_beta_inner = cc, beta_candidate

        # Train on all inner families with best β, predict outer
        graphs_tr = [all_graphs[rt] for rt in train_rts]
        labels_a = [active_dict[rt] for rt in train_rts]
        labels_e = [eff_dict[rt] for rt in train_rts]
        model = train_gnn(graphs_tr, labels_a, labels_e, alpha=0.5)

        graphs_out = [all_graphs[rt] for rt in outer_rts]
        scores = predict_gnn(model, graphs_out, beta=best_beta_inner)

        for i, rt in enumerate(outer_rts):
            all_oof_standalone.append({
                'rt_name': rt, 'active': active_dict[rt],
                'pe_efficiency_pct': eff_dict[rt], 'rt_family': rt_family[rt],
                'predicted_score': scores[i]
            })
        print(f'  {outer_family:<25s} β={best_beta_inner}  inner_cls={best_cls_inner:.4f}')
        del model

    oof_standalone = pd.DataFrame(all_oof_standalone).set_index('rt_name').loc[gt_order].reset_index()
    r_standalone = compute_cls(oof_standalone['active'].values, oof_standalone['predicted_score'].values,
                               oof_standalone['pe_efficiency_pct'].values)
    print(f'  GNN standalone: CLS={r_standalone["cls"]:.4f}  PR={r_standalone["pr_auc"]:.4f}  WSp={r_standalone["w_spearman"]:.4f}')

    # ===================================================================
    # GNN + V6 BLEND (strict nested LOFO, leak-free)
    # ===================================================================
    print('\n[3] GNN + v6 blend (nested LOFO, leak-free)...')

    # Prepare v6 data
    for col in ['connection_mean_pot','triad_best_rmsd','D1_D2_dist','D2_D3_dist',
                'yxdd_hydrophobic_fraction','yxdd_mean_hydrophobicity','yxdd_5A_mean_pot']:
        train[f'{col}_missing'] = train[col].isna().astype(int)
    train['foldseek_gap_MMLV'] = train['foldseek_best_TM'] - train['foldseek_TM_MMLV']
    train['t40_x_foldseek_MMLV'] = train['t40_raw'] * train['foldseek_TM_MMLV']
    train['triad_quality'] = train['triad_found_bin'] * (1 / (train['triad_best_rmsd'].fillna(99) + 1))
    train['seq_struct_compat'] = -train['perplexity'] * train['instability_index']

    print('  Loading ESM2...')
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
    print('  ESM2 done.')

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

    # v6 baseline
    def run_v6_fold(train_df, train_mask, test_mask, inner_splits):
        """Run v6 inner LOFO, return OOF scores for train set and predictions for test set."""
        idf = train_df[train_mask].reset_index(drop=True)
        n = len(idf)
        il12 = np.array([l12_raw[r] for r in idf['rt_name']])
        il33 = np.array([l33_raw[r] for r in idf['rt_name']])
        p12 = PCA(3).fit(il12); p33 = PCA(3).fit(il33)
        il12p = p12.transform(il12); il33p = p33.transform(il33)
        ol12p = p12.transform(np.array([l12_raw[r] for r in train_df.loc[test_mask,'rt_name']]))
        ol33p = p33.transform(np.array([l33_raw[r] for r in train_df.loc[test_mask,'rt_name']]))

        ioof = {'EN': np.full(n, np.nan), 'L12': np.full(n, np.nan), 'L33': np.full(n, np.nan)}
        for ifam, irts in inner_splits.items():
            tm = idf['rt_name'].isin(irts); trm = ~tm
            y = idf.loc[trm, 'pe_efficiency_pct'].values
            ioof['EN'][tm.values] = make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                ElasticNet(alpha=1.0,l1_ratio=0.3,max_iter=10000)).fit(
                idf.loc[trm,FEATURES_BASE].values,y).predict(idf.loc[tm,FEATURES_BASE].values)
            ioof['L12'][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=12.5)).fit(
                il12p[trm.values],y).predict(il12p[tm.values])
            ioof['L33'][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=7.5)).fit(
                il33p[trm.values],y).predict(il33p[tm.values])

        best_c, best_w = -1, (0.33,0.33,0.34)
        for w in iprod(np.arange(0,1.05,0.05), repeat=3):
            if abs(sum(w)-1.0) > 0.025: continue
            b = w[0]*norm(ioof['EN']) + w[1]*norm(ioof['L12']) + w[2]*norm(ioof['L33'])
            cc = compute_cls(idf['active'].values, b, idf['pe_efficiency_pct'].values)['cls']
            if cc > best_c: best_c, best_w = cc, w

        inner_blend = best_w[0]*norm(ioof['EN']) + best_w[1]*norm(ioof['L12']) + best_w[2]*norm(ioof['L33'])

        # Outer prediction
        yi = train_df.loc[train_mask,'pe_efficiency_pct'].values
        pr_en = make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
            ElasticNet(alpha=1.0,l1_ratio=0.3,max_iter=10000)).fit(
            train_df.loc[train_mask,FEATURES_BASE].values,yi).predict(train_df.loc[test_mask,FEATURES_BASE].values)
        pr_l12 = make_pipeline(StandardScaler(),Ridge(alpha=12.5)).fit(il12p,yi).predict(ol12p)
        pr_l33 = make_pipeline(StandardScaler(),Ridge(alpha=7.5)).fit(il33p,yi).predict(ol33p)
        oen = (pr_en-ioof['EN'].min())/max(ioof['EN'].max()-ioof['EN'].min(),1e-12)
        ol12 = (pr_l12-ioof['L12'].min())/max(ioof['L12'].max()-ioof['L12'].min(),1e-12)
        ol33 = (pr_l33-ioof['L33'].min())/max(ioof['L33'].max()-ioof['L33'].min(),1e-12)
        outer_blend = best_w[0]*oen + best_w[1]*ol12 + best_w[2]*ol33

        return inner_blend, outer_blend

    # Full nested LOFO with GNN + v6 blend
    all_oof_blend = []
    beta = 0.5  # fixed for GNN score construction in blend; β was nested in standalone

    for outer_family in splits:
        outer_rts = splits[outer_family]
        om = train['rt_name'].isin(outer_rts); im = ~om
        inner_splits = {f: rts for f, rts in splits.items() if f != outer_family}
        inner_families = list(inner_splits.keys())

        # --- Inner LOFO for GNN OOF scores ---
        inner_rts_all = [rt for rts in inner_splits.values() for rt in rts]
        gnn_inner_oof = np.full(len(inner_rts_all), np.nan)
        inner_rt_to_idx = {rt: i for i, rt in enumerate(inner_rts_all)}

        for inner_held_family in inner_families:
            inner_held_rts = inner_splits[inner_held_family]
            inner_train_rts = [rt for f, rts in inner_splits.items() if f != inner_held_family for rt in rts]

            graphs_itr = [all_graphs[rt] for rt in inner_train_rts]
            labels_a_itr = [active_dict[rt] for rt in inner_train_rts]
            labels_e_itr = [eff_dict[rt] for rt in inner_train_rts]

            model_inner = train_gnn(graphs_itr, labels_a_itr, labels_e_itr, alpha=0.5)
            graphs_held = [all_graphs[rt] for rt in inner_held_rts]
            scores_held = predict_gnn(model_inner, graphs_held, beta=beta)

            for j, rt in enumerate(inner_held_rts):
                gnn_inner_oof[inner_rt_to_idx[rt]] = scores_held[j]
            del model_inner

        # --- v6 inner OOF ---
        v6_inner_oof, v6_outer = run_v6_fold(train, im, om, inner_splits)

        # --- GNN outer prediction ---
        train_rts_outer = [rt for rts in inner_splits.values() for rt in rts]
        graphs_tr_outer = [all_graphs[rt] for rt in train_rts_outer]
        labels_a_outer = [active_dict[rt] for rt in train_rts_outer]
        labels_e_outer = [eff_dict[rt] for rt in train_rts_outer]
        model_outer = train_gnn(graphs_tr_outer, labels_a_outer, labels_e_outer, alpha=0.5)
        graphs_test = [all_graphs[rt] for rt in outer_rts]
        gnn_outer = predict_gnn(model_outer, graphs_test, beta=beta)
        del model_outer

        # --- Optimize blend weight on inner OOF ---
        inner_active = np.array([active_dict[rt] for rt in inner_rts_all])
        inner_eff = np.array([eff_dict[rt] for rt in inner_rts_all])

        best_w_blend, best_cls_blend = 1.0, -1
        for w_v6 in np.arange(0, 1.05, 0.05):
            w_gnn = 1.0 - w_v6
            blend_inner = w_v6 * norm(v6_inner_oof) + w_gnn * norm(gnn_inner_oof)
            cc = compute_cls(inner_active, blend_inner, inner_eff)['cls']
            if cc > best_cls_blend:
                best_cls_blend = cc
                best_w_blend = w_v6

        # --- Outer blend ---
        outer_score = best_w_blend * norm(v6_outer) + (1-best_w_blend) * norm(gnn_outer)

        for i, rt in enumerate(outer_rts):
            all_oof_blend.append({
                'rt_name': rt, 'active': active_dict[rt],
                'pe_efficiency_pct': eff_dict[rt], 'rt_family': rt_family[rt],
                'predicted_score': outer_score[i]
            })

        print(f'  {outer_family:<25s} w_v6={best_w_blend:.2f}  inner_cls={best_cls_blend:.4f}')

    oof_blend = pd.DataFrame(all_oof_blend).set_index('rt_name').loc[gt_order].reset_index()
    r_blend = compute_cls(oof_blend['active'].values, oof_blend['predicted_score'].values,
                           oof_blend['pe_efficiency_pct'].values)

    # v6 baseline for comparison (same script)
    print('\n  Running v6 baseline...')
    all_oof_v6 = []
    for outer_family in splits:
        outer_rts = splits[outer_family]
        om = train['rt_name'].isin(outer_rts); im = ~om
        inner_splits = {f: rts for f, rts in splits.items() if f != outer_family}
        _, v6_outer = run_v6_fold(train, im, om, inner_splits)
        for i, rt in enumerate(outer_rts):
            all_oof_v6.append({
                'rt_name': rt, 'active': active_dict[rt],
                'pe_efficiency_pct': eff_dict[rt], 'rt_family': rt_family[rt],
                'predicted_score': v6_outer[i]
            })
    oof_v6 = pd.DataFrame(all_oof_v6).set_index('rt_name').loc[gt_order].reset_index()
    r_v6 = compute_cls(oof_v6['active'].values, oof_v6['predicted_score'].values,
                        oof_v6['pe_efficiency_pct'].values)

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print(f'\n{"="*70}')
    print('SUMMARY')
    print('='*70)
    print(f'  v6 baseline:        CLS={r_v6["cls"]:.4f}  PR={r_v6["pr_auc"]:.4f}  WSp={r_v6["w_spearman"]:.4f}')
    print(f'  GNN standalone:     CLS={r_standalone["cls"]:.4f}  PR={r_standalone["pr_auc"]:.4f}  WSp={r_standalone["w_spearman"]:.4f}')
    print(f'  GNN + v6 blend:     CLS={r_blend["cls"]:.4f}  PR={r_blend["pr_auc"]:.4f}  WSp={r_blend["w_spearman"]:.4f}  delta={r_blend["cls"]-r_v6["cls"]:+.4f}')

    if r_blend['cls'] > r_v6['cls']:
        print(f'\n  *** GNN + v6 improves over v6 by {r_blend["cls"]-r_v6["cls"]:+.4f} ***')
    if r_blend['cls'] > 0.7088:
        print(f'\n  *** NEW RECORD: CLS={r_blend["cls"]:.4f} ***')
        oof_blend[['rt_name','predicted_score']].to_csv(ROOT / 'submissions/submission_v20_gnn.csv', index=False)
        from src.bootstrap import bootstrap_cls, print_bootstrap_results
        boot = bootstrap_cls(oof_blend, n_bootstrap=10000)
        print(); print_bootstrap_results(boot)
    else:
        print(f'\n  No improvement over 0.7088.')

    print('\nDone.')


if __name__ == '__main__':
    main()
