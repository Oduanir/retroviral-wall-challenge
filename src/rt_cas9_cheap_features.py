#!/usr/bin/env python3
"""
RT-Cas9 compatibility features (cheap version).

Computes 10 features approximating RT fusion compatibility
with the prime editor complex, without full docking.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw"
STRUCT_DIR = DATA_DIR / "structures"
PROCESSED_DIR = ROOT / "data" / "processed"

import sys
sys.path.insert(0, str(ROOT))
from src.esm2_features import find_yxdd


def _load_ca_coords(pdb_path):
    """Load CA atom coordinates from PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rt", pdb_path)
    cas = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    cas.append(residue["CA"].get_vector().get_array())
    return np.array(cas)


def _foldseek_local_tm(ca_query, ca_ref, yxdd_pos_query, yxdd_pos_ref, window=30):
    """
    Approximate local TM-score around YXDD using CA RMSD as proxy.
    Returns 1/(1 + RMSD) as a similarity score in [0,1].
    Not a real TM-score but a cheap structural similarity proxy.
    """
    if yxdd_pos_query is None or yxdd_pos_ref is None:
        return np.nan

    # Extract local windows
    n_q = len(ca_query)
    n_r = len(ca_ref)
    start_q = max(0, yxdd_pos_query - window)
    end_q = min(n_q, yxdd_pos_query + window)
    start_r = max(0, yxdd_pos_ref - window)
    end_r = min(n_r, yxdd_pos_ref + window)

    local_q = ca_query[start_q:end_q]
    local_r = ca_ref[start_r:end_r]

    # Truncate to same length
    min_len = min(len(local_q), len(local_r))
    if min_len < 4:
        return np.nan

    local_q = local_q[:min_len]
    local_r = local_r[:min_len]

    # Center both
    local_q = local_q - local_q.mean(axis=0)
    local_r = local_r - local_r.mean(axis=0)

    # Simple RMSD (no alignment — cheap proxy)
    rmsd = np.sqrt(((local_q - local_r) ** 2).sum(axis=1).mean())
    return 1.0 / (1.0 + rmsd)


def extract_rt_cas9_features():
    """
    Extract 10 RT-Cas9 compatibility features for all 57 RTs.

    Returns pd.DataFrame indexed by rt_name.
    """
    seqs_df = pd.read_csv(DATA_DIR / "rt_sequences.csv")
    train = pd.read_csv(DATA_DIR / "train.csv")
    sequences = dict(zip(seqs_df["rt_name"], seqs_df["sequence"]))

    # Reference: MMLV-RT
    mmlv_seq = sequences.get("MMLV-RT")
    mmlv_yxdd = find_yxdd(mmlv_seq) if mmlv_seq else None
    mmlv_pdb = STRUCT_DIR / "MMLV-RT.pdb"
    mmlv_ca = _load_ca_coords(mmlv_pdb) if mmlv_pdb.exists() else None

    # Also get "best" PE reference — MMLV-PE if available, else MMLV
    best_ref_name = "MMLVPE-RT" if "MMLVPE-RT" in sequences else "MMLV-RT"
    best_ref_seq = sequences.get(best_ref_name, mmlv_seq)
    best_ref_yxdd = find_yxdd(best_ref_seq) if best_ref_seq else None
    best_ref_pdb = STRUCT_DIR / f"{best_ref_name}.pdb"
    best_ref_ca = _load_ca_coords(best_ref_pdb) if best_ref_pdb.exists() else mmlv_ca

    # Get existing FoldSeek TM-scores for global comparison
    foldseek_global_mmlv = dict(zip(train["rt_name"], train["foldseek_TM_MMLV"]))
    foldseek_global_best = dict(zip(train["rt_name"], train["foldseek_best_TM"]))

    results = []

    for rt_name, seq in sequences.items():
        yxdd_pos = find_yxdd(seq)
        seq_len = len(seq)
        feats = {"rt_name": rt_name}

        # === Feature 1: nterm_to_yxdd_frac ===
        if yxdd_pos is not None:
            feats["nterm_to_yxdd_frac"] = yxdd_pos / seq_len
        else:
            feats["nterm_to_yxdd_frac"] = np.nan

        # === Feature 2: cterm_to_yxdd_frac ===
        if yxdd_pos is not None:
            feats["cterm_to_yxdd_frac"] = (seq_len - yxdd_pos - 4) / seq_len
        else:
            feats["cterm_to_yxdd_frac"] = np.nan

        # === Feature 3: yxdd_surface_proxy ===
        # Use structure: distance from YXDD centroid to protein centroid,
        # normalized by Rg. Higher = more surface-exposed.
        pdb_path = STRUCT_DIR / f"{rt_name}.pdb"
        ca_coords = None
        if pdb_path.exists():
            try:
                ca_coords = _load_ca_coords(pdb_path)
            except Exception:
                pass

        if ca_coords is not None and yxdd_pos is not None and len(ca_coords) > yxdd_pos + 3:
            centroid = ca_coords.mean(axis=0)
            yxdd_centroid = ca_coords[yxdd_pos:yxdd_pos+4].mean(axis=0)
            dist_to_center = np.linalg.norm(yxdd_centroid - centroid)
            rg = np.sqrt(((ca_coords - centroid) ** 2).sum(axis=1).mean())
            feats["yxdd_surface_proxy"] = dist_to_center / max(rg, 1e-6)
        else:
            feats["yxdd_surface_proxy"] = np.nan

        # === Feature 4: yxdd_local_compactness ===
        # Mean pairwise distance of residues within 15 residues of YXDD
        if ca_coords is not None and yxdd_pos is not None:
            start = max(0, yxdd_pos - 15)
            end = min(len(ca_coords), yxdd_pos + 4 + 15)
            local_ca = ca_coords[start:end]
            if len(local_ca) > 2:
                pw = pdist(local_ca)
                feats["yxdd_local_compactness"] = pw.mean()
            else:
                feats["yxdd_local_compactness"] = np.nan
        else:
            feats["yxdd_local_compactness"] = np.nan

        # === Feature 5: global_vs_local_mmlv_gap ===
        # Global FoldSeek TM to MMLV minus local structural similarity around YXDD
        global_tm_mmlv = foldseek_global_mmlv.get(rt_name, np.nan)
        if ca_coords is not None and mmlv_ca is not None:
            local_sim = _foldseek_local_tm(ca_coords, mmlv_ca, yxdd_pos, mmlv_yxdd)
            feats["global_vs_local_mmlv_gap"] = global_tm_mmlv - local_sim if not np.isnan(local_sim) else np.nan
        else:
            feats["global_vs_local_mmlv_gap"] = np.nan

        # === Feature 6: global_vs_local_best_gap ===
        global_tm_best = foldseek_global_best.get(rt_name, np.nan)
        if ca_coords is not None and best_ref_ca is not None:
            local_sim_best = _foldseek_local_tm(ca_coords, best_ref_ca, yxdd_pos, best_ref_yxdd)
            feats["global_vs_local_best_gap"] = global_tm_best - local_sim_best if not np.isnan(local_sim_best) else np.nan
        else:
            feats["global_vs_local_best_gap"] = np.nan

        # === Feature 7: rnaseh_present_proxy ===
        # RNase H domain typically adds 120-150 residues after the polymerase domain.
        # If YXDD is far from C-term (large cterm_to_yxdd_frac), likely has RNase H.
        if yxdd_pos is not None:
            cterm_residues = seq_len - yxdd_pos - 4
            feats["rnaseh_present_proxy"] = 1.0 if cterm_residues > 150 else cterm_residues / 150.0
        else:
            feats["rnaseh_present_proxy"] = np.nan

        # === Feature 8: fusion_burden_proxy ===
        # Length × inverse compactness (elongated large proteins are worse for fusion)
        if ca_coords is not None:
            centroid = ca_coords.mean(axis=0)
            rg = np.sqrt(((ca_coords - centroid) ** 2).sum(axis=1).mean())
            feats["fusion_burden_proxy"] = seq_len * rg / 1000.0  # normalized
        else:
            feats["fusion_burden_proxy"] = np.nan

        # === Feature 9: yxdd_confidence_mean ===
        # Mean B-factor (pLDDT for AlphaFold) around YXDD
        if pdb_path.exists() and yxdd_pos is not None:
            try:
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure("rt", pdb_path)
                bfactors = []
                for model in structure:
                    for chain in model:
                        residues = list(chain.get_residues())
                        start = max(0, yxdd_pos - 10)
                        end = min(len(residues), yxdd_pos + 4 + 10)
                        for res in residues[start:end]:
                            for atom in res:
                                bfactors.append(atom.get_bfactor())
                feats["yxdd_confidence_mean"] = np.mean(bfactors) if bfactors else np.nan
            except Exception:
                feats["yxdd_confidence_mean"] = np.nan
        else:
            feats["yxdd_confidence_mean"] = np.nan

        # === Feature 10: termini_asymmetry_to_core ===
        # |nterm_to_yxdd - cterm_to_yxdd| / seq_len — how off-center is the catalytic core
        if yxdd_pos is not None:
            nterm_dist = yxdd_pos
            cterm_dist = seq_len - yxdd_pos - 4
            feats["termini_asymmetry_to_core"] = abs(nterm_dist - cterm_dist) / seq_len
        else:
            feats["termini_asymmetry_to_core"] = np.nan

        results.append(feats)

    df = pd.DataFrame(results)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "rt_cas9_cheap_features.csv", index=False)
    print(f"Saved {len(df)} RTs × {len(df.columns)-1} features to {PROCESSED_DIR / 'rt_cas9_cheap_features.csv'}")
    return df


if __name__ == "__main__":
    df = extract_rt_cas9_features()
    print(df.set_index("rt_name").describe().round(3))
