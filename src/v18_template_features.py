#!/usr/bin/env python3
"""
V18 — Template-based RT-Cas9 compatibility features.

Places each candidate RT into the PE complex (PDB 8WUS) by structural
alignment onto the MMLV-RT chain, then extracts compatibility features.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
import tmtools
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.esm2_features import find_yxdd

DATA_DIR = ROOT / "data" / "raw"
STRUCT_DIR = DATA_DIR / "structures"
TEMPLATE_DIR = DATA_DIR / "templates"
PROCESSED_DIR = ROOT / "data" / "processed"

# 8WUS chain mapping (verified):
# Chain E: MMLV-RT (453 residues, protein)
# Chain A: SpCas9 (1313 residues, protein)
# Chain B: pegRNA (115 nt, RNA)
# Chain C: target DNA strand 1
# Chain D: target DNA strand 2
# Chain F: target DNA strand 3
CHAIN_RT = "E"
CHAIN_CAS9 = "A"
CHAIN_PEGRNA = "B"
CHAIN_DNA = ["C", "D", "F"]


def _get_ca_coords(structure, chain_id):
    """Get CA coordinates from a specific chain."""
    coords = []
    for model in structure:
        chain = model[chain_id]
        for res in chain:
            if "CA" in res:
                coords.append(res["CA"].get_vector().get_array())
    return np.array(coords) if coords else None


def _get_backbone_coords(structure, chain_ids):
    """Get backbone atom coordinates (P for nucleic acids) from chains."""
    coords = []
    for model in structure:
        for cid in chain_ids:
            if cid not in model:
                continue
            for res in model[cid]:
                for atom_name in ["P", "C3'", "C4'"]:  # nucleic acid backbone
                    if atom_name in res:
                        coords.append(res[atom_name].get_vector().get_array())
                        break
    return np.array(coords) if coords else None


def _load_structure(pdb_path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure("s", pdb_path)


def _tm_align(query_ca, ref_ca, query_seq, ref_seq):
    """
    Structural alignment using tmtools (Python TM-align).
    tmtools.tm_align expects: (coords1, coords2, seq1, seq2)
    where coords are float32 arrays and seqs are strings.
    """
    result = tmtools.tm_align(
        query_ca.astype(np.float32),
        ref_ca.astype(np.float32),
        query_seq,
        ref_seq,
    )
    return result


def extract_template_features():
    """
    Extract ~12 template-based compatibility features for all 57 RTs.
    """
    # Load template complex
    template_path = TEMPLATE_DIR / "8WUS.pdb"
    if not template_path.exists():
        raise FileNotFoundError(f"Template {template_path} not found. Download from RCSB first.")

    template = _load_structure(template_path)
    ref_rt_ca = _get_ca_coords(template, CHAIN_RT)
    cas9_ca = _get_ca_coords(template, CHAIN_CAS9)
    pegrna_backbone = _get_backbone_coords(template, [CHAIN_PEGRNA])
    dna_backbone = _get_backbone_coords(template, CHAIN_DNA)

    print(f"Template 8WUS loaded:")
    print(f"  MMLV-RT (chain {CHAIN_RT}): {len(ref_rt_ca)} CA atoms")
    print(f"  Cas9 (chain {CHAIN_CAS9}): {len(cas9_ca)} CA atoms")
    print(f"  pegRNA backbone: {len(pegrna_backbone) if pegrna_backbone is not None else 0} atoms")
    print(f"  DNA backbone: {len(dna_backbone) if dna_backbone is not None else 0} atoms")

    # Sequences for TMalign
    seqs_df = pd.read_csv(DATA_DIR / "rt_sequences.csv")
    sequences = dict(zip(seqs_df["rt_name"], seqs_df["sequence"]))
    mmlv_seq = sequences.get("MMLV-RT", "")
    ref_yxdd_pos = find_yxdd(mmlv_seq)
    # Reference sequence truncated to match structure residue count
    ref_seq_for_align = mmlv_seq[:len(ref_rt_ca)] if len(mmlv_seq) >= len(ref_rt_ca) else mmlv_seq
    print(f"  MMLV YXDD position: {ref_yxdd_pos}")
    print(f"  Ref seq for align: {len(ref_seq_for_align)} aa (structure has {len(ref_rt_ca)} CA)")

    # Reference YXDD centroid in complex space
    if ref_yxdd_pos is not None and ref_yxdd_pos + 4 <= len(ref_rt_ca):
        ref_yxdd_centroid = ref_rt_ca[ref_yxdd_pos:ref_yxdd_pos+4].mean(axis=0)
    else:
        ref_yxdd_centroid = None

    results = []

    for rt_name, seq in sequences.items():
        feats = {"rt_name": rt_name}
        yxdd_pos = find_yxdd(seq)

        # Load candidate structure
        pdb_path = STRUCT_DIR / f"{rt_name}.pdb"
        if not pdb_path.exists():
            print(f"  {rt_name}: PDB not found, skipping")
            for k in _feature_names():
                feats[k] = np.nan
            results.append(feats)
            continue

        candidate = _load_structure(pdb_path)
        cand_ca = _get_ca_coords(candidate, list(candidate[0].get_chains())[0].id)

        if cand_ca is None or len(cand_ca) < 10:
            for k in _feature_names():
                feats[k] = np.nan
            results.append(feats)
            continue

        # --- TM-align candidate onto reference MMLV-RT ---
        try:
            cand_seq = seq[:len(cand_ca)] if len(seq) >= len(cand_ca) else seq
            tm_result = _tm_align(cand_ca, ref_rt_ca, cand_seq, ref_seq_for_align)
            # tmtools returns: t (translation), u (rotation), tm_norm_chain1, tm_norm_chain2
            rotation = tm_result.u
            translation = tm_result.t

            # Apply transformation: aligned = R @ coords + t
            cand_ca_aligned = (rotation @ cand_ca.T).T + translation

            feats["alignment_tm_score"] = float(tm_result.tm_norm_chain2)  # normalized by ref length
            feats["alignment_rmsd"] = float(tm_result.rmsd)

        except Exception as e:
            print(f"  {rt_name}: TMalign failed ({e}), skipping")
            for k in _feature_names():
                feats[k] = np.nan
            results.append(feats)
            continue

        n_aligned = len(cand_ca_aligned)

        # --- Clash features (normalized by aligned region) ---
        dist_to_cas9 = cdist(cand_ca_aligned, cas9_ca)
        min_per_residue = dist_to_cas9.min(axis=1)

        feats["cas9_clash_fraction"] = float((min_per_residue < 3.0).sum() / n_aligned)
        feats["cas9_clash_severe_fraction"] = float((min_per_residue < 2.0).sum() / n_aligned)
        feats["cas9_min_distance"] = float(min_per_residue.min())
        feats["cas9_contact_fraction"] = float((min_per_residue < 5.0).sum() / n_aligned)

        # --- Active-site orientation features ---
        if yxdd_pos is not None and yxdd_pos + 4 <= n_aligned:
            yxdd_centroid = cand_ca_aligned[yxdd_pos:yxdd_pos+4].mean(axis=0)

            # Distance to DNA
            if dna_backbone is not None and len(dna_backbone) > 0:
                dist_yxdd_dna = cdist([yxdd_centroid], dna_backbone).min()
                feats["yxdd_to_dna_distance"] = float(dist_yxdd_dna)
            else:
                feats["yxdd_to_dna_distance"] = np.nan

            # Distance to pegRNA
            if pegrna_backbone is not None and len(pegrna_backbone) > 0:
                dist_yxdd_pegrna = cdist([yxdd_centroid], pegrna_backbone).min()
                feats["yxdd_to_pegrna_distance"] = float(dist_yxdd_pegrna)
            else:
                feats["yxdd_to_pegrna_distance"] = np.nan

            # Orientation: angle between YXDD→DNA vector and ref YXDD→DNA vector
            if ref_yxdd_centroid is not None and dna_backbone is not None:
                # Vector from YXDD to nearest DNA point
                nearest_dna_idx = cdist([yxdd_centroid], dna_backbone).argmin()
                vec_cand = dna_backbone[nearest_dna_idx] - yxdd_centroid
                vec_cand = vec_cand / max(np.linalg.norm(vec_cand), 1e-10)

                nearest_dna_idx_ref = cdist([ref_yxdd_centroid], dna_backbone).argmin()
                vec_ref = dna_backbone[nearest_dna_idx_ref] - ref_yxdd_centroid
                vec_ref = vec_ref / max(np.linalg.norm(vec_ref), 1e-10)

                cos_angle = np.clip(np.dot(vec_cand, vec_ref), -1, 1)
                feats["yxdd_orientation_angle"] = float(np.degrees(np.arccos(cos_angle)))
            else:
                feats["yxdd_orientation_angle"] = np.nan

            # YXDD occlusion by Cas9
            dist_yxdd_cas9 = cdist(cand_ca_aligned[max(0,yxdd_pos-5):yxdd_pos+9], cas9_ca)
            n_yxdd_region = dist_yxdd_cas9.shape[0]
            feats["yxdd_occlusion_by_cas9"] = float((dist_yxdd_cas9.min(axis=1) < 5.0).sum() / max(n_yxdd_region, 1))
        else:
            feats["yxdd_to_dna_distance"] = np.nan
            feats["yxdd_to_pegrna_distance"] = np.nan
            feats["yxdd_orientation_angle"] = np.nan
            feats["yxdd_occlusion_by_cas9"] = np.nan

        # --- Protrusion (fraction of candidate atoms beyond ref RT envelope) ---
        # Approximate: for each candidate CA, check if it's farther from ref centroid
        # than any ref CA in that direction
        ref_centroid = ref_rt_ca.mean(axis=0)
        ref_max_dist = np.linalg.norm(ref_rt_ca - ref_centroid, axis=1).max()
        cand_dists = np.linalg.norm(cand_ca_aligned - ref_centroid, axis=1)
        feats["protrusion_fraction"] = float((cand_dists > ref_max_dist).sum() / n_aligned)

        # --- N-terminus to Cas9 distance (fusion point geometry) ---
        nterm_pos = cand_ca_aligned[0]
        feats["nterm_to_cas9_distance"] = float(cdist([nterm_pos], cas9_ca).min())

        results.append(feats)
        print(f"  {rt_name:<25s} TM={feats['alignment_tm_score']:.3f}  RMSD={feats['alignment_rmsd']:.1f}  "
              f"clash={feats['cas9_clash_fraction']:.3f}  yxdd→DNA={feats.get('yxdd_to_dna_distance', 'N/A')}")

    df = pd.DataFrame(results)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "v18_template_features.csv", index=False)
    print(f"\nSaved {len(df)} RTs × {len(df.columns)-1} features")
    return df


def _feature_names():
    return [
        "alignment_tm_score", "alignment_rmsd",
        "cas9_clash_fraction", "cas9_clash_severe_fraction",
        "cas9_min_distance", "cas9_contact_fraction",
        "yxdd_to_dna_distance", "yxdd_to_pegrna_distance",
        "yxdd_orientation_angle", "yxdd_occlusion_by_cas9",
        "protrusion_fraction", "nterm_to_cas9_distance",
    ]


if __name__ == "__main__":
    df = extract_template_features()
    print("\nFeature statistics:")
    print(df.set_index("rt_name")[_feature_names()].describe().round(3))
