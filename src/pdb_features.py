"""
Extract structural features from AlphaFold-predicted PDB files.

Features extracted:
  - B-factor statistics (7 features)
  - Shape / compactness descriptors (6 features)
  - Pairwise CA distance distribution (4 features)
  - YXDD active-site-centric features (6 features)

Total: 23 structural features per RT.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import skew, kurtosis

from Bio.PDB import PDBParser

from src.esm2_features import find_yxdd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRUCTURES_DIR = PROJECT_ROOT / "data" / "raw" / "structures"
SEQUENCES_CSV = PROJECT_ROOT / "data" / "raw" / "rt_sequences.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "pdb_structural_features.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ca_coords_and_bfactors(structure):
    """Return (N x 3) CA coordinate array and corresponding B-factor array."""
    cas = []
    bfactors = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue  # skip HETATMs
                if "CA" in residue:
                    ca = residue["CA"]
                    cas.append(ca.get_vector().get_array())
                    bfactors.append(ca.get_bfactor())
        break  # first model only
    return np.array(cas), np.array(bfactors)


def _estimate_secondary_structure(ca_coords):
    """
    Rough secondary structure assignment based on CA-CA distance patterns.
    Returns an array of labels: 0 = coil, 1 = helix, 2 = sheet.
    Helix: i to i+3 CA distance ~5.0-5.5 A
    Sheet: i to i+2 CA distance ~6.5-7.0 A
    """
    n = len(ca_coords)
    ss = np.zeros(n, dtype=int)  # default coil
    for i in range(n - 3):
        d_i3 = np.linalg.norm(ca_coords[i] - ca_coords[i + 3])
        if 4.5 <= d_i3 <= 6.0:
            ss[i] = 1  # helix-like
            ss[i + 1] = 1
            ss[i + 2] = 1
            ss[i + 3] = 1
    for i in range(n - 2):
        if ss[i] == 0:  # not already helix
            d_i2 = np.linalg.norm(ca_coords[i] - ca_coords[i + 2])
            if 6.0 <= d_i2 <= 7.5:
                ss[i] = 2  # sheet-like
                ss[i + 1] = 2
                ss[i + 2] = 2
    return ss


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_pdb_features(pdb_path: str, sequence: str = None) -> dict:
    """
    Extract all structural features from a single PDB file.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file.
    sequence : str, optional
        Amino acid sequence for YXDD motif detection.

    Returns
    -------
    dict
        Dictionary of feature_name -> value.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    ca_coords, bfactors = _ca_coords_and_bfactors(structure)
    n_res = len(ca_coords)

    feats = {}

    if n_res < 5:
        warnings.warn(f"Too few CA atoms ({n_res}) in {pdb_path}")
        return feats

    # ------------------------------------------------------------------
    # B-factor features (7)
    # ------------------------------------------------------------------
    feats["b_factor_mean"] = float(np.mean(bfactors))
    feats["b_factor_std"] = float(np.std(bfactors))
    feats["b_factor_median"] = float(np.median(bfactors))
    feats["b_factor_q25"] = float(np.percentile(bfactors, 25))
    feats["b_factor_q75"] = float(np.percentile(bfactors, 75))

    # N-term vs C-term ratio
    third = max(1, n_res // 3)
    nterm_mean = np.mean(bfactors[:third])
    cterm_mean = np.mean(bfactors[-third:])
    feats["b_factor_nterm_ratio"] = float(nterm_mean / cterm_mean) if cterm_mean != 0 else np.nan

    # Core vs surface: use distance from centroid as burial proxy
    centroid = ca_coords.mean(axis=0)
    dists_to_centroid = np.linalg.norm(ca_coords - centroid, axis=1)
    median_dist = np.median(dists_to_centroid)
    core_mask = dists_to_centroid <= median_dist
    surface_mask = ~core_mask
    core_bf = np.mean(bfactors[core_mask]) if core_mask.sum() > 0 else np.nan
    surface_bf = np.mean(bfactors[surface_mask]) if surface_mask.sum() > 0 else np.nan
    feats["b_factor_core_vs_surface"] = float(core_bf / surface_bf) if (surface_bf and surface_bf != 0) else np.nan

    # ------------------------------------------------------------------
    # Shape / compactness features (6)
    # ------------------------------------------------------------------
    rg = float(np.sqrt(np.mean(np.sum((ca_coords - centroid) ** 2, axis=1))))
    feats["radius_of_gyration"] = rg
    feats["rg_normalized"] = rg / np.sqrt(n_res) if n_res > 0 else np.nan

    # Principal component analysis for elongation / asphericity
    centered = ca_coords - centroid
    cov_matrix = np.cov(centered.T)  # 3x3
    eigenvalues = np.sort(np.linalg.eigvalsh(cov_matrix))[::-1]  # descending
    eigenvalues = np.maximum(eigenvalues, 0)  # guard against tiny negatives

    feats["elongation"] = float(eigenvalues[0] / eigenvalues[2]) if eigenvalues[2] > 1e-9 else np.nan
    lam_sum = eigenvalues.sum()
    if lam_sum > 1e-9:
        lam_mean = lam_sum / 3.0
        feats["asphericity"] = float(
            1.5 * np.sum((eigenvalues - lam_mean) ** 2) / (lam_sum ** 2)
        )
    else:
        feats["asphericity"] = np.nan

    # Contact densities
    dist_matrix = squareform(pdist(ca_coords))
    for cutoff in (8, 12):
        contacts = (dist_matrix < cutoff).sum(axis=1) - 1  # subtract self
        feats[f"contact_density_{cutoff}A"] = float(np.mean(contacts))

    # ------------------------------------------------------------------
    # Pairwise distance distribution features (4)
    # ------------------------------------------------------------------
    pw_dists = pdist(ca_coords)
    feats["pairwise_dist_mean"] = float(np.mean(pw_dists))
    feats["pairwise_dist_std"] = float(np.std(pw_dists))
    feats["pairwise_dist_skew"] = float(skew(pw_dists))
    feats["pairwise_dist_kurtosis"] = float(kurtosis(pw_dists))

    # ------------------------------------------------------------------
    # YXDD-centric features (6)
    # ------------------------------------------------------------------
    yxdd_pos = None
    if sequence is not None:
        yxdd_pos = find_yxdd(sequence)

    if yxdd_pos is not None and yxdd_pos + 3 < n_res:
        yxdd_indices = list(range(yxdd_pos, min(yxdd_pos + 4, n_res)))
        yxdd_cas = ca_coords[yxdd_indices]
        yxdd_centroid = yxdd_cas.mean(axis=0)

        # Local B-factor (within 10 A of YXDD centroid)
        dists_to_yxdd = np.linalg.norm(ca_coords - yxdd_centroid, axis=1)
        local_10 = dists_to_yxdd < 10.0
        feats["yxdd_local_bfactor"] = float(np.mean(bfactors[local_10])) if local_10.sum() > 0 else np.nan

        # YXDD contact density (CA within 8 A of any YXDD residue)
        yxdd_contact_counts = 0
        for idx in yxdd_indices:
            yxdd_contact_counts += (dist_matrix[idx] < 8.0).sum() - 1
        # Average per YXDD residue
        feats["yxdd_contact_density"] = float(yxdd_contact_counts / len(yxdd_indices))

        # YXDD burial (mean distance of YXDD CAs from centroid, normalized by Rg)
        yxdd_to_centroid = np.linalg.norm(yxdd_cas - centroid, axis=1).mean()
        feats["yxdd_burial"] = float(yxdd_to_centroid / rg) if rg > 1e-9 else np.nan

        # Local Rg (residues within 15 A of YXDD centroid)
        local_15 = dists_to_yxdd < 15.0
        if local_15.sum() > 2:
            local_coords = ca_coords[local_15]
            local_centroid = local_coords.mean(axis=0)
            feats["yxdd_local_rg"] = float(
                np.sqrt(np.mean(np.sum((local_coords - local_centroid) ** 2, axis=1)))
            )
        else:
            feats["yxdd_local_rg"] = np.nan

        # Secondary structure entropy near YXDD (within 15 A)
        ss_assignments = _estimate_secondary_structure(ca_coords)
        local_ss = ss_assignments[local_15]
        if len(local_ss) > 0:
            counts = np.bincount(local_ss, minlength=3).astype(float)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            feats["yxdd_secondary_structure_entropy"] = float(-np.sum(probs * np.log2(probs)))
        else:
            feats["yxdd_secondary_structure_entropy"] = np.nan

        # Distance from YXDD centroid to nearest terminal residue
        terminal_coords = ca_coords[[0, -1]]
        d_to_terminals = np.linalg.norm(terminal_coords - yxdd_centroid, axis=1)
        feats["yxdd_distance_to_surface"] = float(np.min(d_to_terminals))
    else:
        feats["yxdd_local_bfactor"] = np.nan
        feats["yxdd_contact_density"] = np.nan
        feats["yxdd_burial"] = np.nan
        feats["yxdd_local_rg"] = np.nan
        feats["yxdd_secondary_structure_entropy"] = np.nan
        feats["yxdd_distance_to_surface"] = np.nan

    return feats


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_all_pdb_features() -> pd.DataFrame:
    """
    Extract structural features for all 57 RTs.

    Reads sequences from rt_sequences.csv, maps each rt_name to its PDB file,
    and returns a DataFrame indexed by rt_name.
    """
    seq_df = pd.read_csv(SEQUENCES_CSV)
    seq_map = dict(zip(seq_df["rt_name"], seq_df["sequence"]))

    records = []
    for rt_name, seq in seq_map.items():
        pdb_file = STRUCTURES_DIR / f"{rt_name}.pdb"
        if not pdb_file.exists():
            warnings.warn(f"PDB file not found for {rt_name}: {pdb_file}")
            records.append({"rt_name": rt_name})
            continue
        try:
            feats = extract_pdb_features(str(pdb_file), sequence=seq)
            feats["rt_name"] = rt_name
            records.append(feats)
        except Exception as e:
            warnings.warn(f"Error processing {rt_name}: {e}")
            records.append({"rt_name": rt_name})

    df = pd.DataFrame(records)
    df = df.set_index("rt_name")

    # Ensure output directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV)
    print(f"Saved {len(df)} rows x {len(df.columns)} features to {OUTPUT_CSV}")
    return df


if __name__ == "__main__":
    df = extract_all_pdb_features()
    print(df.head())
    print(f"\nFeature columns ({len(df.columns)}):")
    print(list(df.columns))
