"""
Data loading and preparation for the Retroviral Wall Challenge.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def load_train():
    """Load training data (sequences + features + labels merged)."""
    return pd.read_csv(DATA_DIR / "train.csv")


def load_sequences():
    """Load RT sequences with labels (GitHub format)."""
    return pd.read_csv(DATA_DIR / "rt_sequences.csv")


def load_features():
    """Load handcrafted features (GitHub format)."""
    return pd.read_csv(DATA_DIR / "handcrafted_features.csv")


def load_esm2_embeddings():
    """
    Load ESM2 embeddings and return a DataFrame indexed by rt_name.

    Returns
    -------
    pd.DataFrame, shape (57, 1280), index = rt_name
    """
    data = np.load(DATA_DIR / "esm2_embeddings.npz", allow_pickle=True)
    names = data["names"]
    embeddings = data["embeddings"]
    cols = [f"esm2_{i}" for i in range(embeddings.shape[1])]
    return pd.DataFrame(embeddings, index=names, columns=cols)


def load_family_splits():
    """Load family splits as a dict: family -> list of rt_names."""
    df = pd.read_csv(DATA_DIR / "family_splits.csv")
    splits = {}
    for _, row in df.iterrows():
        family = row["family"]
        rt_names = row["rt_names"].split("|")
        splits[family] = rt_names
    return splits


def load_feature_dictionary():
    """Load feature descriptions."""
    return pd.read_csv(DATA_DIR / "feature_dictionary.csv")


def get_numeric_feature_cols(df):
    """
    Return list of numeric feature column names, excluding metadata and targets.
    """
    exclude = {
        "rt_name", "sequence", "active", "pe_efficiency_pct",
        "rt_family", "yxdd_seq", "protein_length_aa",
    }
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
