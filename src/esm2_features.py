"""
ESM2 feature extraction: per-region embeddings and zero-shot features.

Produces:
- data/processed/esm2_per_region.npz : per-region pooled embeddings (7 regions x 1280D)
- data/processed/esm2_zero_shot_features.csv : pseudo-perplexity, entropy, log-likelihood
"""

import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import EsmTokenizer, EsmModel, EsmForMaskedLM

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def find_yxdd(seq):
    """Find YXDD motif position in amino acid sequence."""
    for i in range(len(seq) - 3):
        if seq[i] in "YF" and seq[i + 2] == "D" and seq[i + 3] == "D":
            return i
    return None


def extract_per_region_embeddings(model_name="facebook/esm2_t33_650M_UR50D"):
    """Extract per-region pooled embeddings from ESM2."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train = pd.read_csv(RAW_DIR / "train.csv")
    sequences = dict(zip(train["rt_name"], train["sequence"]))

    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.eval()

    all_features = []

    for rt_name, seq in sequences.items():
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)

        with torch.no_grad():
            outputs = model(**inputs)
            per_residue = outputs.last_hidden_state[0, 1:-1, :].numpy()

        seq_len = len(seq)
        global_mean = per_residue.mean(axis=0)

        yxdd_pos = find_yxdd(seq)
        if yxdd_pos is not None:
            start = max(0, yxdd_pos - 15)
            end = min(seq_len, yxdd_pos + 4 + 15)
            active_site_emb = per_residue[start:end].mean(axis=0)
            motif_emb = per_residue[yxdd_pos : yxdd_pos + 4].mean(axis=0)
            contrast_emb = active_site_emb - global_mean
        else:
            active_site_emb = np.zeros(1280)
            motif_emb = np.zeros(1280)
            contrast_emb = np.zeros(1280)

        n_third = seq_len // 3
        nterm_emb = per_residue[:n_third].mean(axis=0)
        mid_emb = per_residue[n_third : 2 * n_third].mean(axis=0)
        cterm_emb = per_residue[2 * n_third :].mean(axis=0)

        all_features.append(
            {
                "rt_name": rt_name,
                "global_emb": global_mean,
                "active_site": active_site_emb,
                "motif_yxdd": motif_emb,
                "contrast": contrast_emb,
                "nterm": nterm_emb,
                "mid": mid_emb,
                "cterm": cterm_emb,
            }
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PROCESSED_DIR / "esm2_per_region.npz",
        names=np.array([f["rt_name"] for f in all_features]),
        global_emb=np.array([f["global_emb"] for f in all_features]),
        active_site=np.array([f["active_site"] for f in all_features]),
        motif_yxdd=np.array([f["motif_yxdd"] for f in all_features]),
        contrast=np.array([f["contrast"] for f in all_features]),
        nterm=np.array([f["nterm"] for f in all_features]),
        mid=np.array([f["mid"] for f in all_features]),
        cterm=np.array([f["cterm"] for f in all_features]),
    )
    print(f"Saved per-region embeddings for {len(all_features)} RTs")


def extract_zero_shot_features(model_name="facebook/esm2_t33_650M_UR50D"):
    """Extract pseudo-perplexity and entropy features from ESM2."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train = pd.read_csv(RAW_DIR / "train.csv")
    sequences = dict(zip(train["rt_name"], train["sequence"]))

    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.eval()

    results = []

    for rt_name, seq in sequences.items():
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits[0, 1:-1, :], dim=-1)

            input_ids = inputs["input_ids"][0, 1:-1]
            log_probs = torch.log(probs)
            per_pos_ll = log_probs[range(len(input_ids)), input_ids]
            pseudo_ppl = torch.exp(-per_pos_ll.mean()).item()

            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        results.append(
            {
                "rt_name": rt_name,
                "esm2_pseudo_ppl": pseudo_ppl,
                "esm2_mean_entropy": entropy.mean().item(),
                "esm2_std_entropy": entropy.std().item(),
                "esm2_mean_ll": per_pos_ll.mean().item(),
            }
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(
        PROCESSED_DIR / "esm2_zero_shot_features.csv", index=False
    )
    print(f"Saved zero-shot features for {len(results)} RTs")


if __name__ == "__main__":
    print("Extracting per-region embeddings...")
    extract_per_region_embeddings()
    print("\nExtracting zero-shot features...")
    extract_zero_shot_features()
