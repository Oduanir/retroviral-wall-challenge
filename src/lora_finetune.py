"""
LoRA fine-tuning of ESM2 for RT activity prediction.

Fine-tunes ESM2 with low-rank adapters on the regression task
(predict pe_efficiency_pct from mid-region embeddings), then extracts
task-specific embeddings for downstream use.

Designed for LOFO CV: each fold gets its own LoRA-adapted model
trained on 6 families, producing fold-specific embeddings for all 57 RTs.
"""

import gc
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmModel
from peft import LoraConfig, get_peft_model, TaskType


def _get_device(device=None):
    """Auto-detect best available device (CUDA > MPS > CPU)."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _empty_cache(device):
    """Release GPU/MPS memory."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()


def _mid_region_pool(hidden_states, seq_len):
    """
    Mean-pool the middle third of the sequence from hidden states.

    Parameters
    ----------
    hidden_states : Tensor, shape (1, total_tokens, hidden_dim)
        Includes [CLS] at position 0 and [EOS] at the end.
    seq_len : int
        Original amino acid sequence length (before tokenization).

    Returns
    -------
    Tensor, shape (hidden_dim,)
    """
    # Strip special tokens: positions 1..seq_len are residue embeddings
    residue_emb = hidden_states[0, 1 : seq_len + 1, :]  # (seq_len, hidden_dim)
    n_third = seq_len // 3
    mid = residue_emb[n_third : 2 * n_third]  # middle third
    return mid.mean(dim=0)


class RegressionHead(nn.Module):
    """Simple linear head on top of pooled embeddings."""

    def __init__(self, hidden_dim=1280):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def lora_finetune_and_extract(
    train_names,
    train_sequences,
    train_labels,
    all_names,
    all_sequences,
    model_name="facebook/esm2_t33_650M_UR50D",
    lora_r=4,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=None,
    n_epochs=10,
    lr=1e-4,
    batch_size=4,
    device=None,
):
    """
    Fine-tune ESM2 with LoRA on training data, then extract embeddings for all RTs.

    Parameters
    ----------
    train_names : list of str
        RT names in the training set for this fold.
    train_sequences : list of str
        Amino acid sequences corresponding to train_names.
    train_labels : list or array of float
        pe_efficiency_pct values for training sequences.
    all_names : list of str
        All RT names (train + test) to extract embeddings for.
    all_sequences : list of str
        All amino acid sequences corresponding to all_names.
    model_name : str
        HuggingFace model identifier for ESM2.
    lora_r : int
        LoRA rank.
    lora_alpha : int
        LoRA scaling factor.
    lora_dropout : float
        Dropout rate in LoRA layers.
    target_modules : list of str or None
        Attention modules to apply LoRA to. Default: ["query", "key", "value"].
    n_epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    batch_size : int
        Effective batch size (uses gradient accumulation).
    device : str or None
        Device string. Auto-detected if None.

    Returns
    -------
    dict : rt_name -> numpy array (1280D mid-region embedding from fine-tuned model)
    """
    if target_modules is None:
        target_modules = ["query", "key", "value"]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = _get_device(device)
    # MPS requires float32 (float16 has numerical issues)
    dtype = torch.float32

    print(f"  [LoRA] Device: {device}, dtype: {dtype}")
    print(f"  [LoRA] Training on {len(train_names)} RTs, extracting for {len(all_names)} RTs")

    # --- Load model and tokenizer ---
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    base_model = EsmModel.from_pretrained(model_name, torch_dtype=dtype)

    # --- Apply LoRA ---
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    # --- Regression head ---
    head = RegressionHead(hidden_dim=1280).to(device).to(dtype)

    # --- Prepare training data ---
    train_labels_t = torch.tensor(
        np.array(train_labels, dtype=np.float32), dtype=dtype, device=device
    )

    # Tokenize all training sequences (truncate to 1024)
    train_inputs = []
    for seq in train_sequences:
        enc = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
        train_inputs.append(enc)

    seq_lens = [len(s) for s in train_sequences]

    # --- Training loop ---
    trainable_params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in trainable_params if p.requires_grad], lr=lr, weight_decay=0.01
    )
    loss_fn = nn.MSELoss()

    # Gradient accumulation: process one sequence at a time, step every batch_size
    n_train = len(train_names)
    accum_steps = max(1, batch_size)

    model.train()
    head.train()

    for epoch in range(n_epochs):
        # Shuffle training order
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step_i, idx in enumerate(perm):
            inputs = {k: v.to(device) for k, v in train_inputs[idx].items()}
            outputs = model(**inputs)
            pooled = _mid_region_pool(outputs.last_hidden_state, seq_lens[idx])
            pred = head(pooled)
            loss = loss_fn(pred, train_labels_t[idx]) / accum_steps
            loss.backward()
            epoch_loss += loss.item() * accum_steps

            if (step_i + 1) % accum_steps == 0 or (step_i + 1) == n_train:
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = epoch_loss / n_train
        print(f"  [LoRA] Epoch {epoch + 1}/{n_epochs}  loss={avg_loss:.4f}")

    # --- Extract embeddings from fine-tuned backbone ---
    model.eval()
    embeddings = {}

    with torch.no_grad():
        for name, seq in zip(all_names, all_sequences):
            enc = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**inputs)
            emb = _mid_region_pool(outputs.last_hidden_state, len(seq))
            embeddings[name] = emb.cpu().numpy()

    # --- Cleanup ---
    del model, base_model, head, optimizer, train_inputs, train_labels_t
    _empty_cache(device)

    return embeddings


def lora_lofo_embeddings(
    train_df,
    splits,
    sequences,
    **lora_kwargs,
):
    """
    Run LoRA fine-tuning in each LOFO fold and return per-fold fine-tuned embeddings.

    For each of 7 folds:
    - Train LoRA on the 6 held-in families
    - Extract embeddings for all 57 RTs using the fold-specific model

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe with columns: rt_name, pe_efficiency_pct, (and others).
    splits : dict
        family_name -> list of rt_names (from load_family_splits).
    sequences : dict
        rt_name -> amino acid sequence string.
    **lora_kwargs
        Additional keyword arguments passed to lora_finetune_and_extract
        (e.g., lora_r, lora_alpha, n_epochs, lr, batch_size).

    Returns
    -------
    dict : fold_family -> {rt_name: numpy array (1280D embedding)}
    """
    all_names = list(sequences.keys())
    all_sequences = [sequences[n] for n in all_names]

    fold_embeddings = {}

    for fold_i, (family, held_out_names) in enumerate(splits.items()):
        print(f"\n{'='*60}")
        print(f"Fold {fold_i + 1}/7 — held-out family: {family} ({len(held_out_names)} RTs)")
        print(f"{'='*60}")

        # Training set: everything NOT in the held-out family
        held_out_set = set(held_out_names)
        train_mask = ~train_df["rt_name"].isin(held_out_set)
        fold_train = train_df[train_mask]

        train_names = fold_train["rt_name"].tolist()
        train_seqs = [sequences[n] for n in train_names]
        train_labels = fold_train["pe_efficiency_pct"].values

        # Fine-tune and extract
        embeddings = lora_finetune_and_extract(
            train_names=train_names,
            train_sequences=train_seqs,
            train_labels=train_labels,
            all_names=all_names,
            all_sequences=all_sequences,
            **lora_kwargs,
        )

        fold_embeddings[family] = embeddings
        print(f"  Done. Extracted {len(embeddings)} embeddings for fold '{family}'.")

    return fold_embeddings


if __name__ == "__main__":
    # Quick smoke test: run one fold
    from .data import load_train, load_family_splits

    train_df = load_train()
    splits = load_family_splits()
    sequences = dict(zip(train_df["rt_name"], train_df["sequence"]))

    # Run just the first fold as a test
    first_family = list(splits.keys())[0]
    test_splits = {first_family: splits[first_family]}

    result = lora_lofo_embeddings(
        train_df, test_splits, sequences,
        n_epochs=3, lora_r=4, lr=1e-4,
    )

    for family, embs in result.items():
        sample_name = list(embs.keys())[0]
        print(f"\nFamily '{family}': {len(embs)} embeddings, "
              f"shape={embs[sample_name].shape}, "
              f"sample={sample_name}")
