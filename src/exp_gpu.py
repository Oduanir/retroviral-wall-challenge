#!/usr/bin/env python3
"""GPU-targeted experiments: LoRA binary + PDB feature selection."""
import numpy as np, pandas as pd, torch, warnings, gc, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import average_precision_score
from itertools import product as iprod
from transformers import EsmTokenizer, EsmModel, EsmForMaskedLM
from peft import LoraConfig, get_peft_model
from pathlib import Path

ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(ROOT))
from src.metrics import compute_cls

# --- Data ---
train = pd.read_csv(ROOT / "data/raw/train.csv")
splits_df = pd.read_csv(ROOT / "data/raw/family_splits.csv")
splits = {}
for _, row in splits_df.iterrows():
    splits[row["family"]] = row["rt_names"].split("|")
gt_order = pd.read_csv(ROOT / "data/raw/rt_sequences.csv", usecols=["rt_name"])["rt_name"].tolist()
sequences = dict(zip(train["rt_name"], train["sequence"]))

for col in ["connection_mean_pot","triad_best_rmsd","D1_D2_dist","D2_D3_dist",
            "yxdd_hydrophobic_fraction","yxdd_mean_hydrophobicity","yxdd_5A_mean_pot"]:
    train[f"{col}_missing"] = train[col].isna().astype(int)
train["foldseek_gap_MMLV"] = train["foldseek_best_TM"] - train["foldseek_TM_MMLV"]
train["t40_x_foldseek_MMLV"] = train["t40_raw"] * train["foldseek_TM_MMLV"]
train["triad_quality"] = train["triad_found_bin"] * (1 / (train["triad_best_rmsd"].fillna(99) + 1))
train["seq_struct_compat"] = -train["perplexity"] * train["instability_index"]

pdb = pd.read_csv(ROOT / "data/processed/pdb_structural_features.csv")
train = train.merge(pdb, on="rt_name", how="left")
pdb_cols = [c for c in pdb.columns if c != "rt_name"]
corrs = train[pdb_cols + ["pe_efficiency_pct"]].corr()["pe_efficiency_pct"].drop("pe_efficiency_pct").abs().sort_values(ascending=False)
print("Top PDB correlations:")
for c, v in list(corrs.items())[:8]:
    print(f"  {c:<40s} {v:.3f}")
TOP_PDB5 = corrs.head(5).index.tolist()
TOP_PDB3 = corrs.head(3).index.tolist()

# --- ESM2 ---
print("\nLoading ESM2...")
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").cuda().eval()
l12_raw, l33_raw = {}, {}
for rt, seq in sequences.items():
    inp = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
        n3 = len(seq)//3
        l12_raw[rt] = out.hidden_states[12][0,1:-1,:][n3:2*n3].mean(0).cpu().numpy()
        l33_raw[rt] = out.hidden_states[33][0,1:-1,:][n3:2*n3].mean(0).cpu().numpy()
del model; torch.cuda.empty_cache(); gc.collect()

model_mlm = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").cuda().eval()
for rt, seq in sequences.items():
    inp = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    with torch.no_grad():
        logits = model_mlm(**inp).logits
        probs = torch.softmax(logits[0,1:-1,:], dim=-1)
        ids = inp["input_ids"][0,1:-1]
        ll = torch.log(probs[range(len(ids)), ids])
        ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    train.loc[train["rt_name"]==rt, "esm2_pseudo_ppl"] = torch.exp(-ll.mean()).item()
    train.loc[train["rt_name"]==rt, "esm2_mean_entropy"] = ent.mean().item()
    train.loc[train["rt_name"]==rt, "esm2_mean_ll"] = ll.mean().item()
del model_mlm; torch.cuda.empty_cache(); gc.collect()
print("ESM2 done.")

FEATURES_ORIG = [c for c in [
    "foldseek_TM_MMLV","foldseek_TM_MMLVPE","foldseek_best_TM",
    "foldseek_best_LDDT","foldseek_best_fident","foldseek_TM_HIV1",
    "triad_found_bin","triad_best_rmsd","perplexity","log_likelihood",
    "D1_D2_dist","D2_D3_dist","t40_raw","t45_raw","t50_raw","t55_raw","t60_raw",
    "instability_index","gravy","camsol","net_charge",
] if c in train.columns] + [
    f"{c}_missing" for c in ["connection_mean_pot","triad_best_rmsd","D1_D2_dist",
    "D2_D3_dist","yxdd_hydrophobic_fraction","yxdd_mean_hydrophobicity","yxdd_5A_mean_pot"]
] + ["foldseek_gap_MMLV","t40_x_foldseek_MMLV","triad_quality","seq_struct_compat",
     "esm2_pseudo_ppl","esm2_mean_entropy","esm2_mean_ll"]


def norm(a):
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn) if mx - mn > 1e-12 else np.zeros_like(a)


def lora_binary_finetune(train_names, train_seqs, train_labels, all_names, all_seqs,
                          lora_r=2, n_epochs=5, lr=5e-5):
    base = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").cuda()
    cfg = LoraConfig(r=lora_r, lora_alpha=8, target_modules=["query","value"],
                     lora_dropout=0.1, bias="none")
    lora_model = get_peft_model(base, cfg)
    head = torch.nn.Linear(1280, 1).cuda()
    opt = torch.optim.AdamW(list(lora_model.parameters()) + list(head.parameters()),
                            lr=lr, weight_decay=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    lora_model.train(); head.train()
    for epoch in range(n_epochs):
        total_loss = 0; opt.zero_grad()
        indices = list(range(len(train_names)))
        np.random.shuffle(indices)
        for step, idx in enumerate(indices):
            seq = train_seqs[idx]
            label = torch.tensor([float(train_labels[idx])], device="cuda")
            inp = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            out = lora_model(**inp, output_hidden_states=True)
            n3 = len(seq)//3
            mid = out.hidden_states[33][0,1:-1,:][n3:2*n3].mean(0)
            logit = head(mid)
            loss = loss_fn(logit, label) / 4
            loss.backward()
            total_loss += loss.item() * 4
            if (step + 1) % 4 == 0 or step == len(indices) - 1:
                opt.step(); opt.zero_grad()
        print(f"    Ep {epoch+1}/{n_epochs} BCE={total_loss/len(indices):.4f}")
    lora_model.eval()
    embeddings = {}
    for name in all_names:
        seq = all_seqs[name]
        inp = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        with torch.no_grad():
            out = lora_model(**inp, output_hidden_states=True)
            n3 = len(seq)//3
            embeddings[name] = out.hidden_states[33][0,1:-1,:][n3:2*n3].mean(0).cpu().numpy()
    del lora_model, head; torch.cuda.empty_cache(); gc.collect()
    return embeddings


def run_lofo(features, l12, l33, lora_embs=None, ra12=12.5, ra33=7.5):
    all_oof = []
    model_names = ["EN","L12","L33"]
    if lora_embs is not None:
        model_names.append("LORA")
    for outer_family in splits:
        outer_rts = splits[outer_family]
        om = train["rt_name"].isin(outer_rts); im = ~om
        inner_splits_local = {f: rts for f, rts in splits.items() if f != outer_family}
        idf = train[im].reset_index(drop=True)
        il12 = np.array([l12[n] for n in idf["rt_name"]])
        il33 = np.array([l33[n] for n in idf["rt_name"]])
        p12 = PCA(3).fit(il12); p33 = PCA(3).fit(il33)
        il12p = p12.transform(il12); il33p = p33.transform(il33)
        ol12p = p12.transform(np.array([l12[n] for n in train.loc[om,"rt_name"]]))
        ol33p = p33.transform(np.array([l33[n] for n in train.loc[om,"rt_name"]]))
        if lora_embs is not None:
            il_l = np.array([lora_embs[outer_family][n] for n in idf["rt_name"]])
            pl = PCA(3).fit(il_l); il_lp = pl.transform(il_l)
            ol_lp = pl.transform(np.array([lora_embs[outer_family][n] for n in train.loc[om,"rt_name"]]))
        n = len(idf)
        ioof = {m: np.full(n, np.nan) for m in model_names}
        for ifam, irts in inner_splits_local.items():
            tm = idf["rt_name"].isin(irts); trm = ~tm
            y = idf.loc[trm, "pe_efficiency_pct"].values
            ioof["EN"][tm.values] = make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),ElasticNet(alpha=1.0,l1_ratio=0.3,max_iter=10000)).fit(idf.loc[trm,features].values,y).predict(idf.loc[tm,features].values)
            ioof["L12"][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=ra12)).fit(il12p[trm.values],y).predict(il12p[tm.values])
            ioof["L33"][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=ra33)).fit(il33p[trm.values],y).predict(il33p[tm.values])
            if lora_embs is not None:
                ioof["LORA"][tm.values] = make_pipeline(StandardScaler(),Ridge(alpha=10.0)).fit(il_lp[trm.values],y).predict(il_lp[tm.values])
        nm = len(model_names)
        best_c, best_w = -1, tuple([1.0/nm]*nm)
        for w in iprod(np.arange(0,1.05,0.05), repeat=nm):
            if abs(sum(w)-1.0) > 0.025:
                continue
            b = sum(w[i]*norm(ioof[m]) for i,m in enumerate(model_names))
            cc = compute_cls(idf["active"].values, b, idf["pe_efficiency_pct"].values)["cls"]
            if cc > best_c:
                best_c, best_w = cc, w
        yi = train.loc[im,"pe_efficiency_pct"].values
        pr = {}
        pr["EN"] = make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),ElasticNet(alpha=1.0,l1_ratio=0.3,max_iter=10000)).fit(train.loc[im,features].values,yi).predict(train.loc[om,features].values)
        pr["L12"] = make_pipeline(StandardScaler(),Ridge(alpha=ra12)).fit(il12p,yi).predict(ol12p)
        pr["L33"] = make_pipeline(StandardScaler(),Ridge(alpha=ra33)).fit(il33p,yi).predict(ol33p)
        if lora_embs is not None:
            pr["LORA"] = make_pipeline(StandardScaler(),Ridge(alpha=10.0)).fit(il_lp,yi).predict(ol_lp)
        bl = np.zeros(om.sum())
        for i,m in enumerate(model_names):
            mn,mx = ioof[m].min(), ioof[m].max()
            bl += best_w[i] * ((pr[m]-mn)/max(mx-mn,1e-12))
        fd = train.loc[om,["rt_name","active","pe_efficiency_pct","rt_family"]].copy()
        fd["predicted_score"] = bl
        all_oof.append(fd)
    oof = pd.concat(all_oof).set_index("rt_name").loc[gt_order].reset_index()
    r = compute_cls(oof["active"].values, oof["predicted_score"].values, oof["pe_efficiency_pct"].values)
    return r, oof


# === EXPERIMENTS ===
print("\n" + "="*70)
print("TARGETED EXPERIMENTS")
print("="*70)

r1, _ = run_lofo(FEATURES_ORIG, l12_raw, l33_raw)
print(f"Exp1 Baseline:    CLS={r1['cls']:.4f}  PR={r1['pr_auc']:.4f}  WSp={r1['w_spearman']:.4f}")

r2, _ = run_lofo(FEATURES_ORIG + TOP_PDB5, l12_raw, l33_raw)
print(f"Exp2 +PDB5:       CLS={r2['cls']:.4f}  PR={r2['pr_auc']:.4f}  WSp={r2['w_spearman']:.4f}")

r3, _ = run_lofo(FEATURES_ORIG + TOP_PDB3, l12_raw, l33_raw)
print(f"Exp3 +PDB3:       CLS={r3['cls']:.4f}  PR={r3['pr_auc']:.4f}  WSp={r3['w_spearman']:.4f}")

all_names = list(sequences.keys())

print("\nExp4: LoRA binary r=2 5ep lr=5e-5")
le4 = {}
for fam in splits:
    print(f"  {fam}")
    ir = [rt for f,rts in splits.items() if f!=fam for rt in rts]
    le4[fam] = lora_binary_finetune(ir, [sequences[r] for r in ir],
        [int(train.loc[train["rt_name"]==r,"active"].values[0]) for r in ir],
        all_names, sequences, 2, 5, 5e-5)
r4, o4 = run_lofo(FEATURES_ORIG, l12_raw, l33_raw, lora_embs=le4)
print(f"Exp4 LoRA_bin:    CLS={r4['cls']:.4f}  PR={r4['pr_auc']:.4f}  WSp={r4['w_spearman']:.4f}")

print("\nExp5: LoRA binary r=2 3ep lr=1e-4")
le5 = {}
for fam in splits:
    print(f"  {fam}")
    ir = [rt for f,rts in splits.items() if f!=fam for rt in rts]
    le5[fam] = lora_binary_finetune(ir, [sequences[r] for r in ir],
        [int(train.loc[train["rt_name"]==r,"active"].values[0]) for r in ir],
        all_names, sequences, 2, 3, 1e-4)
r5, o5 = run_lofo(FEATURES_ORIG, l12_raw, l33_raw, lora_embs=le5)
print(f"Exp5 LoRA_fast:   CLS={r5['cls']:.4f}  PR={r5['pr_auc']:.4f}  WSp={r5['w_spearman']:.4f}")

print("\nExp6: LoRA binary r=4 3ep lr=5e-5")
le6 = {}
for fam in splits:
    print(f"  {fam}")
    ir = [rt for f,rts in splits.items() if f!=fam for rt in rts]
    le6[fam] = lora_binary_finetune(ir, [sequences[r] for r in ir],
        [int(train.loc[train["rt_name"]==r,"active"].values[0]) for r in ir],
        all_names, sequences, 4, 3, 5e-5)
r6, o6 = run_lofo(FEATURES_ORIG, l12_raw, l33_raw, lora_embs=le6)
print(f"Exp6 LoRA_r4:     CLS={r6['cls']:.4f}  PR={r6['pr_auc']:.4f}  WSp={r6['w_spearman']:.4f}")

print("\nExp7: LoRA replacing L33")
best_le = le4 if r4["cls"] >= max(r5["cls"],r6["cls"]) else (le5 if r5["cls"] >= r6["cls"] else le6)
lm = {}
for fam, rts in splits.items():
    for rt in rts:
        lm[rt] = best_le[fam][rt]
r7, o7 = run_lofo(FEATURES_ORIG, l12_raw, lm)
print(f"Exp7 LoRA=L33:    CLS={r7['cls']:.4f}  PR={r7['pr_auc']:.4f}  WSp={r7['w_spearman']:.4f}")

print("\nExp8: LoRA + PDB3")
r8, o8 = run_lofo(FEATURES_ORIG + TOP_PDB3, l12_raw, l33_raw, lora_embs=best_le)
print(f"Exp8 LoRA+PDB3:   CLS={r8['cls']:.4f}  PR={r8['pr_auc']:.4f}  WSp={r8['w_spearman']:.4f}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
results = {"1_baseline":r1,"2_pdb5":r2,"3_pdb3":r3,"4_lora":r4,"5_lora_fast":r5,
           "6_lora_r4":r6,"7_lora=l33":r7,"8_lora+pdb3":r8}
for n, r in sorted(results.items(), key=lambda x: -x[1]["cls"]):
    print(f"  {n:<20s} CLS={r['cls']:.4f}  PR={r['pr_auc']:.4f}  WSp={r['w_spearman']:.4f}")
bn = max(results, key=lambda k: results[k]["cls"])
print(f"\nBest: {bn}  CLS={results[bn]['cls']:.4f}")

# Save best if it beats 0.7088
if results[bn]["cls"] > 0.7088:
    best_oof = {"4_lora":o4,"5_lora_fast":o5,"6_lora_r4":o6,"7_lora=l33":o7,"8_lora+pdb3":o8}.get(bn)
    if best_oof is not None:
        best_oof[["rt_name","predicted_score"]].to_csv(ROOT / "submissions/submission_v7.csv", index=False)
        print("Submission saved!")
