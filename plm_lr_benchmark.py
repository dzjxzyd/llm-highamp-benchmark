import os, re, json, argparse, warnings
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, AutoConfig, T5EncoderModel
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix, average_precision_score
)
from sklearn.model_selection import train_test_split


# ----------------------------
# utils
# ----------------------------
def safe_model_tag(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_id)

def clean_aa(seq: str) -> str:
    # keep only standard AA-ish; map rare to X
    return seq.strip().upper().replace("U","X").replace("Z","X").replace("O","X").replace("B","X")

def needs_space_separated(model_id: str) -> bool:
    mid = model_id.lower()
    return any(k in mid for k in ["rostlab", "prot_t5", "prot_bert", "distilprotbert"])

@torch.no_grad()
def mean_pool(last_hidden: torch.Tensor,
              attn_mask: torch.Tensor,
              special_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    last_hidden: [B, L, H]
    attn_mask:  [B, L] 1=valid,0=pad
    special_mask: [B, L] 1=special tokens (CLS/SEP/BOS/EOS...), optional
    """
    mask = attn_mask.bool()

    # special_mask 有些 tokenizer 可能行为怪（甚至全 1），要做“保底”
    if special_mask is not None:
        keep = mask & (~special_mask.bool())
        # 如果某些样本被 special_mask 全剔空，就忽略 special_mask（否则 pooled 会全 0）
        bad = (keep.sum(dim=1) == 0)
        if bad.any():
            keep[bad] = mask[bad]
        mask = keep

    mask_f = mask.unsqueeze(-1).float()            # [B, L, 1]
    summed = (last_hidden * mask_f).sum(dim=1)     # [B, H]
    denom = mask_f.sum(dim=1).clamp(min=1.0)       # [B, 1]
    return summed / denom

def masked_max_pool(last_hidden, attn_mask):
    # last_hidden: [B,L,H], attn_mask: [B,L]
    mask = attn_mask.bool().unsqueeze(-1)  # [B,L,1]
    x = last_hidden.masked_fill(~mask, -1e9)
    return x.max(dim=1).values  # [B,H]

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    ap = average_precision_score(y_true, y_prob)
    return {"ACC": acc, "BACC": bacc, "Sn": sn, "Sp": sp, "MCC": mcc, "AUC": auc, "AP": ap}


# ----------------------------
# model loaders
# ----------------------------
def load_model_tokenizer_transformers(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    torch_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

    # ✅ ANKH: encoder-only (T5EncoderModel)，避免 decoder_input_ids 报错
    if ("ankh" in model_id.lower()) or (getattr(cfg, "model_type", "") == "t5"):
        model = T5EncoderModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
    else:
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )

    model.to(device).eval()
    return tok, model


# ----------------------------
# ESMC / ESM3 (esm lib)
# ----------------------------
def embed_sequences_esmc(model_id: str, seqs: List[str], device: str) -> np.ndarray:
    """
    ESM-C: use esm library (ESMC + LogitsConfig(return_embeddings=True))
    See ESM cookbook quickstart. :contentReference[oaicite:4]{index=4}
    """
    if "300m" in model_id.lower():
        esm_name = "esmc_300m"
    elif "600m" in model_id.lower():
        esm_name = "esmc_600m"
    else:
        raise ValueError(f"Unknown ESMC size in model_id: {model_id}")

    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    client = ESMC.from_pretrained(esm_name).to(device)
    client.eval()

    vecs = []
    with torch.no_grad():
        for seq in tqdm(seqs, desc=f"Embedding (ESMC:{esm_name})"):
            s = clean_aa(seq)
            protein = ESMProtein(sequence=s)
            protein_tensor = client.encode(protein)
            out = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))

            emb = out.embeddings
            seq_emb = getattr(emb, "sequence", None)
            if seq_emb is None and isinstance(emb, dict):
                seq_emb = emb.get("sequence", None)
            if seq_emb is None:
                seq_emb = emb

            if not torch.is_tensor(seq_emb):
                seq_emb = torch.tensor(seq_emb)

            if seq_emb.dim() == 3:
                seq_emb = seq_emb[0]  # [L, D]

            # 常见：len(seq)+2 含 BOS/EOS
            if seq_emb.size(0) == len(s) + 2:
                seq_emb = seq_emb[1:-1]

            vec = seq_emb.mean(dim=0)
            vecs.append(vec.float().cpu().numpy())

    X = np.stack(vecs, axis=0)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def embed_sequences_esm3(seqs: List[str], device: str) -> np.ndarray:
    """
    ESM3: use esm library (ESM3.from_pretrained("esm3_sm_open_v1"))
    Some repos are gated; you must accept license + login. :contentReference[oaicite:5]{index=5}
    """
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, SamplingConfig

    client = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
    client.eval()

    vecs = []
    with torch.no_grad():
        for seq in tqdm(seqs, desc="Embedding (ESM3:esm3_sm_open_v1)"):
            s = clean_aa(seq)
            protein = ESMProtein(sequence=s)
            protein_tensor = client.encode(protein)

            out = client.forward_and_sample(
                protein_tensor,
                SamplingConfig(return_per_residue_embeddings=True)
            )

            emb = getattr(out, "per_residue_embedding", None)
            if emb is None:
                emb = getattr(out, "per_residue_embeddings", None)
            if emb is None:
                raise RuntimeError("ESM3 output has no per_residue_embedding(s).")

            if emb.dim() == 3:
                emb = emb[0]  # [L, D]

            if emb.size(0) == len(s) + 2:
                emb = emb[1:-1]

            vec = emb.mean(dim=0)
            vecs.append(vec.float().cpu().numpy())

    X = np.stack(vecs, axis=0)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------
# DPLM (prefer official byprot; fallback to transformers)
# ----------------------------
def embed_sequences_dplm(model_id: str, seqs: List[str], device: str) -> np.ndarray:
    """
    Official loading in model card uses byprot DiffusionProteinLanguageModel. :contentReference[oaicite:6]{index=6}
    If byprot not installed, fallback to transformers (may be suboptimal).
    """
    try:
        from byprot.models.lm.dplm import DiffusionProteinLanguageModel  # type: ignore
        dplm = DiffusionProteinLanguageModel.from_pretrained(model_id)
        dplm.to(device).eval()

        # NOTE: byprot API varies across versions. We'll try a robust path.
        vecs = []
        with torch.no_grad():
            for seq in tqdm(seqs, desc=f"Embedding (DPLM:{model_id})"):
                s = clean_aa(seq)
                # common: dplm.encode / dplm.get_representation not guaranteed
                if hasattr(dplm, "encode"):
                    rep = dplm.encode([s])
                elif hasattr(dplm, "get_representation"):
                    rep = dplm.get_representation([s])
                else:
                    # last resort: forward with tokenizer inside model (may fail)
                    rep = dplm([s])

                rep = torch.as_tensor(rep)
                rep = rep.squeeze(0)  # [L,D] or [D]
                if rep.dim() == 2:
                    rep = rep.mean(dim=0)
                vecs.append(rep.float().detach().cpu().numpy())

        X = np.stack(vecs, axis=0)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    except Exception as e:
        warnings.warn(f"[DPLM] byprot path failed ({type(e).__name__}: {e}). Fallback to transformers.")
        return embed_sequences_transformers(model_id, seqs, device, batch_size=2, max_length=256)


# ----------------------------
# generic transformers embed
# ----------------------------
def embed_sequences_transformers(model_id: str,
                                seqs: List[str],
                                device: str,
                                batch_size: int,
                                max_length: int) -> np.ndarray:
    tok, model = load_model_tokenizer_transformers(model_id, device)

    all_vecs = []
    for i in tqdm(range(0, len(seqs), batch_size), desc=f"Embedding {model_id}"):
        batch_seqs = [clean_aa(s) for s in seqs[i:i+batch_size]]

        # ✅ ANKH/ESM/ESM2/SaProt/Mistral-Prot：不要空格分隔
        batch_text = [" ".join(list(s)) for s in batch_seqs] if needs_space_separated(model_id) else batch_seqs

        enc = tok(
            batch_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
        # tokenizer-side quick checks (on CPU ok)
        input_ids_cpu = enc["input_ids"]
        unk_id = getattr(tok, "unk_token_id", None)
        if unk_id is not None:
            unk_rate = (input_ids_cpu == unk_id).float().mean().item()
            if unk_rate > 0.2:
                print(f"[WARN] high UNK rate={unk_rate:.3f} for {model_id}")

        special_cpu = enc.get("special_tokens_mask", None)
        if special_cpu is not None:
            all_special = (special_cpu.sum(dim=1) == special_cpu.size(1)).any().item()
            if all_special:
                print(f"[WARN] special_tokens_mask marks ALL tokens as special for {model_id} (will auto-fallback).")

        enc = {k: v.to(device) for k, v in enc.items()}
        attn_mask = enc.get("attention_mask", None)
        special_mask = enc.get("special_tokens_mask", None)

        out = model(**{k: enc[k] for k in ["input_ids", "attention_mask"] if k in enc})
        last_hidden = out.last_hidden_state

        if attn_mask is None:
            attn_mask = torch.ones(last_hidden.shape[:2], device=device, dtype=torch.long)

        if "mistral-prot" in model_id.lower():
            vec = masked_max_pool(last_hidden, attn_mask)
        else:
            vec = mean_pool(last_hidden, attn_mask, special_mask)
        vec = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        # pooled 全零会直接把 LR 逼成“全预测正类”的 0.617 基线
        if (vec.abs().sum(dim=1) == 0).any().item():
            print(f"[WARN] zero pooled embedding exists for {model_id} (check tokenizer/pooling)")

        all_vecs.append(vec.detach().cpu().float().numpy())

    X = np.concatenate(all_vecs, axis=0)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def embed_sequences(model_id: str,
                    seqs: List[str],
                    device: str,
                    batch_size: int,
                    cache_dir: str,
                    max_length: int) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    tag = safe_model_tag(model_id)
    cache_path = os.path.join(cache_dir, f"{tag}.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path)

    # ---- model-specific routing ----
    if model_id.startswith("EvolutionaryScale/esmc-"):
        X = embed_sequences_esmc(model_id, seqs, device)

    elif model_id == "EvolutionaryScale/esm3-sm-open-v1":
        X = embed_sequences_esm3(seqs, device)

    elif model_id == "airkingbd/dplm_650m":
        X = embed_sequences_dplm(model_id, seqs, device)

    elif model_id.startswith("westlake-repl/SaProt_"):
        # SaProt 通常需要 seq+structure tokens；只有 AA seq 时结果可能接近基线。:contentReference[oaicite:7]{index=7}
        print("[NOTE] SaProt usually expects structure-aware tokens (often containing '#'). "
              "If you only have AA sequences, performance may collapse to majority baseline.")
        X = embed_sequences_transformers(model_id, seqs, device, batch_size, max_length)

    else:
        # ESM2 / ANKH / Mistral-Prot
        X = embed_sequences_transformers(model_id, seqs, device, batch_size, max_length)

    np.save(cache_path, X)
    return X


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--seq_col", default="sequence")
    ap.add_argument("--label_col", default="label")  # 你可以传 --label_col positive

    ap.add_argument("--model_id", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=256)

    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data_csv)
    if args.seq_col not in df.columns:
        raise KeyError(f"seq_col '{args.seq_col}' not found. Columns={list(df.columns)}")
    if args.label_col not in df.columns:
        raise KeyError(f"label_col '{args.label_col}' not found. Columns={list(df.columns)}")

    seqs = df[args.seq_col].astype(str).tolist()
    y = df[args.label_col].astype(int).to_numpy()

    X = embed_sequences(args.model_id, seqs, args.device, args.batch_size, args.cache_dir, args.max_length)

    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(
        idx,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
        stratify=y
    )

    split_path = os.path.join(args.out_dir, f"split_seed{args.seed}_test{args.test_size}.npz")
    np.savez(split_path, train_idx=tr_idx, test_idx=te_idx)
    print("Saved split:", split_path)

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)),
        ("scaler", MinMaxScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")),
    ])

    pipe.fit(X_tr, y_tr)
    y_prob = pipe.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    m = compute_metrics(y_te, y_prob, y_pred)
    out = {
        "model_id": args.model_id,
        "n": int(len(df)),
        "dim": int(X.shape[1]),
        "metrics": m,
        "split": {"method": "train_test_split", "test_size": args.test_size, "seed": args.seed},
        "note": "ESMC/ESM3 use esm library; ANKH uses T5EncoderModel; DPLM prefers byprot; SaProt may need structure tokens."
    }

    tag = safe_model_tag(args.model_id)
    out_path = os.path.join(args.out_dir, f"{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Saved:", out_path, flush=True)
    print(json.dumps(m, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
