import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, EsmModel
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score, balanced_accuracy_score


AA_OK = set(list("ACDEFGHIKLMNPQRSTVWYXBZUOJ"))  # 允许的字符集合（非标准会转X）


def clean_aa(seq: str) -> str:
    s = (seq or "").strip().upper()
    # 把非字母去掉
    s = re.sub(r"[^A-Z]", "", s)
    # 非常规AA统一变成X
    s = "".join([c if c in AA_OK else "X" for c in s])
    # ESM 常见处理：U/O/J/Z/B 等也可以当未知
    s = s.replace("U", "X").replace("O", "X").replace("J", "X").replace("Z", "X").replace("B", "X")
    return s


def mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden: [B, L, H]
    mask:        [B, L]  True=参与平均
    """
    mask_f = mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp_min(1e-8)
    return summed / denom


@torch.no_grad()
def embed_sequences_esm_transformers(
    model_id: str,
    seqs: list[str],
    device: str,
    batch_size: int = 2,
    max_aa_len: int | None = None,
    fp16: bool = False,
) -> np.ndarray:
    """
    纯 transformers：EsmModel + AutoTokenizer
    返回 [N, hidden_size] 的 numpy
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = EsmModel.from_pretrained(model_id).to(device).eval()

    # 这个 checkpoint 的 config 显示 max_position_embeddings=1026（含 special tokens）:contentReference[oaicite:3]{index=3}
    # 所以原始AA长度建议 <= 1024（保守点可以 1022）
    if max_aa_len is None:
        # tokenizer.model_max_length 通常等于 1026（含 special tokens）
        # 为避免各版本差异，这里保守减2
        max_aa_len = min(int(getattr(tokenizer, "model_max_length", 1026)), 1026) - 2

    special_ids = {tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.eos_token_id,
                   getattr(tokenizer, "bos_token_id", None), getattr(tokenizer, "sep_token_id", None)}
    special_ids = {i for i in special_ids if i is not None}

    vecs = []
    for i in tqdm(range(0, len(seqs), batch_size), desc=f"Embedding({model_id})"):
        batch = [clean_aa(s)[:max_aa_len] for s in seqs[i:i+batch_size]]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_aa_len + 2,  # +2 给 CLS/EOS 留空间
            add_special_tokens=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if fp16 and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(**inputs)
        else:
            out = model(**inputs)

        last_hidden = out.last_hidden_state  # [B,L,H]

        # mask：先用 attention_mask 去掉padding，再去掉 special tokens（CLS/EOS/PAD...）
        mask = inputs["attention_mask"].bool()
        for sid in special_ids:
            mask &= (inputs["input_ids"] != sid)

        pooled = mean_pool(last_hidden, mask)  # [B,H]
        vecs.append(pooled.float().cpu().numpy())

    X = np.concatenate(vecs, axis=0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def embed_sequences_esm3(
    model_id: str,
    seqs: list[str],
    device: str,
    fp16: bool = False,
) -> np.ndarray:
    """
    ESM3 (esm package) -> [N, D] numpy
    model_id 例子: "esm3_sm_open_v1"
    """
    # 只在用到 ESM3 时才 import，避免没装 esm 的时候影响别的模型
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, SamplingConfig

    client = ESM3.from_pretrained(model_id).to(device)
    client.eval()

    vecs = []
    with torch.no_grad():
        for seq in tqdm(seqs, desc=f"Embedding (ESM3:{model_id})"):
            s = clean_aa(seq)
            protein = ESMProtein(sequence=s)
            protein_tensor = client.encode(protein)

            if fp16 and device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = client.forward_and_sample(
                        protein_tensor,
                        SamplingConfig(return_per_residue_embeddings=True)
                    )
            else:
                out = client.forward_and_sample(
                    protein_tensor,
                    SamplingConfig(return_per_residue_embeddings=True)
                )

            emb = getattr(out, "per_residue_embedding", None)
            if emb is None:
                emb = getattr(out, "per_residue_embeddings", None)
            if emb is None:
                raise RuntimeError("ESM3 output has no per_residue_embedding(s).")

            # emb: [1, L, D] or [L, D]
            if emb.dim() == 3:
                emb = emb[0]

            # 有些版本会包含 BOS/EOS：长度 = len(seq)+2，就去掉两端
            if emb.size(0) == len(s) + 2:
                emb = emb[1:-1]

            vec = emb.mean(dim=0)  # [D]
            vecs.append(vec.float().cpu().numpy())

    X = np.stack(vecs, axis=0)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    return mcc, auc, bacc

import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

def build_model_pipeline(model_name: str, seed: int):
    """
    返回: (pipeline, uses_proba)
    pipeline: sklearn Pipeline
    uses_proba: 是否能直接 predict_proba（否则会做校准）
    """
    model_name = model_name.lower()

    if model_name == "logreg":
        clf = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=5000
        )
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
        return pipe, True

    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        )
        # 树模型不需要标准化，直接 passthrough
        pipe = Pipeline([
            ("scaler", "passthrough"),
            ("clf", clf),
        ])
        return pipe, True

    # 你还可以继续加：SVM/GBDT 等
    raise ValueError(f"Unknown model_name: {model_name}")


def get_prob(pipe, X):
    """
    统一拿到“正类概率”。优先 predict_proba，其次 decision_function + sigmoid，
    再不行就直接报错。
    """
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]

    if hasattr(pipe, "decision_function"):
        s = pipe.decision_function(X)
        # sigmoid 映射到(0,1)，用于AUC/MCC阈值
        return 1 / (1 + np.exp(-s))

    raise ValueError("Model has neither predict_proba nor decision_function.")

def main():
    # ========= VS Code 直接改这里 =========
    data_csv  = r"E:\UCD\new adventure\Thesis\main\data.csv"
    seq_col   = "sequence"
    label_col = "label"
    model_id  = "airkingbd/dplm_650m"

    test_size = 0.2
    seed      = 42

    batch_size = 2
    fp16       = True
    max_aa_len = None

    out_dir    = Path(r"E:\UCD\new adventure\Thesis\main\run_compare_models")
    emb_cache  = "emb_dplm650m.npy"

    # 要对比的模型（名字用 build_model_pipeline 里定义的 key）
    models_to_run = [
        "logreg",
        "rf",
    ]
    # =====================================

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_csv)
    seqs = df[seq_col].astype(str).tolist()
    y = df[label_col].astype(int).to_numpy()
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("当前脚本默认二分类，label 必须是 0/1。")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # embeddings（缓存一次，多模型复用）
    emb_path = out_dir / emb_cache
    if emb_path.exists():
        X = np.load(emb_path)
    else:
        X = embed_sequences_esm_transformers(
            model_id=model_id,
            seqs=seqs,
            device=device,
            batch_size=batch_size,
            max_aa_len=max_aa_len,
            fp16=fp16,
        )
        np.save(emb_path, X)

    # 固定一次 split（保证不同模型公平）
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    all_summary_rows = []

    for model_name in models_to_run:
        print(f"\n===== Running model: {model_name} =====")
        model_run_dir = out_dir / model_name
        model_run_dir.mkdir(parents=True, exist_ok=True)

        fold_rows = []
        t0 = time.perf_counter()

        for fold, (tr, va) in enumerate(cv.split(X_tr, y_tr)):
            pipe, _ = build_model_pipeline(model_name, seed)

            pipe.fit(X_tr[tr], y_tr[tr])
            prob = get_prob(pipe, X_tr[va])

            mcc, auc, bacc = compute_metrics(y_tr[va], prob)
            fold_rows.append({"fold": fold, "mcc": mcc, "auc": auc, "bacc": bacc})

        train_secs = time.perf_counter() - t0

        cv_df = pd.DataFrame(fold_rows)
        cv_df.to_csv(model_run_dir / "cv_metrics.csv", index=False)

        # test
        pipe, _ = build_model_pipeline(model_name, seed)
        pipe.fit(X_tr, y_tr)
        te_prob = get_prob(pipe, X_te)
        te_mcc, te_auc, te_bacc = compute_metrics(y_te, te_prob)

        res = {
            "model_id": model_id,
            "device": device,
            "model_name": model_name,
            "data": {"n": int(len(y)), "test_size": test_size, "seed": seed},
            "cv": {
                "mcc_mean": float(cv_df["mcc"].mean()),
                "mcc_std": float(cv_df["mcc"].std(ddof=1)),
                "auc_mean": float(cv_df["auc"].mean(skipna=True)),
                "auc_std": float(cv_df["auc"].std(ddof=1, skipna=True)),
                "bacc_mean": float(cv_df["bacc"].mean()),
                "bacc_std": float(cv_df["bacc"].std(ddof=1)),
            },
            "test": {"mcc": float(te_mcc), "auc": float(te_auc), "bacc": float(te_bacc)},
            "train_time_sec_cv": float(train_secs),
            "embedding_cache": str(emb_path),
            "out_dir": str(model_run_dir),
        }
        with open(model_run_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        # 汇总行（方便对比）
        all_summary_rows.append({
            "model": model_name,
            "cv_mcc_mean": res["cv"]["mcc_mean"],
            "cv_mcc_std":  res["cv"]["mcc_std"],
            "cv_auc_mean": res["cv"]["auc_mean"],
            "cv_auc_std":  res["cv"]["auc_std"],
            "cv_bacc_mean": res["cv"]["bacc_mean"],
            "cv_bacc_std":  res["cv"]["bacc_std"],
            "test_mcc": res["test"]["mcc"],
            "test_auc": res["test"]["auc"],
            "test_bacc": res["test"]["bacc"],
            "cv_time_sec": res["train_time_sec_cv"],
        })

        print(json.dumps(res, ensure_ascii=False, indent=2))

    summary_df = pd.DataFrame(all_summary_rows).sort_values("cv_mcc_mean", ascending=False)
    summary_df.to_csv(out_dir / "comparison_summary.csv", index=False)
    print("\n=== Saved comparison_summary.csv ===")
    print(summary_df)

if __name__ == "__main__":
    main()