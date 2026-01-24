# summarize_results.py
import argparse
import glob
import json
import os
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd


METRICS = ["ACC", "BACC", "Sn", "Sp", "MCC", "AUC", "AP"]


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def extract_mean_std(obj: Dict[str, Any], k: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Supports:
    1) CV summary:
       obj["summary"][k] = {"mean":..., "std":...}
    2) single split:
       obj["metrics"][k] = value
    3) flat metrics:
       obj[k] = value  (just in case)
    4) folds list:
       obj["folds"] = [{"metrics": {...}}, ...] or [{"ACC":...}, ...]
    """
    # 1) summary dict
    s = obj.get("summary")
    if isinstance(s, dict) and isinstance(s.get(k), dict):
        mean = _safe_float(s[k].get("mean"))
        std = _safe_float(s[k].get("std"))
        return mean, std

    # 4) folds
    folds = obj.get("folds")
    if isinstance(folds, list) and len(folds) > 0:
        vals: List[float] = []
        for f in folds:
            if not isinstance(f, dict):
                continue
            fm = f.get("metrics") if isinstance(f.get("metrics"), dict) else f
            if isinstance(fm, dict) and k in fm:
                v = _safe_float(fm.get(k))
                if v is not None:
                    vals.append(v)
        if len(vals) > 0:
            arr = np.asarray(vals, dtype=float)
            return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    # 2) metrics dict
    m = obj.get("metrics")
    if isinstance(m, dict) and k in m:
        return _safe_float(m.get(k)), None

    # 3) flat
    if k in obj:
        return _safe_float(obj.get(k)), None

    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results", help="Directory containing *.json result files")
    ap.add_argument("--pattern", default="*.json", help="Glob pattern inside results_dir")
    ap.add_argument("--out_prefix", default="summary", help="Output filename prefix (csv/xlsx)")
    ap.add_argument("--include_std", action="store_true", help="Also output *_std columns if available")
    ap.add_argument("--topk", type=int, default=20, help="Print top-k rows to console")
    args = ap.parse_args()

    json_paths = sorted(glob.glob(os.path.join(args.results_dir, args.pattern)))
    if not json_paths:
        print(f"[WARN] No files found: {os.path.join(args.results_dir, args.pattern)}")
        return

    rows = []
    for path in json_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {type(e).__name__}: {e}")
            continue

        row = {
            "model_id": obj.get("model_id"),
            "dim": obj.get("dim"),
            "file": os.path.basename(path),
        }

        # Optional: mark run type
        if isinstance(obj.get("split"), dict):
            row["eval"] = obj["split"].get("method", "split")
        elif isinstance(obj.get("cv"), dict):
            row["eval"] = obj["cv"].get("method", "cv")
        else:
            row["eval"] = None

        # metrics
        for k in METRICS:
            mean, std = extract_mean_std(obj, k)
            row[k] = mean
            if args.include_std:
                row[f"{k}_std"] = std

        rows.append(row)

    df = pd.DataFrame(rows)

    # sort: prioritize MCC, then AP, then AUC
    if not df.empty:
        sort_cols = ["MCC", "AP", "AUC"]
        sort_cols = [c for c in sort_cols if c in df.columns]
        if sort_cols:
            df = df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols), na_position="last")

    out_csv = os.path.join(args.results_dir, f"{args.out_prefix}.csv")
    out_xlsx = os.path.join(args.results_dir, f"{args.out_prefix}.xlsx")

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df.to_excel(out_xlsx, index=False)  # uses openpyxl/xlsxwriter engine

    print("Saved:", out_csv)
    print("Saved:", out_xlsx)
    if not df.empty:
        print(df.head(args.topk).to_string(index=False))


if __name__ == "__main__":
    main()
