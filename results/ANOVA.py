from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy import stats

from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests


def _parse_fold_list(x) -> list[float]:
    """
    支持：
      - "0.1,0.2,0.3"
      - "0.1, 0.2, 0.3"
      - 带空值/None/NaN 的情况（会跳过空项）
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    vals = []
    for p in parts:
        if p == "" or p.lower() == "nan" or p.lower() == "none":
            continue
        vals.append(float(p))
    return vals


def analyze_cv_csv(
    csv_path: str | Path,
    metrics: tuple[str, ...] = ("mcc", "auc", "bacc"),
    alpha: float = 0.05,
    out_root: str | Path = "anova_batch_outputs",
    group_col: str = "solver_group",
    verbose: bool = True,
) -> dict:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if group_col not in df.columns:
        raise KeyError(f"CSV缺少分组列 {group_col}，现有列：{list(df.columns)}")

    out_root = Path(out_root)
    run_dir = out_root / csv_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "file": str(csv_path),
        "n_groups": int(df[group_col].nunique()),
        "groups": sorted(df[group_col].unique().tolist()),
        "metrics": {},
    }

    for metric in metrics:
        fold_col = f"fold_{metric}"
        if fold_col not in df.columns:
            if verbose:
                print(f"[skip] {csv_path.name}: 没有列 {fold_col}")
            continue

        metric_dir = run_dir / metric
        metric_dir.mkdir(parents=True, exist_ok=True)

        # 1) 拆 long
        rows = []
        for _, r in df.iterrows():
            g = r[group_col]
            vals = _parse_fold_list(r[fold_col])
            for i, v in enumerate(vals):
                rows.append({"split": i, group_col: g, metric: v})

        long = pd.DataFrame(rows)
        if long.empty:
            if verbose:
                print(f"[skip] {csv_path.name}/{metric}: long为空（可能 fold 列都是空）")
            continue

        # 2) pivot：split × group
        pivot = long.pivot(index="split", columns=group_col, values=metric)

        # 关键：RM-ANOVA / Friedman 需要“平衡数据”
        # 如果某些组缺少某些fold，会出现NaN；这里直接删掉含NaN的split（保留完全配对的fold）
        pivot_bal = pivot.dropna(axis=0, how="any")
        if pivot_bal.shape[0] < 2 or pivot_bal.shape[1] < 2:
            if verbose:
                print(f"[skip] {csv_path.name}/{metric}: 有效配对split或组数不足（split={pivot_bal.shape[0]}, groups={pivot_bal.shape[1]})")
            continue

        # 同步 long_bal
        long_bal = (
            pivot_bal.reset_index()
            .melt(id_vars=["split"], var_name=group_col, value_name=metric)
            .sort_values(["split", group_col])
            .reset_index(drop=True)
        )

        # 3) RM-ANOVA
        aov = AnovaRM(long_bal, depvar=metric, subject="split", within=[group_col]).fit()

        # 从AnovaRM表里取 F, df1, df2, p（尽量稳健解析）
        aov_table = aov.anova_table.copy()
        # AnovaRM 只有一个within因子时，index通常就是该因子名（比如 solver_group）
        idx = aov_table.index[0]
        F = float(aov_table.loc[idx, "F Value"])
        df1 = float(aov_table.loc[idx, "Num DF"])
        df2 = float(aov_table.loc[idx, "Den DF"])
        p_aov = float(aov_table.loc[idx, "Pr > F"])
        # partial eta squared（常用效应量）
        partial_eta2 = (F * df1) / (F * df1 + df2) if (F * df1 + df2) != 0 else np.nan

        # 4) Friedman（非参）
        fried = stats.friedmanchisquare(*[pivot_bal[c].values for c in pivot_bal.columns])
        p_fried = float(fried.pvalue)

        # 5) Post-hoc：配对t + Holm
        pairs, pvals_t, diffs = [], [], []
        cols = list(pivot_bal.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                t_stat, p = stats.ttest_rel(pivot_bal[a].values, pivot_bal[b].values)
                pairs.append((a, b))
                pvals_t.append(p)
                diffs.append(float(np.mean(pivot_bal[a].values - pivot_bal[b].values)))

        rej_t, p_holm_t, _, _ = multipletests(pvals_t, method="holm", alpha=alpha)
        post_t = pd.DataFrame({
            "A": [x[0] for x in pairs],
            "B": [x[1] for x in pairs],
            "mean_diff(A-B)": diffs,
            "p_raw": pvals_t,
            "p_holm": p_holm_t,
            f"reject_{alpha}": rej_t
        }).sort_values("p_holm", ascending=True)

        # 6) Post-hoc：Wilcoxon + Holm（更稳一点，但对全相等/零差值会报错；做保护）
        pvals_w = []
        ok_pairs = []
        diffs_w = []
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                x = pivot_bal[a].values
                y = pivot_bal[b].values
                d = x - y
                diffs_w.append(float(np.mean(d)))
                ok_pairs.append((a, b))
                try:
                    # zero_method='wilcox'：忽略零差值
                    w_stat, p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                except Exception:
                    p = np.nan
                pvals_w.append(p)

        # Holm 不能处理 NaN：先把NaN剔除做校正，再塞回去
        mask = ~pd.isna(pvals_w)
        pvals_w_np = np.array(pvals_w, dtype=float)
        rej_w = np.array([False] * len(pvals_w), dtype=bool)
        p_holm_w = np.array([np.nan] * len(pvals_w), dtype=float)
        if mask.sum() > 0:
            rej_tmp, p_holm_tmp, _, _ = multipletests(pvals_w_np[mask], method="holm", alpha=alpha)
            rej_w[mask] = rej_tmp
            p_holm_w[mask] = p_holm_tmp

        post_w = pd.DataFrame({
            "A": [x[0] for x in ok_pairs],
            "B": [x[1] for x in ok_pairs],
            "mean_diff(A-B)": diffs_w,
            "p_raw": pvals_w_np,
            "p_holm": p_holm_w,
            f"reject_{alpha}": rej_w
        }).sort_values("p_holm", ascending=True)

        # 7) mean±std
        summ = pivot_bal.agg(["mean", "std"]).T.sort_values("mean", ascending=False)
        summ["mean±std"] = summ["mean"].map(lambda v: f"{v:.6f}") + " ± " + summ["std"].map(lambda v: f"{v:.6f}")

        # 8) 写文件
        long_bal.to_csv(metric_dir / "long.csv", index=False)
        pivot_bal.to_csv(metric_dir / "pivot.csv")
        summ.to_csv(metric_dir / "summary_mean_std.csv")
        post_t.to_csv(metric_dir / "posthoc_paired_t_holm.csv", index=False)
        post_w.to_csv(metric_dir / "posthoc_wilcoxon_holm.csv", index=False)

        with open(metric_dir / "stats.txt", "w", encoding="utf-8") as f:
            f.write(f"FILE: {csv_path}\n")
            f.write(f"METRIC: {metric}\n")
            f.write(f"GROUPS: {cols}\n")
            f.write(f"VALID_SPLITS(after dropna): {pivot_bal.shape[0]}\n\n")
            f.write("=== Repeated-measures ANOVA (AnovaRM) ===\n")
            f.write(str(aov) + "\n\n")
            f.write(f"Parsed: F={F:.6f}, df1={df1:.0f}, df2={df2:.0f}, p={p_aov:.6g}, partial_eta2={partial_eta2:.6f}\n\n")
            f.write("=== Friedman test ===\n")
            f.write(f"stat={fried.statistic:.6f}, p={p_fried:.6g}\n")

        # 9) 汇总到返回值
        results["metrics"][metric] = {
            "rm_anova_F": F,
            "rm_anova_df1": df1,
            "rm_anova_df2": df2,
            "rm_anova_p": p_aov,
            "partial_eta2": partial_eta2,
            "friedman_stat": float(fried.statistic),
            "friedman_p": p_fried,
            "valid_splits": int(pivot_bal.shape[0]),
            "out_dir": str(metric_dir),
            "best_group_by_mean": str(summ.index[0]),
            "best_mean": float(summ.loc[summ.index[0], "mean"]),
            "best_std": float(summ.loc[summ.index[0], "std"]),
        }

        if verbose:
            print(f"[done] {csv_path.name}/{metric}: RM-ANOVA p={p_aov:.4g}, Friedman p={p_fried:.4g}, best={summ.index[0]} mean={summ.loc[summ.index[0],'mean']:.4f}")

    return results


def batch_run(
    csv_files: list[str | Path],
    metrics: tuple[str, ...] = ("mcc", "auc", "bacc"),
    alpha: float = 0.05,
    out_root: str | Path = "anova_batch_outputs",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    批量跑多个csv，返回一个总汇总表（DataFrame），并额外写出 batch_summary.csv
    """
    all_rows = []
    for p in csv_files:
        res = analyze_cv_csv(p, metrics=metrics, alpha=alpha, out_root=out_root, verbose=verbose)
        # 展平成行：每个文件×每个metric一行
        for metric, info in res.get("metrics", {}).items():
            row = {"file": res["file"], "metric": metric, **info}
            all_rows.append(row)

    summary = pd.DataFrame(all_rows)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_root / "batch_summary.csv", index=False)
    if verbose:
        print(f"\n[batch] summary saved -> {out_root / 'batch_summary.csv'}")
    return summary


if __name__ == "__main__":
    # 方式1：手动列出文件
    csv_list = [
        "solver_cv_Synthyra_ANKH_large_seed42_test0.2.csv",
        "solver_cv_EvolutionaryScale_esmc-300m-2024-12_seed42_test0.2.csv",
        "solver_cv_EvolutionaryScale_esmc-600m-2024-12_seed42_test0.2.csv",
        "solver_cv_facebook_esm2_t33_650M_UR50D_seed42_test0.2.csv",
        "solver_cv_EvolutionaryScale_esm3-sm-open-v1_seed42_test0.2.csv",
        "solver_cv_Synthyra_ANKH_base_seed42_test0.2.csv",
        "solver_cv_airkingbd_dplm_650m_seed42_test0.2.csv",
        "solver_cv_RaphaelMourad_Mistral-Prot-v1-134M_seed42_test0.2.csv",
        # "solver_cv_xxx.csv",
        # ...
    ]
    # 方式2：用通配符自动搜（把目录改成你放csv的目录）
    # csv_list = sorted([str(p) for p in Path(".").glob("solver_cv_*.csv")])

    batch_run(csv_list, metrics=("mcc", "auc", "bacc"), alpha=0.05, out_root="anova_batch_outputs")