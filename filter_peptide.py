import pandas as pd

infile = "mic_data.csv"
outfile = "peptide_len_gt25.csv"
peptide_col = "sequence"  # 改成你的真实列名，比如 sequence

df = pd.read_csv(infile)

# 处理缺失值，避免报错；按字符长度筛选
mask = df[peptide_col].fillna("").astype(str).str.len() < 26
df_out = df.loc[mask].copy()

df_out.to_csv(outfile, index=False, encoding="utf-8-sig")
print("原始行数:", len(df), "筛选后行数:", len(df_out))
