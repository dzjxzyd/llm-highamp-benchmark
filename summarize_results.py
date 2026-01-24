import glob, json, os
import pandas as pd

rows = []
for path in glob.glob(os.path.join("results", "*.json")):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    s = obj.get("summary", {})
    def get_mean(k):
        v = s.get(k, {})
        return v.get("mean", None) if isinstance(v, dict) else None

    rows.append({
        "model_id": obj.get("model_id"),
        "dim": obj.get("dim"),
        "ACC": get_mean("ACC"),
        "BACC": get_mean("BACC"),
        "Sn": get_mean("Sn"),
        "Sp": get_mean("Sp"),
        "MCC": get_mean("MCC"),
        "AUC": get_mean("AUC"),
        "AP": get_mean("AP"),
        "file": os.path.basename(path),
    })

df = pd.DataFrame(rows)
if not df.empty:
    df = df.sort_values(by=["MCC","AP","AUC"], ascending=[False, False, False], na_position="last")

out_csv = os.path.join("results", "summary.csv")
out_xlsx = os.path.join("results", "summary.xlsx")
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
df.to_excel(out_xlsx, index=False)

print("Saved:", out_csv)
print("Saved:", out_xlsx)
print(df.head(20).to_string(index=False))
