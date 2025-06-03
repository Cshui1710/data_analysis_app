import pandas as pd

# 正しいエンコーディングで再読み込み
df = pd.read_csv("saitama_data2.csv", encoding="utf-8-sig")

# カンマを除去し、数値変換
for col in df.columns[2:]:  # 「調査年」「地域」以外の列を処理
    df[col] = df[col].astype(str).str.replace(",", "", regex=False)
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 欠損値を前後の平均で補完 → それでも欠けるなら列平均
for col in df.columns[2:]:
    df[col] = df[col].interpolate(method='linear', limit_direction='both')
    df[col] = df[col].fillna(df[col].mean())

# すべて int に変換
for col in df.columns[2:]:
    df[col] = df[col].astype(int)

# 保存（必要に応じて）
df.to_csv("saitama_data2_cleaned.csv", index=False, encoding="utf-8-sig")
