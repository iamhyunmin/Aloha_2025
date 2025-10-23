import pandas as pd

df = pd.read_csv("store_with_rag_text.csv", encoding="utf-8-sig")

# rag_text 결측 여부 확인
missing_count = df["rag_text"].isna().sum()
empty_count = (df["rag_text"].str.strip() == "").sum()

print(f"결측 행: {missing_count}개")
print(f"공백 행: {empty_count}개")
print(f"총 데이터: {len(df):,}개")
