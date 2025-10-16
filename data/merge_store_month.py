# merge_store_month.py
import pandas as pd
from pathlib import Path

# ---- 0) 파일 경로 (본인 환경에 맞게 바꾸세요) ----
dataset1_path = r"C:\Users\LG\Desktop\Bigcontest_Agent\data\big_data_set1_f.csv"  # 개요
dataset2_path = r"C:\Users\LG\Desktop\Bigcontest_Agent\data\big_data_set2_f.csv"  # 월별 이용
dataset3_path = r"C:\Users\LG\Desktop\Bigcontest_Agent\data\big_data_set3_f.csv"  # 월별 고객
out_path      = r"C:\Users\LG\Desktop\Bigcontest_Agent\data\store_month_df.csv"
# ---- 1) 인코딩 자동 판별 로더 (UTF-8 / CP949 / EUC-KR 순차 시도) ----
def read_csv_smart(path, **kwargs):
    tried = []
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            print(f"[INFO] Loaded {Path(path).name} with encoding='{enc}'")
            return df
        except UnicodeDecodeError as e:
            tried.append((enc, str(e)))
        except Exception as e:
            tried.append((enc, str(e)))
    # 마지막으로 python 엔진 + 오류 치환 시도 (따옴표/구분자 문제 대비)
    try:
        df = pd.read_csv(path, encoding="cp949", engine="python", on_bad_lines="skip", **kwargs)
        print(f"[WARN] Fallback: encoding='cp949', engine='python', on_bad_lines='skip'")
        return df
    except Exception as e:
        msg = "\n".join([f" - {enc}: {err}" for enc, err in tried])
        raise RuntimeError(f"Failed to read {path} with common encodings:\n{msg}\nLast error: {e}")

# ---- 2) 공통 클린업 ----
def clean_common(df):
    # -999999.9 같은 sentinel 값 → NaN
    df = df.replace(-999999.9, pd.NA)

    # 키 타입 통일
    if "ENCODED_MCT" in df.columns:
        df["ENCODED_MCT"] = df["ENCODED_MCT"].astype(str)

    # TA_YM 표준화: YYYY-MM 문자열로
    if "TA_YM" in df.columns:
        ym = (
            df["TA_YM"]
            .astype(str)
            .str.replace(r"[^0-9]", "", regex=True)  # 숫자만 남김
        )
        # 우선 YYYYMM 해석
        parsed = pd.to_datetime(ym, format="%Y%m", errors="coerce")
        # 안 되면 YYYY-MM도 시도
        parsed = parsed.fillna(pd.to_datetime(df["TA_YM"].astype(str), format="%Y-%m", errors="coerce"))
        df["TA_YM"] = parsed.dt.to_period("M").astype(str)
    return df

# ---- 3) 로드 ----
d1 = read_csv_smart(dataset1_path)   # 개요(정적)
d2 = read_csv_smart(dataset2_path)   # 월별 이용(본체)
d3 = read_csv_smart(dataset3_path)   # 월별 고객

d1 = clean_common(d1)
d2 = clean_common(d2)
d3 = clean_common(d3)

# ---- 4) 병합 (권장 순서: d2 ← d3 ← d1) ----
# 4-1) d2(월별 이용) ⟵ d3(월별 고객)
key_cols = ["ENCODED_MCT", "TA_YM"]
if not all(k in d2.columns for k in key_cols):
    raise KeyError("dataset2(월별 이용)에 ENCODED_MCT, TA_YM가 있어야 합니다.")
if not all(k in d3.columns for k in key_cols):
    raise KeyError("dataset3(월별 고객)에 ENCODED_MCT, TA_YM가 있어야 합니다.")

use = d2.merge(d3, on=key_cols, how="left", suffixes=("", "_CUST"))

# 4-2) ⟵ d1(개요, 정적) : ENCODED_MCT 기준
if "ENCODED_MCT" not in d1.columns:
    raise KeyError("dataset1(개요)에 ENCODED_MCT가 있어야 합니다.")

store_month_df = use.merge(d1, on="ENCODED_MCT", how="left", suffixes=("", "_OVERVIEW"))

# ---- 5) 중복 키 검사 & 정리 (있으면 최신/마지막 행 기준으로 남김) ----
dup_mask = store_month_df.duplicated(subset=key_cols, keep=False)
if dup_mask.any():
    print(f"[WARN] Duplicated keys found: {dup_mask.sum()} rows")
    # 필요 시 규칙 변경: 여기서는 마지막 행을 남김
    store_month_df = store_month_df.drop_duplicates(subset=key_cols, keep="last")

# ---- 6) 저장 (엑셀 호환을 위해 utf-8-sig 권장) ----
store_month_df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"[OK] Saved merged CSV: {out_path}  (rows={len(store_month_df):,}, cols={store_month_df.shape[1]})")
