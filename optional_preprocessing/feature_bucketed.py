import pandas as pd
import numpy as np
import re
from pathlib import Path

IN_PATH  = "store_month_df.csv"
OUT_PATH = "store_month_df_bucketed.csv"

df = pd.read_csv(IN_PATH, encoding="utf-8-sig")

# =========================
# 0) 전역 NaN/문자 클린업 (추가)
# =========================
NULL_RX = r"^\s*(nan|NaN|null|None|N/A|NA)?\s*$"

# sentinel → NaN
df.replace(-999999.9, np.nan, inplace=True)
# 공백/문자 null → NaN
df.replace(NULL_RX, np.nan, regex=True, inplace=True)

# 문자열 컬럼 공백 제거
obj_cols = df.select_dtypes(include="object").columns
if len(obj_cols):
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

    # --- 퍼센트형 컬럼 숫자화(0~100) ---
pct_cols = [
    "RC_M1_SHC_FLP_UE_CLN_RAT",  # 유동 비중
    "RC_M1_SHC_WP_UE_CLN_RAT",   # 직장 비중
    "RC_M1_SHC_RSD_UE_CLN_RAT",  # 주거 비중
    "MCT_UE_CLN_REU_RAT",        # 재방문 비중
    "MCT_UE_CLN_NEW_RAT",        # 신규 비중
    "DLV_SAA_RAT"                # 배달 매출 비중
]
def to_pct_float(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace("％", "%")
    # "nn%" → "nn"
    if re.fullmatch(r"^\d+(\.\d+)?%$", s):
        s = s[:-1]
    # 기타 텍스트/공백 → NaN
    try:
        v = float(s)
    except:
        return np.nan
    # 합리 범위 밖은 NaN
    if v < 0 or v > 100: 
        return np.nan
    return v

for c in pct_cols:
    if c in df.columns:
        df[c] = df[c].apply(to_pct_float).astype("Float64")

bucket_cols = [
    "MCT_OPE_MS_CN",    # 운영개월수 구간
    "RC_M1_SAA",        # 매출금액 구간
    "RC_M1_TO_UE_CT",   # 매출건수 구간
    "RC_M1_UE_CUS_CN",  # 고객 수 구간
    "RC_M1_AV_NP_AT",   # 객단가 구간
    "APV_CE_RAT"        # 취소율 구간
]

MIDMAP = {1:5.0, 2:17.5, 3:37.5, 4:62.5, 5:82.5, 6:95.0}
EDGES  = [10, 25, 50, 75, 90]

# 하이픈/물결 전부 허용: -, ‒, – , —, ~, 〜
RANGE_SEP = r"[-\u2010\u2011\u2012\u2013\u2014~\u301C]"

def parse_bucket_any(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return (np.nan, np.nan)
    s0 = str(x).strip()
    if s0 == "" or s0.lower() == "nan":
        return (np.nan, np.nan)

    s = s0.replace("％", "%")
    s = re.sub(r"^\s*([1-6])\s*[_-]\s*", "", s)     # 앞 프리픽스 '3_' 제거
    s = re.sub(r"\(.*?\)", "", s)                   # 괄호 설명 제거
    s = s.replace(" ", "").replace("상위", "").replace("하위", "")
    s = s.replace(",", ".")  # 소수점 콤마 대응

    # 숫자 단독 → (1~6 코드) or (0~100 퍼센트)
    if re.fullmatch(r"^-?\d+(\.\d+)?$", s):
        v = float(s)
        if v.is_integer() and 1 <= v <= 6:
            b = int(v); return (b, MIDMAP[b])
        if 0 <= v <= 100:
            b = 1 + sum(v > e for e in EDGES)
            return (b, v)
        return (np.nan, np.nan)

    # 범위: 하이픈/물결 전부 지원
    m = re.fullmatch(rf"(\d+(?:\.\d+)?)%?{RANGE_SEP}(\d+(?:\.\d+)?)%?", s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        mid = np.clip((lo + hi) / 2.0, 0, 100)
        b = 1 + sum(mid > e for e in EDGES)
        return (b, mid)

    # 이하 / 초과
    m = re.fullmatch(r"(\d+(?:\.\d+)?)%?이하", s)
    if m:
        v = float(m.group(1))
        return (1, max(0.0, v / 2.0))
    m = re.fullmatch(r"(\d+(?:\.\d+)?)%?초과", s)
    if m:
        v = float(m.group(1))
        return (6, min(v + 5.0, 100.0))

    # 최후의 시도: 숫자만 추출
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if len(nums) >= 2:
        lo, hi = map(float, nums[-2:])
        mid = np.clip((lo + hi) / 2.0, 0, 100)
        b = 1 + sum(mid > e for e in EDGES)
        return (b, mid)
    if len(nums) == 1:
        v = float(nums[0])
        if 1 <= v <= 6 and v.is_integer():
            b = int(v); return (b, MIDMAP[b])
        if 0 <= v <= 100:
            b = 1 + sum(v > e for e in EDGES)
            return (b, v)

    return (np.nan, np.nan)

for c in bucket_cols:
    if c not in df.columns:
        print(f"[WARN] '{c}' 없음 → 건너뜀"); 
        continue
    if c.endswith("_BIN") or c.endswith("_MID"):
        continue

    # ── 진단: 값 형태 요약 ──
    sample = df[c].astype(str).str.strip().value_counts().head(8)
    print(f"\n[INFO] {c} top values:\n{sample}")

    # ── 케이스 A: 값이 전부 1~6 코드(문자/숫자) ──
    cleaned = df[c].astype(str).str.strip()
    is_code = cleaned.str.fullmatch(r"[1-6](\.0)?").fillna(False)
    if is_code.all():
        bin_ = pd.to_numeric(cleaned.str.replace(".0","", regex=False), errors="coerce").astype("Int64")
        mid_ = bin_.map(MIDMAP).astype(float)
        df[c+"_BIN"] = bin_
        df[c+"_MID"] = mid_
        print(f"[OK] {c}: detected pure 1~6 codes → fixed map applied")
        continue

    # ── 케이스 B: 혼합/텍스트 구간 → 범용 파서 ──
    parsed = df[c].apply(parse_bucket_any)
    df[c+"_BIN"] = parsed.apply(lambda x: x[0]).astype("Float64")
    df[c+"_MID"] = parsed.apply(lambda x: x[1]).astype("Float64")

    # 안전 장치
    mid_col = c+"_MID"
    df.loc[(df[mid_col] < 0) | (df[mid_col] > 100), mid_col] = np.nan

    print(f"[OK] {c}: parsed → BIN non-null={df[c+'_BIN'].notna().sum()}, MID non-null={df[mid_col].notna().sum()}")

# =========================
# 2) 버킷 변환 이후 전역 NaN 정리 (추가)
# =========================
print("\n[INFO] Cleaning residual NaN / sentinel values after bucketing...")
df.replace(-999999.9, np.nan, inplace=True)
df.replace(NULL_RX, np.nan, regex=True, inplace=True)

# 모든 _MID 컬럼은 0~100만 허용
for col in df.filter(like="_MID").columns:
    df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan

# (선택) 너무 큰 수치/이상치 제거
for col in df.select_dtypes(include=[np.number]).columns:
    df.loc[df[col] > 1e12, col] = np.nan

# 저장
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n[DONE] saved → {OUT_PATH}  shape={df.shape}")
print(df[["RC_M1_AV_NP_AT","RC_M1_AV_NP_AT_BIN","RC_M1_AV_NP_AT_MID"]].head(10))

