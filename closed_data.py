import pandas as pd
import numpy as np

# ===== 0) 파일 경로 =====
PATH_DATA = "store_month_df_bucketed.csv"   # 이미 병합된 최종 파일

# ===== 1) 로드 =====
df = pd.read_csv(PATH_DATA, encoding="utf-8-sig")

# ===== 2) 폐점 여부 플래그 =====
# 컬럼명이 'MCT_ME_D' (폐점일)이라고 가정
df["is_closed"] = df["MCT_ME_D"].notna()   # True=폐점, False=영업중

# ===== 3) 숫자형/결측 처리 =====
SV_SENTINELS = {-999999.9, -9999999.9, -999999}  # 문서의 SV 값들

def clean_numeric(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.mask(s.isin(SV_SENTINELS), np.nan)

# 비교할 컬럼들 (폐점 vs 영업중)
compare_cols = [
    # dataset2 쪽 수치형
    "RC_M1_SAA_BIN",           # 매출금액 점수(이미 변환된 수치형)
    "RC_M1_TO_UE_CT_BIN",      # 매출건수 점수
    "RC_M1_UE_CUS_CT_BIN",     # 유니크 고객수 점수
    "RC_M1_AV_NP_AT_BIN",      # 객단가 점수
    "APV_CE_RAT_BIN",          # 취소율 점수(구간→수치 변환본)

    "M1_SME_RY_SAA_RAT",         # 동일 업종 매출금액 비율
    "M1_SME_RY_CNT_RAT",         # 동일 업종 매출건수 비율
    "M12_SME_RY_SAA_PCE_RT",     # 업종 내 매출 순위 비율(0~100, 낮을수록 상위)
    "M12_SME_BZN_SAA_PCE_RT",    # 상권 내 매출 순위 비율(있으면)
    "M12_SME_RY_ME_MCT_RAT",     # 동일 업종 내 해지 가맹점 비중
    "M12_SME_BZN_ME_MCT_RAT",    # 동일 상권 내 해지 가맹점 비중
    "DLV_SAA_RAT",               # 배달매출 비율

    # dataset3 쪽 수치형
    "MCT_UE_CLN_REU_RAT",        # 재방문 고객 비중
    "MCT_UE_CLN_NEW_RAT",        # 신규 고객 비중
    "RC_M1_SHC_RSD_UE_CLN_RAT",  # 거주 고객 비중
    "RC_M1_SHC_WP_UE_CLN_RAT",   # 직장 고객 비중
    "RC_M1_SHC_FLP_UE_CLN_RAT",  # 유동 고객 비중

    # 연령/성별 비중(대표)
    "M12_MAL_30_RAT","M12_MAL_40_RAT","M12_MAL_50_RAT","M12_MAL_60_RAT",
    "M12_FME_30_RAT","M12_FME_40_RAT","M12_FME_50_RAT","M12_FME_60_RAT"
]

# 실제 존재하는 컬럼만 사용 + 숫자/결측 정리
cols_in_df = [c for c in compare_cols if c in df.columns]
for c in cols_in_df:
    df[c] = clean_numeric(df[c])

# ===== 4) 요약 통계 (폐점 vs 영업중) =====
def summarize_by_status(df, cols, status_col="is_closed"):
    closed_df = df[df[status_col]]
    active_df = df[~df[status_col]]
    rows = []
    for col in cols:
        rows.append({
            "Index": col,
            "Closed_mean": closed_df[col].mean(),
            "Open_mean": active_df[col].mean(),
            "Closed_median": closed_df[col].median(),
            "Open_median": active_df[col].median(),
            "Closed_std": closed_df[col].std(),
            "Open_std": active_df[col].std()
        })
    return pd.DataFrame(rows).sort_values("지표")

summary_df = summarize_by_status(df, cols_in_df)

# ===== 5) 저장/확인 =====
summary_df.to_csv("versus_closed.csv", index=False, encoding="utf-8-sig")
print("요약 저장 완료 → 폐점_vs_영업중_요약지표.csv")
print(summary_df.head(15))