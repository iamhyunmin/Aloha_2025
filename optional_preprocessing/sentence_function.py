import pandas as pd
import re
from typing import Any, Dict

# -------------------------
# 0) 파일 경로 설정
# -------------------------
IN_PATH  = "store_month_df_bucketed.csv"     # 입력 파일
OUT_PATH = "store_with_rag_text.csv"         # 출력 파일
CLOSE_COL = "MCT_ME_D"                       # 폐점일 컬럼

# -------------------------
# 1) 유틸 함수들
# -------------------------
def safe_str(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def format_year_month(val: Any) -> str:
    s = str(val).replace("-", "")
    if len(s) >= 6 and s[:6].isdigit():
        y, m = s[:4], s[4:6].lstrip("0") or "0"
        return f"{y}년 {m}월"
    return str(val)

def pct_fmt(x):
    """퍼센트 숫자를 문자열로 포맷. 1% 미만은 None으로 처리."""
    try:
        if pd.isna(x):
            return None
        v = float(x)
        if v < 1:
            return None  # 0% 또는 1% 미만은 정보 없음으로 처리
        return f"{v:.0f}%"
    except:
        return None
    
def josa_eun_neun(word: str) -> str:
    """단어 끝 받침 유무에 따라 '은/는' 자동 선택"""
    if not word:
        return "은"
    last_char = word[-1]
    code = ord(last_char) - 44032
    if code < 0 or code > 11171:
        return "은"
    return "은" if code % 28 != 0 else "는"

def extract_addr_from_MCT_BSE_AR(x: Any) -> str:
    s = safe_str(x)
    if not s:
        return ""
    s = s.split("\n")[0].split(",")[0].strip()
    m = re.search(r"([^,]*?(?:시|군|구)\s*[^\s,]*?(?:로|길)\s*\d+(?:-\d+)?)", s)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    m2 = re.search(r"(?:로|길)\s*\d+(?:-\d+)?", s)
    if m2:
        return s
    return s

def filter_open_stores(df: pd.DataFrame, close_col: str = "MCT_ME_D") -> pd.DataFrame:
    if close_col not in df.columns:
        print(f"[WARN] '{close_col}' 컬럼이 없습니다. 전체 데이터 사용.")
        return df
    s = df[close_col].astype(str).str.strip()
    is_open = s.eq("") | s.eq("0") | s.eq("00000000") | s.str.lower().isin({"nan","none","null"})
    print(f"[INFO] 폐점일 컬럼: {close_col} | 사용 행: {is_open.sum():,}/{len(df):,}")
    return df.loc[is_open].copy()

# -------------------------
# 2) RAG 문장화 함수
# -------------------------
def make_rag_sentence(row: pd.Series) -> str:
    ym_txt = format_year_month(row.get("TA_YM", ""))
    name   = safe_str(row.get("MCT_NM")) or "이름없는 매장"
    addr   = extract_addr_from_MCT_BSE_AR(row.get("MCT_BSE_AR"))
    biz    = safe_str(row.get("HPSN_MCT_ZCD_NM")) or "업종 정보 없음"
    key    = safe_str(row.get("ENCODED_MCT"))
    where  = safe_str(row.get("MCT_SIGUNGU_NM"))

    prefix = f"[ADDR={addr}] [KEY={key}] " if addr or key else ""
    head = f"{ym_txt} 기준, {where}의 {name} 매장은 업종 {biz}입니다."

    sentences = []

    # ① 구간형 (5개)
    bin_fields = {
        "RC_M1_SAA_MID": "매출금액",
        "RC_M1_TO_UE_CT_MID": "거래건수",
        "RC_M1_AV_NP_AT_MID": "평균 객단가",
        "RC_M1_UE_CUS_CN_MID": "순고객수",
        "APV_CE_RAT_MID": "결제취소율",
    }
    for col, label in bin_fields.items():
        val = pct_fmt(row.get(col))
        if not val:
            continue
        josa = josa_eun_neun(label)
        if "APV_CE_RAT" in col:
            sentences.append(f"{label}{josa} 업종 내 하위 {val} 수준으로, 낮을수록 성과가 우수합니다.")
        else:
            sentences.append(f"{label}{josa} 업종 내 상위 {val} 수준으로, 낮을수록 성과가 우수합니다.")

    # ② 비율형 (4개)
    ratio_fields = {
        "DLV_SAA_RAT": "배달 매출금액 비율",
        "M1_SME_RY_SAA_RAT": "동일 업종 매출금액 비율",
        "M1_SME_RY_CNT_RAT": "동일 업종 매출건수 비율",
    }
    for col, label in ratio_fields.items():
        val = pct_fmt(row.get(col))
        if not val: continue
        sentences.append(f"{label}은 동일 업종 평균의 {val} 수준입니다.")

    # ③ 순위비율 (2개)
    rank_fields = {
        "M12_SME_RY_SAA_PCE_RT": "동일 업종 내 매출 순위",
        "M12_SME_BZN_SAA_PCE_RT": "동일 상권 내 매출 순위",
    }
    for col, label in rank_fields.items():
        val = pct_fmt(row.get(col))
        if val:
            sentences.append(f"{label}는 상위 {val} 수준입니다.")

    # ④ 고객 방문유형
    reu = pct_fmt(row.get("MCT_UE_CLN_REU_RAT"))
    new = pct_fmt(row.get("MCT_UE_CLN_NEW_RAT"))
    if reu or new:
        sentences.append(f"재방문 고객은 {reu or '정보없음'}, 신규 고객은 {new or '정보없음'} 비중입니다.")

    # ⑤ 소비자 생활권 유형
    flp = pct_fmt(row.get("RC_M1_SHC_FLP_UE_CLN_RAT"))
    wp  = pct_fmt(row.get("RC_M1_SHC_WP_UE_CLN_RAT"))
    rsd = pct_fmt(row.get("RC_M1_SHC_RSD_UE_CLN_RAT"))
    segs = [f"유동 {flp}" for flp in [flp] if flp] + [f"직장 {wp}" for wp in [wp] if wp] + [f"주거 {rsd}" for rsd in [rsd] if rsd]
    if segs:
        sentences.append(f"소비자 유형은 {', '.join(segs)}입니다.")

    # ⑥ 성별·연령별 고객 비중
    age_cols = [c for c in row.index if re.match(r"M12_(FME|MAL)_(10|20|30|40|50|60)_RAT", c)]
    age_parts = []
    for c in age_cols:
        val = pct_fmt(row[c])
        if not val:
            continue
        gender = "여성" if "FME" in c else "남성"
        age = re.search(r"_(1020|30|40|50|60)_", c).group(1)
        age_parts.append(f"{gender} {age}대 {val}")
    if age_parts:
            sentences.append("성별·연령별 고객 구성은 " + ", ".join(age_parts) + " 비중을 보입니다.")

    return " ".join([prefix + head] + sentences)

# -------------------------
# 3) 실행
# -------------------------
if __name__ == "__main__":
    print("[INFO] Loading:", IN_PATH)
    df = pd.read_csv(IN_PATH, encoding="utf-8-sig")

    df = filter_open_stores(df, CLOSE_COL)

    print("[INFO] Generating rag_text ...")
    df["rag_text"] = df.apply(make_rag_sentence, axis=1)

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved → {OUT_PATH} (rows={len(df):,})")
