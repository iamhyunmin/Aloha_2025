# -*- coding: utf-8 -*-
"""
유니크 매장(ENCODED_MCT 기준) → Google Places API로 '별점(rating)' + '리뷰수(user_ratings_total)'만 수집
- Text Search → place_id 매칭
- Place Details (fields=rating,user_ratings_total) 조회
- 무료 구간 보호를 위한 '하드 스톱' 포함
- 진행률/ETA 출력 + 주기적 체크포인트 저장 + 응답 캐시 + 제한적 백오프 재시도

필수:
  export GOOGLE_MAPS_API_KEY="AIzaSyCYJ6yHOY4ATy1RVGKFI7IwwKZblf2aMbQ"

설치:
  pip install requests pandas
"""

import os, re, time, json, math, requests, pandas as pd
from datetime import timedelta

# ================== 경로/설정 ==================
SRC = "store_month_df_buckedted.csv"           # 입력 (수정 가능)
DST = "store_google_rating.csv"     # 최종 출력
CACHE_DIR = "/mnt/data/.gplaces_cache"; os.makedirs(CACHE_DIR, exist_ok=True)
CHKPT_EVERY = 200   # N건마다 체크포인트 저장

ID_COL   = "ENCODED_MCT"
NAME_COL = "MCT_NM"
ADDR_COL = "MCT_BSE_AR"

# 호출 간 sleep 및 재시도 백오프
SLEEP = 0.18
MAX_RETRY = 2               # 429/5xx에 한해 2회 재시도
BACKOFF_BASE = 0.6          # 지수 백오프 시작

# ================== 무료 한도 하드 스톱 ==================
# Google Places (New) 무료 사용량 캡을 여유 있게 보전
TS_FREE_LIMIT = 5000     # Text Search 월 무료
DT_FREE_LIMIT = 10000    # Place Details (Essentials) 월 무료
TS_HARD_STOP  = 4800     # 안전 여유치로 조기 중단
DT_HARD_STOP  = 9000     # 안전 여유치로 조기 중단

TS_CALLS = 0             # 실행 중 누적 카운터(캐시 히트는 미포함)
DT_CALLS = 0

# ================== Google Places API ==================
API_KEY = "AIzaSyCYJ6yHOY4ATy1RVGKFI7IwwKZblf2aMbQ"
assert API_KEY, "환경변수 GOOGLE_MAPS_API_KEY 가 비었습니다. 먼저 설정하세요."

SESSION = requests.Session()

# ================== 유틸 ==================
def read_csv_safely(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")

def norm_prefix2(name: str) -> str:
    """마스킹 상호에서도 앞 2글자 접두어 추출"""
    name = (name or "").strip()
    m = re.match(r"^[\w가-힣]+", name)
    base = m.group(0) if m else ""
    base = re.sub(r"\s+", "", base)
    base = re.sub(r"[^\w가-힣]", "", base)
    return base[:2]

def cache_path(key: str) -> str:
    key = re.sub(r"[^A-Za-z0-9가-힣]+", "_", key)[:140]
    return os.path.join(CACHE_DIR, key + ".json")

def fmt_hms(seconds: float) -> str:
    if seconds < 0: seconds = 0
    return str(timedelta(seconds=int(seconds)))

def eta_report(start_ts: float, done: int, total: int) -> str:
    elapsed = time.time() - start_ts
    per = elapsed / max(done, 1)
    remain = (total - done) * per
    return f"{done}/{total} avg:{per:.2f}s/store elapsed:{fmt_hms(elapsed)} ETA:{fmt_hms(remain)}"

def pick_representative(df: pd.DataFrame) -> pd.DataFrame:
    """같은 ID 내 대표 주소: 최빈값(mode) → 없으면 첫 유효값. 이름은 첫 유효값."""
    def mode_or_first(s: pd.Series):
        s = s.dropna()
        if s.empty: return None
        m = s.mode()
        return m.iloc[0] if not m.empty else s.iloc[0]
    reps = (df[[ID_COL, NAME_COL, ADDR_COL]]
            .dropna(subset=[ADDR_COL])
            .groupby(ID_COL, as_index=False)
            .agg({NAME_COL: lambda x: x.dropna().iloc[0] if x.dropna().size else None,
                  ADDR_COL: mode_or_first}))
    return reps

# ================== HTTP 호출 래퍼(재시도/백오프) ==================
def _get_with_retry(url: str, params: dict):
    """429/5xx에 한해 MAX_RETRY까지 백오프 재시도"""
    for attempt in range(MAX_RETRY + 1):
        r = SESSION.get(url, params=params, timeout=15)
        if r.status_code < 500 and r.status_code != 429:
            r.raise_for_status()
            return r
        # 재시도 조건(429/5xx)
        if attempt < MAX_RETRY:
            sleep_s = BACKOFF_BASE * (2 ** attempt)
            time.sleep(sleep_s)
            continue
        r.raise_for_status()
    # 논리상 도달하지 않음
    return r

# ================== Places API 함수(하드스톱 포함) ==================
def gplaces_text_search(query: str, region="kr", language="ko"):
    global TS_CALLS
    if TS_CALLS >= TS_HARD_STOP:
        raise RuntimeError(f"[HARD-STOP] TextSearch {TS_CALLS}/{TS_HARD_STOP} 도달. 자동 중단.")
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    r = _get_with_retry(url, {
        "query": query, "key": API_KEY, "region": region, "language": language
    })
    TS_CALLS += 1
    time.sleep(SLEEP)
    return r.json()

def gplaces_details(place_id: str, fields="rating,user_ratings_total", language="ko"):
    global DT_CALLS
    if DT_CALLS >= DT_HARD_STOP:
        raise RuntimeError(f"[HARD-STOP] PlaceDetails {DT_CALLS}/{DT_HARD_STOP} 도달. 자동 중단.")
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    r = _get_with_retry(url, {
        "place_id": place_id, "fields": fields, "key": API_KEY, "language": language
    })
    DT_CALLS += 1
    time.sleep(SLEEP)
    return r.json()

# ================== 매칭/조회 로직 ==================
def match_place_id(name_masked: str, address: str):
    """
    이름 접두어 2글자 + 주소 토큰을 섞어 Text Search → 가장 그럴듯한 후보 선택
    - 캐시: 쿼리 문자열 기준
    - 간단 스코어: 접두어 일치 가산 + 리뷰수 가중
    """
    p2 = norm_prefix2(name_masked)
    addr_tokens = re.findall(r"[가-힣A-Za-z0-9]+", address)[:2]  # 노이즈 감소
    query = f"{p2} {' '.join(addr_tokens)} {address}".strip() if p2 else address

    cp = cache_path("ts_" + query)
    if os.path.exists(cp):
        data = json.load(open(cp, "r", encoding="utf-8"))
    else:
        data = gplaces_text_search(query)
        json.dump(data, open(cp, "w", encoding="utf-8"), ensure_ascii=False)

    results = data.get("results", [])
    if not results:
        return None, "not_found"

    def score(item):
        nm = (item.get("name") or "").replace(" ", "")
        s = 0.0
        if p2 and nm.startswith(p2): s += 5.0
        s += min((item.get("user_ratings_total", 0) or 0) / 100.0, 5.0)
        return s

    best = max(results, key=score)
    return best.get("place_id"), "ok"

def get_rating(place_id: str):
    """Place Details에서 rating/user_ratings_total만 조회(캐시 사용, 하드스톱 연동)"""
    if not place_id:
        return None, None, "no_place_id"
    cp = cache_path("dt_" + place_id)
    if os.path.exists(cp):
        data = json.load(open(cp, "r", encoding="utf-8"))
        status = data.get("status", "OK")
    else:
        data = gplaces_details(place_id, fields="rating,user_ratings_total")
        status = data.get("status", "OK")
        json.dump(data, open(cp, "w", encoding="utf-8"), ensure_ascii=False)
    res = data.get("result", {}) or {}
    return res.get("rating"), res.get("user_ratings_total"), status

# ================== 메인 ==================
def main():
    global TS_CALLS, DT_CALLS
    df = read_csv_safely(SRC)
    for c in [ID_COL, NAME_COL, ADDR_COL]:
        assert c in df.columns, f"필수 컬럼 누락: {c}"

    reps = pick_representative(df)
    total = len(reps)
    print(f"[INFO] unique stores: {total}")
    print(f"[LIMIT] TextSearch hard-stop:{TS_HARD_STOP} (free:{TS_FREE_LIMIT}), "
          f"Details hard-stop:{DT_HARD_STOP} (free:{DT_FREE_LIMIT})")

    rows = []
    start = time.time()
    last_ckpt = time.time()

    for i, r in enumerate(reps.itertuples(index=False), 1):
        sid = str(getattr(r, ID_COL))
        nm  = str(getattr(r, NAME_COL)) if pd.notna(getattr(r, NAME_COL)) else ""
        ad  = str(getattr(r, ADDR_COL)) if pd.notna(getattr(r, ADDR_COL)) else ""

        try:
            pid, st1 = match_place_id(nm, ad)
            rating, cnt, st2 = get_rating(pid) if pid else (None, None, st1)
            rows.append({
                "ENCODED_MCT": sid,
                "MCT_BSE_AR": ad,
                "g_place_id": pid,
                "g_rating": rating,
                "g_user_ratings_total": cnt,
                "status": st2 if pid else st1
            })
        except requests.HTTPError as e:
            rows.append({"ENCODED_MCT": sid, "MCT_BSE_AR": ad,
                         "g_place_id": None, "g_rating": None, "g_user_ratings_total": None,
                         "status": f"http_{getattr(e.response,'status_code','err')}"})
        except RuntimeError as e:
            # 하드스톱 도달 시 현재까지 저장 후 종료
            print(f"[STOP] {e}")
            pd.DataFrame(rows).to_csv(DST, index=False)
            print(f"[SAVED] partial -> {DST} (rows: {len(rows)})")
            print(f"[COUNT] TS:{TS_CALLS} / DT:{DT_CALLS}")
            return
        except Exception as e:
            rows.append({"ENCODED_MCT": sid, "MCT_BSE_AR": ad,
                         "g_place_id": None, "g_rating": None, "g_user_ratings_total": None,
                         "status": f"error:{type(e).__name__}"})

        # 진행률/ETA/카운터 로그
        if i == 1 or i % 20 == 0:
            print("[PROGRESS]", eta_report(start, i, total), f" | [COUNT] TS:{TS_CALLS} DT:{DT_CALLS}")

        # 주기적 체크포인트 저장(건수 또는 3분 간격)
        if i % CHKPT_EVERY == 0 or (time.time() - last_ckpt > 180):
            pd.DataFrame(rows).to_csv(DST, index=False)
            last_ckpt = time.time()
            print(f"[CKPT] saved {i} rows -> {DST}")

    out = pd.DataFrame(rows)
    out.to_csv(DST, index=False)
    print(f"[DONE] saved -> {DST} (rows: {len(out)})")
    print(f"[COUNT] TS:{TS_CALLS} / DT:{DT_CALLS}")
    print("[SUMMARY]", eta_report(start, total, total))

if __name__ == "__main__":
    main()
  
path = "store_google_rating.csv"  
df = pd.read_csv(path)

out_path = "C:\\Users\\LG\\Desktop\\Bigcontest_Agent\\store_google_rating.csv"  # 원하는 경로

# 엑셀에서 한글 안 깨지게 저장
df.to_csv(out_path, index=False, encoding="utf-8-sig")


print("saved ->", out_path)

