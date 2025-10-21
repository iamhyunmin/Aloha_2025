# app.py
# ============================================================
# ReVue 스트림릿 앱 - 커뮤니티 디자인 반영 버전
# - 초기 로딩 최적화(@st.cache_resource)
# - .env 없이도 동작( st.secrets["GEMINI_API_KEY"] → os.environ Fallback )
# - 경로/환경변수(REVUE_OUT_DIR, GEMINI_API_KEY)
# - 선택 데이터(별점/폐점) 자동 반영 + 스키마 검증
# - RAG 검색 + 주소 기반 필터링(정규식)
# - 고정된 출력 포맷(신호등/경로3종/최종경로/운행안내/도착알림)
# - 커뮤니티 시안과 유사한 UI(좌측 네비, 초록 히어로, 트래픽 신호등, 말풍선)
# ============================================================

import os, re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# -----------------------------
# 0) 페이지 설정 & 전역 스타일
# -----------------------------
st.set_page_config(page_title="ReVue — 마케팅 내비게이터", page_icon="🚗", layout="wide")

st.markdown("""
<style>
/* 레이아웃 여백 */
.main .block-container{padding-top:1rem;padding-bottom:1.2rem}

/* 좌측 패널: 스크롤 가능 + 내부 모든 텍스트 포함 */
.left-panel{
  background:#0F172A;color:#E5E7EB;border-radius:14px;padding:20px;
  min-height:86vh;max-height:86vh;overflow:auto;
  display:flex;flex-direction:column;gap:18px;
  box-shadow:0 6px 18px rgba(0,0,0,.18)
}
.left-panel h3{margin:.2rem 0 .6rem 0;letter-spacing:.3px;font-weight:800}
.left-panel .section-title{color:#93C5FD;font-weight:800;margin:.6rem 0 .3rem}
.left-panel .bullet{color:#E5E7EB;opacity:.92;line-height:1.5;font-size:.95rem}
.footer-note{opacity:.6;font-size:.8rem;margin-top:10px}

/* 히어로: 이중 테두리 */
.hero{
  position:relative;background:#16A34A;color:#fff;border-radius:16px;
  padding:18px 20px;font-weight:800;font-size:20px;letter-spacing:.2px;
  box-shadow:0 10px 24px rgba(0,0,0,.10);border:4px solid #22C55E
}
.hero:after{content:"";position:absolute;inset:6px;border-radius:12px;
  border:3px solid rgba(255,255,255,.35);pointer-events:none}
.hero small{display:block;font-weight:500;opacity:.95;margin-top:6px}

/* 좌패널에서 뻗는 브리지 */
.bridge{height:12px;width:36%;background:#374151;border-radius:8px;margin:12px 0 8px}

/* 신호등 칩 */
.traffic{
  display:inline-flex;gap:10px;align-items:center;margin:6px 0 14px 6px;
  padding:8px 12px;border-radius:999px;background:#111827;border:2px solid #1F2937;
  box-shadow:inset 0 -2px 0 rgba(255,255,255,.06),0 4px 10px rgba(0,0,0,.12)
}
.light{width:18px;height:18px;border-radius:50%;box-shadow:inset 0 -3px 0 rgba(0,0,0,.25)}
.light.red{background:#EF4444}.light.yellow{background:#F59E0B}.light.green{background:#22C55E}

/* 말풍선 */
.bubble{
  max-width:760px;display:inline-flex;align-items:center;gap:10px;
  background:#FDECEC;color:#111827;border:1px solid #FCA5A5;border-radius:16px;padding:12px 16px;
  box-shadow:0 2px 12px rgba(0,0,0,.06)
}
.bubble .emoji{font-size:20px}

/* 입력 행: 인풋/버튼을 항상 밝게 */
.query-row{display:flex;gap:10px;align-items:center;margin-top:10px}
.stTextInput>div>div>input {background:#FFFFFF !important;color:#111827 !important;
  border:1px solid #E5E7EB !important;border-radius:12px !important;padding:12px 14px !important}
.stButton>button {height:46px;border-radius:12px;background:#111827 !important;color:#F9FAFB !important;
  border:1px solid #374151}

/* 결과 카드 + 본문 꾸미기 */
.result-card{background:#FFFFFF;border:1px solid #E5E7EB;border-radius:14px;padding:18px;
  box-shadow:0 8px 18px rgba(0,0,0,.04)}
.answer h2{font-size:20px;margin:6px 0 8px}
.answer h3{font-size:18px;margin:10px 0 6px}
.answer .pill{display:inline-block;padding:2px 10px;border-radius:999px;background:#F1F5F9;border:1px solid #E2E8F0;margin-right:6px}
.answer .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.answer .card{background:#F8FAFC;border:1px solid #E5E7EB;border-radius:12px;padding:12px}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 1) 런타임(모델/인덱스/데이터) - 캐시
# -----------------------------
EMB_MODEL = "BAAI/bge-m3"
DEFAULT_OUT_DIR = "artifacts"
TOP_K = 10

def _get_api_key() -> str:
    # 1) 환경변수 우선  2) (있을 때만) secrets 보조
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            key = None
    if not key:
        raise ValueError(
            "GEMINI_API_KEY가 없습니다. PowerShell에서 "
            "$env:GEMINI_API_KEY='AIza...키...'" " 로 넣거나, "
            ".streamlit/secrets.toml 에 GEMINI_API_KEY 를 작성하세요."
        )
    return key

@st.cache_resource(show_spinner=False)
def get_runtime():
    out_dir = os.getenv("REVUE_OUT_DIR", DEFAULT_OUT_DIR)

    # 1) LLM
    genai.configure(api_key=_get_api_key())
    llm = genai.GenerativeModel("gemini-2.5-flash")

    # 2) FAISS & 메타
    index_path = Path(out_dir) / "rag_faiss.index"
    meta_path  = Path(out_dir) / "meta.csv"
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"FAISS/메타가 없습니다: {index_path}, {meta_path}")
    index = faiss.read_index(str(index_path))
    meta  = pd.read_csv(meta_path)

    # 3) 임베딩
    model = SentenceTransformer(EMB_MODEL, device="cpu")

    # 4) 선택: 폐점 비교
    summary_df = pd.DataFrame()
    try:
        p = Path("versus_closed.csv")
        if p.exists():
            tmp = pd.read_csv(p, encoding="utf-8-sig")
            need = {"Index","Closed_mean","Open_mean"}
            if need.issubset(tmp.columns):
                summary_df = tmp[["Index","Closed_mean","Open_mean"]].copy()
            else:
                st.warning("versus_closed.csv 컬럼 요구: Index, Closed_mean, Open_mean")
    except Exception as e:
        st.warning(f"versus_closed.csv 로딩 오류: {e}")

    # 5) 선택: 별점/리뷰수
    rating_map = {}
    try:
        p = Path("store_google_rating.csv")
        if p.exists():
            r = pd.read_csv(p, encoding="utf-8-sig")
            need = {"ENCODED_MCT","g_rating","g_user_ratings_total"}
            if need.issubset(r.columns):
                r["ENCODED_MCT"] = r["ENCODED_MCT"].astype(str)
                rating_map = dict(zip(r["ENCODED_MCT"], zip(r["g_rating"], r["g_user_ratings_total"])))
            else:
                st.warning("store_google_rating.csv 컬럼 요구: ENCODED_MCT, g_rating, g_user_ratings_total")
    except Exception as e:
        st.warning(f"store_google_rating.csv 로딩 오류: {e}")

    return {"llm": llm, "index": index, "meta": meta, "model": model,
            "summary_df": summary_df, "rating_map": rating_map, "out_dir": out_dir}

RT = get_runtime()

# -----------------------------
# 2) 유틸: 검색/필터/요약/프롬프트
# -----------------------------
def retrieve_context(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    q_emb = RT["model"].encode([query], normalize_embeddings=True)
    D, I = RT["index"].search(np.array(q_emb, dtype="float32"), top_k)
    ctx = RT["meta"].iloc[I[0]].copy()
    ctx["score"] = D[0]
    return ctx

_ADDR_REGEX = r"((?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)?[^\\n]{0,15}?(?:로|길|대로)\\s*\\d+[\\-\\d]*)"

def filter_by_address(context_df: pd.DataFrame) -> pd.DataFrame:
    if "rag_text" not in context_df.columns:
        return context_df
    text = "\n\n".join(context_df["rag_text"].astype(str).tolist())
    m = re.search(_ADDR_REGEX, text)
    if not m:
        return context_df
    addr = m.group(0).strip()
    df = context_df.copy()
    df["ADDR_EXTRACT"] = df["rag_text"].str.extract(r"\\[ADDR=([^\\]]+)\\]", expand=False)
    df = df[df["ADDR_EXTRACT"].fillna("").str.contains(addr, na=False)]
    return df if len(df) else context_df

def build_closure_hints(summary_df: pd.DataFrame, max_lines: int = 3) -> str:
    if summary_df is None or summary_df.empty:
        return "폐점 관련 비교 요약 데이터 없음"
    lines = []
    for _, row in summary_df.head(max_lines).iterrows():
        idx, cm, om = str(row["Index"]), float(row["Closed_mean"]), float(row["Open_mean"])
        gap = cm - om
        trend = "↑" if gap > 0 else ("↓" if gap < 0 else "→")
        lines.append(f"- {idx}: 폐점평균 {cm:.2f} vs 영업중 {om:.2f} (차이 {gap:+.2f} {trend})")
    return "\n".join(lines)

def build_rating_summary(mct_list, rating_map: dict, max_lines: int = 5) -> str:
    if not rating_map or not mct_list:
        return "별점 요약 데이터 없음"
    lines = []
    for key in mct_list:
        key = str(key).strip()
        if key in rating_map:
            rating, cnt = rating_map[key]
            lines.append(f"- {key}: ⭐ {rating} / 리뷰수 {int(cnt)}")
            if len(lines) >= max_lines: break
    return "\n".join(lines) if lines else "별점 요약 데이터 없음"

SYSTEM_PROMPT = """
당신은 ‘마케팅 내비게이터 ReVue’입니다. 반드시 아래 마크다운 구조로 답변합니다.
한국어, 간결·구체, 실행 가능한 조언을 제시하세요. 불필요한 면책문구 금지.

[필수 출력 구조]
### 📍 현재 위치 파악
- 우리 가게 신호등: 🟢/🟡/🔴 중 하나 + 한 줄 근거
- 한 줄 요약(핵심 한 문장)

### 🧭 경로 탐색
- **강화 경로(Enhance Line)**: •• 불릿 2~4개
- **보수 경로(Fix Line)**: •• 불릿 2~4개
- **전환 경로(Shift Line)**: •• 불릿 2~4개

### 🏁 최종 경로 제안
- 추천 경로: (강화/보수/전환 중 택1) — 한 문단 이유 설명

### 🛠️ 운행 안내(실행 체크리스트)
- [ ] 실행 항목 3~5개 (간단한 수치·기간 포함)

### ✅ 도착 알림(한 문장)
- 마무리 한 문장

[표현 규칙]
- 가능하면 수치/근거 간단 병기(예: ~%p↑, ~% 감소 등).
- 마크다운 헤딩/불릿을 정확히 사용해 블록이 구분되게 출력.
"""

def generate_revue_answer(query: str, mct_list=None, top_k: int = TOP_K) -> str:
    # 1) RAG + 주소 필터
    ctx_df = filter_by_address(retrieve_context(query, top_k=top_k))
    context_text = "\n\n".join(ctx_df.get("rag_text", "").astype(str).head(10))

    # 2) 보조 요약
    closure_hints = build_closure_hints(RT["summary_df"], max_lines=3)
    rating_summary = build_rating_summary(mct_list or [], RT["rating_map"], max_lines=5)

    # 3) 프롬프트
    full_prompt = f"""{SYSTEM_PROMPT}

[시스템 참고 힌트 - 폐점 데이터]
{closure_hints}

[근거 데이터 - 구글맵 별점]
{rating_summary}

[참고 데이터 문맥]
{context_text}

[사용자 질의]
{query}
"""
    # 4) LLM
    return RT["llm"].generate_content(full_prompt).text

# -----------------------------
# 3) UI 레이아웃 (디자인 반영)
# -----------------------------
left, right = st.columns([0.38, 0.62])

# ——— 좌측 패널 (한 번의 st.markdown으로 열고/닫고/내용/버튼까지) ———
with left:
    st.markdown("""
<div class="left-panel">
  <h3>Revue</h3>

  <div class="section-title">About ReVue</div>
  <div class="bullet">
    “데이터가 알려주는, 우리 가게의 다음 길”<br><br>
    ReVue는 단순히 데이터를 읽는 AI가 아닙니다.<br>
    당신의 가게 데이터를 다시 깊게 (Re-view),<br>
    더 알맞은 가치를 다시 (Re-value) 찾아,<br>
    매출이라는 목적지로 향하는 AI 내비게이터입니다.
  </div>

  <div class="section-title">사용 가이드</div>
  <div class="bullet">
    1. 질문하기 → 가게의 고민을 자유롭게 입력하세요.<br>
    2. 질의 분석 → ReVue가 데이터+내비로 진단합니다.<br>
    3. 전략 맵 → 유지/보수·전환 라인을 보여 드려요.<br>
    4. 실행하기 → 제안된 액션을 실천하고 변화를 지켜보세요.
  </div>

  <div class="section-title">예시 질문</div>
  <div class="bullet">
    • (거래/객단가) 요일별 최적 프로모션은?<br>
    • (재방문 30%↓ 가정) 무엇부터 손볼까?<br>
    • (입지/연령) 우리 상권 핵심 고객은?<br>
    • (리뷰) 리뷰 이벤트 효과는?
  </div>

  <div class="section-title">데이터 근거</div>
  <div class="bullet">
    • 2025 빅콘테스트 기반(비식별/집계)<br>
    • 지역/연령/요일/품목 다각 분석<br>
    • 구글 지도 별점/리뷰 요약(선택)<br>
    • Gemini 2.5 Flash + RAG
  </div>

  <div class="footer-note">2025 빅콘테스트 | Updated Oct 2025 | Team Aloha</div>

  <!-- 패널 안 ‘다른 질문하기’ (쿼리스트링 방식) -->
  <div class="bottom-btn" style="margin-top:12px">
    <a href="?clear=1" style="
      display:block; text-align:center; text-decoration:none;
      background:#111827; color:#F9FAFB; border:1px solid #374151;
      border-radius:10px; padding:10px 12px;">다른 질문하기</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ——— 링크 클릭 처리: 세션 초기화는 우측 컬럼에서 last_*를 읽기 전에! ———
if st.query_params.get("clear") == "1":
    for k in ("last_answer", "last_query"):
        st.session_state.pop(k, None)
    st.query_params.clear()
    st.rerun()  # 즉시 깔끔하게 다시 그리기
    
with right:
    # 히어로 + 브리지 + 트래픽 신호등
    st.markdown("""
    <div class="hero">🚗 마케팅 내비게이터, ReVue
      <small>숫자 속에서 가게의 길을 발견합니다. 다시 바라보면, 매출로 향하는 새로운 길이 열립니다.</small>
    </div>
    <div class="bridge"></div>
    <div class="traffic">
      <div class="light red"></div><div class="light yellow"></div><div class="light green"></div>
    </div>
    """, unsafe_allow_html=True)

    # 말풍선 가이드
    st.markdown("""
    <div class="bubble"><span class="emoji">⚠️</span>
    안녕하세요, 사장님. 저는 데이터를 바탕으로 가게의 숨은 기회를 찾아드리는 AI 비밀상담사 ReVue입니다.<br>
    오늘은 어떤 고민을 함께 찾아볼까요? 😊</div>
    """, unsafe_allow_html=True)

    # 입력
    with st.form("query_form", clear_on_submit=False):
        cols = st.columns([1, 0.15])
        with cols[0]:
            query = st.text_input(
                "질문",
                value=st.session_state.get("last_query", ""),
                placeholder="가맹점 이름(ENCODED_MCT)과 함께 질문을 입력하세요.",
                label_visibility="collapsed",
            )
        with cols[1]:
            submitted = st.form_submit_button("➤", use_container_width=True)
    st.caption("💡 Tip: “가게명+지역(도로명주소)”을 포함하면 문맥 필터링이 더 정확해집니다.")

    # 예시 프롬프트
    with st.expander("예시 프롬프트 펼치기"):
        st.write("- “성동구 왕십리 **○○카페** 재방문율을 올릴 방안은?”")
        st.write("- “평균 객단가가 높은데 순고객수가 적을 때 매출 보완 전략은?”")
        st.write("- “리뷰 이벤트/구독제가 우리 가게에 맞는지?”")

    # 결과
    if submitted and query.strip():
        st.session_state["last_query"] = query
        with st.spinner("진단 중..."):
            try:
                mcts = [x.strip() for x in re.split(r"[ ,;/]+", query) if x.strip().isdigit()]
                st.session_state["last_answer"] = generate_revue_answer(query, mct_list=mcts, top_k=TOP_K)
            except Exception as e:
                st.error(f"⚠️ 오류가 발생했습니다: {e}")

    if st.session_state.get("last_answer"):
        st.markdown("---")
        st.markdown('<div class="result-card answer">', unsafe_allow_html=True)
        # 필요하면 섹션마다 추가 카드 UI를 붙일 수 있도록 기본 wrapper만 둠
        st.markdown(st.session_state["last_answer"])
        st.markdown('</div>', unsafe_allow_html=True)
    