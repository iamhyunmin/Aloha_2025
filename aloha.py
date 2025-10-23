# app.py
# ============================================================
# ReVue – RAG + Gemini 기반 마케팅 네비게이터 (st.chat_message 버전)
# ============================================================

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from rag_gemini import generate_revue_answer  # ✅ 기존 RAG+Gemini 함수 그대로 사용
import re # <-- 1. re 모듈 추가
from config import DATA_DIR, ARTIFACTS_DIR

# -------------------------------
# 환경 변수 로드
# -------------------------------
load_dotenv()

# -------------------------------
# 캐싱된 이미지 로드 함수
# -------------------------------
ASSETS = Path("assets")

@st.cache_data
def load_image(name: str):
    return Image.open(ASSETS / name)

# -------------------------------
# 페이지 설정
# -------------------------------
st.set_page_config(
    page_title="ReVue — 마케팅 네비게이터",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# 박스컨테이너
# -------------------------------
st.markdown("""
<style>
.block-container {
    position: relative;
    max-width: 1200px;
    margin: 0 auto;
    padding: 5rem 5rem;
    box-sizing: border-box;
}
/* 뷰포트(브라우저 폭)가 1000px 이하일 때 적용되는 반응형 규칙 */
@media (max-width: 1000px) {
  .block-container {
      border-left-width: 0px;        /* 모바일에서 좌측 보더를 0px로 얇게 (콘텐츠 공간 확보) */
      border-right-width: 0px;       /* 우측 보더도 0px로 */
  }
}              
</style>
""", unsafe_allow_html=True)


# ============================================================
# 사이드바 커스텀 스타일
# ============================================================
st.markdown("""
<style>       
            
/* 🌌 사이드바 배경 네이비톤 */
section[data-testid="stSidebar"] {
    background-color: #2A313C !important;
    padding-left: 1.65rem !important;   /* ← 좌측 여백 */
    padding-right: 1.65rem !important;  /* → 우측 여백 */
}

/* 🎨 사이드바 전체 텍스트 흰색 */
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* 🚦 ReVue 브랜드 타이틀 */
section[data-testid="stSidebar"] .revue-brand {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  font-size: 2.5rem;
  font-weight: 600;
  color: #ffffff;
  line-height: 1.1;
  letter-spacing: 0.5px;
  margin: 0rem 0 1.8rem 0;
}

/* 🪧 About ReVue 카드 */
section[data-testid="stSidebar"] .about-card{
  background: #586477;                 /* 카드 배경 */
  border-radius: 10px;
  padding: 16px 16px;
  margin-bottom: 1.2rem;   /* 아래 콘텐츠와의 거리 */
}

/* 카드 안 텍스트 기본 */
section[data-testid="stSidebar"] .about-card p{
  margin: 0 0 .45rem 0;
  color: #ffffff;                       /* 흰색 글씨 */
  line-height: 1.2;
  font-size: 0.9rem;
}

/* 마지막 문단은 아래쪽 여백 제거 👇 */            
section[data-testid="stSidebar"] .about-card p:last-child {
  margin-bottom: 0 !important;
}

/* 🧹 답변 초기화 버튼 — 글씨 크기 확실히 적용 */
section[data-testid="stSidebar"] button[kind="secondary"],
section[data-testid="stSidebar"] button[kind="secondary"] * {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    line-height: 1.1 !important;
}

/* 🎨 버튼의 기본 색상/테두리 유지 */
section[data-testid="stSidebar"] button[kind="secondary"] {
    background-color: #586477 !important;
    color: #ffffff !important;
    border: 1px solid #ffffff !important;
    border-radius: 10px !important;
    transition: 0.2s ease-in-out;
    display: inline-flex !important;
    justify-content: center;
    align-items: center;

    /* 🎯 글씨 위아래 여백 */
    padding-top: 0.95rem !important;
    padding-bottom: 0.95rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    
    margin-bottom: 1.2rem !important;   /* 👈 버튼 아래쪽 여백 추가 */
}

/* ✨ Hover 시 색상 변경 */
section[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background-color: #9BA2AD !important;
    color: #ffffff !important;
    /* transform: scale(1.00); */  /* ← 크기 변동 없음 */
}                     

</style>
""", unsafe_allow_html=True)


# ============================================================
# 사이드바 구성
# ============================================================
with st.sidebar:
    # st.image(load_image("logo_revue.png"), width=180)  # ✅ 로고 이미지
    # st.markdown("# 🚦 ReVue")
    st.markdown('<div class="revue-brand">🚦<span>ReVue</span></div>', unsafe_allow_html=True)
    st.markdown("## 🕵️ About ReVue")
    st.caption("“데이터가 말해주는, 우리 가게의 다음 길”")

    st.markdown("""
    <div class="about-card">

    <p><b>ReVue</b>는 단순히 데이터를 읽는 AI가 아닙니다.</p>
    <p>당신의 가게 데이터를 다시 보고(<b>Re-view</b>),</p>
    <p>그 안의 가치를 다시 찾고(<b>Re-value</b>)</p>
    <p>매출이라는 목적지를 향해 안내하는 AI 네비게이터입니다.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📍 사용 가이드")
    st.markdown("""
    1. 질문하기 — 정확한 주소와 가게명과 함께 질문해주세요.  
    2. AI 분석 - ReVue가 데이터를 바탕으로 진단합니다.  
    3. 전략 확인 - 세 가지 노선(강화·보수·전환)을 제시합니다.  
    4. 실행하기 — 제안된 전략을 실행해 매출을 높이세요.
    """)

    st.markdown("## 💡 답변 안내")
    # st.markdown("**🏁 마케팅 최종 경로**")
    st.markdown("**🏁 마케팅 최종 경로**  \n · 강화 경로 : 매장의 **강점**을 극대화해 경쟁 우위를 유지하는 전략  \n · 보수 경로 : **약점**을 보완하여 안정적 성장을 도모하는 전략  \n · 전환 경로 : 강점과 약점을 **재해석**해 새로운 매출 루트를 만드는 전략")
    # st.markdown("**🧩 운행 전략 안내**")
    st.markdown("**🧩 운행 전략 안내**  \n · 실행 방법 : AI가 제시한 전략 가이드를 따라 단계별 실행  \n · 기대효과 : 실행 후 매출·재방문율 등 주요 지표 개선 기대")

    st.markdown("## 💬 예시 질문")
    st.markdown(" · '용답중앙15길 12에 위치한 메가** 매장의 주요 방문 고객 특성에 따른 마케팅 채널 추천 및 홍보안을 작성해줘.'  \n · '성동구 상원길 49 그리** 매장에서 현재 재방문률을 높일 수 있는 마케팅 아이디어와 근거를 제시해줘.'  \n · '마조로 30 프랭**** 매장의 현재 가장 큰 문제점 및 이를 보완할 마케팅 아이디어와 근거를 제시해줘.'")

    st.divider()
    if st.button("답변 초기화", key="reset", use_container_width=True):
        st.session_state.pop("chat_history", None)
        st.session_state.pop("messages", None)
        st.toast("대화가 초기화되었습니다.", icon="🧽")
        st.rerun()


    # st.caption("2025 빅콘테스트 | Updated Oct 2025 | Team Aloha")
    st.markdown(
    "<p style='text-align: center; color: #ffffff; opacity: 0.8;'>2025 빅콘테스트 | Updated Oct 2025 | Team Aloha</p>",
    unsafe_allow_html=True
    )

# ==============================================================================
# LLM 텍스트 파싱 함수
# ==============================================================================
def extract_section(regex_pattern, text):
    """주어진 정규 표현식 패턴을 사용하여 텍스트에서 섹션을 추출"""
    match = re.search(regex_pattern, text, re.DOTALL | re.IGNORECASE)
    # 정규식 그룹 1을 반환하며 공백 제거, 매칭 실패 시 "정보 없음" 반환
    return match.group(1).strip() if match else "정보 없음"

# LLM 텍스트를 파싱하고 Streamlit에 출력하는 함수
def display_revue_report(llm_output_text):
    # 1. LLM 텍스트 파싱
    data = {}
    
    # [현재 위치 파악] 섹션
    data['traffic_light'] = extract_section(r'🚦신호등:\s*(.*?)\n', llm_output_text)
    data['good_area'] = extract_section(r'🚗 잘 가고 있는 구간\n(.*?)\n\n⚠️ 느리게 가고 있는 구간', llm_output_text)
    data['bad_area'] = extract_section(r'⚠️ 느리게 가고 있는 구간\n(.*?)\n\n🎯한줄요약:', llm_output_text)
    data['summary'] = extract_section(r'🎯한줄요약:\s*(.*?)(?:\n\n|\n|$)', llm_output_text)
    
    # [경로 탐색] 섹션
    data['Enhance_line'] = extract_section(r'- 강화 경로 \(Enhance Line\): (.*?)(?:\n|- 보수 경로)', llm_output_text)
    data['Fix_line'] = extract_section(r'- 보수 경로 \(Fix Line\): (.*?)(?:\n|- 전환 경로)', llm_output_text)
    data['Shift_line'] = extract_section(r'- 전환 경로 \(Shift Line\): (.*?)(?:\n|===== 🏁최종 경로)', llm_output_text)

    # [최종 경로] 섹션
    data['recommended_path'] = extract_section(r'추천 경로:\s*(.*?)\n', llm_output_text)
    data['strategy_name'] = extract_section(r'전략명: (.*?)\n', llm_output_text)
    data['core_idea'] = extract_section(r'핵심 아이디어: (.*?)\n', llm_output_text)
    data['reason'] = extract_section(r'채택 근거: (.*?)\n', llm_output_text)

    # [운행 안내] 섹션=
    data['action_plan'] = extract_section(r'<실행 방법>\n(.*?)\n\n<기대효과>', llm_output_text)
    data['expected_effect'] = extract_section(r'<기대효과>\n(.*?)\n\n===== 🏆 도착 알림', llm_output_text)

    # [도착 알림] 섹션
    data['growth_phrase'] = extract_section(
    r'🎉오늘 사장님은 [“"](.*?)[”"](으)?로 성장했습니다!', 
    llm_output_text)
    
    # 2. Streamlit UI 구성
    
    # 1） 현재 위치 파악
    st.subheader("📍 현재 가게 위치 파악")
    st.markdown(f"<p style='font-size:1.4rem; font-weight:700;'>🚦 우리 가게 신호: {data['traffic_light']}</p>", unsafe_allow_html=True)


    col1, col2 = st.columns(2)
    
# --- 잘 가고 있는 구간 ---
    good_area_content = data['good_area']
    with col1:
        # st.expander로 변경: 제목은 잘 가고 있는 구간
        with st.expander("🚗 잘 가고 있는 구간", expanded=True):
            st.markdown(good_area_content)

    # --- 느리게 가고 있는 구간 ---
    bad_area_content = data['bad_area']
    with col2:
        # st.expander로 변경: 제목은 느리게 가고 있는 구간
        with st.expander("⚠️ 느리게 가고 있는 구간", expanded=True):
            st.markdown(bad_area_content)

    st.markdown("🎯**한줄요약:**")
    with st.container(border=True):
        st.markdown(data['summary']) # 일반 텍스트로 내용만 표시
    st.markdown("---")

    # 2） 경로 탐색
    st.subheader("🧭 마케팅 경로 탐색")
    st.markdown(f"**- 🔥 강화 경로 (Enhance Line):** {data['Enhance_line']}")
    st.markdown(f"**- 🛠️ 보수 경로 (Fix Line):** {data['Fix_line']}")
    st.markdown(f"**- 💫 전환 경로 (Shift Line):** {data['Shift_line']}")
    st.markdown("---")
    
    # 3） 최종 경로 (추가된 섹션)
    st.subheader("🏁 최종 마케팅 경로")
    st.markdown(f"**🏆 추천 경로:**  **{data['recommended_path']}**")
    st.markdown(f"<p style='font-size:1.2rem; font-weight:700;'> 전략명: {data['strategy_name']}</p>", unsafe_allow_html=True)
    st.markdown(f"**핵심 아이디어:** {data['core_idea']}")
    st.markdown(f"**채택 근거:** {data['reason']}")
    st.markdown("---")

    # 4） 운행 안내 (st.success/st.warning로 변경)
    st.subheader("🧩 운행 전략 안내")
    
    # === 실행 방법 (제목 분리 후 success 박스) ===
    st.markdown(f"**📝 실행 방법**")
    # ✅ 핵심 수정: st.success 내부에서 f-string 제거하고 data['action_plan']만 전달
    st.success(data['action_plan']) 
    
    # === 기대 효과 (제목 분리 후 warning 박스) ===
    st.markdown(f"**📈 기대효과**")
    # ✅ 핵심 수정: st.warning 내부에서 f-string 제거하고 data['expected_effect']만 전달
    st.warning(data['expected_effect']) 
    st.markdown("---")

    # 5） 도착 알림
    st.subheader("🛣️ 도착 알림")
    st.markdown(f"🚈 **‘{data['strategy_name']}’ 노선에 진입하셨네요.**")
    st.markdown(f"🎉오늘 사장님은 “**{data['growth_phrase']}**”(으)로 성장했습니다!")

# ============================================================
# 메인 대화 영역
# ============================================================

# 🚗 제목 표지판 스타일
st.markdown("""

<style>
.title-sign {
  position: relative;
  isolation: isolate; 
  background-color: #00A86B;              /* 초록색 배경 */
  border: 9px solid #00A86B;              /* 흰색 테두리 */
  border-radius: 16px;                    /* 둥근 모서리 */
  padding: 1rem 2.5rem;                   /* 내부 여백 (위아래, 좌우)*/
  color: #FFFFFF;                         /* 글자 흰색 */
  margin-bottom: 1.5rem;                    /* 아래쪽 여백 */
}
.title-sign::after{
  content:"";
  position:absolute;
  inset:0;
  border-radius:12px;
  background:#00A86B;
  border: 4px solid #FFFFFF;              /* 흰색 테두리 */
  transform: translate(0px,0px);
  z-index:-1;              /* 이제 부모 바깥으로 사라지지 않음 */
}

.title-sign h1 {
  font-size: 2.3rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.title-sign p {
  font-size: 1rem;
  font-weight: 400;  
  margin-bottom: 0.3rem;
  opacity: 0.9;
  line-height: 1.5;
  padding: 0rem 0.2rem;
}
</style>

<div class="title-sign">
  <h1>🚘 마케팅 네비게이터, <b>ReVue</b></h1>
  <p>ReVue는 숫자 속에서 <b>가치를 다시 발견</b>하는 AI입니다. 당신의 가게를 <b>다시 바라보고</b>, <b>매출</b>로 향하는 새로운 길을 안내합니다.</p>
</div>
""", unsafe_allow_html=True)

# --- 대화 초기화 ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {"role": "assistant", "content": "안녕하세요, 사장님. 오늘은 어떤 고민을 함께 풀어볼까요? 😊"}
    ]

# 기존 대화 출력
for msg in st.session_state["chat_history"]:
    if msg["role"] == "assistant":
        with st.chat_message("assistant"):
            # 💡 응답 내용이 보고서 템플릿의 시작점("=====")을 포함하는지 확인하여 분기
            if "===== 📍 현재 위치 파악 =====" in msg["content"]:
                # 보고서 형식의 응답이면, 파싱 함수를 호출하여 구조화합니다.
                display_revue_report(msg["content"])
            else:
                # 일반 텍스트 메시지이거나 오류 메시지이면, 마크다운으로 그대로 표시합니다.
                st.markdown(msg["content"])
    else:
        with st.chat_message("user"):
            st.markdown(msg["content"])

# 사용자 입력
if prompt := st.chat_input("가맹점 이름과 정확한 주소를 함께 질문에 입력하세요."):
    # 사용자 메시지 저장 및 표시
    st.session_state["chat_history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 응답
    with st.chat_message("assistant"):
        with st.spinner("🔍 분석 중입니다..."):
            try:
                # 1. LLM 응답을 받습니다. (주석 해제)
                # 'generate_revue_answer' 함수는 'rag_gemini' 모듈에서 가져옵니다.
                answer = generate_revue_answer(prompt)
                
                # 2. LLM 응답을 파싱하고 구조화된 Streamlit 보고서로 출력합니다.
                display_revue_report(answer) 
                
            except Exception as e:
                # 3. 오류 발생 시, 오류 메시지만 채팅창에 표시합니다.
                answer = f"⚠️ 오류 발생: {e}"
                st.markdown(answer)
                
    # 4. 전체 응답 텍스트 (파싱 전 원본)를 대화 기록에 저장합니다.
    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
    st.rerun()
