# 🚦 **Revue — 당신의 마케팅 네비게이션**
2025 빅콘테스트 출품작 │ Team Aloha  

<br><br>

## 🧭 Revue의 철학

**Revue**는 두 단어의 결합에서 탄생했습니다:

- **Re-** : 다시, 새롭게  
- **Vue (View / Value)** : 시야, 가치  

👉 즉, “Revue = 가치를 새롭게 보다 (Re-Value / Re-View)”  
**Re-view the data, Re-value your business.**

Revue는 단순히 데이터를 읽는 AI가 아니라,  
당신의 가게 데이터를 **다시 보고**, 그 안의 가치를 **다시 찾고**, 매출이라는 목적지를 향해 안내하는 AI 네비게이션입니다.  

<br><br>

## 💡 주요 기능

| 기능 | 설명 |
|------|------|
| 자연어 상담형 인터페이스 | 점주가 직접 “우리 매장 재방문율을 높이려면?” 같은 질문을 입력하면 AI가 즉시 답변 |
| 현재 가게 위치 파악 | 매출, 고객 수, 결제 취소율 등 핵심 지표를 분석하여 가게 진단 도출 |
| 마케팅 경로 탐색 | 단순한 수치 해석이 아닌, 다양한 시각에서 맞춤 전략 제시 |
| 최종 마케팅 경로 | 가게에 가장 알맞은 전략 + 근거 출력 |
| 운행 전략 안내 | 각 전략에 대해 실행 가능한 구체적 액션 플랜 제공 |


<br><br>

## 📊 데이터 및 코드 구성

Revue 프로젝트는 **데이터 전처리 → 인덱스 구축 → 서비스 구동**의 3단계로 구성되어 있습니다.  
각 단계는 아래와 같이 `data` 폴더의 파일과 Python 코드로 연결되어 있습니다.  

🔗 Google Drive (data + artifacts 전체) <br>
https://drive.google.com/drive/folders/13Kq5EvJBd-SP75wmDrNzxjts_9ayj3bH?usp=sharing

<br>

### 1️⃣ 원본 데이터
- **big_data_set1_f.csv** : 가맹점 개요 및 업종별 정보  
- **big_data_set2_f.csv** : 월별 이용금액·거래건수 데이터  
- **big_data_set3_f.csv** : 월별 고객수·신규·재방문 고객 데이터  

📍 출처: 신한카드 빅데이터 (2025 빅콘테스트 제공)

<br>

### 2️⃣ 전처리 데이터 (data 폴더)
| 파일명 | 생성 코드 | 설명 |
|--------|-------------|------|
| **store_month_df.csv** | `merge_store_month.py` | 원본 3개 데이터를 병합하여 매장×월 단위 테이블로 통합 |
| **store_month_df_bucketed.csv** | `feature_bucketed.py` | 매출·건수·고객·객단가·취소율 등 주요 지표를 6단계 구간(BIN)으로 구분 |
| **store_with_rag_text.csv** | `sentence_function.py` | 매장별 데이터를 문장 형태로 가공한 RAG 입력용 텍스트 데이터 (LLM 입력의 핵심) |

🔍 이 세 파일은 Revue의 **기반 데이터셋**으로, 모든 분석 및 질의응답의 출발점입니다.

<br>

### 3️⃣ 인덱스 파일 (artifacts 폴더)
| 파일명 | 생성 코드 | 설명 |
|--------|-------------|------|
| **rag_faiss.index** | `build_index.py` | 문장 임베딩을 기반으로 생성된 FAISS 인덱스 (RAG 검색용) |
| **meta.csv** | `build_index.py` | 인덱스 내 문장 및 매장 메타정보 (검색 결과 매핑용) |
| **document_keys.npy** | `build_index.py` | 인덱스 문서 키 배열 (내부 참조용) |

🔍 이 폴더는 **Streamlit 서비스(aloha.py)** 구동 시 자동으로 참조됩니다.

<br>

### 4️⃣ 주요 코드 파일
| 파일명 | 역할 |
|--------|------|
| **merge_store_month.py** | 원본 3개 데이터 병합 |
| **feature_bucketed.py** | 주요 지표 구간화(Binning) |
| **sentence_function.py** | 매장별 RAG 문장화 수행 |
| **build_index.py** | 임베딩 모델을 이용한 인덱스 구축 |
| **rag_gemini.py** | Gemini + FAISS 기반 질의응답 로직 |
| **aloha.py** | Streamlit 메인 앱 (서비스 인터페이스) |
| **utils/closed_data.py** | 폐점 매장 vs 영업중 매장 비교 통계 생성 |
| **utils/google_rating.py** | Google Maps 리뷰·평점 수집 모듈 |
| **config.py** | 경로 및 폴더 자동 설정 관리 (DATA_DIR, ARTIFACTS_DIR) |

<br><br>

## 주요 코드 실행 순서
1️⃣ `merge_store_month.py` — 원본 데이터 병합  
2️⃣ `feature_bucketed.py` — 주요 지표 버킷화  
3️⃣ `sentence_function.py` — 문장화 (RAG 입력 데이터 생성)  
4️⃣ `build_index.py` — 인덱스 구축 (FAISS, meta.csv 생성)  
5️⃣ `aloha.py` — Streamlit 앱 실행 (챗봇 서비스)

<br><br>

## 🚀 실행 방법

### 1. 가상화경 설정
```bash
conda env create -n bigcontest_2025
conda activate bigcontest_2025
pip install -r requirements.txt
```

### 2. 환경 변수 설정

.env 파일에 API 키를 추가:
```bash
GOOGLE_API_KEY=your_api_key_here
```

### 3. 서비스 실행
```bash
streamlit run aloha.py
```

