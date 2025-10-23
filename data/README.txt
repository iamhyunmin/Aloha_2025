📂 data 폴더 안내
========================

본 폴더에는 Streamlit 서비스 구동에 필요한 데이터 파일이 포함되어 있습니다.
모든 코드는 config.py 내 DATA_DIR 경로를 참조하도록 설정되어 있습니다.

────────────────────────
[1] 파일 구성
────────────────────────
- meta.csv                   : 매장 메타정보 파일
- store_with_rag_text.csv  : RAG 검색용 문맥 데이터 (가맹점별 문장 데이터)

※ 위 데이터들은 Streamlit 서비스 실행에 직접 사용됩니다.

────────────────────────
[2] 원본 데이터 (신한카드 제공)
────────────────────────
용량 제한으로 인해 원본 데이터(big_data_set1_f.csv, big_data_set2_f.csv, big_data_set3_f.csv)는
GitHub에 포함되지 않았습니다.

원본 데이터를 이용해 전처리를 수행할 경우,
아래 순서대로 여러 전처리 코드 파일을 실행해야 합니다.
각 코드 파일은 data 폴더 내에 존재하며, 결과물은 모두 data 폴더 내에 저장됩니다.

예시 실행 순서:
1️⃣ merge_store_month.py   → 월 단위 매장 통합 (store_month_df.csv 생성)
2️⃣ feature_bucketed.py     → 주요 지표 구간화  (store_month_df_bucketed.csv 생성)
3️⃣ sentence_function.py   → 문장화 (store_with_rag_text.csv 생성) <== **최종 전처리 결과

이후, build_index.py 실행 시 위 최종 파일을 참조하여
artifacts 폴더에 인덱스(rag_faiss.index, document_keys.npy, meta.csv)를 생성합니다.

🔗 Google Drive: https://drive.google.com/drive/folders/XXXXX

────────────────────────
[3] 경로 관련 유의사항
────────────────────────
- 모든 경로는 config.py에서 관리됩니다. (DATA_DIR, ARTIFACTS_DIR)
- 절대경로를 사용하지 않고, os.path.join() 기반 상대경로로 통일되어 있습니다.
- data 폴더와 artifacts 폴더는 config.py에서 자동 생성되며,
  폴더가 없을 경우 코드 실행 시 자동으로 만들어집니다.

────────────────────────
[4] 실행 순서 (데이터 기준)
────────────────────────
1. (필요시) 원본 데이터(big_data_set1~3)를 data 폴더에 추가
2. 각 전처리 코드 실행 → store_with_rag_text.csv 생성
3. build_index.py를 실행하여 인덱스(rag_faiss.index, document_keys.npy, meta.csv)를 생성합니다.
4. rag_gemini.py, aloha.py가 동일 디렉토리에 존재해야 합니다.  
5. aloha.py를 실행하여 서비스를 구동합니다. (streamlit run aloha.py)

────────────────────────
작성자: [Team Aloha]
제출일: 2025-10-23
