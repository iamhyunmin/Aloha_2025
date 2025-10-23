
📂 데이터 및 인덱스 파일 안내

========================

※ 본 README.txt는 데이터 및 인덱스 파일 관리용이며,  
  프로젝트 전체 개요는 루트 폴더의 README.md를 참고하시기 바랍니다.

────────────────────────
[1] 데이터 구성
────────────────────────

📁 data 폴더
- big_data_set1_f.csv, big_data_set2_f.csv, big_data_set3_f.csv : 원본 데이터셋
- store_month_df.csv : 원본 3개 데이터 병합 결과 (월 단위 매장 통합 데이터)
- store_month_df_bucketed.csv : 주요 지표 버킷화(구간화) 완료 데이터
- store_with_rag_text.csv : RAG 검색용 문맥 데이터 (가맹점별 문장화 완료, 최종 전처리 결과)

📁 artifacts 폴더
- rag_faiss.index : FAISS 인덱스 파일 (문장 임베딩 기반 검색용)
- meta.csv : 인덱스 문서 메타데이터
- document_keys.npy : 문서-키 매핑 정보

※ artifacts 폴더 내 파일들은 Streamlit 서비스(aloha.py) 실행 시 직접 사용됩니다.


────────────────────────
[2] 원본 데이터 (신한카드 제공)
────────────────────────

용량 제한으로 인해 GitHub에 포함되지 않은 데이터와 인덱스를 아래 링크를 통해 다운 받을 수 있습니다.
아래 순서대로 전처리 코드를 실행하여 동일한 결과를 생성할 수 있습니다.

전처리 코드 실행 순서:
1️⃣ merge_store_month.py → 월 단위 매장 통합 (store_month_df.csv 생성)
2️⃣ feature_bucketed.py → 주요 지표 버킷화 (store_month_df_bucketed.csv 생성)
3️⃣ sentence_function.py → 문장화 (store_with_rag_text.csv 생성) ✅ 최종 결과

이후, build_index.py 실행 시 위 최종 데이터를 참조하여
artifacts/ 폴더 내에 rag_faiss.index, document_keys.npy, meta.csv가 자동 생성됩니다.

🔗 Google Drive (data + artifacts 전체)
https://drive.google.com/drive/folders/13Kq5EvJBd-SP75wmDrNzxjts_9ayj3bH?usp=sharing


────────────────────────
[3] 경로 관련 유의사항
────────────────────────

- 모든 경로는 config.py에서 관리됩니다. (DATA_DIR, ARTIFACTS_DIR)
- 절대경로 대신 os.path.join() 기반 상대경로로 통일되어 있습니다.
- data 및 artifacts 폴더는 config.py 실행 시 자동 생성됩니다.
→ 이미 존재할 경우, 덮어쓰기 없이 그대로 유지됩니다.


작성자: [Team Aloha]
제출일: 2025-10-24

