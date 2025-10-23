import pandas as pd
import numpy as np
import faiss, os, re, math
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import DATA_DIR, ARTIFACTS_DIR
import os

# ----------------------------
# 설정
# ----------------------------
IN_PATH = os.path.join(DATA_DIR, "store_with_rag_text.csv")
OUT_DIR = ARTIFACTS_DIR
EMB_MODEL = "BAAI/bge-m3"              # 임베딩 모델
CHUNK_SIZE = 700                       # 청킹 크기 (토큰 단위 근사)
CHUNK_OVERLAP = 100                    # 청킹 오버랩
TOP_N = 5                              # 검색 시 top-k 기본값

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# 1️⃣ 데이터 로드
# ----------------------------
print(f"[INFO] Loading data from {IN_PATH} ...")
df = pd.read_csv(IN_PATH, encoding="utf-8-sig")
df = df[df["rag_text"].notna() & (df["rag_text"].str.strip() != "")]
print(f"[INFO] Rows loaded: {len(df):,}")

# ----------------------------
# 2️⃣ 청킹 (Chunking)
# ----------------------------
print("[INFO] Splitting text into chunks ...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n", ".", " "]
)

chunks = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    text = str(row["rag_text"])
    parts = splitter.split_text(text)
    for p in parts:
        if len(p.strip()) < 30:
            continue
        chunks.append({
            "ENCODED_MCT": row.get("ENCODED_MCT", ""),
            "rag_text": p.strip(),
            "TA_YM": row.get("TA_YM", ""),
            "HPSN_MCT_ZCD_NM": row.get("HPSN_MCT_ZCD_NM", ""),
            "MCT_SIGUNGU_NM": row.get("MCT_SIGUNGU_NM", "")
        })

meta = pd.DataFrame(chunks)
print(f"[INFO] Total chunks generated: {len(meta):,}")

# ----------------------------
# 3️⃣ 임베딩 (Embedding)
# ----------------------------
print(f"[INFO] Loading embedding model: {EMB_MODEL}")
model = SentenceTransformer(EMB_MODEL, device="cuda")

texts = meta["rag_text"].tolist()
print(f"[INFO] Encoding {len(texts):,} chunks ...")
embs = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
embs = np.array(embs, dtype="float32")

# ----------------------------
# 4️⃣ 파시스 인덱싱 (FAISS Index)
# ----------------------------
print("[INFO] Building FAISS index ...")
dim = embs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embs)
print(f"[INFO] FAISS index built: dim={dim}, vectors={index.ntotal:,}")

# ----------------------------
# 5️⃣ 저장 (Index, Metadata, Keys)
# ----------------------------
index_path = os.path.join(OUT_DIR, "rag_faiss.index")
meta_path  = os.path.join(OUT_DIR, "meta.csv")
keys_path  = os.path.join(OUT_DIR, "document_keys.npy")

faiss.write_index(index, index_path)
meta.to_csv(meta_path, index=False, encoding="utf-8-sig")
np.save(keys_path, meta["ENCODED_MCT"].to_numpy())

print(f"[DONE] All artifacts saved to '{OUT_DIR}'")
print(f"        • Index: {index_path}")
print(f"        • Metadata: {meta_path}")
print(f"        • Keys: {keys_path}")

# ----------------------------
# 6️⃣ 검색 함수 예시 (테스트용)
# ----------------------------
def search(query, top_k=TOP_N):
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_emb, dtype="float32"), top_k)
    results = meta.iloc[I[0]].copy()
    results["score"] = D[0]
    return results[["ENCODED_MCT", "MCT_SIGUNGU_NM", "HPSN_MCT_ZCD_NM", "score", "rag_text"]]

if __name__ == "__main__":
    # 예시 질의 테스트
    q = "성동구 카페 매출이 높은 매장"
    print("\n[TEST SEARCH]", q)
    res = search(q, top_k=3)
    for _, r in res.iterrows():
        print(f"🏪 {r['MCT_SIGUNGU_NM']} | {r['HPSN_MCT_ZCD_NM']} | score={r['score']:.3f}")
        print(f"→ {r['rag_text'][:120]}...\n")



