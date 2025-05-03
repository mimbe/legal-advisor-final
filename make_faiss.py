import json
import numpy as np
import faiss
from openai import OpenAI
import os
from dotenv import load_dotenv  # ⬅️ برای بارگذاری فایل env

load_dotenv()  # ⬅️ اجرای بارگذاری متغیرها از فایل .env

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# مسیر ورودی و خروجی
CHUNKS_FILE = "data/chunk_metadata.json"
INDEX_PATH = "data/faiss_index.index"

# بارگذاری داده‌ها
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Total chunks: {len(chunks)}")

# ساخت embedding برای هر chunk
embeddings = []
for i, chunk in enumerate(chunks):
    text = chunk["content"]
    try:
        embedding = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        ).data[0].embedding
        embeddings.append(embedding)
    except Exception as e:
        print(f"❌ Error at chunk {i}: {e}")
        embeddings.append([0.0] * 1536)  # fallback vector in case of failure

# تبدیل به آرایه numpy
embedding_matrix = np.array(embeddings).astype("float32")

# ساخت index
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)

# ذخیره index
faiss.write_index(index, INDEX_PATH)
print(f"✅ FAISS index saved to {INDEX_PATH}")