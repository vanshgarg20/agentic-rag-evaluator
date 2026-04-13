import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

def get_embedding(text):
    return model.encode(text)

def ingest_documents(folder_path="data"):
    all_chunks = []
    embeddings = []

    for file in os.listdir(folder_path):
        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            text = f.read()
            chunks = chunk_text(text)

            for chunk in chunks:
                all_chunks.append(chunk)
                embeddings.append(get_embedding(chunk))

    embeddings = np.array(embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs("embeddings", exist_ok=True)
    faiss.write_index(index, "embeddings/index.faiss")

    return index, all_chunks