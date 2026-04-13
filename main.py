import faiss
import pickle
import os

from src.ingestion import ingest_documents
from src.retriever import retrieve
from src.router import route_query
from src.generator import generate_answer

# 🔹 Load saved embeddings + chunks OR create new
if os.path.exists("embeddings/index.faiss") and os.path.exists("embeddings/chunks.pkl"):
    print("✅ Loading saved embeddings...")

    index = faiss.read_index("embeddings/index.faiss")

    with open("embeddings/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

else:
    print("⚡ Creating embeddings...")

    index, chunks = ingest_documents("data")


# 🔹 Main loop
while True:
    query = input("Enter query: ")

    # exit option (nice touch for demo)
    if query.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    # 🔹 Retrieve relevant chunks
    retrieved_chunks, distances = retrieve(query, index, chunks)

    # 🔹 Route query (IMPORTANT: pass query also)
    query_type = route_query(query, distances)

    # 🔹 Generate answer
    answer = generate_answer(query, retrieved_chunks, query_type)

    # 🔹 Output
    print(f"\nType: {query_type}")
    print(f"Answer: {answer}\n")