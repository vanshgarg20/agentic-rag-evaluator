import pandas as pd
import faiss
import pickle
import os

from src.ingestion import ingest_documents
from src.retriever import retrieve
from src.router import route_query
from src.generator import generate_answer
from src.evaluator import evaluate_answer


# 🔹 Load saved embeddings OR create new
if os.path.exists("embeddings/index.faiss") and os.path.exists("embeddings/chunks.pkl"):
    print("✅ Loading saved embeddings...")

    index = faiss.read_index("embeddings/index.faiss")

    with open("embeddings/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

else:
    print("⚡ Creating embeddings...")
    index, chunks = ingest_documents("data")


# 🔹 15 test cases (5 each)
test_cases = [
    # -------- FACTUAL --------
    {"query": "What is AI regulation?", "type": "factual", "expected": "AI regulation refers to rules governing AI systems"},
    {"query": "What is GDPR?", "type": "factual", "expected": "GDPR is a data protection law in Europe"},
    {"query": "What is AI governance?", "type": "factual", "expected": "AI governance ensures responsible AI usage"},
    {"query": "Define risk-based AI", "type": "factual", "expected": "AI systems are classified based on risk"},
    {"query": "Purpose of AI regulation?", "type": "factual", "expected": "To ensure safety and compliance"},

    # -------- SYNTHESIS --------
    {"query": "Compare US and EU AI regulation", "type": "synthesis", "expected": "EU is stricter, US is flexible"},
    {"query": "Different AI regulation approaches globally", "type": "synthesis", "expected": "Countries follow different frameworks"},
    {"query": "Explain AI risk frameworks", "type": "synthesis", "expected": "Multiple risk categories exist"},
    {"query": "Compare governance models in AI", "type": "synthesis", "expected": "Different governance structures"},
    {"query": "How countries regulate AI differently?", "type": "synthesis", "expected": "Varied strategies across countries"},

    # -------- OUT OF SCOPE --------
    {"query": "What is India's AI law?", "type": "out_of_scope", "expected": "not available"},
    {"query": "Explain blockchain law", "type": "out_of_scope", "expected": "not available"},
    {"query": "What is Tesla AI policy?", "type": "out_of_scope", "expected": "not available"},
    {"query": "Quantum computing regulation?", "type": "out_of_scope", "expected": "not available"},
    {"query": "AI laws in Africa?", "type": "out_of_scope", "expected": "not available"},
]


results = []

# 🔹 Run evaluation
for test in test_cases:
    query = test["query"]

    # Retrieve
    retrieved_chunks, distances = retrieve(query, index, chunks)

    # 🔥 IMPORTANT FIX (pass query also)
    pred_type = route_query(query, distances)

    # Generate answer
    answer = generate_answer(query, retrieved_chunks, pred_type)

    # Evaluate similarity
    score = evaluate_answer(answer, test["expected"])

    results.append({
        "query": query,
        "true_type": test["type"],
        "predicted_type": pred_type,
        "correct_routing": test["type"] == pred_type,
        "score": round(score, 3)
    })


# 🔹 Save results
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)

print("\n✅ Evaluation Results:\n")
print(df)