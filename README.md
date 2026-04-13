# Agentic RAG Evaluator

## Overview

This project implements an **Agentic Retrieval-Augmented Generation (RAG)** system over a dataset of AI regulation documents.

Unlike traditional RAG systems, this system first **decides how to answer a query** before retrieving information.

The system classifies each query into:
- Factual (direct answer exists)
- Synthesis (requires combining multiple sources)
- Out-of-scope (information not present)

---

## Architecture

Pipeline:

User Query → Query Router → Retriever → Generator → Final Answer

### Components:

- **Ingestion Pipeline**
  - Chunking (500 characters, 50 overlap)
  - Embedding using SentenceTransformers (MiniLM)
  - Vector storage using FAISS

- **Query Router (Agentic Layer)**
  - Hybrid logic:
    - Keyword-based classification (e.g., "compare", "what is")
    - Embedding similarity (distance thresholds)
  - Explicit and inspectable (not LLM-based)

- **Retriever**
  - Top-k nearest neighbor search using FAISS

- **Answer Generator**
  - Uses Groq LLaMA 3.1 model
  - Grounded responses only (no hallucination)
  - Explicit refusal for out-of-scope queries

---

## Chunking Strategy

- Chunk size: 500 characters  
- Overlap: 50 characters  

Reason:
- Maintains context continuity  
- Avoids information loss at boundaries  
- Suitable for policy-style documents  

---

## Routing Logic

The system uses a **hybrid routing approach**:

1. Keyword-based signals:
   - "what is" → factual  
   - "compare", "difference" → synthesis  
   - unknown entities → out_of_scope  

2. Embedding-based fallback:
   - min_distance → strong match  
   - avg_distance → moderate match  

This ensures both **precision and robustness**.

---

## Evaluation Framework

- 15 test queries:
  - 5 factual  
  - 5 synthesis  
  - 5 out_of_scope  

Metrics used:
- Routing accuracy  
- Cosine similarity (answer quality)

Results are saved in `results.csv`.

---

## Results Summary

- Factual accuracy: High  
- Out-of-scope detection: Perfect (no hallucination)  
- Synthesis accuracy: Moderate  

Overall routing accuracy ≈ 85–90%

---

## Failure Analysis

### 1. Synthesis misclassification
Some synthesis queries were classified as factual.

**Reason:**  
Implicit comparison queries were not always captured by keyword rules.

**Fix:**  
Use semantic intent classification or LLM-based routing.

---

### 2. Incomplete synthesis answers
Some answers lacked depth.

**Reason:**  
Top-k retrieval (k=3) may miss full context.

**Fix:**  
Increase k or use reranking.

---

### 3. Evaluation mismatch
Some factual answers scored low.

**Reason:**  
Generated answers were longer than expected answers.

**Fix:**  
Use ROUGE or keyword overlap instead of cosine similarity only.

---

## Key Features

- Explicit agentic routing (no black-box decision)
- No hallucination on out-of-scope queries
- Modular and clean code structure
- Fast inference using Groq LLaMA
- Cached embeddings for efficiency

---

## How to Run

```bash
python main.py