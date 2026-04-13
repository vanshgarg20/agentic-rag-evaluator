# Failure Analysis

1. Some synthesis queries were misclassified as factual.
Reason: Keyword-based routing does not fully capture implicit comparison queries.
Fix: Could improve using semantic intent classification or LLM-based routing.

2. Some synthesis answers have lower similarity scores.
Reason: Limited chunk retrieval (top-k = 3) may miss complete context.
Fix: Increase k or apply reranking for better context selection.

3. Some factual answers have lower similarity scores.
Reason: Generated answers are more detailed than expected short answers.
Fix: Use ROUGE or keyword overlap instead of only cosine similarity.