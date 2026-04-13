from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def evaluate_answer(predicted, expected):
    pred_emb = model.encode(predicted).reshape(1, -1)
    exp_emb = model.encode(expected).reshape(1, -1)

    score = cosine_similarity(pred_emb, exp_emb)[0][0]
    return score