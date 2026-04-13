from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(query, chunks, query_type):

    if query_type == "out_of_scope":
        return "The information is not available in the provided documents."

    context = "\n\n".join(chunks)

    prompt = f"""
Answer the question using ONLY the context below.

Instructions:
- If exact definition is not present, give best possible explanation from context
- Do NOT say "not available" unless completely missing
- Keep answer clear and simple

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content