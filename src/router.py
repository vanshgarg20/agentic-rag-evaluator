def route_query(query, distances):
    """
    Hybrid routing:
    - keyword-based (strong signal)
    - embedding-based (fallback)
    """

    query_lower = query.lower()

    avg_distance = sum(distances) / len(distances)
    min_distance = min(distances)

    # 🔴 Out of scope keywords (strong signal)
    if "india" in query_lower or "africa" in query_lower or "tesla" in query_lower:
        return "out_of_scope"

    # 🟡 Synthesis keywords
    if "compare" in query_lower or "difference" in query_lower or "how do" in query_lower:
        return "synthesis"

    # 🟢 Factual keywords
    if "what is" in query_lower or "define" in query_lower:
        return "factual"

    # 🔹 Fallback (embedding-based)
    if min_distance < 0.6:
        return "factual"
    elif avg_distance < 1.0:
        return "synthesis"
    else:
        return "out_of_scope"