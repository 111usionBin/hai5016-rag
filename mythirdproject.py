import requests
import time
from typing import List, Dict

# Configuration - fill these in with your values
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"        # for Custom Search JSON API
GOOGLE_CX = "YOUR_SEARCH_ENGINE_ID"           # Custom Search engine ID
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"        # Generative Language API key
GEMINI_MODEL = "models/gemini-1.0"             # replace with your target Gemini model name
GOOGLE_CUSTOMSEARCH_URL = "https://www.googleapis.com/customsearch/v1"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta2/{GEMINI_MODEL}:generate"

# 1) Get passages from context7 (assumes a `context7` object is available in your environment)
def fetch_context7(query: str, top_k: int = 5) -> List[Dict]:
    """
    Use context7 to retrieve relevant passages. Adjust call to your context7 API.
    Expected return: list of dicts with keys: 'text' and optionally 'source' or 'id'.
    """
    try:
        # Example call shape; adapt to your actual context7 client API:
        results = context7.get(query, k=top_k)  # <- replace with actual API if different
        # Normalize results
        normalized = []
        for r in results:
            normalized.append({
                "title": r.get("title") or r.get("id") or "context7",
                "snippet": r.get("text") or r.get("snippet") or "",
                "link": r.get("source") or r.get("url") or None
            })
        return normalized
    except Exception:
        # If context7 is not available in runtime, return empty list gracefully
        return []

# 2) Query Google Programmable Search (Custom Search JSON API)
def fetch_google_search(query: str, num: int = 5) -> List[Dict]:
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "q": query,
        "num": min(num, 10)
    }
    resp = requests.get(GOOGLE_CUSTOMSEARCH_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("items", [])
    results = []
    for it in items:
        results.append({
            "title": it.get("title"),
            "snippet": it.get("snippet"),
            "link": it.get("link")
        })
    return results

# 3) Merge & deduplicate sources (prioritize context7 first)
def merge_sources(context_docs: List[Dict], serp_results: List[Dict], max_sources: int = 8) -> List[Dict]:
    seen = set()
    merged = []
    for src in (context_docs + serp_results):
        key = (src.get("link") or src.get("snippet") or "")[:200]
        if key in seen:
            continue
        seen.add(key)
        merged.append(src)
        if len(merged) >= max_sources:
            break
    return merged

# 4) Build grounding prompt for Gemini
def build_grounding_prompt(query: str, sources: List[Dict]) -> str:
    parts = ["You are an assistant. Use ONLY the numbered sources below to answer the user's question. For any factual claim, include an inline citation like [1] or [2]. If the answer cannot be found in the sources, say: \"I don't know.\" Do not invent information.\n"]
    parts.append("SOURCES:")
    for i, s in enumerate(sources, start=1):
        title = s.get("title") or f"source-{i}"
        link = s.get("link") or ""
        snippet = s.get("snippet") or ""
        parts.append(f"[{i}] {title}\n{link}\n{snippet}\n")
    parts.append("\nQUESTION:\n" + query + "\n\nANSWER:")
    return "\n\n".join(parts)

# 5) Call Gemini (Generative Language API)
def call_gemini(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Dict:
    headers = {
        "Content-Type": "application/json"
    }
    params = {"key": GEMINI_API_KEY}
    payload = {
        "prompt": {"text": prompt},
        "maxOutputTokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(GEMINI_ENDPOINT, params=params, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()

# Main helper tying everything together
def grounded_answer(query: str) -> Dict:
    # 1. local KB via context7
    ctx = fetch_context7(query, top_k=5)

    # 2. live web via Google Search
    serp = fetch_google_search(query, num=5)

    # 3. merge
    sources = merge_sources(ctx, serp, max_sources=8)

    # 4. build prompt
    prompt = build_grounding_prompt(query, sources)

    # 5. call Gemini
    gen = call_gemini(prompt)

    # parse output text depending on API response structure
    text = ""
    # try to read typical response field - adapt if your model returns different structure
    if isinstance(gen, dict):
        # v1beta2 generate returns candidate text in gen['candidates'][0]['output'] or text
        cand = gen.get("candidates") or gen.get("outputs") or []
        if cand:
            first = cand[0]
            text = first.get("output") or first.get("content") or first.get("text") or ""
        else:
            text = gen.get("text", "")
    return {"answer": text, "sources": sources, "raw": gen}

# Example usage
if __name__ == "__main__":
    q = "How to ground a Gemini response with Google Search results?"
    out = grounded_answer(q)
    print("Answer:\n", out["answer"])
    print("\nSOURCES:")
    for i, s in enumerate(out["sources"], start=1):
        print(f"[{i}] {s.get('title')} - {s.get('link')}\n  {s.get('snippet')}\n")

