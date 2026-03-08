import json
import csv
from datetime import datetime

import os
import requests
import trafilatura
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from serpapi import GoogleSearch

# ─── Load API keys ───────────────────────────────────────────
load_dotenv()

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/gemini-embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY")
)

from serpapi import GoogleSearch


# ─── Step 1: Extract content from URL ────────────────────────
def extract_text_from_url(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not fetch: {url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    if not text:
        raise ValueError(f"No readable text found at: {url}")
    return text


# ─── Step 2: Build vector index ──────────────────────────────
def build_index(text: str) -> VectorStoreIndex:
    docs = [Document(text=text)]
    return VectorStoreIndex.from_documents(docs)


# ─── Step 3: Simulate retrieval ──────────────────────────────
def simulate_retrieval(index: VectorStoreIndex, query: str, top_k: int = 5):
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)
    return [(r.node.text, getattr(r, "score", None)) for r in results]


# ─── Step 4: Fetch live AI Overview citations ─────────────────
def fetch_ai_overview(query: str) -> list:
    params = {
        "q": query,
        "api_key": os.getenv("SERPAPI_KEY"),
        "engine": "google",
        "gl": "us",
        "hl": "en"
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    # Extract AI Overview sources if present
    ai_overview = results.get("ai_overview") or {}
    sources = ai_overview.get("sources") or []
    urls = [s.get("link") for s in sources if s.get("link")]

    # Fallback to organic results if no AI Overview
    if not urls:
        organic = results.get("organic_results") or []
        urls = [r.get("link") for r in organic[:5] if r.get("link")]

    return urls

    headers = {"x-api-key": FETCHSERP_API_KEY}
    r = requests.get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    ai = data.get("ai_overview") or {}
    citations = ai.get("citations") or []
    return [c.get("url") for c in citations if c.get("url")]


# ─── Step 5: Score retrieval readiness ───────────────────────
def score_content(chunks: list) -> dict:
    scores = [s for _, s in chunks if s is not None]
    if not scores:
        return {"avg_score": 0, "max_score": 0, "chunks_retrieved": len(chunks)}
    return {
        "avg_score": round(sum(scores) / len(scores), 4),
        "max_score": round(max(scores), 4),
        "chunks_retrieved": len(chunks)
    }

# ─── Step 6: Save report ──────────────────────────────────────
def save_report(url: str, query: str, chunks: list, readiness: dict, citations: list):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    domain = url.split("/")[2].replace(".", "_")
    base_filename = f"reports/{domain}_{timestamp}"

    # ── JSON report ───────────────────────────────────────────
    report = {
        "timestamp": timestamp,
        "query": query,
        "url": url,
        "readiness_score": readiness,
        "your_domain_cited": any(domain.replace("_", ".") in u for u in citations),
        "live_citations": citations,
        "retrieved_chunks": [
            {
                "rank": i + 1,
                "score": round(score, 4) if score else None,
                "text": chunk[:500]
            }
            for i, (chunk, score) in enumerate(chunks)
        ]
    }

    json_path = f"{base_filename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n   💾 JSON saved → {json_path}")

    # ── CSV report ────────────────────────────────────────────
    csv_path = f"{base_filename}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "url", "query", "chunk_rank",
                         "chunk_score", "avg_score", "max_score",
                         "chunks_retrieved", "domain_cited", "chunk_preview"])
        for i, (chunk, score) in enumerate(chunks):
            writer.writerow([
                timestamp,
                url,
                query,
                i + 1,
                round(score, 4) if score else None,
                readiness["avg_score"],
                readiness["max_score"],
                readiness["chunks_retrieved"],
                report["your_domain_cited"],
                chunk[:200].replace("\n", " ")
            ])
    print(f"   📊 CSV saved  → {csv_path}")


# ─── Main pipeline ────────────────────────────────────────────
def run_simulation(url: str, query: str, top_k: int = 5):
    print(f"\n{'='*60}")
    print(f"🔍 Query : {query}")
    print(f"🌐 URL   : {url}")
    print(f"{'='*60}")

    print("\n📄 Extracting content...")
    text = extract_text_from_url(url)
    print(f"   Extracted {len(text):,} characters")

    print("🧠 Building vector index...")
    index = build_index(text)

    print(f"🔎 Simulating retrieval (top {top_k} chunks)...")
    chunks = simulate_retrieval(index, query, top_k)

    print("\n─── Retrieved Chunks ────────────────────────────────────")
    for i, (chunk, score) in enumerate(chunks, 1):
        score_str = f"{score:.4f}" if score else "n/a"
        print(f"\n[Chunk {i}] Score: {score_str}")
        print(chunk[:500].strip() + ("…" if len(chunk) > 500 else ""))

    readiness = score_content(chunks)
    print(f"\n─── Retrieval Readiness Score ───────────────────────────")
    print(f"   Avg similarity score : {readiness['avg_score']}")
    print(f"   Max similarity score : {readiness['max_score']}")
    print(f"   Chunks retrieved     : {readiness['chunks_retrieved']}")

    print(f"\n─── Live AI Overview Citations ──────────────────────────")
    live_citations = []
    try:
        live_citations = fetch_ai_overview(query)
        if live_citations:
            for u in live_citations:
                print(f"   • {u}")
            your_domain = url.split("/")[2]
            cited = any(your_domain in u for u in live_citations)
            print(f"\n   ✅ Your domain cited? {'YES 🎉' if cited else 'NO — optimization opportunity!'}")
        else:
            print("   No citations found in AI Overview for this query.")
    except Exception as e:
        print(f"   [FetchSERP skipped] {e}")

    # Save report
    print(f"\n─── Saving Report ───────────────────────────────────────")
    save_report(url, query, chunks, readiness, live_citations if 'live_citations' in dir() else [])

    print(f"\n{'='*60}\n")


# ─── Run it ───────────────────────────────────────────────────
if __name__ == "__main__":
    run_simulation(
        url="https://en.wikipedia.org/wiki/Backpacking_(hiking)",  # ← replace this
        query="what gear do you need for backpacking",            # ← replace this
        top_k=5
    )