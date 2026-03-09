import os
import json
import csv
import requests
import trafilatura
from datetime import datetime
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from serpapi import GoogleSearch
import markdown
from bs4 import BeautifulSoup

# ─── Load API keys ───────────────────────────────────────────
load_dotenv()

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/gemini-embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY")
)

SERPAPI_KEY = os.getenv("SERPAPI_KEY")


# ─── Input Mode 1: Extract from URL ──────────────────────────
def extract_from_url(url: str) -> str:
    print("   📡 Mode: Live URL (most authentic RAG simulation)")
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not fetch: {url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    if not text:
        raise ValueError(f"No readable text found at: {url}")
    return text


# ─── Input Mode 2: Extract from .md file ─────────────────────
def extract_from_md(filepath: str) -> str:
    print("   📝 Mode: Markdown File (pre-publish simulation)")
    if not filepath.endswith(".md"):
        raise ValueError("Only .md files are supported. Please provide a Markdown file.")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        raw_md = f.read()
    # Convert markdown → HTML → plain text
    html = markdown.markdown(raw_md)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    if not text.strip():
        raise ValueError("Markdown file appears to be empty.")
    return text


# ─── Smart input router ───────────────────────────────────────
def extract_content(source: str, input_type: str) -> str:
    if input_type == "url":
        return extract_from_url(source)
    elif input_type == "file":
        return extract_from_md(source)
    else:
        raise ValueError("input_type must be 'url' or 'file'")


# ─── Build vector index ───────────────────────────────────────
def build_index(text: str) -> VectorStoreIndex:
    docs = [Document(text=text)]
    return VectorStoreIndex.from_documents(docs)


# ─── Simulate retrieval ───────────────────────────────────────
def simulate_retrieval(index: VectorStoreIndex, query: str, top_k: int = 5):
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)
    return [(r.node.text, getattr(r, "score", None)) for r in results]


# ─── Fetch live AI Overview citations ────────────────────────
def fetch_ai_overview(query: str) -> list:
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "engine": "google",
        "gl": "us",
        "hl": "en"
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    ai_overview = results.get("ai_overview") or {}
    sources = ai_overview.get("sources") or []
    urls = [s.get("link") for s in sources if s.get("link")]

    if not urls:
        organic = results.get("organic_results") or []
        urls = [r.get("link") for r in organic[:5] if r.get("link")]

    return urls


# ─── Score retrieval readiness ────────────────────────────────
def score_content(chunks: list) -> dict:
    scores = [s for _, s in chunks if s is not None]
    if not scores:
        return {"avg_score": 0, "max_score": 0, "chunks_retrieved": len(chunks)}
    return {
        "avg_score": round(sum(scores) / len(scores), 4),
        "max_score": round(max(scores), 4),
        "chunks_retrieved": len(chunks)
    }


# ─── Save report ──────────────────────────────────────────────
def save_report(source: str, input_type: str, query: str,
                chunks: list, readiness: dict, citations: list):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if input_type == "url":
        label = source.split("/")[2].replace(".", "_")
    else:
        label = os.path.basename(source).replace(".md", "")

    base_filename = f"reports/{label}_{timestamp}"

    # ── JSON ─────────────────────────────────────────────────
    report = {
        "timestamp": timestamp,
        "input_type": input_type,
        "source": source,
        "query": query,
        "readiness_score": readiness,
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
    print(f"   💾 JSON saved → {json_path}")

    # ── CSV ──────────────────────────────────────────────────
    csv_path = f"{base_filename}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "input_type", "source", "query",
                         "chunk_rank", "chunk_score", "avg_score",
                         "max_score", "chunks_retrieved", "chunk_preview"])
        for i, (chunk, score) in enumerate(chunks):
            writer.writerow([
                timestamp,
                input_type,
                source,
                query,
                i + 1,
                round(score, 4) if score else None,
                readiness["avg_score"],
                readiness["max_score"],
                readiness["chunks_retrieved"],
                chunk[:200].replace("\n", " ")
            ])
    print(f"   📊 CSV saved  → {csv_path}")


# ─── Main pipeline ────────────────────────────────────────────
def run_simulation(source: str, query: str,
                   input_type: str = "url", top_k: int = 5):

    print(f"\n{'='*60}")
    print(f"🔍 Query  : {query}")
    print(f"📥 Source : {source}")
    print(f"{'='*60}")

    # Extract content
    print("\n📄 Extracting content...")
    text = extract_content(source, input_type)
    print(f"   Extracted {len(text):,} characters")

    # Build index
    print("🧠 Building vector index...")
    index = build_index(text)

    # Simulate retrieval
    print(f"🔎 Simulating retrieval (top {top_k} chunks)...")
    chunks = simulate_retrieval(index, query, top_k)

    # Display chunks
    print("\n─── Retrieved Chunks ────────────────────────────────────")
    for i, (chunk, score) in enumerate(chunks, 1):
        score_str = f"{score:.4f}" if score else "n/a"
        print(f"\n[Chunk {i}] Score: {score_str}")
        print(chunk[:500].strip() + ("…" if len(chunk) > 500 else ""))

    # Readiness score
    readiness = score_content(chunks)
    print(f"\n─── Retrieval Readiness Score ───────────────────────────")
    print(f"   Avg similarity score : {readiness['avg_score']}")
    print(f"   Max similarity score : {readiness['max_score']}")
    print(f"   Chunks retrieved     : {readiness['chunks_retrieved']}")

    # Live citations
    print(f"\n─── Live AI Overview Citations ──────────────────────────")
    live_citations = []
    try:
        live_citations = fetch_ai_overview(query)
        if live_citations:
            for u in live_citations:
                print(f"   • {u}")
            if input_type == "url":
                your_domain = source.split("/")[2]
                cited = any(your_domain in u for u in live_citations)
                print(f"\n   ✅ Your domain cited? {'YES 🎉' if cited else 'NO — optimization opportunity!'}")
            else:
                print(f"\n   💡 Publish this content then re-run with input_type='url' to check citation status.")
        else:
            print("   No AI Overview citations found for this query.")
    except Exception as e:
        print(f"   [SerpAPI skipped] {e}")

    # Save report
    print(f"\n─── Saving Report ───────────────────────────────────────")
    save_report(source, input_type, query, chunks, readiness, live_citations)

    print(f"\n{'='*60}\n")


# ─── Run it ───────────────────────────────────────────────────
if __name__ == "__main__":

    # ── OPTION A: URL input (live page, most authentic) ──────
    # run_simulation(
    #     source="https://en.wikipedia.org/wiki/Backpacking_(hiking)",
    #     query="what gear do you need for backpacking",
    #     input_type="url",
    #     top_k=5
    # )

    # ── OPTION B: Markdown file (pre-publish testing) ────────
    run_simulation(
        source="drafts/my_article.md",        # ← path to your .md file
        query="your target query here",        # ← your target query
        input_type="file",
        top_k=5
    )