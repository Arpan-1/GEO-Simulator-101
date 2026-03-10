import os
import json
import csv
import trafilatura
import markdown
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv
from serpapi import GoogleSearch
from openai import OpenAI
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# ─── Load API Keys ────────────────────────────────────────────
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPAPI_KEY    = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ─── Gemini Embedding (Dimension 1) ──────────────────────────
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/gemini-embedding-001",
    api_key=GOOGLE_API_KEY
)

# ─── OpenAI Client (Dimensions 2, 3, 4, 5) ───────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL  = "gpt-4o-mini"


# ══════════════════════════════════════════════════════════════
#  HELPER — OpenAI JSON call
# ══════════════════════════════════════════════════════════════

def ask_openai(prompt: str) -> dict:
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a GEO (Generative Engine Optimization) expert. Always respond with valid JSON only. No markdown, no explanation, no code fences."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ══════════════════════════════════════════════════════════════
#  CONTENT EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_from_url(url: str) -> str:
    print("   📡 Mode : Live URL (most authentic RAG simulation)")
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not fetch: {url}")
    text = trafilatura.extract(
        downloaded, include_comments=False, include_tables=True)
    if not text:
        raise ValueError(f"No readable text found at: {url}")
    return text


def extract_from_md(filepath: str) -> str:
    print("   📝 Mode : Markdown File (pre-publish simulation)")
    if not filepath.endswith(".md"):
        raise ValueError("Only .md files are supported.")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        raw_md = f.read()
    html = markdown.markdown(raw_md)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    if not text.strip():
        raise ValueError("Markdown file appears to be empty.")
    return text


def extract_content(source: str, input_type: str) -> str:
    if input_type == "url":
        return extract_from_url(source)
    elif input_type == "file":
        return extract_from_md(source)
    else:
        raise ValueError("input_type must be 'url' or 'file'")


# ══════════════════════════════════════════════════════════════
#  DIMENSION 1 — RETRIEVAL SCORE
# ══════════════════════════════════════════════════════════════

def build_index(text: str) -> VectorStoreIndex:
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    docs     = [Document(text=text)]
    return VectorStoreIndex.from_documents(
        docs, transformations=[splitter])


def simulate_retrieval(index: VectorStoreIndex,
                       query: str, top_k: int = 5):
    retriever = index.as_retriever(similarity_top_k=top_k)
    results   = retriever.retrieve(query)
    return [(r.node.text, getattr(r, "score", None)) for r in results]


def score_retrieval(chunks: list) -> dict:
    scores = [s for _, s in chunks if s is not None]
    if not scores:
        return {"avg_score": 0, "max_score": 0,
                "chunks_retrieved": len(chunks), "dimension_score": 0}
    avg = round(sum(scores) / len(scores), 4)
    mx  = round(max(scores), 4)
    return {
        "avg_score"       : avg,
        "max_score"       : mx,
        "chunks_retrieved": len(chunks),
        "dimension_score" : round(avg * 100)
    }


# ══════════════════════════════════════════════════════════════
#  DIMENSION 2 — STRUCTURE & CLARITY SCORE
# ══════════════════════════════════════════════════════════════

def score_structure(text: str, query: str) -> dict:
    prompt = f"""
You are a GEO expert evaluating content structure for AI search retrieval.

Target query: "{query}"

Score each criterion from 0 to 20:

1. HEADING ALIGNMENT (0-20)
   Do headings directly match query intent and user expectations?

2. ANSWER FRONT-LOADING (0-20)
   Are direct answers placed at the start of sections, not buried?

3. PARAGRAPH EXTRACTABILITY (0-20)
   Can individual paragraphs stand alone as complete answer units?

4. AMBIGUITY LEVEL (0-20)
   Is content free of vague pronouns and unclear references?

5. SCANNABILITY (0-20)
   Is content logically broken into sections an AI can navigate easily?

CONTENT TO EVALUATE:
{text[:3000]}

Respond in this exact JSON format:
{{
  "heading_alignment"       : <0-20>,
  "answer_front_loading"    : <0-20>,
  "paragraph_extractability": <0-20>,
  "ambiguity_level"         : <0-20>,
  "scannability"            : <0-20>,
  "total_score"             : <0-100>,
  "top_issues"              : ["issue 1", "issue 2", "issue 3"],
  "recommendations"         : ["fix 1", "fix 2", "fix 3"]
}}"""
    result = ask_openai(prompt)
    result["dimension_score"] = result.get("total_score", 0)
    return result


# ══════════════════════════════════════════════════════════════
#  DIMENSION 3 — HALLUCINATION RISK SCORE
# ══════════════════════════════════════════════════════════════

def score_hallucination_risk(text: str, query: str) -> dict:
    prompt = f"""
You are a GEO expert evaluating hallucination risk in AI-generated answers.

Target query: "{query}"

Score each factor from 0 to 20 (higher = LOWER risk = better):

1. FACTUAL SPECIFICITY (0-20)
   Does content have specific facts, numbers, dates, named entities?

2. CLAIM CLARITY (0-20)
   Are all claims unambiguous and clearly bounded?

3. INTERNAL CONSISTENCY (0-20)
   Is content free of contradictions or conflicting data?

4. SOURCE GROUNDEDNESS (0-20)
   Are statistics and facts attributed to specific sources?

5. SCOPE CLARITY (0-20)
   Are limitations and caveats clearly stated?

CONTENT TO EVALUATE:
{text[:3000]}

Respond in this exact JSON format:
{{
  "factual_specificity"  : <0-20>,
  "claim_clarity"        : <0-20>,
  "internal_consistency" : <0-20>,
  "source_groundedness"  : <0-20>,
  "scope_clarity"        : <0-20>,
  "total_score"          : <0-100>,
  "high_risk_phrases"    : ["phrase 1", "phrase 2", "phrase 3"],
  "recommendations"      : ["fix 1", "fix 2", "fix 3"]
}}"""
    result = ask_openai(prompt)
    result["dimension_score"] = result.get("total_score", 0)
    return result


# ══════════════════════════════════════════════════════════════
#  DIMENSION 4 — COMPETITIVE GAP SCORE
# ══════════════════════════════════════════════════════════════

def fetch_live_citations(query: str) -> list:
    params = {
        "q"      : query,
        "api_key": SERPAPI_KEY,
        "engine" : "google",
        "gl"     : "np",
        "hl"     : "en"
    }
    search  = GoogleSearch(params)
    results = search.get_dict()

    ai_overview = results.get("ai_overview") or {}
    sources     = ai_overview.get("sources") or []
    urls        = [s.get("link") for s in sources if s.get("link")]

    if not urls:
        organic = results.get("organic_results") or []
        urls    = [r.get("link") for r in organic[:5] if r.get("link")]

    return urls


def fetch_competitor_text(url: str) -> str:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        text = trafilatura.extract(downloaded, include_comments=False)
        return text[:3000] if text else ""
    except:
        return ""


def score_competitive_gap(your_text: str,
                          query: str,
                          citations: list) -> dict:
    if not citations:
        return {
            "dimension_score": 50,
            "gap_analysis"   : "No citations available for comparison.",
            "missing_topics" : [],
            "recommendations": ["Could not fetch competitor data."]
        }

    competitor_texts = []
    for url in citations[:2]:
        t = fetch_competitor_text(url)
        if t:
            competitor_texts.append(f"URL: {url}\n{t}")

    if not competitor_texts:
        return {
            "dimension_score": 50,
            "gap_analysis"   : "Could not extract competitor content.",
            "missing_topics" : [],
            "recommendations": ["Check competitor URLs manually."]
        }

    competitors_combined = "\n\n---\n\n".join(competitor_texts)

    prompt = f"""
You are a GEO expert performing competitive content gap analysis.

Target query: "{query}"

YOUR CONTENT:
{your_text[:2000]}

CURRENTLY CITED COMPETITOR CONTENT:
{competitors_combined[:3000]}

Respond in this exact JSON format:
{{
  "dimension_score" : <0-100>,
  "gap_analysis"    : "<2-3 sentence summary>",
  "missing_topics"  : ["topic 1", "topic 2", "topic 3", "topic 4"],
  "recommendations" : ["action 1", "action 2", "action 3"]
}}"""
    return ask_openai(prompt)


# ══════════════════════════════════════════════════════════════
#  DIMENSION 5 — QUERY COVERAGE SCORE
# ══════════════════════════════════════════════════════════════

def generate_query_variants(query: str) -> list:
    prompt = f"""
Generate 5 different search queries related to: "{query}"

Cover these intents:
- Beginner question
- Specific detail question
- Practical/how-to question
- Cost or logistics question
- Comparison question

Respond as a valid JSON array of exactly 5 strings:
["query 1", "query 2", "query 3", "query 4", "query 5"]"""
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Return only a valid JSON array of 5 strings. No explanation."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.4
    )
    raw      = response.choices[0].message.content.strip()
    raw      = raw.replace("```json", "").replace("```", "").strip()
    variants = json.loads(raw)
    return variants[:5]


def score_query_coverage(index: VectorStoreIndex,
                         seed_query: str) -> dict:
    print("      Generating query variants...")
    variants = generate_query_variants(seed_query)
    results  = {}
    scores   = []

    for variant in variants:
        chunks = simulate_retrieval(index, variant, top_k=3)
        s      = score_retrieval(chunks)
        results[variant] = s["avg_score"]
        scores.append(s["avg_score"])
        print(f"      [{s['avg_score']:.2f}] {variant}")

    avg_coverage    = round(sum(scores) / len(scores), 4) if scores else 0
    dimension_score = round(avg_coverage * 100)
    weak_queries    = [q for q, s in results.items() if s < 0.6]

    return {
        "seed_query"      : seed_query,
        "variant_scores"  : results,
        "avg_coverage"    : avg_coverage,
        "weak_queries"    : weak_queries,
        "dimension_score" : dimension_score
    }


# ══════════════════════════════════════════════════════════════
#  OVERALL SCORE
# ══════════════════════════════════════════════════════════════

def calculate_overall_score(d1, d2, d3, d4, d5) -> dict:
    weights = {
        "retrieval"    : 0.25,
        "structure"    : 0.20,
        "hallucination": 0.20,
        "competitive"  : 0.20,
        "coverage"     : 0.15
    }

    weighted_score = round(
        d1["dimension_score"] * weights["retrieval"]     +
        d2["dimension_score"] * weights["structure"]     +
        d3["dimension_score"] * weights["hallucination"] +
        d4["dimension_score"] * weights["competitive"]   +
        d5["dimension_score"] * weights["coverage"]
    )

    if weighted_score >= 80:
        grade, verdict = "A", "Excellent — Strong GEO readiness"
    elif weighted_score >= 65:
        grade, verdict = "B", "Good — Minor improvements recommended"
    elif weighted_score >= 50:
        grade, verdict = "C", "Average — Significant improvements needed"
    else:
        grade, verdict = "D", "Poor — Major rework required before publishing"

    return {
        "overall_score": weighted_score,
        "grade"        : grade,
        "verdict"      : verdict,
        "weights_used" : weights
    }


# ══════════════════════════════════════════════════════════════
#  DISPLAY REPORT
# ══════════════════════════════════════════════════════════════

def display_report(source, query, d1, d2, d3, d4, d5, overall):
    W = "═" * 60

    print(f"\n{W}")
    print(f"  GEO CONTENT QUALITY REPORT")
    print(f"  Source  : {os.path.basename(source)}")
    print(f"  Query   : {query}")
    print(f"{W}")
    print(f"\n  OVERALL GEO SCORE   {overall['overall_score']}/100"
          f"  |  Grade: {overall['grade']}"
          f"  |  {overall['verdict']}")

    print(f"\n{'─'*60}")
    print(f"  DIMENSION SCORES")
    print(f"{'─'*60}")

    dims = [
        ("① Retrieval Score    ", d1["dimension_score"]),
        ("② Structure Score    ", d2["dimension_score"]),
        ("③ Hallucination Risk ", d3["dimension_score"]),
        ("④ Competitive Gap    ", d4["dimension_score"]),
        ("⑤ Query Coverage     ", d5["dimension_score"]),
    ]

    for label, score in dims:
        filled  = "█" * (score // 5)
        empty   = "░" * (20 - score // 5)
        status  = "✅" if score >= 70 else "⚠️ " if score >= 50 else "❌"
        print(f"  {status} {label} {score:3}/100  {filled}{empty}")

    print(f"\n{'─'*60}")
    print(f"  DIMENSION 1 — RETRIEVAL")
    print(f"{'─'*60}")
    print(f"  Avg Similarity  : {d1['avg_score']}")
    print(f"  Max Similarity  : {d1['max_score']}")
    print(f"  Chunks Retrieved: {d1['chunks_retrieved']}")

    print(f"\n{'─'*60}")
    print(f"  DIMENSION 2 — STRUCTURE ISSUES")
    print(f"{'─'*60}")
    for issue in d2.get("top_issues", []):
        print(f"  ⚠️  {issue}")
    print(f"  Recommendations:")
    for fix in d2.get("recommendations", []):
        print(f"  → {fix}")

    print(f"\n{'─'*60}")
    print(f"  DIMENSION 3 — HALLUCINATION RISK")
    print(f"{'─'*60}")
    for phrase in d3.get("high_risk_phrases", []):
        print(f"  ⚠️  \"{phrase}\"")
    print(f"  Recommendations:")
    for fix in d3.get("recommendations", []):
        print(f"  → {fix}")

    print(f"\n{'─'*60}")
    print(f"  DIMENSION 4 — COMPETITIVE GAP")
    print(f"{'─'*60}")
    print(f"  {d4.get('gap_analysis', '')}")
    for topic in d4.get("missing_topics", []):
        print(f"  ❌ {topic}")
    print(f"  Recommendations:")
    for fix in d4.get("recommendations", []):
        print(f"  → {fix}")

    print(f"\n{'─'*60}")
    print(f"  DIMENSION 5 — QUERY COVERAGE")
    print(f"{'─'*60}")
    for variant, score in d5.get("variant_scores", {}).items():
        status = "✅" if score >= 0.6 else "❌"
        print(f"  {status} [{score:.2f}] {variant}")
    if d5.get("weak_queries"):
        print(f"\n  Weak coverage queries (need more content):")
        for q in d5["weak_queries"]:
            print(f"  ❌ {q}")

    print(f"\n{W}\n")


# ══════════════════════════════════════════════════════════════
#  SAVE REPORT
# ══════════════════════════════════════════════════════════════

def save_report(source, input_type, query,
                chunks, d1, d2, d3, d4, d5, overall):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label     = (os.path.basename(source).replace(".md", "")
                 if input_type == "file"
                 else source.split("/")[2].replace(".", "_"))
    base      = f"reports/{label}_{timestamp}"

    report = {
        "timestamp" : timestamp,
        "input_type": input_type,
        "source"    : source,
        "query"     : query,
        "overall"   : overall,
        "dimensions": {
            "retrieval"    : d1,
            "structure"    : d2,
            "hallucination": d3,
            "competitive"  : d4,
            "coverage"     : d5
        },
        "retrieved_chunks": [
            {"rank" : i + 1,
             "score": round(s, 4) if s else None,
             "text" : c[:500]}
            for i, (c, s) in enumerate(chunks)
        ]
    }

    json_path = f"{base}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  💾 JSON  → {json_path}")

    csv_path = f"{base}_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "source", "query", "overall_score", "grade",
            "d1_retrieval", "d2_structure", "d3_hallucination",
            "d4_competitive", "d5_coverage"
        ])
        writer.writerow([
            timestamp, source, query,
            overall["overall_score"], overall["grade"],
            d1["dimension_score"], d2["dimension_score"],
            d3["dimension_score"], d4["dimension_score"],
            d5["dimension_score"]
        ])
    print(f"  📊 CSV   → {csv_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def run_simulation(source: str, query: str,
                   input_type: str = "url",
                   top_k: int = 5,
                   skip_serp: bool = False):

    print(f"\n{'═'*60}")
    print(f"  GEO SIMULATOR — 5 DIMENSION ANALYSIS")
    print(f"{'═'*60}")
    print(f"  Query  : {query}")
    print(f"  Source : {source}")

    # Extract
    print(f"\n[1/6] Extracting content...")
    text = extract_content(source, input_type)
    print(f"      {len(text):,} characters extracted")

    # Index
    print(f"[2/6] Building vector index...")
    index = build_index(text)

    # D1
    print(f"[3/6] Dimension 1 — Retrieval Score...")
    chunks = simulate_retrieval(index, query, top_k)
    d1     = score_retrieval(chunks)
    print(f"      Score: {d1['dimension_score']}/100")

    # D2
    print(f"[4/6] Dimension 2 — Structure Score...")
    d2 = score_structure(text, query)
    print(f"      Score: {d2['dimension_score']}/100")

    # D3
    print(f"[5/6] Dimension 3 — Hallucination Risk...")
    d3 = score_hallucination_risk(text, query)
    print(f"      Score: {d3['dimension_score']}/100")

    # D4 + D5
    live_citations = []
    if not skip_serp:
        print(f"[6/6] Dimension 4 — Competitive Gap + Dimension 5 — Query Coverage...")
        try:
            live_citations = fetch_live_citations(query)
            print(f"      {len(live_citations)} citations fetched")
        except Exception as e:
            print(f"      [SerpAPI error] {e}")
        d4 = score_competitive_gap(text, query, live_citations)
        print(f"      D4 Score: {d4['dimension_score']}/100")
    else:
        print(f"[6/6] SerpAPI skipped (skip_serp=True)...")
        d4 = {
            "dimension_score": 50,
            "gap_analysis"   : "Skipped — run with skip_serp=False for full analysis.",
            "missing_topics" : [],
            "recommendations": ["Enable SerpAPI for competitive gap analysis."]
        }

    d5 = score_query_coverage(index, query)
    print(f"      D5 Score: {d5['dimension_score']}/100")

    # Overall
    overall = calculate_overall_score(d1, d2, d3, d4, d5)

    # Display
    display_report(source, query, d1, d2, d3, d4, d5, overall)

    # Save
    print(f"  Saving reports...")
    save_report(source, input_type, query,
                chunks, d1, d2, d3, d4, d5, overall)

    print(f"\n{'═'*60}")
    print(f"  ✅ Simulation complete!")
    print(f"{'═'*60}\n")

    return overall


# ══════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── OPTION A: URL input ───────────────────────────────────
    # run_simulation(
    #     source    = "https://en.wikipedia.org/wiki/Backpacking_(hiking)",
    #     query     = "what gear do you need for backpacking",
    #     input_type= "url",
    #     top_k     = 5,
    #     skip_serp = False
    # )

    # ── OPTION B: Markdown file ───────────────────────────────
    run_simulation(
        source    = "drafts/my_article.md",
        query     = "Best trekking destinations in Nepal",
        input_type= "file",
        top_k     = 5,
        skip_serp = False
    )