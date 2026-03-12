import streamlit as st
import os
import tempfile
import json
from datetime import datetime
from geo_simulator import (
    extract_from_url,
    extract_from_md,
    build_index,
    simulate_retrieval,
    score_retrieval,
    score_structure,
    score_hallucination_risk,
    fetch_live_citations,
    score_competitive_gap,
    score_query_coverage,
    calculate_overall_score,
    save_report
)

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="GEO Content Simulator 1.0",
    layout="wide"
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
         font-size: 3rem !important;
        font-weight: 800 !imporant;
        color: #1010HH !important;
        margin-bottom: 0;
        text-align: center;
    }
    .sub-header {
        font-size: 10rem;
        color: #666;
        margin-bottom: 2rem;
        text-align: center;
    }
    .score-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .score-number {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
    }
    .grade-a { color: #00c853; }
    .grade-b { color: #64dd17; }
    .grade-c { color: #ffd600; }
    .grade-d { color: #ff1744; }
    .dim-label {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.2rem;
    }
    .recommendation-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        color: #1a1a1a;
        margin: 0.4rem 0;
    }
    .risk-box {
        background: #fce4ec;
        border-left: 4px solid #e91e63;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        color: #1a1a1a;
        margin: 0.4rem 0;
    }
    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        color: #1a1a1a;
        margin: 0.4rem 0;
    }
    .stProgress > div > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────
st.markdown('<p class="main-header">GEO Content Simulator 1.0</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyze your content across 5 different GEO dimensions before publishing it on the internet</p>',
            unsafe_allow_html=True)
st.divider()


# ─── Sidebar: Inputs ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Simulation Settings")

    input_type = st.radio(
        "Input Type",
        ["Markdown File", "URL"],
        help="File = pre-publish testing | URL = live page simulation"
    )

    source     = None
    tmp_path   = None

    if input_type == "Markdown File":
        uploaded = st.file_uploader(
            "Upload your .md file",
            type=["md"],
            help="Upload the markdown file you want to analyze"
        )
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".md", mode="wb")
            tmp.write(uploaded.read())
            tmp.flush()
            tmp_path = tmp.name
            source   = tmp_path
            st.success(f"✅ {uploaded.name} uploaded")
    else:
        source = st.text_input(
            "Page URL",
            placeholder="https://yourwebsite.com/article",
            help="Enter the live URL to analyze"
        )

    query = st.text_input(
        "Target Query",
        placeholder="Best trekking destinations in Nepal",
        help="The search query you want your content to rank for"
    )

    top_k = st.slider(
        "Chunks to Retrieve (top_k)",
        min_value=3,
        max_value=10,
        value=5,
        help="How many content chunks to retrieve and analyze"
    )

    skip_serp = not st.checkbox(
        "Enable Live Citations (SerpAPI)",
        value=True,
        help="Uncheck to save SerpAPI credits"
    )

    run_button = st.button(
        "🚀 Run Simulation",
        type="primary",
        use_container_width=True,
        disabled=not (source and query)
    )

    st.divider()
    st.caption("💡 Each full simulation costs ~$0.002 in OpenAI credits")
    st.caption("📊 Reports auto-saved to /reports folder")


# ─── Score Color Helper ───────────────────────────────────────
def score_color(score: int) -> str:
    if score >= 80: return "#00c853"
    if score >= 65: return "#64dd17"
    if score >= 50: return "#ffd600"
    return "#ff1744"


def grade_class(grade: str) -> str:
    return f"grade-{grade.lower()}"


# ─── Main Area ────────────────────────────────────────────────
if run_button and source and query:

    # ── Progress bar ─────────────────────────────────────────
    progress = st.progress(0, text="Starting simulation...")
    status   = st.empty()

    try:
        # Step 1
        status.info("📄 Extracting content...")
        if input_type == "Markdown File":
            text = extract_from_md(source)
        else:
            text = extract_from_url(source)
        progress.progress(15, text="Content extracted")

        # Step 2
        status.info("🧠 Building vector index...")
        index = build_index(text)
        progress.progress(30, text="Vector index built")

        # Step 3
        status.info("🔎 Dimension 1 — Retrieval scoring...")
        chunks = simulate_retrieval(index, query, top_k)
        d1     = score_retrieval(chunks)
        progress.progress(45, text="Retrieval scored")

        # Step 4
        status.info("📐 Dimension 2 — Structure analysis...")
        d2 = score_structure(text, query)
        progress.progress(58, text="Structure analyzed")

        # Step 5
        status.info("🧪 Dimension 3 — Hallucination risk check...")
        d3 = score_hallucination_risk(text, query)
        progress.progress(70, text="Hallucination risk checked")

        # Step 6
        live_citations = []
        if not skip_serp:
            status.info("🌐 Fetching live citations...")
            try:
                live_citations = fetch_live_citations(query)
            except Exception as e:
                st.warning(f"SerpAPI error: {e}")
            d4 = score_competitive_gap(text, query, live_citations)
        else:
            d4 = {
                "dimension_score": 50,
                "gap_analysis"   : "Skipped — enable SerpAPI for full analysis.",
                "missing_topics" : [],
                "recommendations": ["Enable live citations for competitive gap analysis."]
            }
        progress.progress(83, text="Competitive gap analyzed")

        # Step 7
        status.info("🔄 Dimension 5 — Query coverage...")
        d5 = score_query_coverage(index, query)
        progress.progress(95, text="Query coverage scored")

        # Overall
        overall = calculate_overall_score(d1, d2, d3, d4, d5)
        progress.progress(100, text="✅ Complete!")
        status.empty()

        # ── Save report ───────────────────────────────────────
        save_report(
            source if input_type == "URL" else "uploaded_file.md",
            "url" if input_type == "URL" else "file",
            query, chunks, d1, d2, d3, d4, d5, overall
        )

        # ══════════════════════════════════════════════════════
        #  RESULTS DISPLAY
        # ══════════════════════════════════════════════════════

        st.divider()

        # ── Overall Score ─────────────────────────────────────
        col_score, col_grade, col_verdict = st.columns([1, 1, 2])

        with col_score:
            color = score_color(overall["overall_score"])
            st.markdown(f"""
            <div class="score-card">
                <div class="dim-label">OVERALL GEO SCORE</div>
                <div class="score-number" style="color:{color}">
                    {overall['overall_score']}
                </div>
                <div class="dim-label">out of 100</div>
            </div>""", unsafe_allow_html=True)

        with col_grade:
            st.markdown(f"""
            <div class="score-card">
                <div class="dim-label">GRADE</div>
                <div class="score-number {grade_class(overall['grade'])}">
                    {overall['grade']}
                </div>
                <div class="dim-label">&nbsp;</div>
            </div>""", unsafe_allow_html=True)

        with col_verdict:
            st.markdown(f"""
            <div class="score-card" style="text-align:left;">
                <div class="dim-label">VERDICT</div>
                <div style="font-size:1.1rem; font-weight:600; margin:0.5rem 0;">
                    {overall['verdict']}
                </div>
                <div class="dim-label">
                    Query: <strong>{query}</strong>
                </div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Dimension Score Bars ──────────────────────────────
        st.subheader("📊 Dimension Scores")

        dims = [
            ("① Retrieval Score",    d1["dimension_score"],
             "Semantic match between content and query"),
            ("② Structure Score",    d2["dimension_score"],
             "Content organization for AI extraction"),
            ("③ Hallucination Risk", d3["dimension_score"],
             "Accuracy risk when AI synthesizes answers"),
            ("④ Competitive Gap",    d4["dimension_score"],
             "Coverage vs currently cited competitors"),
            ("⑤ Query Coverage",     d5["dimension_score"],
             "Breadth across related search intents"),
        ]

        for label, score, desc in dims:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{label}** — {desc}")
                st.progress(score / 100)
            with col2:
                color = score_color(score)
                st.markdown(
                    f"<div style='text-align:center;"
                    f"font-size:1.5rem; font-weight:700;"
                    f"color:{color}'>{score}/100</div>",
                    unsafe_allow_html=True
                )
            st.write("")

        st.divider()

        # ── Detailed Tabs ─────────────────────────────────────
        t1, t2, t3, t4, t5, t6 = st.tabs([
            "🔎 Retrieval",
            "📐 Structure",
            "🧪 Hallucination",
            "🌐 Competitive",
            "🔄 Coverage",
            "📄 Raw Report"
        ])

        # Tab 1 — Retrieval
        with t1:
            st.subheader("Retrieved Chunks")
            st.caption(
                f"Avg score: **{d1['avg_score']}** | "
                f"Max score: **{d1['max_score']}** | "
                f"Chunks: **{d1['chunks_retrieved']}**"
            )
            for i, (chunk, score) in enumerate(chunks, 1):
                score_val = f"{score:.4f}" if score else "n/a"
                with st.expander(f"Chunk {i} — Score: {score_val}"):
                    st.write(chunk)

        # Tab 2 — Structure
        with t2:
            st.subheader("Structure Analysis")
            sub_scores = {
                "Heading Alignment"       : d2.get("heading_alignment", 0),
                "Answer Front-loading"    : d2.get("answer_front_loading", 0),
                "Paragraph Extractability": d2.get("paragraph_extractability", 0),
                "Ambiguity Level"         : d2.get("ambiguity_level", 0),
                "Scannability"            : d2.get("scannability", 0),
            }
            for label, score in sub_scores.items():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{label}**")
                    st.progress(score / 20)
                with col2:
                    st.markdown(f"**{score}/20**")

            st.subheader("Issues Found")
            for issue in d2.get("top_issues", []):
                st.markdown(
                    f'<div class="risk-box">⚠️ {issue}</div>',
                    unsafe_allow_html=True)

            st.subheader("Recommendations")
            for fix in d2.get("recommendations", []):
                st.markdown(
                    f'<div class="recommendation-box">→ {fix}</div>',
                    unsafe_allow_html=True)

        # Tab 3 — Hallucination
        with t3:
            st.subheader("Hallucination Risk Analysis")
            sub_scores = {
                "Factual Specificity"  : d3.get("factual_specificity", 0),
                "Claim Clarity"        : d3.get("claim_clarity", 0),
                "Internal Consistency" : d3.get("internal_consistency", 0),
                "Source Groundedness"  : d3.get("source_groundedness", 0),
                "Scope Clarity"        : d3.get("scope_clarity", 0),
            }
            for label, score in sub_scores.items():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{label}**")
                    st.progress(score / 20)
                with col2:
                    st.markdown(f"**{score}/20**")

            st.subheader("High-Risk Phrases")
            phrases = d3.get("high_risk_phrases", [])
            if phrases:
                for phrase in phrases:
                    st.markdown(
                        f'<div class="risk-box">⚠️ "{phrase}"</div>',
                        unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="success-box">✅ No high-risk phrases detected</div>',
                    unsafe_allow_html=True)

            st.subheader("Recommendations")
            for fix in d3.get("recommendations", []):
                st.markdown(
                    f'<div class="recommendation-box">→ {fix}</div>',
                    unsafe_allow_html=True)

        # Tab 4 — Competitive
        with t4:
            st.subheader("Competitive Gap Analysis")
            st.info(d4.get("gap_analysis", ""))

            if live_citations:
                st.subheader("Currently Cited Pages")
                for url in live_citations:
                    st.markdown(f"🔗 {url}")

            st.subheader("Missing Topics")
            for topic in d4.get("missing_topics", []):
                st.markdown(
                    f'<div class="risk-box">❌ {topic}</div>',
                    unsafe_allow_html=True)

            st.subheader("Recommendations")
            for fix in d4.get("recommendations", []):
                st.markdown(
                    f'<div class="recommendation-box">→ {fix}</div>',
                    unsafe_allow_html=True)

        # Tab 5 — Coverage
        with t5:
            st.subheader("Query Coverage Analysis")
            st.metric("Average Coverage Score",
                      f"{d5['avg_coverage']:.3f}",
                      help="Average retrieval score across all query variants")

            st.subheader("Query Variant Scores")
            for variant, score in d5.get("variant_scores", {}).items():
                col1, col2 = st.columns([4, 1])
                with col1:
                    status = "✅" if score >= 0.6 else "❌"
                    st.markdown(f"{status} {variant}")
                    st.progress(score)
                with col2:
                    st.markdown(f"**{score:.3f}**")

            if d5.get("weak_queries"):
                st.subheader("⚠️ Weak Coverage Areas")
                for q in d5["weak_queries"]:
                    st.markdown(
                        f'<div class="risk-box">❌ {q}</div>',
                        unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="success-box">✅ All query variants have adequate coverage</div>',
                    unsafe_allow_html=True)

        # Tab 6 — Raw JSON
        with t6:
            st.subheader("Full JSON Report")
            full_report = {
                "timestamp" : datetime.now().strftime("%Y%m%d_%H%M%S"),
                "query"     : query,
                "overall"   : overall,
                "dimensions": {
                    "retrieval"    : d1,
                    "structure"    : d2,
                    "hallucination": d3,
                    "competitive"  : d4,
                    "coverage"     : d5
                }
            }
            report_json = json.dumps(full_report, indent=2)
            st.download_button(
                label="⬇️ Download Full JSON Report",
                data=report_json,
                file_name=f"geo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            st.json(full_report)

    except Exception as e:
        progress.empty()
        status.empty()
        st.error(f"❌ Simulation failed: {str(e)}")
        st.exception(e)

else:
    # ── Empty state ───────────────────────────────────────────
    st.markdown("""
    ### <<<  Get Started Here
    1. Choose your input type in the sidebar
    2. Upload a markdown file or enter a URL
    3. Enter your target search query
    4. Click **Run Simulation**

    ---

    ### About The System
    A system developed by Arpan Chaudhary, to understand the working mechanism of the Generative Engine in a controlled environment. The system helps to predict how the content may actually perform for the real generative engine, before they are published on the internet.

    ---

    ### What This Simulator Measures
    | Dimension | What it checks |
    |---|---|
    | ① Retrieval Score | Can AI retrievers find your content? |
    | ② Structure Score | Is your content organized for AI extraction? |
    | ③ Hallucination Risk | Will AI misrepresent your content? |
    | ④ Competitive Gap | What do cited competitors have that you don't? |
    | ⑤ Query Coverage | Does your content cover related search intents? |
    """)