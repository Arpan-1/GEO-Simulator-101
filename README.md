1. How the System Works — The Logic
Think of your simulator as a miniature version of what Google's AI does when it decides whose content to cite in an AI Overview. Here's each stage:
Stage 1 — Content Extraction (Trafilatura)
When you give the simulator a URL, trafilatura fetches the page and strips away all the HTML noise — navigation menus, ads, footers, cookie banners — and returns only the clean readable text. This is important because AI search systems don't index raw HTML either; they work with clean text.
Stage 2 — Chunking (LlamaIndex)
The clean text is too long to process as one block, so LlamaIndex automatically splits it into smaller chunks — typically 512 tokens (~400 words) each, with slight overlaps so context isn't lost at boundaries. Each chunk becomes an independently retrievable unit. This mirrors how real RAG systems work — they don't retrieve whole pages, they retrieve specific passages.
Stage 3 — Embedding (Google Gemini)
Each chunk is passed through Google's gemini-embedding-001 model, which converts it into a vector — a list of numbers that represents its meaning in mathematical space. Your query is also converted into a vector the same way. The key insight is: semantically similar text produces similar vectors, regardless of exact wording.
Stage 4 — Retrieval Simulation
The simulator compares your query's vector against every chunk's vector using cosine similarity — essentially measuring the angle between them in high-dimensional space. Chunks closest in meaning to the query get the highest scores. The top 5 are returned. This is the core of what RAG-based AI search systems do before generating an answer.
Stage 5 — Live Citation Comparison (SerpAPI)
The simulator then calls Google via SerpAPI and fetches what the real AI Overview is actually citing for that same query today. This gives you a ground truth to compare against your simulated results.
Stage 6 — Report Saving
Everything — chunks, scores, citations, timestamp — is saved to JSON and CSV so you can track how your content performs over time and measure the impact of edits.

The Full Flow in Simple Terms
Your URL
   ↓
Clean text extracted          (Trafilatura)
   ↓
Text split into chunks        (LlamaIndex)
   ↓
Chunks converted to vectors   (Gemini Embeddings)
   ↓
Query also converted to vector
   ↓
Similarity measured between query ↔ each chunk
   ↓
Top 5 most relevant chunks returned
   ↓
Compared against what Google AI actually cites today
   ↓
Results + scores saved to file

2. What the Output Means — SEO Professional's Guide
Retrieved Chunks
These are the exact passages an AI retriever would extract from your page to answer a query. As an SEO professional, this tells you:

Which parts of your content are doing the heavy lifting — if your most important section never appears in the top chunks, it's invisible to AI search regardless of how well-written it is
Which parts are ignored — if Chunk 5 is always your references section, that's noise and signals your content structure may need work
Whether your key facts are front-loaded — AI retrievers favor passages where the answer appears early in the chunk, not buried in the middle

Similarity Scores
These are cosine similarity scores between 0 and 1. Here's how to read them as an SEO professional:
Score RangeMeaningAction0.85 – 1.00Excellent matchContent is highly retrievable for this query0.70 – 0.84Good matchContent is competitive but improvable0.50 – 0.69Weak matchContent needs restructuring for this queryBelow 0.50Poor matchPage likely won't be cited for this query
Avg vs Max Score

Max score tells you your single best chunk — your strongest passage for that query
Avg score tells you the overall retrieval health of the whole page — a high max but low average means only one section is doing the work, which is a structural weakness

Live AI Overview Citations
This is your competitive intelligence. It shows exactly who Google's AI is trusting right now for your target query. As an SEO professional you can:

Analyze cited pages — what structure, depth, and format do they use that yours doesn't?
Identify the gap — is your score high but still not cited? The bottleneck is trust/authority, not content quality
Track citation shifts — run the same query weekly and watch if competitors enter or leave the citations list
Benchmark your score against cited pages — run the simulator on a cited competitor URL with the same query and compare scores

"Your Domain Cited? YES/NO"
This is the bottom line metric. It directly answers: is your content making it into AI-generated answers? Over time, moving this from NO to YES for your target queries is the primary KPI of your GEO efforts.

3. How Authentic Are the Insights?
This is the most important question, and the honest answer is: highly directional, not perfectly precise. Here's why:
What's Authentic ✅
The retrieval simulation is genuinely meaningful. You're using the same embedding model family (Gemini) that powers Google's AI search infrastructure. When a chunk scores 0.85, it genuinely is semantically close to your query — that's real math, not guesswork.
The live citations are 100% real. What SerpAPI returns is the actual AI Overview Google is serving right now. There's no simulation in that part — it's live data.
The gap analysis is real. When your page scores 0.72 but isn't cited while a competitor scoring lower is, that's a genuine signal that non-content factors (authority, trust, entity recognition) are influencing the outcome.
Trends over time are reliable. If you edit your content to better answer a query and your avg score jumps from 0.65 to 0.78, that improvement is real and meaningful.
What's an Approximation ⚠️
Google's exact retrieval model is proprietary. Your simulator uses Gemini embeddings as a proxy, but Google's internal retrieval system for AI Overviews uses undisclosed model versions with additional layers — entity linking, trust scoring, freshness signals — that your simulator doesn't replicate.
Chunk boundaries may differ. Your simulator uses LlamaIndex's default chunking (512 tokens). Google's system may chunk differently — by heading, by paragraph, or dynamically based on query type.
Scores aren't directly comparable to Google's internal scores. A 0.74 in your simulator doesn't mean Google internally scores it 0.74. The numbers are relative indicators, not absolute measures.
Context window and synthesis logic aren't simulated. After retrieval, the generative layer decides how to use the retrieved chunks. Your simulator stops at retrieval — it doesn't model whether the LLM would actually include your passage in the final answer.
The Right Mental Model
Think of your simulator the way a weather forecast works — it's based on real atmospheric data and genuine physics, and it's directionally accurate, but it's not the actual weather. A 70% retrieval readiness score means your content is well-positioned to be cited, not guaranteed. The simulator gives you a calibrated leading indicator, not a definitive verdict.

The teams that use this tool best will treat it as a hypothesis engine — form a hypothesis ("if I restructure this section as a direct Q&A, my score will improve"), test it in the simulator, then verify against live citation data over time.


Summary for Your GEO Workflow
Simulator tells you    →  Is my content structured for AI retrieval?
Live citations tell you →  Who is Google actually trusting right now?
The gap between them   →  Where to focus optimization efforts
Reports over time      →  Did my edits actually improve performance?