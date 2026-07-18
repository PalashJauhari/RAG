# Dataset research — non-toy, PDF-native, ingestion→answer→eval fit

> Researched against the actual pipeline (`ingestion/` Unstructured hi-res PDF → chunk → Qdrant,
> `graph/` fact-decompose → retrieve → recall-verify → repair-loop → answer,
> `benchmarking/hotpotqa/` RAGAS eval: context precision/recall, faithfulness, answer correctness).
> No code changed — this is dataset selection research only.

## What "good fit" means for *this* codebase specifically

| Requirement | Why it matters here |
|---|---|
| Source is real PDFs (not pre-chunked text/HTML) | `ingestion/unstructured_pipeline.py` is built around Unstructured's hi-res partitioner + `chunk_by_title` — it wants raw PDFs, ideally with real tables/layout to actually exercise `infer_table_structure` |
| Has QA pairs **with evidence** (text span + page/doc id) | Your benchmark harness (`benchmarking/hotpotqa/download_process_hotpotqa.py` → `run_evaluation.py`) needs `question`, `reference_answer`, `contexts[].is_supporting` to compute RAGAS context precision/recall — datasets with an evidence string + page slot into this shape with minimal glue code |
| Questions require **multiple facts** / multi-hop reasoning | Your `fact_decomposition` → `recall_check` → repair loop is the whole point of the project; a dataset of single-fact lookups (most FAQ-style sets) never exercises the repair loop, gap-fill, or strategy escalation |
| Domain reads well on a CV / to a non-technical reviewer | "Legal contract review assistant" or "financial filings analyst" lands better in an interview than another biomedical RAG demo |
| Not already saturated in every RAG tutorial | Ruled out SQuAD, Natural Questions, MS MARCO, TriviaQA, and (since you already use it) HotpotQA itself |

---

## Ranked shortlist

### 1. Climate Finance Bench — top recommendation

- **Link:** https://github.com/pladifes/climate_finance_bench (paper: https://arxiv.org/abs/2505.22752)
- **What it is:** 33 real corporate sustainability/ESG report **PDFs** (all 11 GICS sectors) + **330 expert-validated QA pairs** (10 per report). Each QA row has: gold answer, source document, **page number**, verbatim **extract**, and a label for **extraction / numerical reasoning / logical reasoning** question type.
- **Why it fits *your* pipeline especially well:**
  - Question-type taxonomy (extraction vs numerical vs logical) is a natural stand-in for "how many facts does this need" — logical-reasoning questions are exactly the multi-fact, repair-loop-exercising case.
  - Gold answer + page + extract maps almost 1:1 onto your `contexts[].is_supporting` / `reference_contexts` shape.
  - Small (33 PDFs) — you can ingest the *entire* corpus in one pass and demo end-to-end, unlike FinanceBench's 361 filings.
  - Freshest and least reused of everything found (May 2025 paper, small GitHub footprint) — good for standing out.
- **Caveat:** small QA set (330) — fine for a demo/eval report, not for statistical significance claims.
- **Narrative:** "climate-disclosure compliance / ESG analyst assistant" — timely, distinct from every finance-bot demo.

### 2. FinanceBench

- **Link:** https://github.com/patronus-ai/financebench (paper: https://arxiv.org/abs/2311.11944)
- **What it is:** Open subset of **150** annotated question/answer/evidence triples over real **10-K/10-Q/8-K/earnings-report PDFs** (40 public companies), plus the source **PDFs themselves ship in the repo** (`/pdfs/`). Full paper dataset is 10,231 Q&A but only 150 + PDFs are public.
- **Why it fits:** Evidence string + page number per answer (same shape as your HotpotQA harness), real financial-filing PDFs with tables — good stress test for `unstructured_infer_table_structure`. Widely cited in RAG literature as "the" finance RAG benchmark, but far less commonly used as an actual demo corpus than the toy sets — most people just cite the paper's numbers rather than ingest the PDFs themselves.
- **Caveat:** questions are largely single-document, single-fact extraction ("What was Boeing's FY2022 COGS?") — you'd want to hand-write a few comparison/multi-hop questions across companies (e.g. "Compare Boeing's and Lockheed's FY2022 gross margins") to actually exercise `query_splitter`/repair loop; the dataset alone under-uses your architecture's most interesting part.

### 3. CUAD — Contract Understanding Atticus Dataset

- **Link:** https://www.atticusprojectai.org/cuad · https://huggingface.co/datasets/theatticusproject/cuad-qa · https://github.com/TheAtticusProject/cuad
- **What it is:** **510 real commercial contract PDFs** + **13,000+** expert clause annotations across **41 clause categories** (termination, exclusivity, IP assignment, change-of-control, etc.), released as a SQuAD-2.0-style JSON. CC BY 4.0.
- **Why it fits:** Legal contract review is an extremely intuitive "why does this exist" pitch. 41 clause categories per contract is a natural source of **multi-fact questions** you can synthesize yourself: "Does this agreement have both an exclusivity clause and a most-favored-customer clause?" is exactly a 2-fact `fact_decomposition` question your graph is built for.
- **Caveat:** the dataset's native task is span extraction per clause category, not open-book QA with a single reference answer — you'd need a light transform step (or an LLM pass) to turn clause-label pairs into natural-language questions + reference answers before feeding your benchmarking harness. Not a blocker, just extra glue code (no code was written for this, just flagging the work).

### 4. QASPER (scientific papers)

- **Link:** https://huggingface.co/datasets/allenai/qasper · paper: https://arxiv.org/abs/2105.03011
- **What it is:** 5,049 questions over **1,585 real NLP research papers** (fetchable as PDFs from arXiv by paper id), each answered with supporting evidence explicitly tagged as coming from **multiple paragraphs** (55.5% of questions require multi-paragraph evidence) or tables/figures (13%).
- **Why it fits:** This is the dataset most purpose-built for testing exactly what your recall-check + repair loop does — a huge fraction of questions are *designed* to require evidence synthesis across non-adjacent passages, which is precisely the "does the retrieved context alone support this fact" test your `recall_check_node` runs.
- **Caveat:** papers ship as parsed text (S2ORC) in the HF dataset, not as bundled PDFs — you'd fetch the actual PDFs from arXiv yourself using the paper ids (straightforward, arXiv PDFs are freely downloadable) to exercise your PDF-native ingestion pipeline.

### 5. DocFinQA (harder-mode financial multi-hop)

- **Link:** https://huggingface.co/datasets/kensho/DocFinQA-Test (also linked from the FinQA/DocFinQA papers; OpenReview PDF: https://openreview.net/attachment?id=eVZZRWhggb&name=pdf)
- **What it is:** Extension of FinQA using **full multi-page SEC 10-K filings as PDFs** (not just the curated paragraph FinQA normally provides) — average context length ~123K tokens vs FinQA's ~687. Answers require multi-step arithmetic derivations (e.g. `200,657-50,565`).
- **Why it fits:** This is a genuine stress test of retrieval-then-verify at scale — full, real, messy SEC filings rather than pre-selected paragraphs. Good for demonstrating that your fixed-top-k, HasId-exclusion repair loop can actually locate the right numbers inside a 100+ page filing rather than a curated snippet.
- **Caveat:** heavier lift — long documents, and answers are computed values (arithmetic over extracted numbers) rather than free text, so your `FinalAnswer`/RAGAS answer-correctness scoring would need numeric-tolerance comparison rather than semantic similarity.

### 6. TAT-DQA (financial tables + text, visually-rich PDFs)

- **Link:** https://github.com/NExTplusplus/TAT-DQA · https://huggingface.co/datasets/next-tat/TAT-DQA
- **What it is:** 16,558 questions over **2,758 real financial-report PDF pages** with heavy table/text mixing, ships with the **actual PDF pages** (`tatdqa_docs_*.zip`) plus OCR-derived JSON.
- **Why it fits:** Directly tests `unstructured_infer_table_structure=True` in your ingestion config — questions frequently require reading numbers out of tables, not prose.
- **Caveat:** documents are short (mostly single pages, ~550 words) — less "big corpus to ingest" feel than the others; best used as a table-reasoning stress test alongside a primary dataset rather than alone.

### 7. AeroEngQA + NASA Technical Reports Server (STELLA corpus) — most novel/uncommon

- **Link:** AeroEngQA dataset: https://doi.org/10.5281/zenodo.14215677 · paper: https://www.southampton.ac.uk/~sem03/aiaa-2025-preprint.pdf · NTRS corpus/benchmark framework (STELLA): https://arxiv.org/abs/2601.03496 (corpus source: https://ntrs.nasa.gov)
- **What it is:** AeroEngQA is 80 small, high-quality human-annotated QA pairs sourced from real **NASA technical reports, NTSB reports, and patents (all PDF)**. Separately, NASA's own **NTRS** (Technical Reports Server) is a huge, freely downloadable, copyright-clear PDF corpus (aeronautics/astronautics categories) that the STELLA paper already filtered for text-centric, post-2000, non-copyrighted documents — you could pair NTRS as your ingestion corpus with AeroEngQA as a small gold eval set, or hand-augment with a few more QA pairs.
- **Why it fits:** Genuinely uncommon domain for a RAG portfolio project (aerospace engineering) — very likely nobody else on the interview panel has seen this pitched before. Real technical PDFs, real layout complexity (figures, tables, equations).
- **Caveat:** AeroEngQA alone is tiny (80 pairs) — good for a qualitative demo, not enough for a statistically meaningful benchmark table unless you also generate additional QA pairs (e.g. LLM-assisted, human-checked) against the wider NTRS corpus.

### 8. ObliQA (regulatory/compliance)

- **Link:** paper/dataset pointers: https://arxiv.org/abs/2409.05677 (RIRAG / ObliQA)
- **What it is:** 27,869 questions over Abu Dhabi Global Markets financial-regulation documents, each mapped to the specific regulatory obligation passages that answer it — explicitly multi-document, multi-passage by design.
- **Why it fits:** "Regulatory compliance assistant" is a strong enterprise narrative, and the dataset is inherently multi-passage (good repair-loop exercise).
- **Caveat:** narrow single-jurisdiction domain (Abu Dhabi financial regulation) may need framing/context for a general audience; check exact download path/license before committing (paper linked above; dataset access details should be confirmed from the ObliQA repo referenced in the paper).

### 9. MAUD (merger agreements) — for reference, likely not the best primary pick

- **Link:** https://www.atticusprojectai.org/maud · https://github.com/TheAtticusProject/maud
- **What it is:** 152 real merger-agreement PDFs, 92 standardized "deal point" questions per agreement, 47,000+ annotations, from the same Atticus Project as CUAD.
- **Why it's here:** Same appealing "legal AI" narrative as CUAD, arguably richer (deal points genuinely require cross-referencing multiple clauses).
- **Caveat:** native task is closer to multiple-choice/classification over pre-extracted deal-point text than open-book generative QA — more adaptation work than CUAD to fit your `FinalAnswer` free-text schema.

### Also considered, not shortlisted (with reason)

| Dataset | Why not top-listed |
|---|---|
| PatentMatch (patent prior art) | Source is 200GB of raw EPO XML, not PDF-native; heavy data-engineering lift before it even reaches your ingestion pipeline |
| Aviation Safety QA (NTSB/ASRS, ~350K pairs) | Source narratives are extractive-span QA (SQuAD-style), single-fact by construction — doesn't exercise multi-hop/repair loop without rework |
| UniDoc-Bench | Excellent and very broad (70K PDF pages, 8 domains) but built for **multimodal** (image+table+text) RAG evaluation — bigger scope than your current text-only pipeline; worth a look later if you add image/table-aware retrieval |
| Synthetic corpora (e.g. "Aether Dynamics Corp", dataset-factory) | Not real-world PDFs, weaker CV story ("I generated my own eval set" reads very differently from "I benchmarked against a published academic dataset") |

---

## My actual recommendation

For a CV/portfolio demo specifically:

1. **Primary corpus + eval: Climate Finance Bench.** Small enough to fully ingest in one sitting, ships gold answers with page/extract evidence that maps directly onto your existing HotpotQA-style benchmark harness, freshest and least-seen dataset on this list, and "AI for climate-risk disclosure" is a strong, current narrative.
2. **Stretch/secondary demo: CUAD.** Reuse the same ingestion pipeline on 510 real contract PDFs; hand-craft ~20–30 multi-clause questions (e.g. "Does Contract X have both an exclusivity clause and a change-of-control clause?") to specifically showcase `fact_decomposition` + the repair loop — this is the dataset best suited to *showing off the architecture*, even though the out-of-the-box QA format needs a small transform step first.
3. If you want a "hard mode" latency/robustness story for the write-up, **DocFinQA** (full 100+ page SEC filings) is the best available stress test of retrieval-then-verify at real document scale.

None of this required or made any code changes — purely dataset research, per your instruction.
