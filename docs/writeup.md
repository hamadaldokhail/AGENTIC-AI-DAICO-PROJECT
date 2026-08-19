# DAICO Navigator - Capstone Write-up

One paragraph per rubric section. Every claim below is backed by captured output in
`DAICO_Navigator.ipynb` (run top-to-bottom; execution counts 1-20, in order).

**Programme:** SDAIA Academy - *Building Agentic AI Systems*, DAICO / King Saud University, sponsored by SDAIA.
**Cohort:** 16-20 August 2026. **Declared track:** Track A - Supervisor (with a Track B human handoff).
**Team:** Hamed Ahmed Aldkhyyal, Saif Fawaz Alanazi, Fahad Abdullah Alanazi, Yousef Saleh Alanzi.

---

### §1 - Agent fundamentals
Four tools (`search_courses`, `get_course_details`, `check_seat_availability`, `get_prerequisites`) read the
real `data/courses.json` and act on their arguments - `search_courses` tokenises the query and matches against
title/description/level/track. In the demo the model *chose* to call `search_courses`, then
`check_seat_availability("DL-301")`, then `get_prerequisites("DL-301")` on its own, and answered "3 seats …
complete ML-201 first" from live data. Separately, `with_structured_output(CourseRecommendation)` returns a
Pydantic object that code reads (`rec.course_id == "AGT-401"`, `rec.prerequisites == ["ML-201"]`), not a human.

### §2 - Multi-agent / routing architecture (Track A: Supervisor)
Routing is an LLM decision via `with_structured_output`. A `Route` Pydantic model constrains the destination to
four workers (`course_advisor`, `enrollment`, `capstone_mentor`, `faq`) with a one-sentence reason. The demo
routed four different messages correctly, and the end-to-end run routed live traffic to `course_advisor`,
`faq`, and `capstone_mentor`. There is no `if ... in question.lower()` anywhere - the classification is the
model's.

### §3 - RAG pipeline (Hybrid)
Four DAICO documents are **loaded**, **split** with `RecursiveCharacterTextSplitter` (19 chunks), **embedded**
with `text-embedding-3-small`, **stored** in `InMemoryVectorStore`, and **retrieved**. The verbatim-answer probe
("what attendance percentage…") returned the exact `policies.md` chunk stating "minimum of 80% attendance", and
the grounded `rag_answer` correctly reported 80% attendance and a 60/100 pass mark. We chose **Hybrid RAG**:
structured facts (seats, prerequisites) are served by the §1 tools, while unstructured prose (policies, FAQ,
rubric) is served by vector retrieval, and the supervisor decides per message which to use.

### §4 - Context & state management
`InMemorySaver` provides short-term, per-conversation state keyed by `thread_id`; a **separate** `InMemoryStore`
holds durable learner facts. The cross-thread test writes an interest in thread `conv-A` and reads it back in a
different thread `conv-B`, with assertions confirming both the interest ("agentic AI") and a counter (1 → 2)
persisted across threads. The end-to-end run reinforces this: one learner's question counter reaches 3 across
three different threads. This is genuine long-term memory, not a growing message list.

### §5 - Human-in-the-loop
`enroll_task` calls `interrupt()` before reserving a sponsored seat. The notebook shows the run **paused**,
returning the approval payload (`confirm_enrollment` for Saif into ML-201), and then `Command(resume="approve")`
**completing** the run - enrollment confirmed and written to long-term memory (`recall("saif",
"enrolled_course") == "ML-201"`). Both the interrupt and the resume execute, with output.

### §6 - LangGraph Functional API & error handling
Every workflow is built with `@task` / `@entrypoint` (no `StateGraph`). Two of the four error strategies are
implemented: (1) **transient retry** - a real `RetryPolicy(max_attempts=3)` on `fetch_seat_snapshot`, which
fails twice and recovers on the third attempt; (2) **LLM-recoverable** - `classify_message` catches
`ValidationError`, feeds the error back into the prompt, and retries before falling back to a safe default.

### §7 - Workflow pattern: Evaluator-Optimizer
We use the **Evaluator-Optimizer** pattern for the Capstone Mentor and name it explicitly. `draft_feedback`
produces a brief first pass, `judge_feedback` scores it against the rubric with structured output, and when the
verdict is `revise`, `improve_feedback` rewrites it. In the captured run the evaluator returned **revise** ("lacks
specific, actionable fixes for each of the 8 sections") and the optimizer produced an expanded, section-by-section
critique - the generate → judge → refine loop is visible in the output.

### §8 - LangSmith observability
Tracing is wired to the correct variable **`LANGCHAIN_TRACING_V2`** and activates automatically only when a
`LANGCHAIN_API_KEY` is present, so it never fails silently. The notebook was run **with tracing ON**, sending
traces to the `daico-navigator` project. What the trace showed: each `daico_navigator` run decomposes into
`bump_and_get_count` → the supervisor `classify_message` call → the selected worker. The supervisor
classification is a small, fast call (~170-380 prompt / ~20-40 completion tokens, under ~1s), whereas the
**Course Advisor path is the clear hotspot** - its tool-calling loop issues several ChatOpenAI calls, pushing a
single root run to roughly 1,000 tokens and several seconds, while FAQ/Capstone answers cost only a classify
plus one grounded call (~150 tokens, ~1s). The trace made the cost of the multi-step tool loop obvious at a
glance, and is the first thing we would optimise (e.g. capping tool rounds or caching catalog lookups).
