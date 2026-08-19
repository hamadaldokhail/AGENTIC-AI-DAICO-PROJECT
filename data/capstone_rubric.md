# AGT-401 Capstone - Grading Rubric

The capstone is graded on 8 sections, 100 points total. Passing requires 60/100,
and no single section may score below 40% of its points.

## 1. Agent fundamentals - 15 points
Full marks: real tool calls the model chooses to make, plus structured output
(`with_structured_output` + a Pydantic model) wherever a result is parsed by code.
Commonly lost on: returning a hardcoded string from a "tool" that ignores its arguments.

## 2. Multi-agent / routing architecture - 15 points
Full marks: the routing decision is made by the LLM - a supervisor that classifies with
structured output, or a real handoff - matching one of the four track patterns.
Commonly lost on: keyword matching such as `if "email" in question.lower()`. That is not routing.

## 3. RAG pipeline - 15 points
Full marks: documents genuinely loaded, split, embedded, stored, and retrieved, plus a
written justification of 2-Step vs. Agentic vs. Hybrid RAG for the problem.
Commonly lost on: a retriever that returns nothing; ask a question whose answer is verbatim
in the documents and confirm it comes back.

## 4. Context & state management - 15 points
Full marks: a checkpointer with a thread_id for short-term state, and a separate Store for
long-term facts, with a cross-thread test proving a fact written in one thread is readable in another.
Commonly lost on: calling a growing list of chat messages "long-term memory."

## 5. Human-in-the-loop - 10 points
Full marks: a real `interrupt()` that pauses before something irreversible, and a
`Command(resume=...)` that completes the run - both demonstrated with output.
Commonly lost on: pausing but never resuming.

## 6. LangGraph Functional API & error handling - 15 points
Full marks: built with `@task` / `@entrypoint`, plus at least two of the four error strategies.
Commonly lost on: building on StateGraph instead of the Functional API, or a hand-written
retry loop instead of a real RetryPolicy object.

## 7. Workflow pattern - 10 points
Full marks: implement one of Prompt Chaining, Parallelization, Routing, Orchestrator-Worker,
or Evaluator-Optimizer - and name it explicitly with a sentence on why it fits.
Commonly lost on: implementing one correctly but never naming it.

## 8. LangSmith observability - 5 points
Full marks: tracing genuinely on, plus a short write-up of something the trace actually showed.
Commonly lost on: using the wrong variable name; it is `LANGCHAIN_TRACING_V2`.

## GitHub & documentation requirements
A professional README, proper technical documentation, good incremental Git commits, a
`.gitignore` that excludes secrets, a statement of the programme name and cohort dates, and
every trainee's full name in the README or notebook header.
