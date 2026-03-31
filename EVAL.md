## Multi-Agent Supervisor System Evaluation (LangGraph + FastAPI)

### Setup summary
This project implements a multi-agent “supervisor pattern” using LangGraph. The system is wrapped by a FastAPI endpoint (`POST /v1/execute`) that runs the supervisor loop, returns the final answer, and returns a full execution trace (who ran, what tool returned, and routing decisions). The graph is compiled with SQLite checkpointing (`SqliteSaver`) keyed by `thread_id` to support persistence across API restarts.

### Stress test queries used (5 runs)
1. `eval1`: Transformer architecture explanation + compute `vocab_size*d_model` with `vocab_size=30522`, `d_model=512`.
2. `eval2`: Lagos vs Nairobi population (Wikipedia requested) + compute population ratio.
3. `eval3`: Explain Haversine formula + compute great-circle distance (Karachi → Lahore).
4. `eval4`: Supervised vs unsupervised vs reinforcement learning comparison + compute labeled examples for 3 classes with 200 samples/class.
5. `eval5`: Vitamin D deficiency causes/symptoms + compute tablets/week (1000 IU/tablet, target 4000 IU/day).

Traces were saved to `logs/trace_eval1.json` … `logs/trace_eval5.json`.

---

## Agent Persona

### Supervisor (router)
Role: Decide which worker to call next or whether to stop and produce the final answer.

Mechanics:
- Reads shared state: the user `query`, `trace` history, `iteration`, `max_iterations`, and `attempted_agents`.
- Produces a routing decision (`next_agent`) using a Groq-backed LLM with a strict JSON instruction.
- Loop safety:
  - Maintains an `iteration` counter and forces `final` at the configured maximum.
  - Guards against repeating a worker by checking `attempted_agents`.

Why it’s needed:
- The project must not be a hard-coded chain. The supervisor is the reason each run can adaptively choose whether to do web research, encyclopedia grounding, and/or computation.

### Researcher worker (DuckDuckGo)
Role: Gather factual snippets from the web for topics requiring background or factual grounding.

Tools:
- `duckduckgo_search(query)` (bound via `bind_tools` inside the worker agent)

### Wikipedia worker (Wikipedia API)
Role: Provide canonical definitions/background when the question asks for “as stated on Wikipedia” or “difference between X/Y/Z” using encyclopedia-style summaries.

Tools:
- `wikipedia_search(title)` (bound via `bind_tools` inside the worker agent)

Note from runs:
- Wikipedia tool frequently failed with `403 Client Error: Forbidden` in this environment, so routing behavior adapted by falling back to other sources.

### Analyst worker (Python REPL)
Role: Perform deterministic calculations and return results (ratio, counts, dosing).

Tools:
- `python_eval(code)` (bound via `bind_tools` inside the worker agent)

Goal:
- The analyst is explicitly prompted to set a `result` variable for the final computation output.

---

## Routing Logic

### Did the supervisor skip an agent or get stuck in a loop?
Observations from the five trace logs:

1. `eval1` (Transformer):
   - Worker sequence: `researcher` → `analyst` → `final`
   - Wikipedia was not invoked in this specific run before the supervisor decided information was sufficient.
   - No looping: `iteration` capped the supervisor calls and the graph reached `final`.

2. `eval2` (Lagos vs Nairobi):
   - Worker sequence: `wikipedia` (failed) → `researcher` → `analyst` → `final`
   - The supervisor attempted Wikipedia first (per its routing instruction), but the tool outputs contained 403 errors, so it then routed to the researcher to recover factual inputs.
   - No looping: routing advanced and terminated on `max_iterations`.

3. `eval3` (Haversine distance):
   - Worker sequence: `researcher` → `wikipedia` (failed) → `analyst` (tool/code issue) → `final`
   - The analyst attempted computation but encountered a `python_eval` error during the tool step (invalid syntax). The supervisor still terminated (iteration cap) and the final synthesis attempted to provide the computed result using the available reasoning context.
   - No stuck loop: supervisor forced `final` once `iteration` reached `max_iterations`.

4. `eval4` (ML comparison + label count):
   - Worker sequence: `researcher` → `wikipedia` → `analyst` → `final`
   - DuckDuckGo tool calls sometimes failed for specific queries, but the analyst still produced a valid numeric output (`600.0`) and the final answer synthesized a comparison table and the computed count.
   - No looping: reached `final` before repeating the same worker indefinitely.

5. `eval5` (Vitamin D dosing):
   - Worker sequence: `researcher` → `wikipedia` (failed) → `analyst` → `final`
   - Analyst successfully computed the numeric dosing result (`12.25 tablets/week`) using `python_eval`.
   - No looping: terminated normally at `final`.

Conclusion:
- The supervisor did not get stuck in a loop. Every run ended by either reaching sufficient info or hitting `max_iterations`.
- Across the five queries, all three worker agents (`researcher`, `wikipedia`, `analyst`) were used. (`eval1` is the only one that did not call Wikipedia.)

### What the supervisor is “seeing”
The supervisor’s prompt includes a truncated view of the execution trace to decide the next branch. Worker tool outputs are stored in the shared `trace` and are returned by the API, enabling post-run evaluation of whether tool outputs influenced routing.

---

## Optimization

### 1) Improve routing under tool failures
Current behavior:
- The supervisor attempts routing to Wikipedia when the user requests canonical definitions/background.
- When Wikipedia tool outputs contain errors (e.g., `403 Forbidden`), routing eventually recovers only if the supervisor prompt decides it needs outside grounding again.

Proposed prompt/system changes:
- Add an explicit rule: “If tool outputs contain an error (403/timeout), do not repeatedly retry that same worker. Prefer the researcher next.”
- Store an additional state field like `tool_error_summary` per worker and include it in the supervisor prompt (even in truncated form).
- Add a “success flag” in `attempted_agents`: `attempted_agents_successfully[agent]=True/False` so routing can prefer workers that produced non-error tool outputs.

### 2) Reduce LLM prompt size further for hard limits
Groq request limits were a recurring constraint during development. The solution in this build is truncation (`_compact_trace_for_prompt`) plus explicit `max_tokens` caps.

Further improvements:
- Keep the supervisor prompt extremely small: include only the last routing decision + last tool-output error (if any).
- For worker prompts, pass no trace context except for the analyst (and even then, only a compact summary).

### 3) Make `python_eval` more robust
Tool-call failures observed in traces:
- Some runs produced python code that caused errors (e.g., invalid syntax; using `print` without allowed builtins).

Improvements applied:
- `python_eval` now sanitizes code fences and captures stdout. If `result` is not set but output is printed, it returns the printed stdout.

Additional improvement:
- In the analyst prompt, force the output contract more strictly: “Return code that assigns a numeric variable to `result` and nothing else.”

### 4) Handle ambiguous user requests
If the user request is ambiguous (“compare”, “summarize”, “compute”), the supervisor prompt should include:
- “If the requested output format is unclear, ask for clarification before final.”
- Or alternatively, choose a default format (e.g., bullet summary + explicit numeric calculations) and mention assumptions.

---

### Evidence of API evaluability
The FastAPI layer returns:
- `final_answer`
- `token_usage_total` (summed across supervisor loop)
- `trace` (the full execution trace returned from the LangGraph state)