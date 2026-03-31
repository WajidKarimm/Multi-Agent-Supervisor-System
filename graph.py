import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

# Groq LLM provider (LangChain integration)
from langchain_groq import ChatGroq


load_dotenv()


RouterNextAgent = Literal["researcher", "wikipedia", "analyst", "final"]


class StepTrace(TypedDict):
    step_id: str
    agent: str
    input: Dict[str, Any]
    routing_decision: Optional[Dict[str, Any]]
    tool_outputs: List[Dict[str, Any]]
    llm_token_total: int
    summary: str


class SupervisorState(TypedDict, total=False):
    thread_id: str
    query: str
    trace: List[StepTrace]
    iteration: int
    max_iterations: int
    token_usage_total: int
    final_answer: str
    # Internal: used to guard against routing loops.
    attempted_agents: List[str]
    route_log: List[Dict[str, Any]]


def _extract_token_total_from_messages(messages: List[BaseMessage]) -> int:
    total = 0
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        rm = getattr(m, "response_metadata", None) or {}
        tu = rm.get("token_usage") or {}
        # Groq usually provides total_tokens.
        val = tu.get("total_tokens") or tu.get("total") or tu.get("completion_tokens")
        try:
            if val is not None:
                total += int(val)
        except Exception:
            pass
    return total


def _estimate_tokens(text: str) -> int:
    # Reasonable heuristic fallback if token_usage is missing.
    return max(1, int(len(text) / 4))


def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON parser. Supervisor prompts instruct "ONLY JSON", but models can be messy.
    """
    if not s:
        return None
    s = s.strip()
    # If the model wrapped JSON in text, try to extract the first {...} block.
    match = re.search(r"\{[\s\S]*\}", s)
    if match:
        s = match.group(0)
    try:
        return json.loads(s)
    except Exception:
        return None


@tool
def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return short snippets as JSON."""
    # Late import so the tool still loads even if deps are missing at import time.
    try:
        from ddgs import DDGS  # newer package name
    except Exception as e:
        try:
            from duckduckgo_search import DDGS  # backward-compatible fallback
        except Exception as e2:
            return json.dumps({"error": f"duckduckgo_search dependency missing: {e2}"})

    results: List[Dict[str, Any]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": r.get("title"),
                        "url": r.get("href"),
                        "snippet": r.get("body"),
                    }
                )
    except Exception as e:
        return json.dumps({"error": f"duckduckgo_search failed: {e}"})
    return json.dumps({"query": query, "results": results}, ensure_ascii=False)


@tool
def wikipedia_search(title: str, sentences: int = 3) -> str:
    """Fetch a short summary from Wikipedia given a page title."""
    import requests

    try:
        url_title = title.strip().replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{url_title}?redirect=true"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        summary = data.get("extract", "") or ""
        # Light truncation to keep tool output small.
        summary = " ".join(summary.split()[: max(20, sentences * 40)])
        return json.dumps(
            {
                "title": data.get("title") or title,
                "url": data.get("content_urls", {}).get("desktop", {}).get("page") or "",
                "summary": summary,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": f"wikipedia_search failed: {e}", "title": title}, ensure_ascii=False)


@tool
def python_eval(code: str) -> str:
    """
    Execute a small Python snippet and return output.
    The analyst should set a variable named `result`.
    """
    import contextlib
    import io
    import math

    def _sanitize_python_code(raw: str) -> str:
        c = (raw or "").strip()
        if not c:
            return ""
        # If the model returned a code fence, extract just the fenced content.
        m = re.search(r"```(?:python)?\s*([\s\S]*?)```", c, flags=re.IGNORECASE)
        if m:
            c = (m.group(1) or "").strip()
        return c

    # Keep the environment deliberately small.
    allowed_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "round": round,
        "int": int,
        "float": float,
        "str": str,
        "len": len,
        "range": range,
        # Allow print in case the analyst uses it; we capture stdout below.
        "print": print,
    }
    safe_globals = {"__builtins__": allowed_builtins, "math": math}
    local_vars: Dict[str, Any] = {}
    try:
        code = _sanitize_python_code(code)
        if not code:
            return "python_eval error: empty code"

        stdout_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, safe_globals, local_vars)  # noqa: S102 (intentional sandbox)
        if "result" in local_vars:
            return str(local_vars["result"])
        stdout_val = stdout_buf.getvalue().strip()
        if stdout_val:
            return stdout_val
        return json.dumps(local_vars, ensure_ascii=False)
    except Exception as e:
        return f"python_eval error: {e}"


def _get_router_llm() -> ChatGroq:
    model = os.getenv("GROQ_ROUTER_MODEL", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    max_tokens = int(os.getenv("GROQ_MAX_TOKENS_ROUTER", "512"))
    return ChatGroq(model=model, temperature=0, max_tokens=max_tokens)


def _get_worker_llm() -> ChatGroq:
    model = os.getenv("GROQ_WORKER_MODEL", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    max_tokens = int(os.getenv("GROQ_MAX_TOKENS_WORKER", "1024"))
    return ChatGroq(model=model, temperature=0, max_tokens=max_tokens)


def _tool_outputs_from_messages(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for m in messages:
        # Tool execution results typically appear as ToolMessage content.
        tool_name = getattr(m, "name", None)
        if tool_name:
            outputs.append({"tool": tool_name, "output": getattr(m, "content", "")})
        else:
            # Some LangChain versions attach tool name inside additional_kwargs.
            ak = getattr(m, "additional_kwargs", None) or {}
            if "tool_call_id" in ak:
                outputs.append({"tool": "tool", "output": getattr(m, "content", "")})
    return outputs


def _summarize_last_ai(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return str(getattr(m, "content", "") or "")
    return ""


def _compact_trace_for_prompt(trace_entries: List[StepTrace], max_steps: int = 3, max_tool_outputs: int = 1) -> List[Dict[str, Any]]:
    """
    Create a small prompt-friendly view of the trace.
    We still keep the full trace in the returned state, but we truncate tool outputs here
    to avoid Groq "Request too large" errors.
    """
    compact: List[Dict[str, Any]] = []
    for e in trace_entries[-max_steps:]:
        tool_outputs = (e.get("tool_outputs") or [])[:max_tool_outputs]
        compact_tool_outputs: List[Dict[str, Any]] = []
        for t in tool_outputs:
            out = t.get("output", "")
            out = str(out)[:250]  # hard truncate to keep prompts small
            compact_tool_outputs.append({"tool": t.get("tool", "tool"), "output": out})

        compact.append(
            {
                "agent": e.get("agent"),
                "summary": str(e.get("summary", ""))[:320],
                "tool_outputs": compact_tool_outputs,
            }
        )
    return compact


def _router_prompt(available_next: List[str], max_iterations: int) -> str:
    return f"""
You are the Supervisor of a multi-agent system.
You must decide which agent to call next: one of {available_next}.

Rules:
1) You MUST eventually choose "final".
2) If the user request needs outside factual grounding, call "researcher" first.
3) If the request needs canonical definitions or entity background, call "wikipedia".
4) If the request needs computation/analysis, call "analyst".
5) Never call the same worker repeatedly. Each worker should be called at most once.
6) When information is sufficient to answer, choose "final".
7) You are not the worker: you only route.

Return ONLY strict JSON:
{{"next_agent": "researcher|wikipedia|analyst|final", "reason": string}}

Also consider: max_iterations={max_iterations}. If close to the limit, choose "final".
""".strip()


def _supervisor_router_decide(
    llm: ChatGroq,
    state: SupervisorState,
    available_next: List[RouterNextAgent],
) -> Dict[str, Any]:
    trace_compact = state.get("trace", [])[-6:]
    trace_text = json.dumps(trace_compact, ensure_ascii=False)
    prompt = _router_prompt([str(x) for x in available_next], state.get("max_iterations", 6))
    user_text = f"""
User query:
{state.get("query","")}

Already attempted agents:
{state.get("attempted_agents",[])}

Recent trace:
{trace_text}
""".strip()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_text},
    ]
    resp = llm.invoke(messages)  # AIMessage
    content = getattr(resp, "content", "") or ""
    parsed = _safe_json_loads(content)
    if parsed and "next_agent" in parsed:
        return parsed
    # Fallback heuristic to keep graph progressing.
    attempted = set(state.get("attempted_agents", []))
    for cand in ["researcher", "wikipedia", "analyst", "final"]:
        if cand == "final":
            if state.get("iteration", 0) >= state.get("max_iterations", 6) - 1:
                return {"next_agent": "final", "reason": "iteration limit reached (fallback)"}
            continue
        if cand not in attempted:
            return {"next_agent": cand, "reason": "router JSON parse failed; fallback to first missing agent"}
    return {"next_agent": "final", "reason": "fallback final"}


def _build_worker_agent(tools: List[Any], system_prompt: str):
    llm = _get_worker_llm()
    llm = llm.bind_tools(tools)
    agent = create_react_agent(
        llm,
        tools=tools,
        prompt=system_prompt,
        debug=False,
    )
    return agent


def build_graph(checkpoint_path: str) -> Any:
    """
    Build and return a compiled supervisor graph with SQLite checkpointing.
    """
    conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    researcher_tools = [duckduckgo_search]
    wikipedia_tools = [wikipedia_search]
    analyst_tools = [python_eval]

    researcher_agent = _build_worker_agent(
        researcher_tools,
        system_prompt=(
            "You are a web research agent. Use duckduckgo_search to gather relevant sources. "
            "Return a compact factual summary and include the most relevant snippets."
        ),
    )
    wikipedia_agent = _build_worker_agent(
        wikipedia_tools,
        system_prompt=(
            "You are a Wikipedia research agent. "
            "Extract the key entity or concept from the user query as a short 1-3 word title "
            "(e.g. 'Haversine formula', 'Data center', 'Elon Musk', 'Transformer deep learning'). "
            "Call wikipedia_search EXACTLY ONCE with that clean short title. "
            "NEVER call wikipedia_search more than once. "
            "NEVER pass the full user question as the title. "
            "After receiving the result, immediately return a concise factual summary. Stop."
        ),
    )
    analyst_agent = _build_worker_agent(
        analyst_tools,
        system_prompt=(
            "You are an analyst agent. Use python_eval for computations. "
            "When calling python_eval, keep the code VERY simple (numbers + basic math only). "
            "Do NOT use f-strings or curly braces. "
            "Set a variable named `result` to the final output, then explain briefly."
        ),
    )

    def supervisor_node(state: SupervisorState) -> SupervisorState:
        router_llm = _get_router_llm()
        state_iteration = int(state.get("iteration", 0))
        max_iterations = int(state.get("max_iterations", 6))
        thread_id = state.get("thread_id") or "unknown"

        attempted_agents = state.get("attempted_agents", [])
        available: List[RouterNextAgent] = ["researcher", "wikipedia", "analyst", "final"]

        # If we hit the iteration cap, force final.
        if state_iteration >= max_iterations:
            decision = {"next_agent": "final", "reason": "iteration cap reached"}
        else:
            # We need both decision + token usage; we re-run invoke here for tokens
            # (kept separate from _supervisor_router_decide to avoid changing its signature
            # in multiple places).
            trace_compact = _compact_trace_for_prompt(state.get("trace", []) or [], max_steps=3, max_tool_outputs=1)
            trace_text = json.dumps(trace_compact, ensure_ascii=False)
            prompt = _router_prompt([str(x) for x in available], state.get("max_iterations", 6))
            user_text = f"""
User query:
{state.get("query","")}

Already attempted agents:
{state.get("attempted_agents",[])}

Recent trace:
{trace_text}
""".strip()
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text},
            ]
            resp = router_llm.invoke(messages)
            content = getattr(resp, "content", "") or ""
            parsed = _safe_json_loads(content) or {}
            decision = parsed if "next_agent" in parsed else _supervisor_router_decide(router_llm, state, available)

            llm_tokens = _extract_token_total_from_messages([resp])
            if llm_tokens <= 0:
                llm_tokens = _estimate_tokens(state.get("query", ""))

        # Guard runs BEFORE step creation so trace always has the corrected decision
        next_agent = decision.get("next_agent", "final")
        next_agent = next_agent if next_agent in ["researcher", "wikipedia", "analyst", "final"] else "final"
        if next_agent in attempted_agents and next_agent != "final":
            for cand in ["researcher", "wikipedia", "analyst"]:
                if cand not in attempted_agents:
                    next_agent = cand
                    decision = {"next_agent": cand, "reason": "guard: avoid repeated worker"}
                    break
            else:
                next_agent = "final"
                decision = {"next_agent": "final", "reason": "guard: no workers left"}

        step_id = str(uuid.uuid4())
        step: StepTrace = {
            "step_id": step_id,
            "agent": "supervisor",
            "input": {"query": state.get("query", "")},
            "routing_decision": decision,
            "tool_outputs": [],
            "llm_token_total": llm_tokens if state_iteration < max_iterations else _estimate_tokens(state.get("query", "")),
            "summary": f"Routing decision: {decision.get('next_agent')}",
        }

        # Track routing decisions in state.
        route_log = state.get("route_log", [])
        route_log = route_log + [{"step_id": step_id, "decision": decision}]

        return {
            **state,
            "trace": state.get("trace", []) + [step],
            "route_log": route_log,
            "token_usage_total": int(state.get("token_usage_total", 0))
            + (step["llm_token_total"] if state_iteration < max_iterations else 0),
        }

    def route_fn(state: SupervisorState) -> str:
        # Read the latest supervisor routing decision.
        trace = state.get("trace", [])
        if not trace:
            return "final_node"
        last = trace[-1]
        if last.get("agent") != "supervisor":
            return "final_node"
        decision = last.get("routing_decision") or {}
        nxt = decision.get("next_agent", "final")
        if nxt == "researcher":
            return "researcher_node"
        if nxt == "wikipedia":
            return "wikipedia_node"
        if nxt == "analyst":
            return "analyst_node"
        return "final_node"

    def worker_node(agent_name: str, agent: Any, tools: List[Any]):
        def _fn(state: SupervisorState) -> SupervisorState:
            # Invoke worker agent over the user query with recent trace context.
            query = state.get("query", "")
            recent_trace = state.get("trace", []) or []
            # To reduce prompt size (Groq has a strict per-request token cap), only
            # provide trace context to the analyst. Researchers can work from the user query alone.
            if agent_name == "analyst":
                context = json.dumps(_compact_trace_for_prompt(recent_trace, max_steps=3, max_tool_outputs=1), ensure_ascii=False)
            else:
                context = "[]"
            input_messages = [
                (
                    "user",
                    f"User query:\n{query}\n\nRecent supervisor trace (may include tool outputs):\n{context}\n\n"
                    f"As {agent_name}, use your tools to produce the best next contribution.",
                )
            ]
            try:
                result = agent.invoke({"messages": input_messages})
                messages = result.get("messages", [])
            except Exception as e:
                # Groq can intermittently fail tool calling (tool_use_failed).
                # If that happens for the analyst, fall back to running python_eval directly
                # using a tiny snippet drafted by the LLM (no tool calling).
                if agent_name == "analyst":
                    llm = _get_worker_llm()
                    compact = json.dumps(
                        _compact_trace_for_prompt(recent_trace, max_steps=2, max_tool_outputs=1),
                        ensure_ascii=False,
                    )
                    fallback_system = (
                        "Return ONLY python code.\n"
                        "Keep it extremely short.\n"
                        "No import statements — math functions (sin, cos, asin, sqrt, radians, pi) are already available.\n"
                        "No f-strings, no curly braces.\n"
                        "Use math.sin, math.cos, math.asin, math.sqrt, math.radians, math.pi directly.\n"
                        "Always set result = <final numeric value>.\n"
                        "If inputs are missing, set result = 'missing_inputs'."
                    )
                    resp = llm.invoke(
                        [
                            {"role": "system", "content": fallback_system},
                            {"role": "user", "content": f"Query:\n{query}\n\nContext:\n{compact}"},
                        ]
                    )
                    code = (getattr(resp, "content", "") or "").strip()
                    out = python_eval.invoke({"code": code})

                    attempted = state.get("attempted_agents", [])
                    if agent_name not in attempted:
                        attempted = attempted + [agent_name]

                    llm_tokens = _extract_token_total_from_messages([resp])
                    if llm_tokens <= 0:
                        llm_tokens = _estimate_tokens(code)

                    step: StepTrace = {
                        "step_id": str(uuid.uuid4()),
                        "agent": agent_name,
                        "input": {"query": query, "fallback_error": str(e)},
                        "routing_decision": None,
                        "tool_outputs": [{"tool": "python_eval", "output": str(out)}],
                        "llm_token_total": llm_tokens,
                        "summary": f"Analyst fallback ran python_eval. Output: {out}",
                    }
                    iteration = int(state.get("iteration", 0)) + 1
                    return {
                        **state,
                        "trace": state.get("trace", []) + [step],
                        "iteration": iteration,
                        "token_usage_total": int(state.get("token_usage_total", 0)) + llm_tokens,
                        "attempted_agents": attempted,
                    }

                # Non-analyst failure: record an empty messages list.
                messages = []

            tool_outputs = _tool_outputs_from_messages(messages)
            llm_tokens = _extract_token_total_from_messages(messages)
            if llm_tokens <= 0:
                llm_tokens = _estimate_tokens(query)

            summary = _summarize_last_ai(messages)[:4000]

            attempted = state.get("attempted_agents", [])
            if agent_name not in attempted:
                attempted = attempted + [agent_name]

            step_id = str(uuid.uuid4())
            step: StepTrace = {
                "step_id": step_id,
                "agent": agent_name,
                "input": {"query": query},
                "routing_decision": None,
                "tool_outputs": tool_outputs,
                "llm_token_total": llm_tokens,
                "summary": summary,
            }

            iteration = int(state.get("iteration", 0)) + 1
            return {
                **state,
                "trace": state.get("trace", []) + [step],
                "iteration": iteration,
                "token_usage_total": int(state.get("token_usage_total", 0)) + llm_tokens,
                "attempted_agents": attempted,
            }

        return _fn

    def final_node(state: SupervisorState) -> SupervisorState:
        final_llm = _get_worker_llm()
        trace = state.get("trace", [])
        trace_text = json.dumps(_compact_trace_for_prompt(trace, max_steps=4, max_tool_outputs=1), ensure_ascii=False)
        system_prompt = (
            "You are the final answer synthesizer. Use the provided trace (including tool outputs) "
            "to answer the user's query clearly and concisely. "
            "If tool outputs contain data, reference them in your answer. "
            "If information is missing, say so explicitly."
        )
        user_prompt = f"User query:\n{state.get('query','')}\n\nTrace:\n{trace_text}\n\nWrite final answer."
        resp = final_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        content = getattr(resp, "content", "") or ""
        llm_tokens = _extract_token_total_from_messages([resp])
        if llm_tokens <= 0:
            llm_tokens = _estimate_tokens(content)

        step: StepTrace = {
            "step_id": str(uuid.uuid4()),
            "agent": "final",
            "input": {"query": state.get("query", "")},
            "routing_decision": None,
            "tool_outputs": [],
            "llm_token_total": llm_tokens,
            "summary": content[:4000],
        }
        return {
            **state,
            "trace": state.get("trace", []) + [step],
            "final_answer": content,
            "token_usage_total": int(state.get("token_usage_total", 0)) + llm_tokens,
            "iteration": int(state.get("iteration", 0)),
        }

    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor_node", supervisor_node)
    builder.add_node("researcher_node", worker_node("researcher", researcher_agent, [duckduckgo_search]))
    builder.add_node(
        "wikipedia_node", worker_node("wikipedia", wikipedia_agent, [wikipedia_search])
    )
    builder.add_node("analyst_node", worker_node("analyst", analyst_agent, [python_eval]))
    builder.add_node("final_node", final_node)

    builder.add_edge(START, "supervisor_node")
    builder.add_conditional_edges("supervisor_node", route_fn, {
        "researcher_node": "researcher_node",
        "wikipedia_node": "wikipedia_node",
        "analyst_node": "analyst_node",
        "final_node": "final_node",
    })

    # After a worker finishes, go back to supervisor for the next routing decision.
    builder.add_edge("researcher_node", "supervisor_node")
    builder.add_edge("wikipedia_node", "supervisor_node")
    builder.add_edge("analyst_node", "supervisor_node")
    builder.add_edge("final_node", END)

    graph = builder.compile(checkpointer=checkpointer)
    return graph


# A small convenience for the FastAPI app.
def get_graph() -> Any:
    base_dir = os.path.dirname(__file__)
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    checkpoint_path = os.path.join(base_dir, "data", "checkpoints.sqlite3")
    return build_graph(checkpoint_path)