import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from graph import get_graph


load_dotenv()

app = FastAPI(
    title="Multi-Agent Supervisor (LangGraph + FastAPI)",
    version="1.0.0",
)


class ExecuteRequest(BaseModel):
    query: str = Field(..., description="Problem statement to solve using the multi-agent supervisor system.")
    thread_id: Optional[str] = Field(
        None,
        description="Optional thread_id for LangGraph checkpointing. Use the same thread_id to resume across restarts.",
    )


class ExecuteResponse(BaseModel):
    thread_id: str
    final_answer: str
    token_usage_total: int
    trace: List[Dict[str, Any]]
    persisted_trace_path: Optional[str] = None


_graph = get_graph()
_base_dir = os.path.dirname(__file__)
_logs_dir = os.path.join(_base_dir, "logs")
os.makedirs(_logs_dir, exist_ok=True)


def _persist_trace(thread_id: str, trace: List[Dict[str, Any]]) -> str:
    path = os.path.join(_logs_dir, f"trace_{thread_id}.json")
    payload = {
        "thread_id": thread_id,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "trace": trace,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


@app.post("/v1/execute", response_model=ExecuteResponse)
def execute(req: ExecuteRequest) -> ExecuteResponse:
    thread_id = req.thread_id or str(uuid4())

    initial_state: Dict[str, Any] = {
        "thread_id": thread_id,
        "query": req.query,
        "trace": [],
        "iteration": 0,
        "max_iterations": int(os.getenv("SUPERVISOR_MAX_ITERATIONS", "6")),
        "token_usage_total": 0,
        "final_answer": "",
        "attempted_agents": [],
        "route_log": [],
    }

    result = _graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
    )

    final_answer = result.get("final_answer", "") or ""
    token_usage_total = int(result.get("token_usage_total", 0) or 0)
    trace = result.get("trace", []) or []

    trace_path = _persist_trace(thread_id, trace)

    return ExecuteResponse(
        thread_id=thread_id,
        final_answer=final_answer,
        token_usage_total=token_usage_total,
        trace=trace,
        persisted_trace_path=trace_path,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
