# Multi-Agent Supervisor System

A sophisticated multi-agent system built with LangGraph and FastAPI that implements a supervisor pattern for intelligent task delegation and execution. The system uses specialized agents (Researcher, Wikipedia, Analyst) to handle different types of queries through adaptive routing.

## Project Structure

### Core Files

#### `main.py`
**Purpose**: FastAPI server implementation that provides the REST API endpoint for the multi-agent system.

**Key Features**:
- Defines the `/v1/execute` POST endpoint that accepts queries and returns responses
- Handles request/response models using Pydantic
- Manages trace persistence to the `logs/` directory
- Integrates with the LangGraph system through the `get_graph()` function
- Supports thread-based conversations for state persistence across requests

**API Endpoint**:
- `POST /v1/execute` - Execute a query through the multi-agent system
  - Request: `{"query": "your question", "thread_id": "optional"}`
  - Response: `{"thread_id": "id", "final_answer": "answer", "token_usage_total": 123, "trace": [...], "persisted_trace_path": "path"}`

#### `graph.py`
**Purpose**: Core LangGraph implementation defining the multi-agent supervisor pattern and worker agents.

**Key Components**:
- **Supervisor Agent**: Intelligent router that decides which worker agent to call next based on the query and execution history
- **Worker Agents**:
  - **Researcher**: Uses DuckDuckGo search for web research and factual information
  - **Wikipedia Agent**: Queries Wikipedia API for encyclopedia-style information
  - **Analyst Agent**: Performs actual analysis, calculations, and evaluation of information (not just summarization)
- **State Management**: Uses TypedDict for shared state across agents
- **Checkpointing**: SQLite-based persistence for conversation continuity
- **Tool Integration**: Custom tools for each agent (search, Wikipedia lookup, Python execution)

**Architecture**:
- Implements a state graph with conditional routing
- Uses Groq LLM for agent decision-making
- Maintains execution traces for debugging and evaluation
- Includes loop prevention and iteration limits

#### `client.py`
**Purpose**: Command-line client for interacting with the FastAPI server.

**Features**:
- Interactive CLI for sending queries to the running server
- Thread management for persistent conversations
- Response formatting with trace summaries
- Error handling for network issues
- Simple usage: `python client.py` (requires server running on localhost:8000)

### Configuration & Dependencies

#### `requirements.txt`
**Purpose**: Python package dependencies for the project.

**Key Dependencies**:
- `fastapi` & `uvicorn` - Web framework and ASGI server
- `langgraph` & `langchain-core` - Graph-based agent orchestration
- `langchain-groq` - Groq LLM integration
- `duckduckgo-search` - Web search functionality
- `langgraph-checkpoint-sqlite` - SQLite checkpointing for LangGraph
- `python-dotenv` & `pydantic` - Configuration and data validation

#### `.env`
**Purpose**: Environment variables configuration file.

**Contents**:
- `GROQ_API_KEY` - API key for Groq language model service (required for LLM functionality)

### Data & Persistence

#### `data/` Directory
**Purpose**: SQLite database storage for LangGraph checkpoints.

**Files**:
- `checkpoints.sqlite3` - Main SQLite database file containing graph state checkpoints
- `checkpoints.sqlite3-shm` - SQLite shared memory file for concurrent access
- `checkpoints.sqlite3-wal` - SQLite write-ahead log for atomic transactions

**Function**: Enables conversation persistence across application restarts. Each `thread_id` maintains its own checkpoint, allowing users to resume conversations.

#### `logs/` Directory
**Purpose**: Execution trace storage for debugging and evaluation.

**Files**: JSON files containing detailed execution traces for each API call, including:
- Agent routing decisions
- Tool outputs
- Token usage
- Step-by-step execution flow

**Naming Convention**:
- `trace_{thread_id}.json` - Traces from API calls with specific thread IDs
- `trace_eval{1-5}.json` - Evaluation traces from stress testing
- `trace_api-{test/smoke}-{number}.json` - API testing traces

### Evaluation & Documentation

#### `EVAL.md`
**Purpose**: Comprehensive evaluation documentation and system analysis.

**Contents**:
- Setup summary and architecture overview
- Stress test results (5 evaluation queries)
- Agent persona descriptions and roles
- Routing logic analysis and optimization recommendations
- Performance observations and improvement suggestions
- API evaluability evidence

**Key Insights**:
- Documents the supervisor pattern implementation
- Analyzes routing behavior under various conditions
- Provides optimization recommendations for production use
- Includes evidence of system robustness and adaptability

### Development Environment

#### `env/` Directory
**Purpose**: Python virtual environment containing all project dependencies.

**Structure**:
- Standard Python virtual environment layout
- Contains all packages from `requirements.txt`
- Isolated environment to prevent dependency conflicts

#### `__pycache__/` Directory
**Purpose**: Python bytecode cache files.

**Contents**: Compiled Python bytecode (.pyc files) for faster module loading.

## System Architecture

### Multi-Agent Supervisor Pattern
1. **User Query** → FastAPI endpoint
2. **Supervisor Agent** analyzes query and decides next agent
3. **Worker Agents** execute specialized tasks (research, Wikipedia lookup, calculations)
4. **Supervisor** reviews results and routes to next agent or finalizes
5. **Response** returned with final answer and execution trace

### Key Features
- **Adaptive Routing**: Supervisor dynamically chooses agents based on query needs
- **State Persistence**: SQLite checkpoints maintain conversation context
- **Tool Integration**: Specialized tools for different information sources
- **Trace Logging**: Complete execution history for debugging and evaluation
- **Error Handling**: Robust error handling for tool failures and network issues
- **Token Management**: Tracks LLM token usage across the execution pipeline

## Usage

### Running the Server
```bash
# Install dependencies
pip install -r requirements.txt

# Set your Groq API key in .env
# GROQ_API_KEY=your_key_here

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Using the Client
```bash
# In another terminal
python client.py
```

### API Usage
```python
import requests

response = requests.post("http://localhost:8000/v1/execute",
    json={"query": "What is the population ratio between Lagos and Nairobi?"})
print(response.json())
```

## Development Notes

- The system uses Groq for LLM capabilities - ensure API key is configured
- SQLite checkpointing provides persistence but may have concurrency limitations
- Wikipedia API occasionally returns 403 errors - system gracefully falls back to other sources
- Python evaluation tool has safety restrictions to prevent system access
- All traces are saved to `logs/` for post-execution analysis</content>
<parameter name="filePath">c:\Users\Wajid\Desktop\intership\README.md