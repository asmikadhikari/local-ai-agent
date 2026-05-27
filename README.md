# Local AI Agent

A local-first AI assistant router with a web UI, memory system, code generation, web search, and autonomous browsing — all running on your own machine via Ollama.

Designed for low-VRAM environments (tested on RTX 4050 6GB). The entire stack stays local: no data leaves your computer, no API keys required, no cloud dependencies.

## Features

**Intent Routing** — Automatically detects what you're trying to do (chat, search, code, research, browse) and dispatches to the right handler or tool.

**Web Search** — Queries DuckDuckGo with automatic fallback. Results are summarized by the AI.

**Deep Research** — Three-phase pipeline: plan, draft, critique. Generates a researched answer with citations, self-corrects, and produces a final reviewed response.

**Code Generation** — Generates single files or full dependency-aware multi-file projects. Uses a dedicated code model (DeepSeek Coder) for better output. Includes backup/restore and file undo.

**Autonomous Browser Agent** — Playwright-based browser control. Can navigate, click, type, extract content, log into sites, and complete multi-step tasks on your behalf.

**Conversation History** — SQLite-backed with 1-year retention. Full-text search via FTS5. Optional vector search (sqlite-vec) for semantic recall across all past conversations.

**Persistent Memory** — Key-value store for facts you want the assistant to remember across sessions.

**Web UI** — Flask-based web interface with SSE streaming, conversation management, activity logs, and database viewer.

**Resource Monitoring** — Real-time VRAM/RAM tracking with NVIDIA GPU support. Loads/unloads models as needed to stay within hardware limits.

**Standalone Mode** — If the full router is not available, the web server runs in a lightweight chat-only mode with Ollama.

## Architecture

```
Browser (static/index.html)
    │
    ▼ HTTP/SSE
server.py  ─── Flask web server (port 5001)
    │
    ├── ai_router.py   →  AIAssistant router (intent detection, tool dispatch, memory)
    │       │
    │       ├── code_agent.py   →  CodeAgent (dependency graph, multi-file generation)
    │       ├── web_search.py   →  DuckDuckGo search + fallback scraper
    │       ├── ollama          →  Local LLM inference
    │       └── playwright      →  Browser automation (/task agent)
    │
    └── standalone fallback → ollama chat (no router loaded)
```

### How the Router Works

1. **Intent Detection** — Your message is analyzed to determine intent: `CHAT`, `/search`, `/code`, `/research`, `/agent`, `/news`, etc. If no explicit command is given, the router infers intent from context.

2. **Pipeline Execution** — Each command follows a structured pipeline:
   - **Search**: query → DuckDuckGo → summarize results with AI
   - **Research**: plan → draft → critique → final (three-phase with self-correction)
   - **Code**: instruction → plan → generate → validate → fix loop
   - **Task Agent**: goal → plan → browser steps → extract → done

3. **Tool Calling** — The AI can invoke tools (web search, browser actions, file operations) via structured JSON output. The router parses the JSON, executes the tool, and feeds results back.

4. **Memory** — Both short-term (conversation history, 60-message window) and long-term (persistent key-value memory + optional vector search across all conversations).

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.10+ | [python.org](https://www.python.org/downloads/) |
| Ollama | Latest | [ollama.ai](https://ollama.ai) or `winget install Ollama.Ollama` |
| Node.js | 18+ (optional, for browser tools) | [nodejs.org](https://nodejs.org/) |

### Required Python Packages

```
pip install flask flask-cors ollama psutil ddgs tiktoken
```

Optional but recommended:

```
pip install playwright sqlite-vec cryptography
playwright install chromium      # For /task browser agent
```

### Recommended Ollama Models

```bash
# Chat / reasoning
ollama pull llama3.1:8b-instruct-q4_K_M

# Code generation
ollama pull deepseek-coder:6.7b-instruct-q4_K_M

# Embeddings (for semantic search)
ollama pull nomic-embed-text
```

## Quick Start

```bash
# 1. Activate virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\activate          # Windows
source venv/bin/activate         # Linux / macOS

# 2. Install dependencies
pip install flask flask-cors ollama psutil ddgs tiktoken

# 3. Make sure Ollama is running
ollama serve

# 4. Start the web server
python server.py
```

Open **http://localhost:5001** in your browser.

## Usage

### Web UI

The interface provides a chat window with SSE streaming — responses appear token-by-token. The Activity tab shows all intermediate steps (search results, planning, model loads). Sidebar lists conversation history.

### Commands

Type these in the chat window:

| Command | Description |
|---|---|
| `/search <query>` | Web search with AI summary |
| `/research <topic>` | Deep 3-phase research (plan → draft → critique) |
| `/news [topic]` | Latest news headlines |
| `/code <instruction>` | Generate code (uses DeepSeek Coder) |
| `/agentcode <request>` | Autonomous multi-file project builder |
| `/analyze <files> <question>` | Audit source files |
| `/task <goal>` | Autonomous multi-step browser agent |
| `/agent <goal>` | Same as /task |
| `/memory add <fact>` | Store a fact in persistent memory |
| `/memory recall` | Show all stored memories |
| `/memory clear` | Clear all memories |
| `/models` | List available Ollama models |
| `/undo <file>` | Restore a file from backup |
| `/help` | Show help |

You can also just type naturally — the router detects intent and dispatches automatically.

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/api/status` | GET | Router and Ollama status |
| `/api/health` | GET | Health check |
| `/api/chat` | POST | Send message (SSE streaming response) |
| `/api/conversations` | GET | List conversations |
| `/api/conversations` | POST | Create new conversation |
| `/api/conversations/<id>` | DELETE | Delete conversation |
| `/api/conversations/<id>/rename` | POST | Rename conversation |
| `/api/conversations/<id>/messages` | GET | Get messages |
| `/api/db/overview` | GET | Database statistics |
| `/api/db/memory` | GET | Memory items |
| `/api/db/memory` | DELETE | Clear memory |

## Project Structure

```
├── server.py              # Flask web server with SSE streaming
├── ai_router.py           # Intent router, tool dispatch, memory, pipeline
├── code_agent.py          # Dependency-graph-aware code generator
├── web_search.py          # DuckDuckGo search + lite fallback
├── test_browser.py        # Playwright browser setup utility
├── start.bat              # Windows launcher
├── start.sh               # Unix launcher
├── static/
│   ├── index.html         # Web UI (chat, activity log, conversation sidebar)
│   ├── manifest.json      # PWA manifest
│   └── sw.js              # Service worker
├── models/                # (placeholder for local model files)
└── venv/                  # Virtual environment (local, not tracked)
```

## Configuration

Key settings in `ai_router.py`:

| Variable | Default | Description |
|---|---|---|
| `CHAT_MODEL` | `llama3.1:8b-instruct-q4_K_M` | Model for chat, planning, review |
| `CODE_MODEL` | `deepseek-coder:6.7b-instruct-q4_K_M` | Model for code generation |
| `EMBED_MODEL` | `nomic-embed-text` | Model for semantic embeddings |
| `PORT` (env var) | `5001` | Web server port |
| `RETENTION_DAYS` | `365` | Conversation history retention |
| `MAX_HISTORY_TOKENS` | `6000` | Token limit for conversation context |

## Security

This is a local development tool designed to run on your own machine. It has no authentication, no rate limiting, and allows arbitrary file access through the code agent and task agent. Do not expose it to the internet or untrusted networks without adding authentication, CORS restrictions, and access controls.

## License

MIT
