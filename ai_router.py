
"""
AI Assistant Router — v3 (Structured Pipeline Upgrade)

KEY CHANGES FROM v2:
  ⚙️ 1. HARD JSON output formats for all planning/review phases
  ⚙️ 2. Split thinking into strict phases (Plan → Execute → Review)
  ⚙️ 3. One-step execution with per-step validation
  ⚙️ 4. Self-correction loop (Generate → Validate → Fix → Continue)
  ⚙️ 5. Reduced context size per step (only relevant info)
  ⚙️ 6. Explicit format constraints in ALL prompts
  ⚙️ 7. Deterministic generation (temp 0.2-0.4, top_p 0.9)
  ⚙️ 8. Strict role reinforcement ("system crash" compliance)
  ⚙️ 9. Memory checkpoints after every step
  ⚙️ 10. Generate → Validate → Fix → Continue as core loop

Manages:
  • Conversation history (SQLite, 3-day retention)
  • Persistent memory (/memory) and temp memory (/tempmem)
  • Model loading / unloading via Ollama
  • Command dispatch: /chat /code /search /research /agent
        /cd /refresh /history /new /load
        /memory /tempmem /endtempmem /news
        /agentcode /analyze /undo
  • VRAM / RAM monitoring and display
  • Interactive /code loop with plan preview and edit cycle
  • Stop signal: ==== halts all processing immediately

Requires:
    pip install ollama psutil ddgs tiktoken playwright sqlite-vec
    playwright install chromium  (for /agent browser automation)
"""

import os
import re
import sys
import glob
import json
import time
import shutil
import signal
import logging
import sqlite3
import platform
import subprocess
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import base64

try:
    import readline  # noqa: F401
except ImportError:
    pass

# sqlite-vec: optional vector search — graceful fallback if not installed
try:
    import sqlite_vec
    _SQLITE_VEC_OK = True
except ImportError:
    sqlite_vec = None  # type: ignore
    _SQLITE_VEC_OK = False

import psutil
import ollama
from code_agent import CodeAgent, count_tokens
from web_search import search_duckduckgo, format_results_for_ai



    # ═══════════════════════════════════════════════════════════════
    # Commands
    # ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════
CHAT_MODEL = 'llama3.1:8b-instruct-q4_K_M'
CODE_MODEL = 'deepseek-coder:6.7b-instruct-q4_K_M'
EMBED_MODEL = 'nomic-embed-text'  # 768-dim embeddings for semantic search
EMBED_DIM = 768

DB_PATH = 'conversations.db'
MEMORY_DB = 'memory.db'
LOG_FILE = 'ai_assistant.log'
RETENTION_DAYS = 365  # Keep history for 1 year
MAX_HISTORY_TOKENS = 6000  # token-based history trimming
MAX_EMBED_CHARS = 4000
MAX_TOOL_RESULT_CHARS = 4000
TOOL_RESULT_PREFIX = '[Tool result from '

# ⚙️ CHANGE 7: Deterministic generation defaults
DEFAULT_CHAT_TEMP = 0.3      # was 0.7 — more stable
DEFAULT_CODE_TEMP = 0.15     # was 0.2 — more deterministic
DEFAULT_PLAN_TEMP = 0.2      # strict planning
DEFAULT_REVIEW_TEMP = 0.1    # strict reviewing

STOP_SIGNAL = re.compile(r'={4,}')
SEP = '=' * 60

# ─── Task Agent Constants ────────────────────────────────────
MAX_AGENT_STEPS = 15
AGENT_TIMEOUT_SEC = 300          # 5 minutes
MAX_OBSERVATION_CHARS = 2000     # per observation
MAX_SCRATCHPAD_CHARS = 4000      # total scratchpad sent to LLM
MAX_PAGE_EXTRACT_CHARS = 6000    # raw page extract before compression
CONSECUTIVE_FAIL_LIMIT = 3

def get_chrome_user_data_dir():
    if platform.system() == 'Windows':
        return os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Google', 'Chrome', 'User Data')
    elif platform.system() == 'Darwin':  # macOS
        return os.path.expanduser('~/Library/Application Support/Google/Chrome')
    else:  # Linux
        return os.path.expanduser('~/.config/google-chrome')

CHROME_USER_DATA_DIR = get_chrome_user_data_dir()
BROWSER_PROFILE_DIR = os.path.join(os.path.expanduser('~'), '.ai_assistant_browser')
CREDENTIALS_DB = os.path.join(os.path.expanduser('~'), '.ai_assistant_credentials.db')

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

AGENT_TOOLS = [
    {"name": "browser_open", "input": {"url": "string"}, "desc": "Navigate to a URL. Returns visible page text. Auto-recovers from crashes."},
    {"name": "browser_type", "input": {"selector": "string (CSS)", "text": "string"}, "desc": "Type text into an input field, then press Enter."},
    {"name": "browser_click", "input": {"text": "string (visible button/link text)"}, "desc": "Click an element by its visible text."},
    {"name": "browser_click_selector", "input": {"selector": "string (CSS)"}, "desc": "Click element by CSS selector."},
    {"name": "browser_extract", "input": {"goal": "string (what to find)"}, "desc": "Extract specific info from current page."},
    {"name": "browser_scroll", "input": {}, "desc": "Scroll down to load more content."},
    {"name": "browser_back", "input": {}, "desc": "Go back to previous page."},
    {"name": "browser_wait", "input": {"selector": "string (optional)", "seconds": "number"}, "desc": "Wait for element or fixed duration."},
    {"name": "browser_login", "input": {"domain": "string", "username": "string (optional)", "password": "string (optional)"}, "desc": "Log into a website. Uses saved credentials if available."},
    {"name": "browser_screenshot", "input": {}, "desc": "Take screenshot for debugging."},
    {"name": "web_search", "input": {"query": "string"}, "desc": "Search DuckDuckGo. Returns titles + snippets."},
    {"name": "file_write", "input": {"path": "string", "content": "string"}, "desc": "Write content to a file (overwrites)."},
    {"name": "file_append", "input": {"path": "string", "content": "string"}, "desc": "Append content to a file."},
    {"name": "file_read", "input": {"path": "string"}, "desc": "Read a file's contents."},
    {"name": "store", "input": {"key": "string", "value": "string"}, "desc": "Save a value for later use."},
    {"name": "done", "input": {"answer": "string"}, "desc": "Task complete. Provide the final answer."},
]
# ⚙️ CHANGE 8: Strict role reinforcement in base prompt
BASE_SYSTEM_PROMPT = """\
You are a structured AI agent in a deterministic pipeline system.
Today is {date}.

CRITICAL SYSTEM RULES (violation = system crash):
- You MUST follow the exact output format specified in each prompt
- You MUST NOT add explanations unless the format requires it
- You MUST NOT hallucinate facts — say "unknown" if unsure
- You MUST resolve follow-up references from conversation context
- You MUST return COMPLETE outputs — never partial, never truncated

BEHAVIORAL RULES:
- Think → plan → act → validate → adapt
- Use simple fixes first — avoid unnecessary complexity
- If modifying files: never destroy useful existing data
- Merge old + new intelligently, keep it clean and structured
- If the user asks from a false premise, correct it directly

OUTPUT RULES:
- When JSON is requested: output ONLY valid JSON, no text before/after
- When text is requested: be clear, structured, use sections/bullets
- When code is requested: output ONLY the code, no markdown fences
"""


# ⚙️ CHANGE 8: Strict role-based self-awareness prompt
SELF_AWARENESS = """\
## YOUR IDENTITY

You are a local AI assistant running on a laptop with 6GB VRAM (RTX 4050) and 8GB RAM.
You have two models:
- Chat model: llama3.1:8b-instruct-q4_K_M (reasoning, planning, review)
- Code model: deepseek-coder:6.7b-instruct-q4_K_M (writing code)

## TOOL CALLING FORMAT

You are a strict tool-calling agent. When you need to use a tool, respond with
ONLY a JSON object. No text before or after.

Available tools:
{tools_json}

### How to call a tool
Output EXACTLY this JSON format (nothing else):
{{"tool": "tool_name", "params": {{"key": "value"}}}}

Rules:
- Output MUST be valid JSON
- No explanations before or after the JSON
- No extra text
- Follow the schema exactly
- If you want to just answer (no tool needed), respond with normal text
- After a tool result, always produce a final answer — never ask what to do

## CONSTRAINTS
- Only one model loaded at a time (limited VRAM)
- Only use memory_add for facts the user wants you to remember (e.g., their name, role, or specific project rules)
- LONG-TERM MEMORY: You have access to 365 days of conversation history.
- GLOBAL RECALL: Semantic retrieval (RAG) now searches across ALL past conversations to provide better context.
- Current project root: {project_root}
"""

# ⚙️ CHANGE 1: Structured tool definitions as JSON schema
TOOL_DEFINITIONS = [
    {"name": "web_search", "params": {"query": "string"}, "description": "Search the web"},
    {"name": "news", "params": {"topic": "string (optional)"}, "description": "Get latest news"},
    {"name": "research", "params": {"topic": "string"}, "description": "Deep research with critique"},
    {"name": "browser_fetch", "params": {"url": "string"}, "description": "Fetch URL content"},
    {"name": "code_edit", "params": {"file": "string", "instruction": "string"}, "description": "Modify a file"},
    {"name": "code_analyze", "params": {"files": "list[string]", "question": "string"}, "description": "Audit files"},
    {"name": "agentcode", "params": {"request": "string"}, "description": "Autonomous project builder"},
    {"name": "memory_add", "params": {"fact": "string"}, "description": "Store persistent fact"},
    {"name": "memory_recall", "params": {}, "description": "Retrieve stored memories"},
    {"name": "memory_clear", "params": {}, "description": "Clear all persistent memory"},
    {"name": "tempmem_add", "params": {"rule": "string"}, "description": "Add temporary rule"},
    {"name": "tempmem_clear", "params": {}, "description": "Clear temporary rules"},
    {"name": "list_conversations", "params": {}, "description": "List saved conversations"},
    {"name": "load_conversation", "params": {"id": "string"}, "description": "Load a conversation"},
    {"name": "new_conversation", "params": {}, "description": "Start fresh conversation"},
    {"name": "set_project_root", "params": {"path": "string"}, "description": "Change working directory"},
    {"name": "refresh_graph", "params": {}, "description": "Rebuild dependency graph"},
    {"name": "undo", "params": {"file": "string"}, "description": "Restore file from backup"},
    {"name": "task_agent", "params": {"goal": "string"}, "description": "Run autonomous multi-step agent for complex tasks (web browsing, comparisons, extract+save)"},
]

# ═══════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════
def _setup_logging() -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))

    root.addHandler(fh)
    root.addHandler(ch)
    return logging.getLogger(__name__)

logger = _setup_logging()

# ═══════════════════════════════════════════════════════════════
# Resource monitoring
# ═══════════════════════════════════════════════════════════════

def get_system_resources() -> Dict[str, Any]:
    ram = psutil.virtual_memory()
    info: Dict[str, Any] = {
        'ram_total_gb': ram.total / 1e9,
        'ram_free_gb': ram.available / 1e9,
        'ram_used_pct': ram.percent,
    }
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            used = int(parts[0].strip())
            total = int(parts[1].strip())
            info.update(vram_used_mb=used, vram_total_mb=total,
                        vram_free_mb=total - used)
    except Exception:
        pass
    return info


def warn_resources(needed_gb: float = 4.5):
    info = get_system_resources()
    free_ram = info['ram_free_gb']
    if free_ram < 0.5:
        logger.warning("Low RAM: %.1fGB free.", free_ram)
        print(f"  ⚠️ [WARNING] Low RAM: {free_ram:.1f}GB free.")
    if 'vram_free_mb' in info and info['vram_free_mb'] < needed_gb * 1024:
        free_vram = info['vram_free_mb']
        logger.warning("Low VRAM: %dMB free, need ~%dMB.", free_vram, int(needed_gb * 1024))
        print(f"  ⚠️ [WARNING] Low VRAM: {free_vram}MB free, need ~{int(needed_gb*1024)}MB.")


def render_vram_bar(info: Dict[str, Any]) -> str:
    if 'vram_used_mb' not in info:
        return ''
    used = info['vram_used_mb']
    total = info['vram_total_mb']
    frac = used / total if total else 0
    bar = '█' * int(frac * 20) + '░' * (20 - int(frac * 20))
    return f"[{bar}] {used}MB / {total}MB"


def print_model_box(model_name: str, status: str = 'loaded'):
    info = get_system_resources()
    bar = render_vram_bar(info)
    print(f"  🤖 MODEL: {model_name}")
    if bar:
        print(f"  📊 VRAM: {bar} ({status})")
    print(f"  {'─'*50}")


# ═══════════════════════════════════════════════════════════════
# SQLite Conversation Manager
# ═══════════════════════════════════════════════════════════════

class ConversationManager:

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()
        self._cleanup_old()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as con:
            if _SQLITE_VEC_OK:
                con.enable_load_extension(True)
                sqlite_vec.load(con)

            con.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conv_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (conv_id) REFERENCES conversations(id)
                );
                CREATE TABLE IF NOT EXISTS message_embeddings (
                    message_id INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    model TEXT NOT NULL,
                    created_at TEXT
                );
            """)

            if _SQLITE_VEC_OK:
                try:
                    con.execute(f"""
                        CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages
                        USING vec0(embedding float[{EMBED_DIM}])
                    """)
                except Exception as e:
                    logger.warning("Could not create vec_messages table: %s", e)

    def _cleanup_old(self):
        cutoff = (datetime.now() - timedelta(days=RETENTION_DAYS)).isoformat()
        with sqlite3.connect(self.db_path) as con:
            if _SQLITE_VEC_OK:
                con.enable_load_extension(True)
                sqlite_vec.load(con)

            old = con.execute(
                "SELECT id FROM conversations WHERE updated_at < ?", (cutoff,)
            ).fetchall()
            for (cid,) in old:
                msg_ids = [
                    r[0] for r in con.execute(
                        "SELECT id FROM messages WHERE conv_id=?", (cid,)
                    ).fetchall()
                ]
                for mid in msg_ids:
                    con.execute(
                        "DELETE FROM message_embeddings WHERE message_id=?", (mid,)
                    )
                    if _SQLITE_VEC_OK:
                        try:
                            con.execute(
                                "DELETE FROM vec_messages WHERE rowid=?", (mid,)
                            )
                        except Exception:
                            pass
                con.execute("DELETE FROM messages WHERE conv_id=?", (cid,))
                con.execute("DELETE FROM conversations WHERE id=?", (cid,))
            if old:
                logger.info("Cleaned up %d old conversation(s)", len(old))
            con.commit()

    def new_conversation(self) -> str:
        cid = datetime.now().strftime('%Y%m%d_%H%M%S')
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO conversations VALUES (?,?,?,?)",
                (cid, f"Conversation {cid}", now, now),
            )
        return cid

    def rename_conversation(self, cid: str, name: str):
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "UPDATE conversations SET name=? WHERE id=?", (name, cid)
            )

    def save_message(self, cid: str, role: str, content: str):
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as con:
            if _SQLITE_VEC_OK:
                con.enable_load_extension(True)
                sqlite_vec.load(con)

            cur = con.execute(
                "INSERT INTO messages (conv_id, role, content, timestamp) VALUES (?,?,?,?)",
                (cid, role, content, now),
            )
            msg_id = cur.lastrowid

            con.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?", (now, cid)
            )

            if _SQLITE_VEC_OK:
                self._store_embedding(con, msg_id, content, now)

    def _store_embedding(self, con, msg_id: int, content: str, now: str):
        if not content or not content.strip():
            return
        if len(content) > MAX_EMBED_CHARS:
            content = content[:MAX_EMBED_CHARS]
        try:
            resp = ollama.embed(model=EMBED_MODEL, input=content)
            if not resp.get('embeddings') or len(resp['embeddings']) == 0:
                logger.warning("No embedding returned for msg %s", msg_id)
                return
            emb = resp['embeddings'][0]
        except Exception as e:
            logger.warning("Embedding failed for msg %s: %s", msg_id, e)
            return

        try:
            emb_bytes = sqlite_vec.serialize_float32(emb)
            con.execute(
                "INSERT OR REPLACE INTO message_embeddings "
                "(message_id, embedding, model, created_at) VALUES (?,?,?,?)",
                (msg_id, emb_bytes, EMBED_MODEL, now),
            )
            con.execute(
                "INSERT INTO vec_messages(rowid, embedding) VALUES (?, ?)",
                (msg_id, emb_bytes),
            )
        except Exception as e:
            logger.warning("Vector insert failed for msg %s: %s", msg_id, e)

    def load_messages(self, cid: str, limit: int = 50) -> List[Dict]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT role, content FROM messages "
                "WHERE conv_id=? ORDER BY id DESC LIMIT ?",
                (cid, limit),
            ).fetchall()
        return [{'role': r, 'content': c} for r, c in reversed(rows)]

    def list_conversations(self) -> List[Tuple[str, str, str]]:
        with sqlite3.connect(self.db_path) as con:
            return con.execute(
                "SELECT id, name, updated_at FROM conversations ORDER BY updated_at DESC"
            ).fetchall()


# ═══════════════════════════════════════════════════════════════
# Memory Manager (persistent + temporary)
# ═══════════════════════════════════════════════════════════════

class MemoryManager:

    _EXTRACT_PATTERNS = [
        (r"(?:my|the user(?:'s)?) name is ['\"]?([\w][\w\s]*?)(?:['\"])?(?:\s|$)", 'user_name'),
        (r"(?:your|the assistant(?:'s)?) name is ['\"]?([\w][\w\s]*?)(?:['\"])?(?:\s|$)", 'assistant_name'),
        (r"act (?:like|as) (?:a |an )?(.+?)(?:\s*$)", 'behavior'),
        (r"(?:i am|i'm) (?:a |an )?([\w][\w\s]{1,30}?)(?:\s*$)", 'user_role'),
        (r"(?:i (?:work|works) (?:at|for)|employed (?:at|by)) ([^.\n]+)", 'user_employer'),
    ]

    def __init__(self, db_path: str = MEMORY_DB):
        self.db_path = db_path
        self._temp: List[str] = []
        self._temp_active = False
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS memory (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TEXT,
                    updated_at TEXT
                );
            """)

    def add(self, raw: str) -> str:
        stored = []
        for pattern, key in self._EXTRACT_PATTERNS:
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                value = match.group(1).strip().rstrip("'\"")
                if value:
                    self._store(key, value)
                    stored.append(f"{key} = {value}")
        if not stored:
            key = f"note_{datetime.now().strftime('%H%M%S')}"
            self._store(key, raw.strip())
            stored.append(f"{key} = {raw.strip()}")
        return "🧠 Stored: " + " | ".join(stored)

    def _store(self, key: str, value: str):
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO memory (key, value, created_at, updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, now, now),
            )

    def recall(self) -> str:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT key, value FROM memory ORDER BY updated_at DESC"
            ).fetchall()
        if not rows:
            return "📭 No memories stored."
        lines = ["🧠 Persistent memory:"]
        for key, value in rows:
            lines.append(f"  {key} = {value}")
        return "\n".join(lines)

    def clear(self) -> str:
        with sqlite3.connect(self.db_path) as con:
            con.execute("DELETE FROM memory")
        return "🗑️ All persistent memory cleared."

    def get_all(self) -> Dict[str, str]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute("SELECT key, value FROM memory ORDER BY updated_at DESC").fetchall()
        return dict(rows)

    def _store_entity(self, subject: str, relation: str, obj: str):
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO entities (subject, relation, object, updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(subject, relation, object) DO UPDATE SET updated_at=excluded.updated_at",
                (subject, relation, obj, now),
            )

    def get_entities(self) -> List[tuple]:
        with sqlite3.connect(self.db_path) as con:
            return con.execute("SELECT subject, relation, object FROM entities ORDER BY updated_at DESC LIMIT 50").fetchall()

    def format_entities(self) -> str:
        entities = self.get_entities()
        if not entities:
            return ""
        lines = ["KNOWLEDGE GRAPH (Known relationships):"]
        for s, r, o in entities[:20]: # Limit context to top 20
            lines.append(f"  - {s} {r} {o}")
        return "\n".join(lines)

    def tempmem_start(self, rule: str) -> str:
        self._temp.append(rule.strip())
        self._temp_active = True
        return f"⏱️ Temp rule active: \"{rule.strip()}\""

    def tempmem_end(self) -> str:
        self._temp.clear()
        self._temp_active = False
        return "⏱️ Temporary memory cleared. Back to normal behaviour."

    def tempmem_status(self) -> str:
        if not self._temp:
            return "📭 No active temporary rules."
        lines = ["⏱️ Active temp rules:"]
        for r in self._temp:
            lines.append(f"  • {r}")
        return "\n".join(lines)

    def build_context_block(self) -> str:
        parts = []
        mem = self.get_all()
        if mem:
            lines = [f"  - {k}: {v}" for k, v in mem.items()]
            mem_text = "\n".join(lines)
            parts.append(
                f"PERSISTENT MEMORY (always apply these facts):\n{mem_text}"
            )
        
        ent_text = self.format_entities()
        if ent_text:
            parts.append(ent_text)
        if self._temp_active and self._temp:
            rules = "\n".join(f"  - {r}" for r in self._temp)
            parts.append(
                f"TEMPORARY RULES (HIGHEST PRIORITY — override everything):\n{rules}"
            )
        if not parts:
            return ""
        block = "\n\n".join(parts)
        return (
            "=== AGENT MEMORY ===\n"
            + block
            + "\n=== END MEMORY ===\n\n"
            "You MUST apply all memory and rules above in your response.\n"
            "Temporary rules override everything. Memory overrides default behaviour.\n"
        )


# ═══════════════════════════════════════════════════════════════
# Model Manager
# ═══════════════════════════════════════════════════════════════

class ModelManager:

    def __init__(self):
        self.current: Optional[str] = None
        self.embed_loaded = False

    def _warm_up(self, model: str):
        ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': '.'}],
            options={'num_predict': 1},
            keep_alive=-1,
        )

    def _evict(self, model: str):
        """Signals Ollama to unload the model immediately."""
        try:
            # Send keep_alive=0 to trigger eviction
            ollama.generate(model=model, prompt='', keep_alive=0)
            logger.info("Eviction signal sent for %s", model)
        except Exception as e:
            logger.debug("Evict warning for %s: %s", model, e)

    def load(self, model: str):
        if self.current == model:
            return
        if self.current:
            logger.info("Unloading %s...", self.current)
            print(f"  ⏳ Unloading {self.current}...")
            self._evict(self.current)
            self.current = None

        vram_needed = 4.8 if 'llama' in model else 4.1
        warn_resources(vram_needed)

        logger.info("Loading %s...", model)
        print(f"  ⏳ Loading {model}...", end='', flush=True)
        try:
            self._warm_up(model)
            self.current = model
            print(" ready")
            logger.info("Model %s loaded and ready", model)
            print_model_box(model, status='loaded')
        except Exception as e:
            print(f" FAILED: {e}")
            logger.error("Failed to load %s: %s", model, e)
            raise

    def ensure_chat(self):
        self.load(CHAT_MODEL)

    def ensure_code(self):
        self.load(CODE_MODEL)

    def ensure_embed(self):
        """Ensures embedding model is sticky in VRAM (keep_alive=-1)."""
        if self.embed_loaded:
            return
            
        logger.info("Loading Sticky Embedding: %s...", EMBED_MODEL)
        try:
            # Note: We use keep_alive=-1 to lock it in VRAM alongside Chat/Code model
            ollama.embed(model=EMBED_MODEL, input="warmup", keep_alive=-1)
            self.embed_loaded = True
            logger.info("Embedding model %s is now STICKY in VRAM.", EMBED_MODEL)
        except Exception as e:
            logger.warning("Failed to make embedding model sticky: %s", e)

    def embed(self, text: str) -> List[float]:
        """Get embeddings with automatic VRAM recovery and sticky loading."""
        self.ensure_embed()
        try:
            resp = ollama.embed(model=EMBED_MODEL, input=text)
            return resp['embeddings'][0]
        except Exception as e:
            if "500" in str(e) or "resources" in str(e).lower():
                logger.warning("Embedding VRAM peak: evicting %s and retrying", self.current)
                if self.current:
                    self._evict(self.current)
                    self.current = None
                # After eviction, retry the embedding (still sticky)
                resp = ollama.embed(model=EMBED_MODEL, input=text, keep_alive=-1)
                return resp['embeddings'][0]
            raise

    # ⚙️ CHANGE 7: All generation uses deterministic defaults
    def chat_complete(
        self,
        messages: List[Dict],
        *,
        num_predict: int = 2048,
        temperature: float = DEFAULT_CHAT_TEMP,
        stream: bool = False,
    ) -> str:
        self.ensure_chat()
        if stream:
            return self._stream(CHAT_MODEL, messages, num_predict, temperature)
        resp = ollama.chat(
            model=CHAT_MODEL,
            messages=messages,
            options={'num_predict': num_predict, 'temperature': temperature,
                     'top_p': 0.9, 'repeat_penalty': 1.1},
            keep_alive=-1,
        )
        return resp['message']['content'].strip()

    # ⚙️ CHANGE 1: JSON-enforced completion for structured outputs
    def chat_complete_json(
        self,
        messages: List[Dict],
        *,
        num_predict: int = 2048,
        temperature: float = DEFAULT_PLAN_TEMP,
    ) -> Optional[Dict]:
        """Call chat model and parse JSON from response. Returns None on failure."""
        self.ensure_chat()
        resp = ollama.chat(
            model=CHAT_MODEL,
            messages=messages,
            options={'num_predict': num_predict, 'temperature': temperature,
                     'top_p': 0.9, 'repeat_penalty': 1.1},
            format='json',
            keep_alive=-1,
        )
        raw = resp['message']['content'].strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            match = re.search(r'\{[\s\S]+\}', raw)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            logger.warning("JSON parse failed: %s", raw[:200])
            return None

    def code_complete(
        self,
        messages: List[Dict],
        *,
        num_predict: int = 4096,
        temperature: float = DEFAULT_CODE_TEMP,
    ) -> str:
        self.ensure_code()
        resp = ollama.chat(
            model=CODE_MODEL,
            messages=messages,
            options={'num_predict': num_predict, 'temperature': temperature,
                     'top_p': 0.9, 'repeat_penalty': 1.1},
            keep_alive=-1,
        )
        return resp['message']['content'].strip()

    def _stream(self, model: str, messages: List[Dict],
                num_predict: int, temperature: float) -> str:
        chunks = []
        for chunk in ollama.chat(
            model=model,
            messages=messages,
            options={'num_predict': num_predict, 'temperature': temperature,
                     'top_p': 0.9, 'repeat_penalty': 1.1},
            keep_alive=-1,
            stream=True,
        ):
            token = chunk['message']['content']
            print(token, end='', flush=True)
            chunks.append(token)
        print()
        return ''.join(chunks).strip()


# ═══════════════════════════════════════════════════════════════
# UI helpers
# ═══════════════════════════════════════════════════════════════

def print_banner():
    print(SEP)
    print("  🤖 AI Assistant v3 · Structured Pipeline · Ollama")
    print(SEP)


def print_help():
    print("""
  🔧 Code ═══════════════════════════════════════
    /agentcode    – autonomous project builder
    /code         – create or edit files
    /analyze      – audit files, no edits
    /undo         – restore file from backup
    /cd           – set project root
    /refresh      – rebuild dependency graph

  💬 Chat & Research ═════════════════════════════
    /chat         – switch to chat mode
    /search       – web search + summarise
    /research     – deep research
    /news [topic] – top news

  🤖 Agent ═══════════════════════════════════════
    /task         – autonomous multi-step agent
                    (browsing, comparison, extract+save)

  🌐 Browser & Login ═════════════════════════════
    /credential save <domain> <user> <pass>  – save login
    /credential list                         – show saved logins
    /credential delete <domain>              – remove login
    /credential show <domain>                – show saved login

    Smart commands (just type naturally):
    • "open youtube and play lofi beats"
    • "log into gmail and check emails"
    • "search amazon for headphones"
    • "go to github.com and find trending repos"

  🧠 Memory ══════════════════════════════════════
    /memory add   – store persistent memory
    /memory recall – show stored memory
    /tempmem      – add session-only rule
    /endtempmem   – clear all temp rules

  💾 Conversations ═══════════════════════════════
    /history      – list saved conversations
    /new          – start new conversation
    /load         – load previous conversation
    /quit         – save and exit

  ⚡ Special
    ==== (4+ equals)  – STOP immediately
    Ctrl+C once=save  twice=exit
""")


def _input(prompt: str) -> str:
    return input(prompt).strip()


# ═══════════════════════════════════════════════════════════════
# ⚙️ CHANGE 9: Step Checkpoint Manager
# ═══════════════════════════════════════════════════════════════

class StepCheckpoint:
    """Saves state after each step so the system can resume and avoid repeating work."""

    def __init__(self, project_root: str):
        self.path = os.path.join(project_root, '.step_checkpoint.json')
        self._data: List[Dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
            except Exception:
                self._data = []

    def _save(self):
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def save_step(self, step_id: str, status: str, output: str = ''):
        """⚙️ CHANGE 9: Checkpoint after every step."""
        self._data.append({
            'step': step_id,
            'status': status,
            'output': output[:500],  # truncate for storage
            'timestamp': datetime.now().isoformat(),
        })
        self._save()

    def get_completed_steps(self) -> set:
        return {s['step'] for s in self._data if s['status'] == 'done'}

    def clear(self):
        self._data = []
        if os.path.exists(self.path):
            os.remove(self.path)


# ═══════════════════════════════════════════════════════════════
# BrowserManager — Persistent headless browser for TaskAgent
# ═══════════════════════════════════════════════════════════════
# CredentialVault — Encrypted credential storage for logins
# ═══════════════════════════════════════════════════════════════

KNOWN_LOGIN_CONFIGS = {
    'google.com': {
        'login_url': 'https://accounts.google.com/signin',
        'username_selector': '#identifierId',
        'password_selector': 'input[type="password"]',
        'submit_selector': '#identifierNext, #passwordNext',
        'multi_step': True,
        'steps': [
            {'action': 'type', 'selector': '#identifierId', 'field': 'username'},
            {'action': 'click', 'selector': '#identifierNext'},
            {'action': 'wait', 'seconds': 3},
            {'action': 'type', 'selector': 'input[type="password"]', 'field': 'password'},
            {'action': 'click', 'selector': '#passwordNext'},
            {'action': 'wait', 'seconds': 5},
        ]
    },
    'youtube.com': {
        'login_url': 'https://accounts.google.com/signin?service=youtube',
        'inherits': 'google.com',
    },
    'github.com': {
        'login_url': 'https://github.com/login',
        'username_selector': '#login_field',
        'password_selector': '#password',
        'submit_selector': 'input[type="submit"]',
    },
    'twitter.com': {
        'login_url': 'https://twitter.com/i/flow/login',
        'username_selector': 'input[autocomplete="username"]',
        'password_selector': 'input[type="password"]',
    },
}


class CredentialVault:
    """
    Credential storage for website logins.
    Uses SQLite with base64 obfuscation (for local use).
    For production: use keyring or OS credential store.
    """

    def __init__(self, db_path: str = CREDENTIALS_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS credentials (
                    domain TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    password_b64 TEXT NOT NULL,
                    login_url TEXT,
                    username_selector TEXT DEFAULT 'input[type="email"], input[name="username"], input[name="email"], #identifierId',
                    password_selector TEXT DEFAULT 'input[type="password"], input[name="password"]',
                    submit_selector TEXT DEFAULT 'button[type="submit"], input[type="submit"]',
                    extra_steps TEXT DEFAULT '[]',
                    last_used TEXT,
                    created_at TEXT
                );
            """)

    def save_credential(self, domain: str, username: str, password: str,
                        login_url: str = '', username_sel: str = '',
                        password_sel: str = '', submit_sel: str = '',
                        extra_steps: list = None) -> str:
        now = datetime.now().isoformat()
        pwd_b64 = base64.b64encode(password.encode()).decode()
        with sqlite3.connect(self.db_path) as con:
            con.execute("""
                INSERT INTO credentials
                    (domain, username, password_b64, login_url,
                     username_selector, password_selector, submit_selector,
                     extra_steps, last_used, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(domain) DO UPDATE SET
                    username=excluded.username,
                    password_b64=excluded.password_b64,
                    login_url=COALESCE(NULLIF(excluded.login_url,''), login_url),
                    username_selector=CASE WHEN excluded.username_selector != ''
                        THEN excluded.username_selector ELSE username_selector END,
                    password_selector=CASE WHEN excluded.password_selector != ''
                        THEN excluded.password_selector ELSE password_selector END,
                    submit_selector=CASE WHEN excluded.submit_selector != ''
                        THEN excluded.submit_selector ELSE submit_selector END,
                    extra_steps=CASE WHEN excluded.extra_steps != '[]'
                        THEN excluded.extra_steps ELSE extra_steps END,
                    last_used=excluded.last_used
            """, (domain, username, pwd_b64,
                  login_url or '', username_sel or '', password_sel or '',
                  submit_sel or '', json.dumps(extra_steps or []), now, now))
        return f"✓ Credentials saved for {domain}"

    def get_credential(self, domain: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as con:
            row = con.execute(
                "SELECT * FROM credentials WHERE domain=?", (domain,)
            ).fetchone()
        if not row:
            with sqlite3.connect(self.db_path) as con:
                row = con.execute(
                    "SELECT * FROM credentials WHERE ? LIKE '%' || domain || '%'",
                    (domain,)
                ).fetchone()
        if not row:
            return None
        return {
            'domain': row[0], 'username': row[1],
            'password': base64.b64decode(row[2]).decode(),
            'login_url': row[3], 'username_selector': row[4],
            'password_selector': row[5], 'submit_selector': row[6],
            'extra_steps': json.loads(row[7]) if row[7] else [],
        }

    def list_domains(self) -> List[str]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute("SELECT domain FROM credentials").fetchall()
        return [r[0] for r in rows]

    def delete_credential(self, domain: str) -> str:
        with sqlite3.connect(self.db_path) as con:
            con.execute("DELETE FROM credentials WHERE domain=?", (domain,))
        return f"✓ Credentials deleted for {domain}"


# ═══════════════════════════════════════════════════════════════
# BrowsingMemory — Learn from past browsing mistakes & successes
# ═══════════════════════════════════════════════════════════════

class BrowsingMemory:
    """Persistent memory that stores successful patterns and failed approaches."""

    def __init__(self):
        self.path = os.path.join(BROWSER_PROFILE_DIR, 'browser_memory.json')
        self.data = self._load()

    def _load(self) -> Dict:
        try:
            if os.path.exists(self.path):
                with open(self.path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {'successes': [], 'failures': [], 'site_tips': {}}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug("BrowsingMemory save failed: %s", e)

    def remember_success(self, site: str, goal: str, steps: List[Dict]):
        """Store a successful action pattern."""
        entry = {
            'site': site, 'goal': goal,
            'steps': [f"{s.get('action','?')}: {s.get('input','')}" for s in steps[-8:]],
            'time': datetime.now().isoformat()
        }
        self.data['successes'] = self.data.get('successes', [])[-19:] + [entry]
        # Build site tips from success patterns
        if site not in self.data.get('site_tips', {}):
            self.data.setdefault('site_tips', {})[site] = []
        tips = self.data['site_tips'][site]
        tip_text = f"For '{goal}': " + ' → '.join(entry['steps'][:5])
        if tip_text not in tips:
            self.data['site_tips'][site] = (tips + [tip_text])[-10:]
        self._save()

    def remember_failure(self, site: str, action: str, error: str):
        """Store a failed action so we avoid repeating it."""
        entry = {
            'site': site, 'action': action,
            'error': str(error)[:200],
            'time': datetime.now().isoformat()
        }
        self.data['failures'] = self.data.get('failures', [])[-29:] + [entry]
        self._save()

    def get_tips(self, site: str, goal: str) -> str:
        """Get relevant tips for a site/goal combination."""
        tips = []
        # Successes for this site
        for s in self.data.get('successes', []):
            if site and site.lower() in s.get('site', '').lower():
                tips.append(f"✅ Previously worked for '{s['goal']}': {' → '.join(s['steps'][:4])}")
        # Recent failures for this site
        site_failures = [f for f in self.data.get('failures', []) if site and site.lower() in f.get('site', '').lower()]
        for f in site_failures[-3:]:
            tips.append(f"❌ AVOID: {f['action']} failed with: {f['error'][:100]}")
        # Site tips
        for t in self.data.get('site_tips', {}).get(site, [])[-3:]:
            tips.append(f"💡 {t}")
        return '\n'.join(tips[-6:]) if tips else ''


# ═══════════════════════════════════════════════════════════════
# BrowserManager v2 — Visible Chrome via CDP, Resilient
# ═══════════════════════════════════════════════════════════════

class BrowserManager:
    """
    Browser controller that connects to user's REAL Chrome via CDP:
    - Visible — user sees every action live in their Chrome window
    - Uses existing cookies/logins (Instagram, Gmail, etc.)
    - Auto-recovery from crashes (3-level: new page → new context → full restart)
    - Persistent sessions via storage_state.json
    - Smart navigation with popup dismissal
    """

    def __init__(self, headless: bool = False):
        self._pw = None
        self._browser = None
        self._context = None
        self._pages: List = []
        self._active: int = 0
        self._launched = False
        self._headless = headless
        self._user_data_dir = CHROME_USER_DATA_DIR
        self._agent_profile_dir = BROWSER_PROFILE_DIR
        self._credential_vault = CredentialVault()
        self._recovery_count = 0
        self._max_recoveries = 5
        self._last_url: str = ''
        os.makedirs(self._agent_profile_dir, exist_ok=True)

    # ─── Core: Health Check & Recovery ───

    def _is_page_alive(self) -> bool:
        if not self._launched or not self._pages:
            return False
        try:
            page = self._pages[self._active]
            page.evaluate("() => document.readyState")
            return True
        except Exception:
            return False

    def _is_browser_alive(self) -> bool:
        try:
            return self._browser is not None and self._browser.is_connected()
        except Exception:
            return False

    def _recover(self):
        self._recovery_count += 1
        if self._recovery_count > self._max_recoveries:
            logger.error("Too many recovery attempts (%d). Doing full restart.", self._recovery_count)
            self._full_restart()
            self._recovery_count = 0
            return
        logger.warning("Browser recovery attempt #%d", self._recovery_count)
        print(f"  🔄 Browser recovery #{self._recovery_count}...")

        # Strategy 1: New page in existing context
        if self._is_browser_alive() and self._context:
            try:
                new_page = self._context.new_page()
                self._pages = [new_page]
                self._active = 0
                if self._last_url:
                    try:
                        new_page.goto(self._last_url, timeout=15000, wait_until='domcontentloaded')
                    except Exception:
                        pass
                logger.info("Recovery: new page in existing context")
                return
            except Exception as e:
                logger.warning("New page failed: %s", e)

        # Strategy 2: New context in existing browser
        if self._is_browser_alive():
            try:
                storage_state = self._get_storage_state_path()
                storage_arg = {}
                if os.path.exists(storage_state):
                    try:
                        with open(storage_state, 'r') as f:
                            json.load(f)
                        storage_arg = {'storage_state': storage_state}
                    except Exception:
                        pass
                self._context = self._browser.new_context(
                    user_agent=BROWSER_USER_AGENT,
                    viewport={'width': 1280, 'height': 720},
                    java_script_enabled=True,
                    **storage_arg,
                )
                new_page = self._context.new_page()
                self._pages = [new_page]
                self._active = 0
                if self._last_url:
                    try:
                        new_page.goto(self._last_url, timeout=15000, wait_until='domcontentloaded')
                    except Exception:
                        pass
                logger.info("Recovery: new context in existing browser")
                return
            except Exception as e:
                logger.warning("New context failed: %s", e)

        # Strategy 3: Full restart
        self._full_restart()

    def _full_restart(self):
        logger.info("Full browser restart...")
        print("  🔄 Full browser restart...")
        self._save_storage_state()
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._pw:
                self._pw.stop()
        except Exception:
            pass
        self._browser = None
        self._pw = None
        self._context = None
        self._pages = []
        self._launched = False
        self._ensure()

    def _setup_context_from_browser(self):
        """Extract context and pages from an existing browser connection."""
        contexts = self._browser.contexts
        if contexts:
            self._context = contexts[0]
            pages = self._context.pages
            if pages:
                self._pages = list(pages)
                self._active = 0
            else:
                self._pages = [self._context.new_page()]
                self._active = 0
        else:
            self._context = self._browser.new_context(
                viewport={'width': 1280, 'height': 720},
                java_script_enabled=True,
            )
            self._pages = [self._context.new_page()]
            self._active = 0
        self._launched = True
        self._recovery_count = 0
        self._is_cdp = True

    def _find_chrome_executable(self) -> Optional[str]:
        """Find the Chrome executable on the system."""
        if platform.system() == 'Windows':
            paths = [
                os.path.expandvars(r'%PROGRAMFILES%\Google\Chrome\Application\chrome.exe'),
                os.path.expandvars(r'%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe'),
                os.path.expandvars(r'%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe'),
            ]
        elif platform.system() == 'Darwin':
            paths = ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome']
        else:
            paths = ['/usr/bin/google-chrome', '/usr/bin/google-chrome-stable']
            
        for p in paths:
            if os.path.exists(p):
                return p
        return None

    def get_active_page(self):
        """Public accessor for the active page, ensuring browser is alive."""
        self._ensure()
        if not self._is_page_alive():
            self._recover()
        return self._pages[self._active]

    def _fix_chrome_exit_type(self):
        """Prevents the 'Restore pages?' bubble by marking the profile as closed cleanly."""
        if not self._user_data_dir or not os.path.exists(self._user_data_dir):
            return
        
        # Check Default profile and any other profile folders
        profiles = ['Default']
        try:
            for p in os.listdir(self._user_data_dir):
                if p.startswith('Profile '):
                    profiles.append(p)
        except Exception: pass
                
        for profile in profiles:
            pref_path = os.path.join(self._user_data_dir, profile, 'Preferences')
            if os.path.exists(pref_path):
                try:
                    with open(pref_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    changed = False
                    if 'profile' in data:
                        if data['profile'].get('exit_type') != 'Normal':
                            data['profile']['exit_type'] = 'Normal'
                            changed = True
                        if data['profile'].get('exited_cleanly') is not True:
                            data['profile']['exited_cleanly'] = True
                            changed = True
                    
                    if changed:
                        with open(pref_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f)
                        logger.info("Fixed exit_type for profile: %s", profile)
                except Exception as e:
                    logger.debug("Could not fix Preferences for %s: %s", profile, e)

    def _ensure(self):
        if self._launched and self._is_page_alive():
            return
        if self._launched and self._is_browser_alive() and not self._is_page_alive():
            self._recover()
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise RuntimeError("playwright not installed. Run:\n  pip install playwright && playwright install chromium")
        if self._pw is None:
            self._pw_cm = sync_playwright()
            self._pw = self._pw_cm.start()

        cdp_url = 'http://127.0.0.1:9222'
        # ── Try CDP connection to user's real Chrome first ──
        if not self._headless:
            try:
                # 1. Try to connect if port 9222 is already open
                self._browser = self._pw.chromium.connect_over_cdp(cdp_url)
                logger.info("Connected to user's Chrome via CDP at %s", cdp_url)
                print("  🌐 Connected to your active Chrome session.")
                self._setup_context_from_browser()
                return
            except Exception:
                logger.info("CDP connection failed, checking if Chrome is running...")

            # 2. Check if Chrome is already running
            is_running = False
            try:
                for proc in psutil.process_iter(['name']):
                    if proc.info['name'] and 'chrome' in proc.info['name'].lower():
                        is_running = True
                        break
            except Exception:
                pass
            
            if is_running:
                logger.info("Chrome detected. Restarting for debug access...")
                print("  ⚠️ Your Chrome is open. Restarting to access your profile...")
                try:
                    if platform.system() == 'Windows':
                        subprocess.run(['taskkill', '/F', '/IM', 'chrome.exe', '/T'], capture_output=True)
                    else:
                        subprocess.run(['pkill', '-f', '(chrome|chromium)'], capture_output=True)
                    time.sleep(10)  # Give MORE time to release file locks on the profile
                except Exception as e:
                    logger.warning("Kill failed: %s", e)

            # 3. Fix exit_type and launch with persistent context
            self._fix_chrome_exit_type()
            chrome_exe = self._find_chrome_executable()
            if chrome_exe and os.path.exists(self._user_data_dir):
                try:
                    logger.info("Launching persistent context: %s with %s", chrome_exe, self._user_data_dir)
                    print("  🚀 Connecting to your Chrome profile...")
                    
                    # Retry a few times if the profile is still locked
                    last_err = ""
                    for attempt in range(3):
                        try:
                            self._context = self._pw.chromium.launch_persistent_context(
                                user_data_dir=self._user_data_dir,
                                executable_path=chrome_exe,
                                headless=self._headless,
                                args=[
                                    '--no-first-run',
                                    '--no-default-browser-check',
                                    '--disable-infobars',
                                    '--disable-search-engine-choice-screen',
                                ],
                                user_agent=BROWSER_USER_AGENT,
                                viewport={'width': 1280, 'height': 720},
                            )
                            break # Success!
                        except Exception as e:
                            last_err = str(e)
                            logger.warning("Launch attempt %d failed: %s", attempt+1, e)
                            time.sleep(2)
                    
                    if not self._context:
                        raise RuntimeError(f"Could not launch after retries. Last error: {last_err}")
                    
                    self._browser = self._context.browser
                    self._pages = self._context.pages
                    if not self._pages:
                        self._pages = [self._context.new_page()]
                    self._active = 0
                    self._launched = True
                    self._is_cdp = False
                    self._recovery_count = 0
                    print("  🌐 Your Chrome is now connected and ready.")
                    return
                except Exception as e:
                    logger.warning("Native persistent launch failed: %s", e)
                    print(f"  ⚠️ Could not launch your Chrome: {e}. Falling back...")

        # ── Fallback: launch Playwright Chromium ──

        # ── Fallback: launch Playwright Chromium ──
        logger.info("Falling back to Playwright Chromium (headless=%s)", self._headless)
        print("  ⚠️ Could not connect to Chrome. Launching Playwright Chromium...")

        storage_state = self._get_storage_state_path()
        storage_arg = {}
        if os.path.exists(storage_state):
            try:
                with open(storage_state, 'r') as f:
                    json.load(f)
                storage_arg = {'storage_state': storage_state}
                logger.info("Restoring browser session from %s", storage_state)
            except Exception:
                logger.warning("Corrupt storage state, starting fresh")

        self._browser = self._pw.chromium.launch(
            headless=self._headless,
            args=[
                '--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage',
                '--disable-extensions', '--disable-background-networking',
                '--disable-sync', '--no-first-run',
            ],
        )
        self._context = self._browser.new_context(
            user_agent=BROWSER_USER_AGENT,
            viewport={'width': 1280, 'height': 720},
            java_script_enabled=True,
            **storage_arg,
        )
        self._context.on("page", lambda p: p.on("dialog", lambda d: d.dismiss()))
        page = self._context.new_page()
        self._pages = [page]
        self._active = 0
        self._launched = True
        self._is_cdp = False
        self._recovery_count = 0
        logger.info("Browser launched (persistent profile, headless=%s)", self._headless)

    def _get_storage_state_path(self) -> str:
        return os.path.join(self._agent_profile_dir, 'storage_state.json')

    def _save_storage_state(self):
        if not self._context:
            return
        try:
            path = self._get_storage_state_path()
            self._context.storage_state(path=path)
            logger.info("Browser session saved to %s", path)
        except Exception as e:
            logger.warning("Could not save storage state: %s", e)

    # ─── Core: Safe Page Access ───

    @property
    def page(self):
        self._ensure()
        if not self._is_page_alive():
            self._recover()
        return self._pages[self._active]

    def _safe_execute(self, operation_name: str, func, *args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._ensure()
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                is_crash = any(x in error_str for x in [
                    'target closed', 'browser has been closed',
                    'context or browser', 'target page',
                    'connection refused', 'not connected',
                    'execution context was destroyed',
                    'frame was detached',
                ])
                if is_crash and attempt < max_retries - 1:
                    logger.warning(
                        "%s failed (attempt %d/%d): %s — recovering...",
                        operation_name, attempt + 1, max_retries, str(e)[:100]
                    )
                    self._recover()
                    # Exponential backoff
                    time.sleep(2 ** attempt)
                    continue
                else:
                    logger.error("%s failed permanently: %s", operation_name, e)
                    return f"ERROR: {operation_name} failed — {str(e)[:200]}"
        return f"ERROR: {operation_name} failed after {max_retries} attempts"

    # ─── Navigation ───

    def open_url(self, url: str, timeout: int = 20000) -> str:
        def _do_open():
            self.page.goto(url, timeout=timeout, wait_until='domcontentloaded')
            self._last_url = url
            time.sleep(1.5)
            self._try_dismiss_popups()
            self._save_storage_state()
            return self._extract_visible_text()
        result = self._safe_execute('browser_open', _do_open)
        if isinstance(result, str) and result.startswith("ERROR"):
            logger.warning("browser_open failed for %s: %s", url, result)
        return result

    def type_text(self, selector: str, text: str, press_enter: bool = True) -> str:
        def _do_type():
            for sel in [selector] + self._get_fallback_selectors(selector):
                try:
                    el = self.page.wait_for_selector(sel, timeout=5000)
                    if el:
                        el.click()
                        time.sleep(0.5)
                        el.fill('') # Clear field
                        time.sleep(0.2)
                        # Switch to sequential typing for better compatibility
                        el.press_sequentially(text, delay=50)
                        time.sleep(0.5)
                        if press_enter:
                            el.press('Enter')
                            time.sleep(2)
                        return f"Typed '{text}' into {sel} and pressed Enter."
                except Exception:
                    continue
            available = self._list_inputs()
            return f"ERROR: Could not find input '{selector}'. Available: {available}"
        return self._safe_execute('browser_type', _do_type)

    def click(self, text: str) -> str:
        def _do_click():
            try:
                locator = self.page.get_by_text(text, exact=False).first
                locator.click(timeout=5000)
                time.sleep(2)
                self._save_storage_state()
                return f"Clicked '{text}'. Page may have changed."
            except Exception:
                pass
            for role in ['button', 'link', 'menuitem', 'tab']:
                try:
                    locator = self.page.get_by_role(role, name=text).first
                    locator.click(timeout=3000)
                    time.sleep(2)
                    self._save_storage_state()
                    return f"Clicked {role} '{text}'."
                except Exception:
                    continue
            if any(c in text for c in ['#', '.', '[', '>']):
                try:
                    self.page.click(text, timeout=5000)
                    time.sleep(2)
                    return f"Clicked selector '{text}'."
                except Exception:
                    pass
            return f"ERROR: Could not find clickable element '{text}'."
        return self._safe_execute('browser_click', _do_click)

    def click_selector(self, selector: str) -> str:
        def _do_click_sel():
            try:
                self.page.click(selector, timeout=5000)
                time.sleep(2)
                self._save_storage_state()
                return f"Clicked selector '{selector}'."
            except Exception as e:
                return f"ERROR: Could not click '{selector}': {e}"
        return self._safe_execute('browser_click_selector', _do_click_sel)

    def wait_for(self, selector: str = '', seconds: float = 3) -> str:
        def _do_wait():
            if selector:
                try:
                    self.page.wait_for_selector(selector, timeout=int(seconds * 1000))
                    return f"Element '{selector}' appeared."
                except Exception:
                    return f"Element '{selector}' not found after {seconds}s."
            else:
                time.sleep(seconds)
                return f"Waited {seconds}s."
        return self._safe_execute('browser_wait', _do_wait)

    def extract(self, goal: str = "") -> str:
        def _do_extract():
            return self._extract_visible_text(goal=goal)
        return self._safe_execute('browser_extract', _do_extract)

    def scroll_down(self) -> str:
        def _do_scroll():
            self.page.evaluate("window.scrollBy(0, window.innerHeight)")
            time.sleep(1)
            return "Scrolled down."
        return self._safe_execute('browser_scroll', _do_scroll)

    def go_back(self) -> str:
        def _do_back():
            self.page.go_back(timeout=10000)
            time.sleep(1.5)
            return f"Went back. Now on: {self.page.url}"
        return self._safe_execute('browser_back', _do_back)

    def current_url(self) -> str:
        try:
            self._ensure()
            return self.page.url
        except Exception:
            return 'unknown'

    def screenshot(self, path: str = '') -> str:
        def _do_screenshot():
            if not path:
                p = os.path.join(self._agent_profile_dir,
                                 f"screenshot_{datetime.now().strftime('%H%M%S')}.png")
            else:
                p = path
            self.page.screenshot(path=p)
            return f"Screenshot saved: {p}"
        return self._safe_execute('browser_screenshot', _do_screenshot)

    # ─── Login System ───

    def login(self, domain: str, username: str = '', password: str = '') -> str:
        creds = self._credential_vault.get_credential(domain)
        if not creds and (not username or not password):
            return (f"ERROR: No saved credentials for '{domain}'. "
                    f"Provide username and password, or save with: "
                    f"/credential save {domain} <username> <password>")
        if username and password:
            self._credential_vault.save_credential(domain, username, password)
            creds = self._credential_vault.get_credential(domain)
        elif creds:
            username = creds['username']
            password = creds['password']
        config = self._get_login_config(domain)
        login_url = creds.get('login_url') or config.get('login_url', f'https://{domain}/login')
        print(f"  🔐 Logging into {domain}...")
        result = self.open_url(login_url)
        if isinstance(result, str) and result.startswith("ERROR"):
            return result
        page_text = self._extract_visible_text()
        if self._detect_logged_in(page_text, domain):
            self._save_storage_state()
            return f"Already logged in to {domain}!"
        if config.get('multi_step') or config.get('steps'):
            return self._execute_login_steps(config, username, password, domain)
        return self._standard_login(config, creds, username, password, domain)

    def _standard_login(self, config, creds, username, password, domain) -> str:
        username_sels = [
            creds.get('username_selector', '') if creds else '',
            config.get('username_selector', ''),
            'input[type="email"]', 'input[name="username"]',
            'input[name="email"]', 'input[name="login"]',
            '#username', '#email', '#login_field', 'input[type="text"]',
        ]
        filled_user = False
        for sel in username_sels:
            if not sel: continue
            try:
                el = self.page.wait_for_selector(sel, timeout=3000)
                if el and el.is_visible():
                    el.click(); el.fill(username)
                    filled_user = True; break
            except Exception: continue
        if not filled_user:
            return f"ERROR: Could not find username field on {domain}"
        time.sleep(0.5)
        password_sels = [
            creds.get('password_selector', '') if creds else '',
            config.get('password_selector', ''),
            'input[type="password"]', 'input[name="password"]', '#password',
        ]
        filled_pass = False
        for sel in password_sels:
            if not sel: continue
            try:
                el = self.page.wait_for_selector(sel, timeout=3000)
                if el and el.is_visible():
                    el.click(); el.fill(password)
                    filled_pass = True; break
            except Exception: continue
        if not filled_pass:
            return f"ERROR: Could not find password field on {domain}"
        time.sleep(0.5)
        submit_sels = [
            creds.get('submit_selector', '') if creds else '',
            config.get('submit_selector', ''),
            'button[type="submit"]', 'input[type="submit"]',
            'button:has-text("Sign in")', 'button:has-text("Log in")',
            'button:has-text("Login")', 'button:has-text("Submit")',
        ]
        for sel in submit_sels:
            if not sel: continue
            try:
                el = self.page.wait_for_selector(sel, timeout=2000)
                if el and el.is_visible():
                    el.click(); time.sleep(3); break
            except Exception: continue
        self._save_storage_state()
        page_text = self._extract_visible_text()
        if self._detect_logged_in(page_text, domain):
            return f"✓ Successfully logged into {domain}!"
        elif self._detect_login_error(page_text):
            return f"Login failed for {domain}. Check credentials."
        else:
            return f"Login attempted on {domain}. Please verify."

    def _execute_login_steps(self, config, username, password, domain) -> str:
        steps = config.get('steps', [])
        for i, step in enumerate(steps):
            action = step.get('action', '')
            selector = step.get('selector', '')
            try:
                if action == 'type':
                    field = step.get('field', '')
                    value = username if field == 'username' else password
                    el = self.page.wait_for_selector(selector, timeout=8000)
                    if el:
                        el.click(); el.fill(value); time.sleep(0.5)
                elif action == 'click':
                    try:
                        el = self.page.wait_for_selector(selector, timeout=5000)
                        if el and el.is_visible(): el.click()
                    except Exception:
                        try: self.page.get_by_text("Next").first.click(timeout=3000)
                        except Exception: pass
                elif action == 'wait':
                    time.sleep(step.get('seconds', 3))
                elif action == 'press':
                    self.page.keyboard.press(step.get('key', 'Enter'))
                    time.sleep(1)
            except Exception as e:
                logger.warning("Login step %d failed: %s", i, e)
        time.sleep(3)
        self._save_storage_state()
        page_text = self._extract_visible_text()
        if self._detect_logged_in(page_text, domain):
            return f"✓ Successfully logged into {domain}!"
        else:
            return f"Login steps completed for {domain}. Session saved."

    def _get_login_config(self, domain: str) -> Dict:
        for known_domain, config in KNOWN_LOGIN_CONFIGS.items():
            if known_domain in domain:
                if 'inherits' in config:
                    parent = KNOWN_LOGIN_CONFIGS.get(config['inherits'], {})
                    merged = {**parent, **config}
                    merged.pop('inherits', None)
                    return merged
                return config
        return {}

    def _detect_logged_in(self, page_text: str, domain: str) -> bool:
        indicators = [
            'sign out', 'log out', 'logout', 'my account',
            'profile', 'dashboard', 'inbox', 'settings',
            'compose', 'upload', 'your channel',
        ]
        text_lower = page_text.lower()
        return any(ind in text_lower for ind in indicators)

    def _detect_login_error(self, page_text: str) -> bool:
        indicators = [
            'wrong password', 'incorrect password', 'invalid credentials',
            'account not found', "couldn't find", 'try again',
            'authentication failed', 'login failed',
        ]
        text_lower = page_text.lower()
        return any(ind in text_lower for ind in indicators)

    @property
    def vault(self) -> CredentialVault:
        return self._credential_vault

    # ─── Cleanup ───

    def close(self):
        self._save_storage_state()
        # If connected via CDP, just disconnect — do NOT close user's Chrome
        if getattr(self, '_is_cdp', False):
            logger.info("Disconnecting from Chrome (keeping browser open).")
            print("  🔌 Disconnected from Chrome (your browser stays open)")
            try:
                if self._browser: self._browser.close()  # disconnect only
            except Exception: pass
            try:
                if self._pw: self._pw.stop()
            except Exception: pass
            self._browser = None
            self._pw = None
            self._pages = []
            self._context = None
            self._launched = False
            return
        # Regular Playwright: actually close
        try:
            if self._browser: self._browser.close()
        except Exception: pass
        try:
            if self._pw: self._pw.stop()
        except Exception: pass
        self._browser = None
        self._pw = None
        self._pages = []
        self._context = None
        self._launched = False
        logger.info("Browser closed (session saved).")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ─── Helpers ───

    def _extract_visible_text(self, goal: str = "") -> str:
        try:
            self.page.evaluate("""() => {
                const remove = 'script,style,nav,footer,header,aside,iframe,' +
                    '[role="banner"],[role="navigation"],[role="complementary"],' +
                    '.cookie-banner,.modal,.popup,.overlay,.ad,.advertisement';
                document.querySelectorAll(remove).forEach(e => e.remove());
            }""")
        except Exception: pass
        text = ""
        for sel in ['main', 'article', '#content', '#main-content', '.content',
                     '.main', '[role="main"]', '.product-info', '.search-results']:
            try:
                el = self.page.query_selector(sel)
                if el:
                    candidate = el.inner_text().strip()
                    if len(candidate) > 100:
                        text = candidate; break
            except Exception: continue
        if not text or len(text) < 100:
            try:
                text = self.page.evaluate("() => document.body ? document.body.innerText.trim() : ''")
            except Exception:
                text = "ERROR: Could not extract page text."
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{4,}', '  ', text)
        if len(text) > MAX_PAGE_EXTRACT_CHARS:
            text = text[:MAX_PAGE_EXTRACT_CHARS] + "\n... (truncated)"
        return text

    def _try_dismiss_popups(self):
        for sel in ['button:has-text("Accept")', 'button:has-text("Accept All")',
                     'button:has-text("Close")', 'button:has-text("Got it")',
                     'button:has-text("OK")', 'button:has-text("Dismiss")',
                     '[aria-label="Close"]', '.cookie-close', '#cookie-accept']:
            try:
                btn = self.page.wait_for_selector(sel, timeout=1000)
                if btn and btn.is_visible():
                    btn.click(); time.sleep(0.5); return
            except Exception: continue

    def _list_inputs(self) -> str:
        try:
            inputs = self.page.evaluate("""() => {
                return Array.from(document.querySelectorAll('input,textarea'))
                    .slice(0, 8).map(el => {
                    const id = el.id ? '#' + el.id : '';
                    const name = el.name ? '[name="' + el.name + '"]' : '';
                    const type = el.type ? '[type="' + el.type + '"]' : '';
                    const ph = el.placeholder ? ' placeholder="' + el.placeholder + '"' : '';
                    return 'input' + id + name + type + ph;
                });
            }""")
            return ', '.join(inputs) if inputs else 'none found'
        except Exception:
            return 'could not list inputs'

    def _get_fallback_selectors(self, selector: str) -> List[str]:
        return [
            'input[type="search"]', 'input[name="q"]',
            'input[name="search"]', 'input[type="text"]',
            '#search', '.search-input', 'textarea',
        ]


# ═══════════════════════════════════════════════════════════════
# SmartIntentClassifier — NLU layer for natural language commands
# ═══════════════════════════════════════════════════════════════

class SmartIntentClassifier:
    """Understands what the user actually wants from natural language."""

    INTENT_PATTERNS = {
        'play_video': {
            'patterns': [
                r'(?:play|watch|open)\s+(?:.*?)\s+(?:on|in|from)\s+(?:youtube|yt)',
                r'(?:play|watch)\s+(?:.*?)(?:video|song|music|lofi)',
                r'(?:youtube|yt)\s+(?:play|search|find)\s+(.*)',
                r'(?:open|go to)\s+youtube\s+(?:and\s+)?(?:play|search|find)\s+(.*)',
            ],
            'handler': 'handle_play_video',
        },
        'login': {
            'patterns': [
                r'(?:log\s*in|sign\s*in)\s+(?:to|into)\s+(\S+)',
                r'(?:open|go to)\s+(\S+)\s+(?:and\s+)?(?:log\s*in|sign\s*in)',
                r'(?:login|signin)\s+(\S+)',
                r'(?:access|enter)\s+(?:my\s+)?(\S+)\s+account',
            ],
            'handler': 'handle_login',
        },
        'check_email': {
            'patterns': [
                r'(?:check|read|open)\s+(?:my\s+)?(?:email|gmail|inbox|mail)',
                r'(?:go to|open)\s+(?:gmail|mail|email)',
            ],
            'handler': 'handle_check_email',
        },
        'browse_and_do': {
            'patterns': [
                r'(?:go to|open|visit|navigate to)\s+(https?://\S+|[\w.]+\.(?:com|org|net|io)\S*)\s+(?:and|then)\s+(.*)',
                r'(?:on|at|from)\s+([\w.]+\.(?:com|org|net|io))\s+(?:find|search|look for|get|check)\s+(.*)',
            ],
            'handler': 'handle_browse_and_do',
        },
        'search_site': {
            'patterns': [
                r'search\s+(?:for\s+)?(.+?)\s+on\s+(\S+)',
                r'(?:find|look for)\s+(.+?)\s+(?:on|at|in)\s+(\S+)',
            ],
            'handler': 'handle_search_site',
        },
        'save_credentials': {
            'patterns': [
                r'(?:save|store|remember)\s+(?:my\s+)?(?:credentials?|password|login)\s+(?:for\s+)?(\S+)',
                r'(?:credentials?|password)\s+(?:for\s+)?(\S+)\s+(?:is|are)\s+(.*)',
            ],
            'handler': 'handle_save_credentials',
        },
    }

    @classmethod
    def classify(cls, text: str) -> Optional[Dict]:
        text_lower = text.lower().strip()
        for intent_name, config in cls.INTENT_PATTERNS.items():
            for pattern in config['patterns']:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    return {
                        'intent': intent_name,
                        'handler': config['handler'],
                        'captures': match.groups(),
                        'full_text': text,
                    }
        return None


class SmartBrowserHandler:
    """High-level handlers for smart browser intents."""

    def __init__(self, browser: BrowserManager, models, memory):
        self.browser = browser
        self.models = models
        self.memory = memory

    def handle_play_video(self, captures, full_text) -> str:
        query = ' '.join(captures).strip() if captures else ''
        if not query:
            match = re.search(r'(?:play|watch|search)\s+(.+?)(?:\s+on\s+youtube)?$', full_text, re.IGNORECASE)
            query = match.group(1) if match else full_text
        print(f"  🎵 Playing on YouTube: {query}")
        result = self.browser.open_url(
            f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        )
        if isinstance(result, str) and result.startswith("ERROR"):
            return result
        time.sleep(2)
        try:
            self.browser.page.wait_for_selector(
                'ytd-video-renderer, ytd-video-renderer a#video-title', timeout=10000
            )
            first_video = self.browser.page.query_selector('ytd-video-renderer a#video-title')
            if first_video:
                first_video.click()
                time.sleep(3)
                title = self.browser.page.title()
                url = self.browser.page.url
                return f"▶ Now playing: {title}\n🔗 {url}"
        except Exception as e:
            logger.warning("Could not auto-play: %s", e)
        return f"Searched YouTube for '{query}'. Results are displayed."

    def handle_login(self, captures, full_text) -> str:
        domain = captures[0] if captures else ''
        domain = domain.strip().lower()
        domain = re.sub(r'^(https?://|www\.)', '', domain)
        domain = domain.rstrip('/')
        if not domain:
            return "Please specify which site to log into."
        return self.browser.login(domain)

    def handle_check_email(self, captures, full_text) -> str:
        creds = self.browser.vault.get_credential('google.com')
        if not creds:
            creds = self.browser.vault.get_credential('gmail.com')
        if creds:
            result = self.browser.open_url('https://mail.google.com')
            page_text = self.browser.extract()
            if not self.browser._detect_logged_in(page_text, 'google.com'):
                login_result = self.browser.login('google.com')
                if 'ERROR' in login_result:
                    return login_result
                result = self.browser.open_url('https://mail.google.com')
            return f"Gmail opened.\n\n{result[:2000]}"
        else:
            return ("No Google credentials saved. Save them first:\n"
                    "/credential save google.com your@email.com yourpassword")

    def handle_browse_and_do(self, captures, full_text) -> str:
        if len(captures) >= 2:
            url, action = captures[0], captures[1]
            if not url.startswith('http'):
                url = 'https://' + url
            result = self.browser.open_url(url)
            if isinstance(result, str) and result.startswith("ERROR"):
                return result
            return f"Opened {url}. Page content:\n{result[:1500]}\n\nRequested action: {action}"
        return "Could not parse the browse command."

    def handle_search_site(self, captures, full_text) -> str:
        if len(captures) >= 2:
            query, site = captures[0], captures[1]
            site = site.strip().lower()
            if not site.startswith('http'):
                site = 'https://' + site
            result = self.browser.open_url(site)
            if isinstance(result, str) and result.startswith("ERROR"):
                return result
            type_result = self.browser.type_text('input[type="search"]', query)
            if 'ERROR' in type_result:
                type_result = self.browser.type_text('input[name="q"]', query)
            if 'ERROR' in type_result:
                type_result = self.browser.type_text('input[type="text"]', query)
            return f"Searched for '{query}' on {site}.\n{type_result}"
        return "Could not parse search command."

    def handle_save_credentials(self, captures, full_text) -> str:
        domain = captures[0] if captures else ''
        return f"Use: /credential save {domain} <username> <password>"



# ═══════════════════════════════════════════════════════════════
# TaskAgent — Autonomous ReAct agent for multi-step tasks
# ═══════════════════════════════════════════════════════════════

class TaskAgent:
    """
    ReAct agent: think → act → observe → repeat.
    Stays on chat model (no swaps). Max 15 steps, 5 min timeout.
    """

    def __init__(self, goal: str, models: 'ModelManager', memory: 'MemoryManager', project_root: str):
        self.goal = goal
        self.models = models
        self.memory = memory
        self.project_root = project_root
        self.browser = BrowserManager(headless=False)   # Visible Chrome
        self.browsing_memory = BrowsingMemory()
        self.scratchpad: List[Dict[str, str]] = []
        self.store: Dict[str, str] = {}
        self.step_count = 0
        self.start_time = 0.0
        self.consecutive_failures = 0
        self.finished = False
        self.final_answer = ""
        self._state_path = os.path.join(project_root, '.task_agent_state.json')
        self._current_site = ''

    def run(self) -> str:
        self.start_time = time.time()
        print(f"\n{'─'*60}")
        print(f"  🤖 TASK AGENT — Starting")
        print(f"  Goal: {self.goal}")
        print(f"  Budget: {MAX_AGENT_STEPS} steps, {AGENT_TIMEOUT_SEC}s")
        print(f"{'─'*60}\n")

        try:
            self.models.ensure_chat()
            while not self.finished and self.step_count < MAX_AGENT_STEPS:
                elapsed = time.time() - self.start_time
                if elapsed > AGENT_TIMEOUT_SEC:
                    print(f"  ⏰ Timeout ({AGENT_TIMEOUT_SEC}s). Producing answer...")
                    break
                if self.consecutive_failures >= CONSECUTIVE_FAIL_LIMIT:
                    print(f"  ❌ Too many failures. Producing answer...")
                    break

                self.step_count += 1
                print(f"  ── Step {self.step_count}/{MAX_AGENT_STEPS} ({elapsed:.0f}s elapsed) ──")

                action = self._get_next_action()
                if action is None:
                    self.consecutive_failures += 1
                    print("  ❌ Model failed to output JSON. Retrying...")
                    self.scratchpad.append({
                        'step': str(self.step_count), 'thought': '(Model produced invalid output)',
                        'action': 'invalid_format', 'input': 'N/A',
                        'observation': 'ERROR: Your previous response was rejected. You MUST output ONLY a valid JSON object starting with { and ending with }. Do NOT write conversational text like "Opening Instagram". ONLY JSON.',
                    })
                    continue

                action_name = action.get('action', '')
                action_input = action.get('input', {})
                thought = action.get('thought', '')

                if thought: print(f"  💭 {thought[:120]}")
                print(f"  🔧 {action_name}: {json.dumps(action_input)[:100]}")

                if action_name == 'done':
                    self.final_answer = action_input.get('answer', '')
                    self.finished = True
                    print(f"  ✅ Agent finished.")
                    break

                observation = self._execute_action(action_name, action_input)
                obs_compressed = self._compress_observation(observation)
                print(f"  👁 {obs_compressed[:150]}...")

                # Track current site for memory
                if action_name == 'browser_open':
                    url = action_input.get('url', '')
                    try:
                        from urllib.parse import urlparse
                        self._current_site = urlparse(url if '://' in url else 'https://'+url).netloc
                    except Exception:
                        self._current_site = url[:30]

                # Learn from failures
                if isinstance(observation, str) and 'ERROR' in observation:
                    self.browsing_memory.remember_failure(
                        self._current_site, action_name,
                        observation[:200]
                    )

                self.scratchpad.append({
                    'step': str(self.step_count), 'thought': thought,
                    'action': action_name, 'input': json.dumps(action_input)[:200],
                    'observation': obs_compressed,
                })
                self.consecutive_failures = 0

                if len(self.scratchpad) >= 5 and len(self.scratchpad) % 3 == 0:
                    self._summarize_scratchpad()
                self._persist_state()

            if not self.finished:
                self.final_answer = self._synthesize_final_answer()
            else:
                # Remember success pattern
                self.browsing_memory.remember_success(
                    self._current_site, self.goal, self.scratchpad
                )

        except KeyboardInterrupt:
            print("\n  ⚠ Agent interrupted. Producing partial answer...")
            self.final_answer = self._synthesize_final_answer()
        except Exception as e:
            logger.exception("TaskAgent error")
            self.final_answer = f"Agent encountered an error: {e}"
        finally:
            self.browser.close()  # Disconnects from CDP, keeps Chrome open
            self._cleanup_state()

        elapsed = time.time() - self.start_time
        print(f"\n{'─'*60}")
        print(f"  🤖 TASK AGENT — Complete")
        print(f"  Steps: {self.step_count} | Time: {elapsed:.1f}s")
        print(f"  Stored values: {list(self.store.keys())}")
        print(f"{'─'*60}\n")
        return self.final_answer

    def _get_next_action(self) -> Optional[Dict]:
        messages = self._build_react_messages()
        result = self.models.chat_complete_json(messages, num_predict=300, temperature=0.2)
        if result and 'action' in result:
            return result
        raw = self.models.chat_complete(messages, num_predict=300, temperature=0.2)
        return self._parse_action_from_text(raw)

    def _build_react_messages(self) -> List[Dict]:
        tools_desc = "\n".join(
            f"  - {t['name']}: {t['desc']} Input: {json.dumps(t['input'])}"
            for t in AGENT_TOOLS
        )
        scratchpad_text = self._format_scratchpad()
        store_text = ""
        if self.store:
            store_lines = [f"  {k} = {v}" for k, v in self.store.items()]
            store_text = "\nSTORED VALUES:\n" + "\n".join(store_lines)

        # Get browsing memory tips
        memory_tips = ''
        if self._current_site:
            memory_tips = self.browsing_memory.get_tips(self._current_site, self.goal)
        if not memory_tips:
            # Try to guess site from goal
            for site in ['instagram', 'gmail', 'youtube', 'twitter', 'facebook', 'amazon']:
                if site in self.goal.lower():
                    memory_tips = self.browsing_memory.get_tips(site, self.goal)
                    break
        memory_section = f"\nLEARNED FROM PREVIOUS ATTEMPTS:\n{memory_tips}" if memory_tips else ''

        system = f"""You are an autonomous browser agent controlling the user's REAL Chrome browser.
The user can SEE everything you do. Complete their goal step by step.

CRITICAL RULES:
1. You MUST output ONLY a single valid JSON object.
2. DO NOT output any conversational text, pleasantries, or explanations outside the JSON.
3. Your JSON must have exactly this format: {{"thought": "...", "action": "action_name", "input": {{...}}}}
4. Do not hallucinates steps. You must literally execute "browser_open" or "browser_type" one at a time.
5. If something fails, try a DIFFERENT selector or approach — do NOT repeat the same failed action.
6. The user is likely already logged into sites like Instagram, Gmail, etc.
7. For messaging apps: look for DM/message icons, search for the person, then type and send.
8. Wait after navigation for pages to load (use browser_wait).
9. All files are relative to: {self.project_root}
{memory_section}

AVAILABLE ACTIONS:
{tools_desc}

EXAMPLES:
1. {{"thought": "I need to open Instagram DMs", "action": "browser_open", "input": {{"url": "https://www.instagram.com/direct/inbox/"}}}}
2. {{"thought": "Let me search for the person", "action": "browser_click", "input": {{"text": "Search"}}}}
3. {{"thought": "Type the person's name", "action": "browser_type", "input": {{"selector": "input[placeholder*='Search']", "text": "aditya"}}}}
4. {{"thought": "Message sent successfully", "action": "done", "input": {{"answer": "Sent a joke to Aditya on Instagram!"}}}}
5. {{"thought": "I need to find iPhone 15 prices", "action": "web_search", "input": {{"query": "iPhone 15 price"}}}}
6. {{"thought": "Found the price, saving it", "action": "store", "input": {{"key": "amazon_price", "value": "$799"}}}}
"""

        user_msg = f"""GOAL: {self.goal}

PROGRESS SO FAR:
{scratchpad_text if scratchpad_text else "(just started — no actions taken yet)"}
{store_text}

Steps remaining: {MAX_AGENT_STEPS - self.step_count}
What is your next action? Output ONLY the JSON object."""

        return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user_msg}]

    def _format_scratchpad(self) -> str:
        if not self.scratchpad: return ""
        lines = []
        total_chars = 0
        for entry in self.scratchpad:
            line = f"Step {entry['step']}: [{entry['action']}] → {entry['observation'][:300]}"
            if total_chars + len(line) > MAX_SCRATCHPAD_CHARS:
                lines.insert(0, "... (earlier steps summarized) ...")
                break
            lines.append(line)
            total_chars += len(line)
        return "\n".join(lines)

    def _execute_action(self, action: str, params: Dict) -> str:
        try:
            if action == 'browser_open':
                url = params.get('url', '')
                if not url.startswith(('http://', 'https://')): url = 'https://' + url
                return self.browser.open_url(url)
            elif action == 'browser_type':
                return self.browser.type_text(params.get('selector', 'input'), params.get('text', ''))
            elif action == 'browser_click':
                return self.browser.click(params.get('text', ''))
            elif action == 'browser_click_selector':
                return self.browser.click_selector(params.get('selector', ''))
            elif action == 'browser_extract':
                return self.browser.extract(params.get('goal', ''))
            elif action == 'browser_scroll':
                return self.browser.scroll_down()
            elif action == 'browser_back':
                return self.browser.go_back()
            elif action == 'browser_wait':
                return self.browser.wait_for(
                    params.get('selector', ''),
                    params.get('seconds', 3)
                )
            elif action == 'browser_login':
                return self.browser.login(
                    params.get('domain', ''),
                    params.get('username', ''),
                    params.get('password', '')
                )
            elif action == 'browser_screenshot':
                return self.browser.screenshot()
            elif action == 'web_search':
                return self._do_web_search(params.get('query', ''))
            elif action == 'file_write':
                return self._do_file_write(params.get('path', ''), params.get('content', ''))
            elif action == 'file_append':
                return self._do_file_append(params.get('path', ''), params.get('content', ''))
            elif action == 'file_read':
                return self._do_file_read(params.get('path', ''))
            elif action == 'store':
                key = params.get('key', f'val_{len(self.store)}')
                value = params.get('value', '')
                self.store[key] = value
                return f"Stored: {key} = {value}"
            else:
                return f"ERROR: Unknown action \'{action}\'."
        except Exception as e:
            self.consecutive_failures += 1
            logger.warning("Action '%s' failed: %s", action, e)
            return f"ERROR: {action} failed — {str(e)[:200]}"

    def _do_web_search(self, query: str) -> str:
        try:
            results = search_duckduckgo(query, max_results=5)
            if not results: return f"No results found for '{query}'."
            lines = []
            for r in results:
                title = r.get('title', '?')
                url = r.get('url', '')
                snippet = r.get('body', r.get('snippet', ''))[:150]
                lines.append(f"• {title}\n  {url}\n  {snippet}")
            return "\n".join(lines)
        except Exception as e:
            return f"Search error: {e}"

    def _do_file_write(self, path: str, content: str) -> str:
        if not path: return "ERROR: No file path provided."
        abs_path = os.path.join(self.project_root, path)
        try:
            os.makedirs(os.path.dirname(abs_path) or '.', exist_ok=True)
            with open(abs_path, 'w', encoding='utf-8') as f: f.write(content)
            return f"Written {len(content)} chars to {path}."
        except Exception as e:
            return f"ERROR writing {path}: {e}"

    def _do_file_append(self, path: str, content: str) -> str:
        if not path: return "ERROR: No file path provided."
        abs_path = os.path.join(self.project_root, path)
        try:
            os.makedirs(os.path.dirname(abs_path) or '.', exist_ok=True)
            with open(abs_path, 'a', encoding='utf-8') as f: f.write(content)
            return f"Appended {len(content)} chars to {path}."
        except Exception as e:
            return f"ERROR appending to {path}: {e}"

    def _do_file_read(self, path: str) -> str:
        if not path: return "ERROR: No file path provided."
        abs_path = os.path.join(self.project_root, path)
        try:
            with open(abs_path, 'r', encoding='utf-8', errors='replace') as f: content = f.read()
            if len(content) > MAX_OBSERVATION_CHARS:
                content = content[:MAX_OBSERVATION_CHARS] + "\n... (truncated)"
            return content or "(file is empty)"
        except FileNotFoundError:
            return f"File not found: {path}"
        except Exception as e:
            return f"ERROR reading {path}: {e}"

    def _compress_observation(self, obs: str) -> str:
        if len(obs) <= MAX_OBSERVATION_CHARS: return obs
        head = int(MAX_OBSERVATION_CHARS * 0.6)
        tail = int(MAX_OBSERVATION_CHARS * 0.2)
        return obs[:head].rstrip() + "\n... (content trimmed) ...\n" + obs[-tail:].lstrip()

    def _summarize_scratchpad(self):
        if len(self.scratchpad) <= 4: return
        to_summarize = self.scratchpad[:-3]
        kept = self.scratchpad[-3:]
        summary_parts = [f"Step {e['step']}: {e['action']} → {e['observation'][:80]}" for e in to_summarize]
        summary_entry = {
            'step': '0', 'thought': 'Summary of earlier steps',
            'action': 'summary', 'input': '',
            'observation': "SUMMARY: " + " | ".join(summary_parts),
        }
        summary_entry['observation'] = summary_entry['observation'][:MAX_OBSERVATION_CHARS]
        self.scratchpad = [summary_entry] + kept

    def _synthesize_final_answer(self) -> str:
        if not self.scratchpad: return "I couldn't complete the task — no data was gathered."
        store_text = "\nStored values: " + json.dumps(self.store) if self.store else ""
        observations = "\n".join(
            f"Step {e['step']} [{e['action']}]: {e['observation'][:400]}" for e in self.scratchpad
        )
        prompt = f"""You were working on this goal: {self.goal}

Here is everything you observed:
{observations}
{store_text}

Now produce the FINAL ANSWER. Be concise and directly answer the goal.
If you have incomplete data, report what you found and what's missing."""

        self.models.ensure_chat()
        return self.models.chat_complete(
            [{'role': 'system', 'content': "You are producing a final answer from an agent's observations. Be direct, factual, and structured. No tool calls."},
             {'role': 'user', 'content': prompt}],
            num_predict=1024, temperature=0.3,
        )

    def _parse_action_from_text(self, text: str) -> Optional[Dict]:
        match = re.search(r'\{[\s\S]*?"action"[\s\S]*?\}', text)
        if match:
            try:
                data = json.loads(match.group(0))
                if 'action' in data: return data
            except json.JSONDecodeError: pass
        for m in re.finditer(r'\{[^{}]*"action"\s*:\s*"(\w+)"[^{}]*\}', text):
            try: return json.loads(m.group(0))
            except json.JSONDecodeError: continue
        logger.warning("Could not parse action from: %s", text[:200])
        return None

    def _persist_state(self):
        state = {
            'goal': self.goal, 'step_count': self.step_count,
            'store': self.store, 'scratchpad': self.scratchpad[-5:],
            'finished': self.finished, 'timestamp': datetime.now().isoformat(),
        }
        try:
            with open(self._state_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug("State persist failed: %s", e)

    def _cleanup_state(self):
        try:
            if os.path.exists(self._state_path): os.remove(self._state_path)
        except Exception: pass


# ─── Task Detection ──────────────────────────────────────────
def detect_agent_task(text: str) -> bool:
    """Detect if user input requires the autonomous agent."""
    text_lower = text.lower()

    # Smart intent detection first
    intent = SmartIntentClassifier.classify(text)
    if intent:
        return True

    multi_step_patterns = [
        r'compare.*(?:price|cost).*(?:on|from|between)',
        r'(?:find|get|extract|scrape).*(?:from|on)\s+\w+.*(?:save|write|append|store)',
        r'(?:go to|visit|open)\s+\w+.*(?:and|then).*(?:find|get|extract|search|type)',
        r'search.*on\s+(?:amazon|flipkart|google|ebay|youtube)',
        r'(?:fill|submit)\s+(?:a |the )?form',
        r'(?:log\s*in|sign\s*in|sign\s*up)',
        r'(?:download|fetch).*(?:save|write)',
        r'(?:check|look\s*up).*(?:on|at)\s+(?:https?://|www\.)',
        r'(?:browse|navigate).*(?:and|then)',
        r'(?:play|watch).*(?:youtube|video|song)',
        r'(?:check|read|open).*(?:email|gmail|inbox)',
        r'(?:open|go\s+to)\s+\w+\.\w+',
    ]
    for pattern in multi_step_patterns:
        if re.search(pattern, text_lower): return True
    action_verbs = ['search', 'find', 'open', 'go to', 'extract', 'save',
                    'write', 'compare', 'check', 'get', 'download', 'click',
                    'play', 'watch', 'login', 'log in', 'sign in']
    if sum(1 for v in action_verbs if v in text_lower) >= 2: return True
    if re.search(r'https?://\S+', text) and any(
        v in text_lower for v in ['and', 'then', 'save', 'extract', 'find']
    ): return True
    return False



# ═══════════════════════════════════════════════════════════════
# Main Assistant
# ═══════════════════════════════════════════════════════════════

class AIAssistant:

    def __init__(self):
        self.models = ModelManager()
        self.conv_mgr = ConversationManager()
        self.memory = MemoryManager()
        self.conv_id: Optional[str] = None
        self.history: List[Dict] = []
        self.project_root: str = os.getcwd()
        self.code_agent: Optional[CodeAgent] = None
        self.analyze_log_path: Optional[str] = None
        self.last_research_topic: Optional[str] = None

        self._stop = False
        self._ctrl_c_count = 0
        self._ctrl_c_time = 0.0
        signal.signal(signal.SIGINT, self._on_ctrl_c)

    def _on_ctrl_c(self, sig, frame):
        now = time.time()
        if now - self._ctrl_c_time < 2.0:
            self._ctrl_c_count += 1
        else:
            self._ctrl_c_count = 1
        self._ctrl_c_time = now

        if self._ctrl_c_count >= 2:
            print("\n👋 Exiting.")
            self._stop = True
            sys.exit(0)
        else:
            if self.conv_id and self.history:
                try:
                    self.conv_mgr.save_message(
                        self.conv_id, 'system', '[session interrupted]'
                    )
                except Exception:
                    pass
            print("\n💾 Saved. (Ctrl+C again to exit)")

    def route_intent(self, text: str) -> Optional[Dict]:
        """Classifies plain text to specific tools: /search, /news, /research, /code, /agent, /analyze, or CHAT."""
        prompt = [
            {'role': 'system', 'content': (
                "You are an INTENT ROUTER. Decide which internal tool best satisfies the user query.\n"
                "Output MUST be valid JSON. Schema:\n"
                "{\"command\": \"/search\"|\"/news\"|\"/research\"|\"/code\"|\"/agent\"|\"/analyze\"|\"CHAT\", \"param\": \"extracted arguments without the command\"}\n"
                "Use /search for quick web lookups or factual questions.\n"
                "Use /news for current events, headlines, or trending topics.\n"
                "Use /research for deep learning/information gathering.\n"
                "Use /code for coding tasks, writing scripts, or fixing code.\n"
                "Use /agent for browser automation tasks (open websites, send messages, fill forms, interact with web apps).\n"
                "Use /analyze for reviewing or auditing files.\n"
                "Use CHAT for conversational, abstract, greetings, or unknown intents."
            )},
            {'role': 'user', 'content': f"Query: {text}\n"}
        ]
        try:
            self.models.ensure_chat()
            res = self.models.chat_complete_json(prompt, num_predict=150, temperature=0.1)
            return res
        except Exception:
            return None

    def _trimmed_history(self) -> List[Dict]:
        total = 0
        trimmed = []
        for msg in reversed(self.history):
            t = count_tokens(msg['content'])
            if total + t > MAX_HISTORY_TOKENS:
                break
            trimmed.insert(0, msg)
            total += t
        return trimmed

    def _truncate_text(self, text: Any, limit: int) -> str:
        text = '' if text is None else str(text)
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "\n... (truncated)"

    def _is_tool_result_message(self, content: str) -> bool:
        return content.startswith(TOOL_RESULT_PREFIX) or content.startswith('[Tool result]')

    def _is_tool_call_message(self, msg: Dict) -> bool:
        return msg.get('role') == 'assistant' and bool(self._extract_tool_calls(msg.get('content', '')))

    def _is_context_eligible_message(self, msg: Dict, include_tool_results: bool = False) -> bool:
        role = msg.get('role', '')
        content = msg.get('content', '')
        if not content or not content.strip() or role == 'system':
            return False
        if self._is_tool_call_message(msg):
            return False
        if not include_tool_results and self._is_tool_result_message(content):
            return False
        return True

    def _filtered_history(self, include_tool_results: bool = False) -> List[Dict]:
        return [
            msg for msg in self.history
            if self._is_context_eligible_message(msg, include_tool_results=include_tool_results)
        ]

    def _last_real_user_message(self) -> str:
        for msg in reversed(self.history):
            if msg.get('role') != 'user':
                continue
            content = msg.get('content', '')
            if self._is_tool_result_message(content):
                continue
            return content
        return ''

    def _memory_add_allowed(self) -> bool:
        # User requested to "remember everything", so we trust the model's decision to call memory_add
        return True

    # ⚙️ CHANGE 5: Reduced context — only relevant messages via semantic search
    def _get_relevant_context(self, query: str, top_k: int = 4) -> List[Dict]:
        filtered_history = self._filtered_history(include_tool_results=False)

        if not _SQLITE_VEC_OK or not self.conv_id or len(filtered_history) < 6:
            return filtered_history[-6:] if filtered_history else self._trimmed_history()

        try:
            query_emb = self.models.embed(query)
        except Exception as e:
            # ⚙️ STABILITY FALLBACK: If embedding fails again after eviction, skip RAG
            logger.warning("VRAM CRITICAL: Skipping Global Recall for this turn (%s)", e)
            return filtered_history[-6:] if filtered_history else self._trimmed_history()

        try:
            with sqlite3.connect(DB_PATH) as con:
                con.enable_load_extension(True)
                sqlite_vec.load(con)

                query_emb_bytes = sqlite_vec.serialize_float32(query_emb)

                # ⚙️ V4 NEXUS: Recency-Weighted Search
                # We combine semantic distance (80%) with time decay (20%)
                rows = con.execute(
                    """
                    SELECT m.role, m.content, m.timestamp
                    FROM vec_messages v
                    JOIN messages m ON v.rowid = m.id
                    ORDER BY 
                        (vec_distance_cosine(v.embedding, ?) * 0.8) + 
                        (COALESCE(julianday('now') - julianday(m.timestamp), 365) / 365.0 * 0.2)
                    LIMIT ?
                    """,
                    (query_emb_bytes, max(top_k * 4, top_k)),
                ).fetchall()
                retrieved = [
                    {
                        'role': r[0],
                        'content': f"[Memory from {r[2][:10] if r[2] else 'unknown date'}]: {r[1]}"
                    } for r in rows
                ]
        except Exception as e:
            logger.warning("Vector search failed: %s — falling back", e)
            return filtered_history[-6:] if filtered_history else self._trimmed_history()

        retrieved = [
            msg for msg in retrieved
            if self._is_context_eligible_message(msg, include_tool_results=False)
        ][:top_k]

        recent = filtered_history[-3:]
        seen = set()
        combined: List[Dict] = []
        for msg in retrieved + recent:
            key = (msg['role'], msg['content'][:80])
            if key not in seen:
                seen.add(key)
                combined.append(msg)

        total = 0
        capped: List[Dict] = []
        for msg in reversed(combined):
            t = count_tokens(msg['content'])
            if total + t > MAX_HISTORY_TOKENS:
                break
            capped.insert(0, msg)
            total += t

        logger.debug(
            "Semantic context: %d retrieved + %d recent → %d final",
            len(retrieved), len(recent), len(capped),
        )
        return capped or (filtered_history[-6:] if filtered_history else self._trimmed_history())

    def _new_conversation(self) -> str:
        self.conv_id = self.conv_mgr.new_conversation()
        self.history = []
        logger.info("New conversation: %s", self.conv_id)
        return self.conv_id

    def _load_conversation(self, cid: str):
        rows = self.conv_mgr.list_conversations()
        names = {r[0]: r[1] for r in rows}
        self.conv_id = cid
        self.history = self.conv_mgr.load_messages(cid)
        name = names.get(cid, cid)
        logger.info("Loaded conversation %s", cid)
        print(f"\n📂 Loaded: '{name}' ({len(self.history)} messages)")

    def _record(self, role: str, content: str):
        self.history.append({'role': role, 'content': content})
        if self.conv_id:
            self.conv_mgr.save_message(self.conv_id, role, content)

    # ⚙️ CHANGE 6+8: Strict final-answer prompt with explicit constraints
    def _build_final_answer_messages(self, user_input: str) -> List[Dict]:
        final_system = self._build_system_prompt(
            "FINAL ANSWER MODE — STRICT RULES:\n"
            "- You MUST answer the original user request NOW\n"
            "- Output format: plain text answer, well-structured\n"
            "- Do NOT output JSON\n"
            "- Do NOT call tools\n"
            "- Do NOT ask for confirmation\n"
            "- Use tool results from conversation as your primary source\n"
            "- If a tool result contains JSON/raw data, parse and present key facts\n"
            "- If the user's premise is wrong, correct it directly\n"
            f"- Original request: {user_input!r}\n"
            "- Failure to follow these rules = system crash"
        )
        return [{'role': 'system', 'content': final_system}] + self._filtered_history(include_tool_results=True)

    # ⚙️ CHANGE 4: Self-correction loop for final answers
    def _run_final_answer_pass(self, user_input: str, depth: int = 0) -> str:
        if depth > 5:
            return "Too many tool calls, stopping."

        self.models.ensure_chat()
        final_response = self.models.chat_complete(
            self._build_final_answer_messages(user_input),
            stream=False,
            temperature=DEFAULT_REVIEW_TEMP,
        )
        self._record('assistant', final_response)

        more_calls = self._extract_tool_calls(final_response)
        if not more_calls:
            return final_response

        for tc in more_calls:
            print(f"  🔧 Executing tool: {tc['tool']}…")
            result = self._truncate_text(
                self._execute_tool(tc['tool'], tc['params']),
                MAX_TOOL_RESULT_CHARS,
            )
            self._record('user', f"[Tool result from {tc['tool']}]\n{result}")

        return self._run_final_answer_pass(user_input, depth + 1)

    def _get_agent(self) -> CodeAgent:
        if self.code_agent is None:
            print(f"  🔍 Building dependency graph for: {self.project_root}")
            self.code_agent = CodeAgent(self.project_root)
            print(f"  ✅ Graph ready: {len(self.code_agent.file_list)} files indexed.")
        return self.code_agent

    # ⚙️ CHANGE 8: System prompt with strict role + tool definitions as JSON
    def _build_system_prompt(self, extra: str = '') -> str:
        base = BASE_SYSTEM_PROMPT.format(date=datetime.now().strftime('%Y-%m-%d'))
        mem_block = self.memory.build_context_block()
        tools_json = json.dumps(TOOL_DEFINITIONS, indent=2)
        parts = []
        if mem_block:
            parts.append(mem_block)
        parts.append(base)
        parts.append(SELF_AWARENESS.format(
            project_root=self.project_root,
            tools_json=tools_json,
        ))
        if extra:
            parts.append(extra)
        return "\n\n".join(parts)

    def _detect_code_intent(self, text: str) -> Optional[Tuple[str, str]]:
        ext_pattern = r'\b([\w./\\-]+\.(?:py|js|css|html|c|cpp|h|json|java|go|rs|rb|php|ts|jsx|tsx|md))\b'
        m = re.search(ext_pattern, text, re.IGNORECASE)
        if m:
            fname = m.group(1)
            abs_path = os.path.join(self.project_root, fname)
            if os.path.exists(abs_path):
                code_words = ['fix', 'add', 'change', 'update', 'modify', 'edit',
                              'remove', 'delete', 'refactor', 'bug', 'error',
                              'implement', 'create', 'write', 'make']
                if any(w in text.lower() for w in code_words):
                    return fname, text
        return None

    # ═════════════════════════════════════════════════════════
    # Main menu
    # ═════════════════════════════════════════════════════════

    def _show_menu(self):
        print(SEP)
        print("  🤖 AI Assistant v3")
        print(SEP)
        convs = self.conv_mgr.list_conversations()
        if convs:
            print("  📁 Recent conversations:")
            for i, (cid, name, upd) in enumerate(convs[:5], 1):
                print(f"    {i}. {name[:50]} ({upd[:10]})")
        print()
        print("  n – New conversation")
        print("  l – Load conversation")
        print("  q – Quit")
        print()

    def _run_menu(self) -> bool:
        self._show_menu()
        convs = self.conv_mgr.list_conversations()
        choice = _input("  Select: ").lower()
        print()

        if choice in ('q', 'quit', '/quit'):
            return False
        if choice in ('n', 'new', '/new'):
            self._new_conversation()
            return True
        if choice in ('l', 'load', '/load'):
            if not convs:
                print("  📭 No saved conversations found.")
                return True
            for i, (cid, name, upd) in enumerate(convs, 1):
                print(f"    {i}. [{cid}] {name}")
            idx = _input("  Enter number: ")
            try:
                self._load_conversation(convs[int(idx) - 1][0])
            except (ValueError, IndexError):
                print("  ❌ Invalid selection — starting new conversation.")
                self._new_conversation()
            return True
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(convs):
                self._load_conversation(convs[idx][0])
                return True
        self._new_conversation()
        return True

    # ═════════════════════════════════════════════════════════
    # Top-level run loop
    # ═════════════════════════════════════════════════════════

    def run(self):
        print_banner()

        if not self._run_menu():
            return

        if not self.conv_id:
            self._new_conversation()

        try:
            warn_resources(5.0)
            self.models.ensure_chat()
        except Exception as e:
            print(f"  ❌ [ERROR] Could not load chat model: {e}")
            print("  Make sure Ollama is running: ollama serve")
            return

        print(SEP)
        label = "Resumed Chat" if self.history else "New Chat"
        print(f"  💬 {label}")
        print(SEP)

        mem = self.memory.get_all()
        if mem:
            print("  🧠 Memory active:")
            for k, v in mem.items():
                print(f"    {k} = {v}")
            print()

        if _SQLITE_VEC_OK:
            print("  🧠 Semantic memory: ACTIVE (sqlite-vec + nomic-embed-text)")
        else:
            print("  🧠 Semantic memory: OFF (install with: pip install sqlite-vec)")
            print("     Then pull model: ollama pull nomic-embed-text")
        print()

        print_help()

        while not self._stop:
            try:
                raw = _input("You: ")
            except (EOFError, KeyboardInterrupt):
                break
            if not raw:
                continue

            cmd = raw.lower()

            if STOP_SIGNAL.search(raw):
                print("  🛑 Stop signal received.")
                continue

            if cmd.strip() == 'again' and self.last_research_topic:
                self._cmd_research(self.last_research_topic)
                continue

            if cmd in ('/back', '/menu'):
                if not self._run_menu():
                    break
                continue

            if cmd in ('/quit', '/exit', 'exit', 'quit'):
                self._cleanup()
                print("  👋 Exiting.")
                break

            if cmd in ('/help', 'help', '?'):
                print_help()
                continue

            if cmd == '/new':
                self._new_conversation()
                print("  🆕 New conversation started.")
                continue

            if cmd == '/history':
                self._cmd_history()
                continue

            if cmd.startswith('/load'):
                self._cmd_load(raw)
                continue

            if cmd == '/chat':
                try:
                    self.models.ensure_chat()
                    print("  💬 Switched to chat mode.")
                except Exception as e:
                    print(f"  ❌ [ERROR] {e}")
                continue

            if cmd == '/refresh':
                self._cmd_refresh()
                continue

            if cmd.startswith('/cd '):
                self._cmd_cd(raw[4:])
                continue

            if cmd.startswith('/memory'):
                self._cmd_memory(raw)
                continue

            if cmd.startswith('/tempmem'):
                rest = raw[8:].strip()
                if rest:
                    print(self.memory.tempmem_start(rest))
                else:
                    print(self.memory.tempmem_status())
                continue

            if cmd.startswith('/endtempmem'):
                print(self.memory.tempmem_end())
                continue

            if cmd.startswith('/credential'):
                self._cmd_credential(raw)
                continue

            if cmd.startswith('/search '):
                self._cmd_search(raw[8:])
                continue

            if cmd.startswith('/research '):
                self._cmd_research(raw[10:])
                continue

            if cmd.startswith('/news'):
                self._cmd_news(raw[5:].strip())
                continue

            if cmd.startswith('/task ') or (cmd.startswith('/agent ') and ' ' in raw[7:].strip()):
                instruction = raw.split(None, 1)[1] if ' ' in raw else ''
                self._cmd_task_agent(instruction)
                continue

            if cmd.startswith('/agent '):
                self._cmd_agent(raw[7:])
                continue

            if cmd.startswith('/analyze') or cmd.startswith('/analyse'):
                self._cmd_analyze(raw.split(None, 1)[1] if ' ' in raw else '')
                continue

            if cmd.startswith('/undo'):
                self._cmd_undo(raw[5:].strip())
                continue

            if cmd.startswith('/agentcode'):
                self._cmd_agentcode(raw[10:].lstrip())
                continue

            if cmd.startswith('/code'):
                self._cmd_code(raw[5:].lstrip())
                continue

            code_intent = self._detect_code_intent(raw)
            if code_intent:
                fname, instruction = code_intent
                print(f"  🔧 Detected code request for {fname}. Running /code…")
                self._cmd_code(f"{fname} {instruction}")
                continue

            # ─── Smart browser intent detection ───
            intent = SmartIntentClassifier.classify(raw)
            if intent:
                intent_name = intent['intent']
                print(f"  🧠 Detected intent: {intent_name}")

                if intent_name == 'save_credentials':
                    print("  Use: /credential save <domain> <username> <password>")
                    continue

                confirm = _input(f"  Run smart browser action '{intent_name}'? (y/n): ").lower()
                if confirm == 'y':
                    with BrowserManager(headless=False) as browser:
                        handler = SmartBrowserHandler(
                            browser=browser,
                            models=self.models,
                            memory=self.memory,
                        )
                        method = getattr(handler, intent['handler'], None)
                        if method:
                            self._record('user', raw)
                            result = method(intent['captures'], intent['full_text'])
                            self._record('assistant', result)
                            print(f"\nAssistant: {result}\n")
                    continue

            # Auto-detect multi-step tasks
            if detect_agent_task(raw):
                print(f"  🤖 Detected multi-step task. Launching agent...")
                confirm = _input("  Use autonomous agent? (y/n): ").lower()
                if confirm == 'y':
                    self._cmd_task_agent(raw)
                    continue

            self._cmd_chat(raw)

    # ═════════════════════════════════════════════════════════
    # /history
    # ═════════════════════════════════════════════════════════

    def _cmd_history(self, return_string: bool = False):
        convs = self.conv_mgr.list_conversations()
        if not convs:
            msg = "  📭 No saved conversations."
            if return_string:
                return msg
            print(msg)
            return
        lines = [""]
        for i, (cid, name, upd) in enumerate(convs, 1):
            m = '▶' if cid == self.conv_id else ' '
            lines.append(f"  {m} {i:2}. [{cid}] {name[:50]:50s} {upd[:16]}")
        lines.append("")
        output = "\n".join(lines)
        if return_string:
            return output
        print(output)

    # ═════════════════════════════════════════════════════════
    # /load
    # ═════════════════════════════════════════════════════════

    def _cmd_load(self, raw: str, return_string: bool = False):
        parts = raw.split(None, 1) if isinstance(raw, str) and raw.strip() else ['']
        convs = self.conv_mgr.list_conversations()
        cid_to_load = parts[1].strip() if len(parts) == 2 and parts[1].strip() else None
        if cid_to_load:
            self._load_conversation(cid_to_load)
            msg = f"Loaded conversation {cid_to_load}"
            if return_string:
                return msg
        else:
            if not convs:
                msg = "  📭 No saved conversations."
                if return_string:
                    return msg
                print(msg)
                return
            if return_string:
                return self._cmd_history(return_string=True)
            self._cmd_history()
            idx = _input("  Enter number: ")
            try:
                self._load_conversation(convs[int(idx) - 1][0])
            except (ValueError, IndexError):
                print("  ❌ Invalid selection.")

    # ═════════════════════════════════════════════════════════
    # /cd
    # ═════════════════════════════════════════════════════════

    def _cmd_cd(self, path: str, return_string: bool = False):
        path = path.strip().strip('"').strip("'")
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path):
            self.project_root = abs_path
            self.code_agent = None
            logger.info("Project root set to: %s", abs_path)
            msg = f"📁 Project root set to: {abs_path}"
            if return_string:
                return msg
            print(f"\n{msg}")
        else:
            msg = f"❌ [ERROR] Directory not found: {abs_path}"
            if return_string:
                return msg
            print(f"  {msg}")

    # ═════════════════════════════════════════════════════════
    # /refresh
    # ═════════════════════════════════════════════════════════

    def _cmd_refresh(self, return_string: bool = False):
        self.code_agent = CodeAgent(self.project_root)
        msg = f"  🔄 Rebuilt dependency graph — {len(self.code_agent.file_list)} files indexed."
        if return_string:
            return msg
        print("  🔄 Rebuilding dependency graph…")
        print(f"  ✅ Done — {len(self.code_agent.file_list)} files indexed.")

    # ═════════════════════════════════════════════════════════
    # /memory
    # ═════════════════════════════════════════════════════════

    def _cmd_memory(self, raw: str):
        parts = raw.strip().split(None, 2)
        if len(parts) < 2:
            print("  Usage: /memory add <info> | /memory recall | /memory clear")
            return
        sub = parts[1].lower()
        if sub == 'add':
            if len(parts) < 3:
                print("  Usage: /memory add <info>")
                return
            print(self.memory.add(parts[2]))
        elif sub in ('recall', 'show', 'list'):
            print(self.memory.recall())
        elif sub == 'clear':
            confirm = _input("  ⚠️ Clear ALL persistent memory? (y/n): ").lower()
            if confirm == 'y':
                print(self.memory.clear())
            else:
                print("  Cancelled.")
        else:
            print(f"  Unknown sub-command '{sub}'. Use: add | recall | clear")

    # ═════════════════════════════════════════════════════════
    # /undo
    # ═════════════════════════════════════════════════════════

    def _cmd_undo(self, filename: str, return_string: bool = False):
        filename = filename.strip()
        if not filename:
            msg = "  Usage: /undo <filename>"
            if return_string:
                return msg
            print(msg)
            return

        abs_path = os.path.join(self.project_root, filename)
        pattern = abs_path + '.*.bak'
        baks = sorted(glob.glob(pattern), reverse=True)

        if not baks:
            msg = f"  ❌ No backups found for {filename}"
            if return_string:
                return msg
            print(msg)
            return

        if return_string:
            backup = baks[0]
        else:
            print(f"  📂 Found {len(baks)} backup(s):")
            for i, b in enumerate(baks[:5], 1):
                ts = os.path.basename(b).replace(os.path.basename(abs_path) + '.', '').replace('.bak', '')
                print(f"    {i}. {ts}")

            choice = _input("  Restore which? (1 = latest, or number): ").strip()
            try:
                idx = int(choice) - 1 if choice else 0
                backup = baks[idx]
            except (ValueError, IndexError):
                backup = baks[0]

        shutil.copy2(backup, abs_path)
        msg = f"  ✅ Restored {filename} from {os.path.basename(backup)}"
        if self.code_agent:
            self.code_agent.refresh()
        if return_string:
            return msg
        print(msg)

    # ═════════════════════════════════════════════════════════
    # /news
    # ═════════════════════════════════════════════════════════

    def _cmd_news(self, topic: str, return_string: bool = False):
        queries = []
        if topic:
            queries = [f"{topic} news today", f"{topic} Nepal news"]
        else:
            queries = ["top world news today", "Nepal news today",
                       "ekantipur.com news"]

        if not return_string:
            print(f"\n📰 Fetching news{' about ' + topic if topic else ''}…\n")
        all_results = []
        for q in queries:
            try:
                results = search_duckduckgo(q, max_results=3)
                all_results.extend(results)
            except Exception as e:
                logger.warning("News search failed for '%s': %s", q, e)

        if not all_results:
            msg = "  ❌ [ERROR] Could not fetch news."
            if return_string:
                return msg
            print(msg)
            return

        try:
            ek_results = search_duckduckgo("site:ekantipur.com latest news", max_results=2)
            all_results.extend(ek_results)
        except Exception:
            pass

        context = format_results_for_ai("news", all_results)
        news_prompt = (
            f"Here are raw search results:\n{context}\n\n"
            "Format a clean news digest with these 3 sections:\n"
            "📌 TOP GLOBAL NEWS\n"
            "🇳🇵 NEPALI NEWS\n"
            "🔥 INTERESTING / TRENDING\n\n"
            "For each section list 2-3 items. One line per item with source."
            + (f"\n\nFocus on: {topic}" if topic else "")
        )
        # ⚙️ CHANGE 8: Strict role
        system = self._build_system_prompt(
            "You are a STRICT NEWS EDITOR. Output ONLY the formatted digest.\n"
            "Do NOT add commentary. Do NOT add tool calls.\n"
            "Follow the 3-section format exactly."
        )
        try:
            self.models.ensure_chat()
            reply = self.models.chat_complete(
                [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': news_prompt},
                ],
                num_predict=1024, temperature=DEFAULT_CHAT_TEMP,
                stream=(not return_string),
            )
            if return_string:
                return reply
            print()
            self._record('user', f"/news {topic}")
            self._record('assistant', reply)
        except Exception as e:
            logger.exception("News LLM call failed")
            msg = f"  ❌ [ERROR] {e}"
            if return_string:
                return msg
            print(msg)
            for r in all_results[:5]:
                print(f"  • {r.get('title', '?')} — {r.get('url', '')}")

    # ═════════════════════════════════════════════════════════
    # Cleanup
    # ═════════════════════════════════════════════════════════

    def _cmd_credential(self, raw: str):
        """Manage saved credentials."""
        parts = raw.strip().split(None, 4)
        vault = CredentialVault()

        if len(parts) < 2:
            print("  Usage:")
            print("    /credential save <domain> <username> <password>")
            print("    /credential list")
            print("    /credential delete <domain>")
            print("    /credential show <domain>")
            return

        sub = parts[1].lower()

        if sub == 'save':
            if len(parts) < 5:
                print("  Usage: /credential save <domain> <username> <password>")
                return
            domain, username, password = parts[2], parts[3], parts[4]
            print(vault.save_credential(domain, username, password))

        elif sub == 'list':
            domains = vault.list_domains()
            if domains:
                print("  🔐 Saved credentials:")
                for d in domains:
                    print(f"    • {d}")
            else:
                print("  No saved credentials.")

        elif sub == 'delete':
            if len(parts) < 3:
                print("  Usage: /credential delete <domain>")
                return
            print(vault.delete_credential(parts[2]))

        elif sub == 'show':
            if len(parts) < 3:
                print("  Usage: /credential show <domain>")
                return
            cred = vault.get_credential(parts[2])
            if cred:
                print(f"  Domain: {cred['domain']}")
                print(f"  Username: {cred['username']}")
                print(f"  Password: {'*' * len(cred['password'])}")
            else:
                print(f"  No credentials found for {parts[2]}")

        else:
            print(f"  Unknown sub-command '{sub}'")

    def _cleanup(self):
        print("  🧹 Cleaning up…")
        if self.models.current:
            try:
                self.models._evict(self.models.current)
                self.models.current = None
                print("  ✅ Model unloaded.")
            except Exception as e:
                logger.warning("Cleanup evict failed: %s", e)

        try:
            if platform.system() == 'Windows':
                subprocess.run(['taskkill', '/F', '/IM', 'chromium.exe'],
                               capture_output=True, timeout=3)
            else:
                subprocess.run(['pkill', '-f', 'chromium'],
                               capture_output=True, timeout=3)
        except Exception:
            pass


    # ⚙️ CHANGE 1: Structured tool call extraction with strict JSON parsing
    def _extract_tool_calls(self, text: str) -> List[Dict]:
        pattern = r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"params"\s*:\s*(\{[^{}]*\})\s*\}'
        matches = re.findall(pattern, text, re.DOTALL)
        tool_calls = []
        for tool_name, params_str in matches:
            try:
                params = json.loads(params_str)
                tool_calls.append({"tool": tool_name, "params": params})
            except json.JSONDecodeError:
                continue
        return tool_calls

    def _execute_tool(self, tool_name: str, params: Dict) -> str:
        if tool_name == "web_search":
            return self._cmd_search(params.get("query", ""), return_string=True)
        elif tool_name == "news":
            return self._cmd_news(params.get("topic", ""), return_string=True)
        elif tool_name == "research":
            return self._cmd_research(params.get("topic", ""), return_string=True)
        elif tool_name == "browser_fetch":
            url = params.get("url")
            if url:
                return self._browser_fetch(url) or "Could not fetch page."
            return "Missing URL"
        elif tool_name == "code_edit":
            file = params.get("file", "")
            instruction = params.get("instruction", "")
            return self._cmd_code(f"{file} {instruction}", return_string=True)
        elif tool_name == "code_analyze":
            files = params.get("files", [])
            question = params.get("question", "")
            args = " ".join(files) + " " + question
            return self._cmd_analyze(args, return_string=True)
        elif tool_name == "agentcode":
            return self._cmd_agentcode(params.get("request", ""), return_string=True)
        elif tool_name == "memory_add":
            if not self._memory_add_allowed():
                return "Skipped memory_add: user did not explicitly ask to store memory."
            return self.memory.add(params.get("fact", ""))
        elif tool_name == "memory_recall":
            return self.memory.recall()
        elif tool_name == "memory_clear":
            return self.memory.clear()
        elif tool_name == "tempmem_add":
            return self.memory.tempmem_start(params.get("rule", ""))
        elif tool_name == "tempmem_clear":
            return self.memory.tempmem_end()
        elif tool_name == "list_conversations":
            return self._cmd_history(return_string=True)
        elif tool_name == "load_conversation":
            return self._cmd_load(params.get("id", ""), return_string=True)
        elif tool_name == "new_conversation":
            self._new_conversation()
            return "New conversation started."
        elif tool_name == "set_project_root":
            return self._cmd_cd(params.get("path", ""), return_string=True)
        elif tool_name == "refresh_graph":
            return self._cmd_refresh(return_string=True)
        elif tool_name == "undo":
            return self._cmd_undo(params.get("file", ""), return_string=True)
        elif tool_name == "task_agent":
            goal = params.get("goal", "")
            agent = TaskAgent(
                goal=goal, models=self.models,
                memory=self.memory, project_root=self.project_root,
            )
            return agent.run()
        else:
            return f"Unknown tool: {tool_name}"

    # ⚙️ CHANGE 2+3+4+10: Structured phased processing with validation loop
    def _process_with_tools(self, user_input: str, depth: int = 0) -> str:
        """
        PHASE 1: Classify intent (structured JSON)
        PHASE 2: Execute tool if needed
        PHASE 3: Generate final answer
        PHASE 4: Validate answer quality
        """
        if depth > 5:
            return "Too many tool calls, stopping."

        # ── PHASE 1: Intent Classification (JSON) ────────────
        classify_system = self._build_system_prompt(
            "You are a STRICT INTENT CLASSIFIER in an autonomous pipeline.\n"
            "Your ONLY job is to classify the user's intent.\n"
            "Output MUST be valid JSON. No other text.\n"
            "Failure to output valid JSON = system crash.\n\n"
            "Output schema:\n"
            '{"intent": "chat|tool_call", "tool": "tool_name or null", '
            '"params": {...} or null, "reasoning": "one sentence why"}'
        )

        # ⚙️ CHANGE 5: Only pass current message + last 3 relevant messages
        recent = self._filtered_history(include_tool_results=False)[-3:]
        classify_msgs = [
            {'role': 'system', 'content': classify_system},
            *recent,
            {'role': 'user', 'content': user_input},
        ]

        intent = self.models.chat_complete_json(
            classify_msgs,
            num_predict=256,
            temperature=DEFAULT_PLAN_TEMP,
        )

        # Fallback: if JSON parsing failed, try direct chat
        if not intent:
            logger.warning("Intent classification failed — falling back to direct chat")
            return self._direct_chat(user_input)

        logger.info("Intent: %s", json.dumps(intent, ensure_ascii=False)[:200])

        # ── PHASE 2: Execute tool if classified as tool_call ──
        if intent.get('intent') == 'tool_call' and intent.get('tool'):
            tool_name = intent['tool']
            params = intent.get('params') or {}
            print(f"  🔧 Executing tool: {tool_name}…")
            result = self._truncate_text(
                self._execute_tool(tool_name, params),
                MAX_TOOL_RESULT_CHARS,
            )
            self._record('user', f"[Tool result from {tool_name}]\n{result}")

            # ── PHASE 3: Generate answer from tool result ─────
            return self._run_final_answer_pass(user_input, depth)

        # ── PHASE 3: Direct answer (no tool needed) ──────────
        return self._direct_chat(user_input)

    def _direct_chat(self, user_input: str) -> str:
        """Generate a direct answer without tool use."""
        # ⚙️ CHANGE 5: Minimal context — only relevant messages
        context_msgs = self._get_relevant_context(user_input)

        system = self._build_system_prompt(
            "You are a helpful, knowledgeable AI assistant.\n"
            "Rules:\n"
            "- Be concise but thorough\n"
            "- Structure your response with sections if complex\n"
            "- Do NOT output JSON tool calls\n"
            "- Answer directly and completely"
        )
        messages = [{'role': 'system', 'content': system}] + context_msgs

        self.models.ensure_chat()
        response = self.models.chat_complete(
            messages, stream=True, temperature=DEFAULT_CHAT_TEMP,
        )
        return response

    def _reflect(self, user_text: str, assistant_text: str):
        """Autonomous reflection to extract entities, facts and tags."""
        try:
            # ⚙️ STABILITY: Fast, simple reflection prompt
            reflect_prompt = f"""
            Identify New Facts:
            Turn: {user_text} | {assistant_text[:100]}...
            Output JSON: {{"tags":[], "facts":{{}}, "entities":[]}}
            """
            self.models.ensure_chat()
            res = self.models.chat_complete_json(
                [{'role': 'system', 'content': reflect_prompt}],
                num_predict=512, temperature=0.1
            )
            if not res: return

            # 1. Update tags for the last message in conversations.db
            tags_str = ",".join(res.get('tags', []))
            if tags_str and self.conv_id:
                with sqlite3.connect(DB_PATH) as con:
                    # Get the ID of the last message in this conversation
                    last_id = con.execute(
                        "SELECT id FROM messages WHERE conv_id=? ORDER BY id DESC LIMIT 1",
                        (self.conv_id,)
                    ).fetchone()
                    if last_id:
                        con.execute("UPDATE messages SET tags=? WHERE id=?", (tags_str, last_id[0]))

            # 2. Update persistent facts
            facts = res.get('facts', {})
            for k, v in facts.items():
                self.memory._store(k, str(v))

            # 3. Update Knowledge Graph (entities)
            entities = res.get('entities', [])
            for ent in entities:
                if isinstance(ent, list) and len(ent) == 3:
                    self.memory._store_entity(ent[0], ent[1], ent[2])
                    
            logger.info("Reflected on turn: stored %d facts, %d entities", len(facts), len(entities))
        except Exception as e:
            logger.warning("Reflection failed: %s", e)

    # ═════════════════════════════════════════════════════════
    # /task — autonomous task agent
    # ═════════════════════════════════════════════════════════

    def _cmd_task_agent(self, instruction: str):
        """Run the autonomous task agent."""
        instruction = instruction.strip()
        if not instruction:
            print("  Usage: /task <instruction>")
            print("  Example: /task compare iPhone 15 prices on Amazon and Flipkart")
            return

        self._record('user', f"/task {instruction}")

        try:
            agent = TaskAgent(
                goal=instruction, models=self.models,
                memory=self.memory, project_root=self.project_root,
            )
            result = agent.run()

            self._record('assistant', result)
            print(f"\nAssistant: {result}\n")

            if len(self.history) == 2 and self.conv_id:
                self.conv_mgr.rename_conversation(self.conv_id, instruction[:60])
        except Exception as e:
            logger.exception("Task agent failed")
            print(f"  ❌ [ERROR] {e}")

    # ═════════════════════════════════════════════════════════
    # Chat (default mode) — ⚙️ CHANGE 10: Generate → Validate → Fix
    # ═════════════════════════════════════════════════════════

    def _cmd_chat(self, text: str):
        self._record('user', text)

        try:
            final = self._process_with_tools(text)

            # ⚙️ CHANGE 4: Validate response quality
            if not self._is_tool_call_message({'role': 'assistant', 'content': final}):
                self._record('assistant', final)

            print(f"\nAssistant: {final}\n")
            
            # ⚙️ STABILITY: Run reflection sequentially (no threading)
            # This prevents VRAM peaks on 6GB cards
            print("  🧠 Reflecting on memory...", end='', flush=True)
            self._reflect(text, final)
            print(" done.")

            if len(self.history) == 2 and self.conv_id:
                self.conv_mgr.rename_conversation(self.conv_id, text[:60])
        except Exception as e:
            logger.exception("Chat error")
            print(f"  ❌ [ERROR] {e}")

    # ═════════════════════════════════════════════════════════
    # /search
    # ═════════════════════════════════════════════════════════

    def _cmd_search(self, query: str, return_string: bool = False):
        query = query.strip()
        if not query:
            msg = "  Usage: /search <query>"
            if return_string:
                return msg
            print(msg)
            return

        if not return_string:
            print(f"  🔍 Searching: {query} …")
        try:
            results = search_duckduckgo(query, max_results=5)
        except Exception as e:
            msg = f"  ❌ [ERROR] Search failed: {e}"
            if return_string:
                return msg
            print(msg)
            return

        context = format_results_for_ai(query, results)
        if not results:
            msg = "  📭 No results found."
            if return_string:
                return msg
            print(msg)
            return

        prompt = (
            f"The user searched for: '{query}'\n\n"
            f"Search results:\n{context}\n\n"
            "Summarise the key information clearly. Cite source titles where helpful."
        )
        # ⚙️ CHANGE 8: Strict role
        system = self._build_system_prompt(
            "You are a STRICT SEARCH SUMMARISER.\n"
            "Rules:\n"
            "- Summarise ONLY the search results provided\n"
            "- Do NOT add tool calls\n"
            "- Cite sources by title\n"
            "- Be concise and factual"
        )

        try:
            self.models.ensure_chat()
            reply = self.models.chat_complete(
                [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': prompt},
                ],
                num_predict=2048, temperature=DEFAULT_CHAT_TEMP,
                stream=(not return_string),
            )
            if return_string:
                return reply
            print(f"\nAssistant: {reply}\n" if not return_string else "")
            self._record('user', f"/search {query}")
            self._record('assistant', reply)
        except Exception as e:
            logger.exception("Search LLM summarisation failed")
            msg = f"  ❌ [ERROR] {e}"
            if return_string:
                return msg
            print(msg)
            for r in results[:3]:
                print(f"  • {r['title']}")
                print(f"    {r['url']}")

    # ═════════════════════════════════════════════════════════
    # /research — ⚙️ CHANGE 2: Strict phased research pipeline
    # ═════════════════════════════════════════════════════════

    def _cmd_research(self, topic: str, return_string: bool = False):
        topic = topic.strip()
        if not topic:
            msg = "  Usage: /research <topic>"
            if return_string:
                return msg
            print(msg)
            return

        self.last_research_topic = topic

        if not return_string:
            print(f"\n🔬 Researching: {topic}\n")

        # ⚙️ CHANGE 2: Phase 0 — Tool Decision (JSON structured)
        def llm(prompt: str, temp: float = DEFAULT_CHAT_TEMP) -> str:
            system = self._build_system_prompt(
                "You are a STRICT RESEARCH ASSISTANT.\n"
                "Be structured, accurate, and factual.\n"
                "Do NOT output tool calls.\n"
                "Follow the exact format requested."
            )
            return self.models.chat_complete(
                [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': prompt},
                ],
                num_predict=2048, temperature=temp,
                stream=(not return_string)
            )

        try:
            search_context = ""

            ALWAYS_SEARCH_PATTERNS = [
                r'\blatest\b', r'\bnewest\b', r'\brecent\b',
                r'\btoday\b', r'\bnow\b', r'\bcurrent\b',
                r'\bthis week\b', r'\bthis month\b', r'\bthis year\b',
                r'\b202[456]\b',
                r'\bnews\b', r'\bupdate\b', r'\brelease\b', r'\bannouncement\b',
                r'\bjust released\b', r'\bjust launched\b',
                r'\btrending\b', r'\bbreaking\b',
                r'\bprice of\b', r'\bstock price\b',
                r'\bweather\b', r'\bscore\b', r'\bwho won\b',
                r'\bstill (free|available|working|alive|running)\b',
                r'^is .+ (still|currently|now)\b',
                r'\bnew in \d{4}\b',
            ]

            topic_lower = topic.lower()
            forced_search = any(
                re.search(pat, topic_lower) for pat in ALWAYS_SEARCH_PATTERNS
            )

            if forced_search:
                if not return_string:
                    print(f"  🌐 Topic requires current info — searching web…")
                try:
                    results = search_duckduckgo(topic, max_results=6)
                    search_context = format_results_for_ai(topic, results)
                    if not return_string:
                        print(f"    Found {len(results)} result(s).")
                except Exception as e:
                    if not return_string:
                        print(f"  ⚠️ [WARN] Search failed: {e}")
            else:
                # ⚙️ CHANGE 1: Structured JSON tool decision
                if not return_string:
                    print("  🧠 [Phase 0] Deciding tools needed (JSON)…")

                tool_decision = self.models.chat_complete_json(
                    [
                        {'role': 'system', 'content': (
                            "You are a STRICT TOOL DECISION AGENT.\n"
                            "Output MUST be valid JSON. Nothing else.\n"
                            "Failure to output JSON = system crash.\n\n"
                            "Schema: {\"action\": \"SEARCH|NEWS|NONE\", \"query\": \"search query or null\"}"
                        )},
                        {'role': 'user', 'content': (
                            f"Today is {datetime.now().strftime('%Y-%m-%d')}.\n"
                            f"Topic: \"{topic}\"\n\n"
                            "Do I need web search to answer accurately?\n"
                            "- Recent events/products/data → SEARCH\n"
                            "- Changed in last 2 years → SEARCH\n"
                            "- Purely conceptual/historical → NONE"
                        )},
                    ],
                    num_predict=100,
                    temperature=DEFAULT_PLAN_TEMP,
                )

                if tool_decision:
                    action = tool_decision.get('action', 'NONE').upper()
                    query = tool_decision.get('query', topic)

                    if action == 'SEARCH':
                        if not return_string:
                            print(f"  🔍 Searching: {query}")
                        try:
                            results = search_duckduckgo(query or topic, max_results=5)
                            search_context = format_results_for_ai(query or topic, results)
                            if not return_string:
                                print(f"    Found {len(results)} result(s).")
                        except Exception as e:
                            if not return_string:
                                print(f"  ⚠️ [WARN] Search failed: {e}")
                    elif action == 'NEWS':
                        news_topic = query or topic
                        if not return_string:
                            print(f"  📰 Fetching news: {news_topic}")
                        try:
                            results = search_duckduckgo(
                                f"{news_topic} news {datetime.now().strftime('%Y')}", max_results=5
                            )
                            search_context = format_results_for_ai(news_topic, results)
                            if not return_string:
                                print(f"    Found {len(results)} result(s).")
                        except Exception as e:
                            if not return_string:
                                print(f"  ⚠️ [WARN] News fetch failed: {e}")
                    else:
                        if not return_string:
                            print("  🧠 Answering from knowledge (no web search needed).")
                else:
                    if not return_string:
                        print("  🧠 Tool decision failed — answering from knowledge.")

            # ⚙️ CHANGE 2: Phase 1 — PLANNER ONLY (no code, no execution)
            if not return_string:
                print("  📝 [Phase 1] PLANNER: Drafting structured answer…")
            draft_prompt = (
                f"You are a PLANNER. Your ONLY job is to draft an answer.\n"
                f"Do NOT write code. Do NOT execute anything.\n\n"
                f"Topic: {topic}\n"
            )
            if search_context:
                draft_prompt += (
                    f"\nCURRENT WEB DATA ({datetime.now().strftime('%Y-%m-%d')}):\n"
                    f"{search_context}\n\n"
                    "Prioritise this data over training knowledge."
                )
            draft_prompt += "\nWrite a thorough, structured draft answer."
            initial = llm(draft_prompt)
            if not return_string:
                print(f"\n  ── Draft ──────────────────────\n{initial}\n")

            # ⚙️ CHANGE 2: Phase 2 — REVIEWER ONLY (structured critique)
            if not return_string:
                print("  🔍 [Phase 2] REVIEWER: Structured critique (JSON)…")

            critique_json = self.models.chat_complete_json(
                [
                    {'role': 'system', 'content': (
                        "You are a STRICT REVIEWER. Your ONLY job is to find issues.\n"
                        "Output MUST be valid JSON. Nothing else.\n"
                        "Schema: {\"issues\": [{\"type\": \"missing|outdated|unclear|incorrect\", "
                        "\"description\": \"what is wrong\", \"fix\": \"how to fix\"}], "
                        "\"overall_quality\": \"good|needs_improvement|poor\"}"
                    )},
                    {'role': 'user', 'content': (
                        f"Topic: {topic}\n\nDraft answer:\n{initial}\n\n"
                        "Find ALL issues: missing info, outdated claims, unclear sections, errors."
                    )},
                ],
                num_predict=500,
                temperature=DEFAULT_REVIEW_TEMP,
            )

            critique_text = ""
            if critique_json and 'issues' in critique_json:
                issues = critique_json.get('issues', [])
                quality = critique_json.get('overall_quality', 'unknown')
                critique_text = f"Quality: {quality}\n"
                for i, issue in enumerate(issues, 1):
                    critique_text += f"{i}. [{issue.get('type', '?')}] {issue.get('description', '?')} → Fix: {issue.get('fix', '?')}\n"
                if not return_string:
                    print(f"\n  ── Critique ────────────────────\n{critique_text}")
            else:
                if not return_string:
                    print("  ⚠️ JSON critique failed – using text fallback")
                fallback_prompt = f"""Critique this draft. List missing info, outdated claims, unclear sections, errors.
Draft: {initial[:1500]}
Return a plain text critique with bullet points."""
                critique_text = self.models.chat_complete(
                    [{'role': 'system', 'content': "You are a strict reviewer. Output only plain text critique."},
                     {'role': 'user', 'content': fallback_prompt}],
                    num_predict=500, temperature=DEFAULT_REVIEW_TEMP
                )
                if not return_string:
                    print(f"\n  ── Critique (Text Fallback) ──\n{critique_text}")

            # ⚙️ CHANGE 2: Phase 3 — FINAL WRITER (improved answer)
            if not return_string:
                print("  ✨ [Phase 3] WRITER: Final improved answer…")
            improve_prompt = (
                f"You are a FINAL WRITER. Produce the best possible answer.\n\n"
                f"Topic: {topic}\n\n"
                f"Draft:\n{initial}\n\n"
                f"Critique:\n{critique_text}\n\n"
            )
            if search_context:
                improve_prompt += f"Current web data:\n{search_context}\n\n"
            improve_prompt += (
                "Write the FINAL, improved answer.\n"
                "Rules:\n"
                "- Fix ALL issues from the critique\n"
                "- Be accurate, structured, up-to-date\n"
                "- Use sections and bullet points\n"
                "- Do NOT mention the critique process"
            )

            final = llm(improve_prompt, temp=DEFAULT_CHAT_TEMP)

            if return_string:
                return final

            print("\n  ── Final Answer ────────────────")
            print(f"{final}")
            print(f"\n{'─'*60}\n")
            print("  💡 Type 'again' to repeat this research.\n")

            self._record('user', f"/research {topic}")
            self._record('assistant', final)

        except Exception as e:
            logger.exception("Research failed")
            msg = f"  ❌ [ERROR] {e}"
            if return_string:
                return msg
            print(msg)

    # ═════════════════════════════════════════════════════════
    # /agent — browser automation
    # ═════════════════════════════════════════════════════════

    def _cmd_agent(self, instruction: str, return_string: bool = False):
        instruction = instruction.strip()
        if not instruction:
            msg = "  Usage: /agent <instruction>"
            if return_string:
                return msg
            print(msg)
            return

        url_match = re.search(r'https?://\S+', instruction)
        url = url_match.group(0).rstrip('.,;)') if url_match else None

        if not url:
            if return_string:
                return "Missing URL in instruction."
            url = _input("  🌐 URL to visit: ")
            if not url:
                print("  No URL provided — cancelled.")
                return

        if not return_string:
            print(f"\n  🌐 Fetching: {url}")
        content = self._browser_fetch(url)
        if not content:
            msg = "  ❌ [ERROR] Could not fetch page content."
            if return_string:
                return msg
            print(msg)
            return

        prompt = (
            f"The user's instruction is: {instruction}\n\n"
            f"Page content from {url}:\n\n"
            f"{content[:10000]}\n\n"
            "Please fulfil the instruction based on this content."
        )
        try:
            system = self._build_system_prompt(
                "You are a STRICT PAGE PROCESSOR.\n"
                "Rules:\n"
                "- Answer based ONLY on the page content provided\n"
                "- Do NOT add tool calls\n"
                "- Be concise and accurate"
            )
            self.models.ensure_chat()
            reply = self.models.chat_complete(
                [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': prompt},
                ],
                num_predict=2048, temperature=DEFAULT_CHAT_TEMP,
                stream=(not return_string),
            )
            if return_string:
                return reply
            print(f"\nAssistant: {reply}\n")
            self._record('user', f"/agent {instruction}")
            self._record('assistant', reply)
        except Exception as e:
            logger.exception("Agent LLM call failed")
            msg = f"  ❌ [ERROR] {e}"
            if return_string:
                return msg
            print(msg)

    def _browser_fetch(self, url: str) -> Optional[str]:
        try:
            # Use BrowserManager to ensure we use the user's Chrome/profile
            if not hasattr(self, '_shared_browser'):
                self._shared_browser = BrowserManager(headless=False)
            
            self._shared_browser._ensure()
            page = self._shared_browser.get_active_page()
            
            print(f"  🌐 Navigating to {url}...")
            page.goto(url, timeout=30_000, wait_until='domcontentloaded')
            time.sleep(2)  # Give it a moment to settle
            
            text = page.evaluate("""() => {
                const unwanted = 'script,style,nav,footer,header,aside,[role="banner"],[role="navigation"]';
                document.querySelectorAll(unwanted).forEach(e => e.remove());
                return document.body ? document.body.innerText.trim() : '';
            }""")
            
            raw_len = len(text)
            text = self._truncate_text(text, MAX_TOOL_RESULT_CHARS)
            logger.info("Persistent browser fetched %s (%d chars)", url, raw_len)
            print(f"  📄 Fetched {raw_len:,} chars from {url}")
            return text
        except ImportError:
            print("  ℹ️ [INFO] playwright not installed — falling back to urllib.")
        except Exception as e:
            logger.warning("Playwright fetch failed: %s", e)
            print(f"  ⚠️ [WARN] Browser error ({e}). Falling back to urllib.")

        try:
            import urllib.request
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': (
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                        'AppleWebKit/537.36 (KHTML, like Gecko) '
                        'Chrome/125.0.0.0 Safari/537.36'
                    )
                },
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                html = resp.read().decode('utf-8', errors='ignore')
                text = re.sub(r'<script[^>]*>.*?</script>', '', html,
                              flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text,
                              flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                raw_len = len(text)
                text = self._truncate_text(text, MAX_TOOL_RESULT_CHARS)
                logger.info("urllib fetched %s (%d chars)", url, raw_len)
                return text
        except Exception as e:
            logger.error("urllib fetch failed: %s", e)
        return None

    # ═════════════════════════════════════════════════════════
    # /analyze — ⚙️ CHANGE 2: Structured review output
    # ═════════════════════════════════════════════════════════

    def _cmd_analyze(self, args: str, return_string: bool = False):
        args = args.strip()
        if not args:
            msg = "  Usage: /analyze <file1> [file2 …] [question]\n  Tip: /analyze log=path/to/log.txt [question]"
            if return_string:
                return msg
            print(msg)
            return

        log_path = self.analyze_log_path

        tokens = args.split()
        remaining_tokens = []
        for tok in tokens:
            if tok.lower().startswith('log='):
                log_path = tok[4:].strip('"\'')
                self.analyze_log_path = log_path
            else:
                remaining_tokens.append(tok)

        if log_path is None and not return_string:
            log_input = _input(
                "  📄 Log file path (Enter to skip, or type a path): "
            ).strip().strip('"\'')
            if log_input:
                log_path = log_input
                self.analyze_log_path = log_path

        ext_pattern = r'\.(c|py|js|css|html|txt|cpp|h|json|java|go|rs|rb|php|ts|jsx|tsx|md)$'
        files = []
        q_parts = []
        for tok in remaining_tokens:
            if re.search(ext_pattern, tok, re.IGNORECASE):
                files.append(tok)
            else:
                q_parts.append(tok)

        if not files:
            msg = "  ❌ [ERROR] Please specify at least one filename."
            if return_string:
                return msg
            print(msg)
            return

        question = ' '.join(q_parts).strip() or (
            "Do a full audit: list every bug, missing link, syntax error, or "
            "improvement you can find. Be specific with line references."
        )

        agent = self._get_agent()
        file_sections = []
        for fname in files:
            content = agent._read_file(fname)
            if content is None:
                if not return_string:
                    print(f"  ⚠️ [WARN] Could not read {fname} — skipping.")
                continue
            file_sections.append(f"--- {fname} ---\n```\n{content}\n```")

        if not file_sections:
            msg = "  ❌ [ERROR] No readable files found."
            if return_string:
                return msg
            print(msg)
            return

        # ⚙️ CHANGE 5: Reduced context — only last 4 messages
        recent_ctx = ""
        if self.history:
            last_few = self.history[-4:]
            recent_ctx = "\n".join(
                f"{m['role'].upper()}: {m['content'][:200]}" for m in last_few
            )

        prompt = (
            "You are a STRICT CODE REVIEWER. Audit the code thoroughly.\n\n"
            + "\n\n".join(file_sections)
            + (f"\n\nRECENT CONTEXT:\n{recent_ctx}" if recent_ctx else "")
            + f"\n\nQUESTION: {question}\n\n"
            "Rules:\n"
            "- List EVERY issue with exact fix\n"
            "- Reference line content (not numbers, they may shift)\n"
            "- If everything is fine, say 'ALL CLEAR'\n"
            "- Do NOT add tool calls\n"
            "- Be specific and actionable"
        )

        system = self._build_system_prompt(
            "You are a STRICT SENIOR CODE REVIEWER.\n"
            "Your ONLY job is to find bugs and issues.\n"
            "Do NOT add tool calls. Do NOT suggest tools."
        )

        if not return_string:
            print(f"\n🔍 Analyzing {', '.join(files)}…\n")
        try:
            self.models.ensure_chat()
            reply = self.models.chat_complete(
                [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': prompt},
                ],
                num_predict=2048, temperature=DEFAULT_REVIEW_TEMP,
                stream=(not return_string),
            )

            if return_string:
                return reply

            print(f"\n{'─'*60}\n")

            if log_path:
                self._write_analyze_log(log_path, files, question, reply)

            self._record('user', f"/analyze {args}")
            self._record('assistant', reply)
        except Exception as e:
            logger.exception("Analyze failed")
            msg = f"  ❌ [ERROR] {e}"
            if return_string:
                return msg
            print(msg)

    def _write_analyze_log(self, log_path: str, files: List[str],
                           question: str, reply: str):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"FILES: {', '.join(files)}\n")
                f.write(f"QUESTION: {question}\n")
                f.write(f"{'─'*60}\n")
                f.write(reply)
                f.write(f"\n{'='*60}\n")
            print(f"  📝 Logged to: {log_path}")
        except Exception as e:
            print(f"  ⚠️ [WARN] Could not write log: {e}")

    # ═════════════════════════════════════════════════════════
    # /code — ⚙️ CHANGE 1+2+3+4: Structured JSON plan + step execution + validation
    # ═════════════════════════════════════════════════════════

    def _try_simple_fix(
        self, primary_file: str, plan: str, agent: CodeAgent,
    ) -> Optional[Dict]:
        plan_lower = plan.lower()

        if re.search(r'css\s*link\s*(missing|wrong|incorrect|not\s*found)', plan_lower):
            content = agent._read_file(primary_file) or ''
            css_ref = re.search(r'([\w./]+\.css)', plan)
            href = css_ref.group(1) if css_ref else 'style.css'
            link_tag = f'  <link rel="stylesheet" href="{href}">'
            if link_tag.strip() in content:
                return None
            fixed = re.sub(
                r'(</head>)', f'{link_tag}\n\\1', content,
                count=1, flags=re.IGNORECASE,
            )
            if fixed != content:
                logger.info("Simple-fix: injected CSS link into %s", primary_file)
                print(f"  ⚡ [Simple fix] Injected <link> for {href} — DeepSeek not needed.")
                return {'primary': fixed}

        return None

    def _cmd_code(self, args: str, return_string: bool = False):
        args = args.strip()
        if not args:
            msg = "  Usage: /code <filename> [extra_files…] <instruction>"
            if return_string:
                return msg
            print(msg)
            return

        ext_pattern = r'\.(c|py|js|css|html|txt|cpp|h|json|java|go|rs|rb|php|ts|jsx|tsx|md)$'
        filenames = []
        rest_parts = []
        for token in args.split():
            if re.search(ext_pattern, token, re.IGNORECASE):
                filenames.append(token)
            else:
                rest_parts.append(token)

        if not filenames:
            msg = "  ❌ [ERROR] Please specify at least one filename.\n  Example: /code main.py Add error handling to the main function"
            if return_string:
                return msg
            print(msg)
            return

        primary_file = filenames[0]
        extra_files = filenames[1:]
        instruction = ' '.join(rest_parts).strip()

        if not instruction:
            if return_string:
                return "Missing instruction for code edit."
            instruction = _input("  📝 Instruction: ")
            if not instruction:
                print("  Cancelled.")
                return

        agent = self._get_agent()
        abs_primary = os.path.join(self.project_root, primary_file)
        all_files = [primary_file] + extra_files
        is_new_file = not os.path.exists(abs_primary)

        # ------------------------------------------------------------------
        # NEW FILE path — ⚙️ CHANGE 1: JSON-structured spec
        # ------------------------------------------------------------------
        if is_new_file:
            print(f"\n🆕 [New project] {len(all_files)} file(s) to create: {', '.join(all_files)}")

            # ⚙️ CHANGE 2: Phase 1 — PLANNER generates JSON spec
            print("  📋 [Phase 1] PLANNER: Generating structured spec (JSON)…")
            try:
                self.models.ensure_chat()
            except Exception as e:
                print(f"  ❌ [ERROR] Could not load chat model: {e}")
                return

            spec_json = self.models.chat_complete_json(
                [
                    {'role': 'system', 'content': (
                        "You are a STRICT PLANNER. Output ONLY valid JSON.\n"
                        "Failure to output valid JSON = system crash.\n\n"
                        "Schema:\n"
                        '{"files": [{"name": "filename", "purpose": "what it does", '
                        '"imports": ["list of imports"], '
                        '"functions": [{"name": "fn_name", "params": "param list", '
                        '"returns": "return type", "description": "what it does"}], '
                        '"connections": ["how it links to other files"]}]}'
                    )},
                    {'role': 'user', 'content': (
                        f"Create specification for files: {', '.join(all_files)}\n"
                        f"User request: \"{instruction}\"\n\n"
                        "Rules:\n"
                        "- ALL files in SAME directory (flat, no subfolders)\n"
                        "- Be SPECIFIC — exact names, signatures, return types\n"
                        "- Do NOT write code, only the specification"
                    )},
                ],
                num_predict=1500,
                temperature=DEFAULT_PLAN_TEMP,
            )

            if not spec_json:
                print("  ⚠️ JSON spec failed — falling back to text spec.")
                spec = self._fallback_text_spec(all_files, instruction)
            else:
                # Convert JSON spec to text for the code model
                spec = json.dumps(spec_json, indent=2)
                print(f"\n[SPEC (JSON)]\n{spec}\n")

            # ⚙️ CHANGE 3: One-step execution — build each file individually
            print("  🔧 [Phase 2] BUILDER: Writing each file…")
            try:
                self.models.ensure_code()
            except Exception as e:
                print(f"  ❌ [ERROR] Could not load coding model: {e}")
                return

            created = []
            failed = []

            for fname in all_files:
                print(f"\n  ✍️ Writing {fname}…")

                # Extract file-specific spec
                file_spec = spec
                if spec_json and 'files' in spec_json:
                    for f_info in spec_json['files']:
                        if f_info.get('name') == fname:
                            file_spec = json.dumps(f_info, indent=2)
                            break

                creation_plan = (
                    f"Full project spec:\n{spec}\n\n"
                    f"This file's spec:\n{file_spec}\n\n"
                    f"Create {fname} from scratch following the spec exactly.\n"
                    f"ALL files are in the SAME directory. Use relative paths."
                )

                result = agent.apply_plan_with_code_model(
                    fname, creation_plan, instruction,
                    extra_files=[f for f in all_files if f != fname],
                )

                if 'error' in result:
                    print(f"  ❌ [ERROR] {fname}: {result['error']}")
                    failed.append(fname)
                    continue

                # ⚙️ CHANGE 4: Validate before applying
                report = agent.apply_plan(result, fname, [])
                print(f"  {report.strip()}")
                created.append(fname)

            agent.refresh()

            summary = f"Created: {', '.join(created)}" if created else "No files created."
            if failed:
                summary += f"\nFailed: {', '.join(failed)}"
            self._record('user', f"/code {args}")
            self._record('assistant', summary)
            return

        # ------------------------------------------------------------------
        # EXISTING FILE — ⚙️ CHANGE 1+2: JSON-structured plan
        # ------------------------------------------------------------------
        print("\n📋 [Phase 1] PLANNER: Analyzing file (JSON plan)…")
        try:
            self.models.ensure_chat()
        except Exception as e:
            print(f"  ❌ [ERROR] Could not load chat model: {e}")
            return

        primary_content = agent._read_file(primary_file) or ""
        related = agent._get_related(primary_file)
        all_ctx = (set(extra_files) | related) - {primary_file}

        # ⚙️ CHANGE 5: Reduced context — only relevant files within budget
        context_parts = []
        token_budget = 4000
        for rel in sorted(all_ctx):
            content = agent._read_file(rel)
            if not content:
                continue
            if len(content) > 4000:
                content = content[:4000] + "\n... (truncated)"
            tok = count_tokens(content)
            if tok > token_budget:
                continue
            token_budget -= tok
            context_parts.append((rel, content))

        # ⚙️ CHANGE 1: JSON-structured plan output
        plan_json = self.models.chat_complete_json(
            [
                {'role': 'system', 'content': (
                    "You are a STRICT CODE PLANNER. Output ONLY valid JSON.\n"
                    "Failure to output valid JSON = system crash.\n\n"
                    "Schema:\n"
                    '{"status": "problem_found|no_problem", '
                    '"analysis": "one sentence describing what you found", '
                    '"changes": [{"location": "function/section name", '
                    '"current": "what exists now (quote the code)", '
                    '"proposed": "what it should become", '
                    '"reason": "why this change is needed"}]}'
                )},
                {'role': 'user', 'content': (
                    f"PRIMARY FILE ({primary_file}):\n{primary_content[:3000]}\n\n"
                    + "".join(f"--- {fn} ---\n{fc[:1000]}\n" for fn, fc in context_parts[:3])
                    + f"\nTASK: {instruction}\n\n"
                    "Analyze the file. If there's a clear problem, describe the exact fix.\n"
                    "If NO problem, set status to 'no_problem'."
                )},
            ],
            num_predict=600,
            temperature=DEFAULT_PLAN_TEMP,
        )

        # Convert JSON plan to text for display and code model
        if plan_json:
            if plan_json.get('status') == 'no_problem':
                print("  ℹ️ [INFO] Planner found no problems.")
                print("  💡 Try being more specific.")
                return
            plan = json.dumps(plan_json, indent=2)
            print(f"\n[PLAN (JSON)]\n{plan}\n")
        else:
            # Fallback to text plan
            print("  ⚠️ JSON plan failed — falling back to text plan.")
            system = self._build_system_prompt()
            planner_prompt = f"""You are a STRICT CODE PLANNER. Analyze the file and create a clear plan.

PRIMARY FILE ({primary_file}):
{primary_content[:3000]}

RELATED FILES:
"""
            for fname, fcontent in context_parts:
                planner_prompt += f"--- {fname} ---\n```\n{fcontent[:1000]}\n```\n"

            planner_prompt += f"""
TASK: {instruction}

Rules:
- Read the file carefully
- If clear problem → describe precisely with exact fix
- If NO problem → say exactly: "NO PROBLEM FOUND"
- Keep changes minimal
- No code, only plan text
"""

            plan = self.models.chat_complete(
                [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': planner_prompt},
                ],
                num_predict=512, temperature=DEFAULT_PLAN_TEMP,
            )
            print(f"\n[PLAN]\n{plan}\n")

            if re.search(r'no\s+problem\s+found', plan, re.IGNORECASE):
                print("  ℹ️ [INFO] Planner could not identify a specific problem.")
                print("  💡 Try being more specific.")
                return

        # ── Shortcut: simple template fix ────────────────────
        shortcut = self._try_simple_fix(primary_file, plan, agent)
        if shortcut is not None:
            result = shortcut
        else:
            # ── Phase 2: Execute with coding model ───────────
            print("  🔧 [Phase 2] BUILDER: Executing plan…")
            try:
                self.models.ensure_code()
            except Exception as e:
                print(f"  ❌ [ERROR] Could not load coding model: {e}")
                return

            result = agent.apply_plan_with_code_model(
                primary_file, plan, instruction, extra_files,
            )

        if 'error' in result:
            msg = f"  ❌ [ERROR] {result['error']}"
            if return_string:
                return msg
            print(f"\n{msg}")

            # ⚙️ CHANGE 4: Self-correction — auto-retry once
            print("  🔄 Auto-retrying with simplified instruction…")
            try:
                self.models.ensure_code()
                simplified_plan = f"SIMPLE FIX for {primary_file}: {instruction}\nKeep changes minimal."
                result = agent.apply_plan_with_code_model(
                    primary_file, simplified_plan, instruction, extra_files,
                )
                if 'error' in result:
                    print(f"  ❌ Auto-retry also failed: {result['error']}")
                    retry = _input("  🔄 Retry with different instruction? (y/n): ").lower()
                    if retry == 'y':
                        new_instruction = _input("  📝 New instruction: ")
                        if new_instruction:
                            self._cmd_code(f"{primary_file} {new_instruction}")
                    return
            except Exception:
                retry = _input("  🔄 Retry with different instruction? (y/n): ").lower()
                if retry == 'y':
                    new_instruction = _input("  📝 New instruction: ")
                    if new_instruction:
                        self._cmd_code(f"{primary_file} {new_instruction}")
                return

        # ── When called as tool: auto-apply ──────────────────
        if return_string:
            report = agent.apply_plan(result, primary_file, extra_files)
            agent.refresh()
            self._record('user', f"/code {args}")
            self._record('assistant', f"Applied:\n{report}")
            return report

        # ── Preview and apply loop ───────────────────────────
        print("\n📋 Proposed changes:\n")
        while True:
            if 'primary' in result:
                print(f"  --- {primary_file} ---")
                preview = result['primary']
                if len(preview) > 1200:
                    print(preview[:1200])
                    print(f"  … ({len(preview) - 1200} more chars)")
                else:
                    print(preview)
                print()

            if 'extra' in result:
                for fname, content in result['extra'].items():
                    print(f"  --- {fname} ---")
                    if len(content) > 400:
                        print(content[:400])
                        print(f"  … ({len(content) - 400} more chars)")
                    else:
                        print(content)
                    print()

            choice = _input(
                "  Apply changes? (y) / Edit instruction (e) / Reject (n): "
            ).lower()

            if choice == 'y':
                report = agent.apply_plan(result, primary_file, extra_files)
                print(f"\n{report}")
                agent.refresh()
                self._record('user', f"/code {args}")
                self._record('assistant', f"Applied:\n{report}")
                break
            elif choice == 'e':
                new_instruction = _input("  📝 Enter revised instruction: ")
                if not new_instruction:
                    print("  Cancelled.")
                    break
                self._cmd_code(f"{primary_file} {new_instruction}")
                break
            else:
                print("  ❌ Changes rejected — nothing written.")
                break

    def _fallback_text_spec(self, all_files: List[str], instruction: str) -> str:
        """Fallback when JSON spec generation fails."""
        spec_prompt = f"""You are a senior developer. Write a DETAILED SPEC.

Files to create: {', '.join(all_files)}
User request: "{instruction}"

For EACH file write:
=== <filename> ===
Purpose:
Structure:
Links to other files:

Rules:
- Be SPECIFIC — exact names, function signatures
- ALL files in SAME directory
- No code, only specification"""

        system = self._build_system_prompt()
        return self.models.chat_complete(
            [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': spec_prompt},
            ],
            num_predict=1500, temperature=DEFAULT_PLAN_TEMP,
        )

    # ═════════════════════════════════════════════════════════
    # /agentcode — ⚙️ CHANGE 1+2+3+4+9+10: Full structured pipeline
    # ═════════════════════════════════════════════════════════

    def _cmd_agentcode(self, request: str, return_string: bool = False):
        request = request.strip()
        if not request:
            msg = "  Usage: /agentcode <description>\n  Example: /agentcode build a calculator CLI in Python"
            if return_string:
                return msg
            print(msg)
            return

        if return_string:
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            try:
                with redirect_stdout(f):
                    runner = AgentCodeRunner(
                        request=request,
                        project_root=self.project_root,
                        models=self.models,
                        memory=self.memory,
                    )
                    runner.run()
            except Exception as e:
                return f"AgentCode error: {e}\n{f.getvalue()}"
            output = f.getvalue()
            return output[-2000:] if len(output) > 2000 else output

        runner = AgentCodeRunner(
            request=request,
            project_root=self.project_root,
            models=self.models,
            memory=self.memory,
        )
        runner.run()


# ═══════════════════════════════════════════════════════════════
# ProjectMemory — ⚙️ CHANGE 9: Enhanced with step checkpoints
# ═══════════════════════════════════════════════════════════════

class ProjectMemory:

    STATUS_PENDING = 'pending'
    STATUS_RUNNING = 'running'
    STATUS_DONE = 'done'
    STATUS_FAILED = 'failed'

    def __init__(self, project_root: str):
        self.path = os.path.join(project_root, '.agentcode_state.json')
        self._data: Dict[str, Any] = {}

    def init(self, goal: str, tech_stack: List[str], file_structure: List[str],
             tasks: List[Dict]) -> None:
        now = datetime.now().isoformat()
        self._data = {
            'goal': goal,
            'tech_stack': tech_stack,
            'file_structure': file_structure,
            'tasks': tasks,
            'completed': [],
            'errors': [],
            'iteration': 0,
            'started_at': now,
            'updated_at': now,
            # ⚙️ CHANGE 9: Step-level checkpoints
            'step_log': [],
        }
        self._save()

    def load(self) -> bool:
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
            return True
        except Exception:
            return False

    def _save(self):
        self._data['updated_at'] = datetime.now().isoformat()
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def reset(self):
        self._data = {}

    def has_state(self) -> bool:
        return bool(self._data)

    @property
    def goal(self) -> str:
        return self._data.get('goal', '')

    @property
    def tech_stack(self) -> List[str]:
        return self._data.get('tech_stack', [])

    @property
    def file_structure(self) -> List[str]:
        return self._data.get('file_structure', [])

    @property
    def tasks(self) -> List[Dict]:
        return self._data.get('tasks', [])

    @property
    def completed(self) -> List[str]:
        return self._data.get('completed', [])

    @property
    def errors(self) -> List[Dict]:
        return self._data.get('errors', [])

    @property
    def iteration(self) -> int:
        return self._data.get('iteration', 0)

    @property
    def started_at(self) -> str:
        return self._data.get('started_at', '?')

    def pending_tasks(self) -> List[Dict]:
        return [t for t in self.tasks if t['status'] == self.STATUS_PENDING]

    def failed_tasks(self) -> List[Dict]:
        return [t for t in self.tasks if t['status'] == self.STATUS_FAILED]

    def all_done(self) -> bool:
        return all(t['status'] == self.STATUS_DONE for t in self.tasks)

    def set_task_status(self, task_id: str, status: str, error: str = ''):
        for t in self._data['tasks']:
            if t['id'] == task_id:
                t['status'] = status
                if error:
                    t['error'] = error
                break
        if status == self.STATUS_DONE:
            if task_id not in self._data['completed']:
                self._data['completed'].append(task_id)
        # ⚙️ CHANGE 9: Log every status change
        self._data.setdefault('step_log', []).append({
            'task': task_id,
            'status': status,
            'time': datetime.now().isoformat(),
            'error': error,
        })
        self._save()

    def log_error(self, task_id: str, error: str, fix: str = ''):
        self._data['errors'].append({
            'task': task_id, 'error': error, 'fix': fix,
            'time': datetime.now().isoformat(),
        })
        self._save()

    def bump_iteration(self):
        self._data['iteration'] = self._data.get('iteration', 0) + 1
        self._save()

    def summary(self) -> str:
        total = len(self.tasks)
        done = len([t for t in self.tasks if t['status'] == self.STATUS_DONE])
        failed = len([t for t in self.tasks if t['status'] == self.STATUS_FAILED])
        pending = total - done - failed
        return (
            f"📊 Progress: {done}/{total} done | {pending} pending | "
            f"{failed} failed | iteration {self.iteration}"
        )


# ═══════════════════════════════════════════════════════════════
# AgentCodeRunner — ⚙️ ALL CHANGES APPLIED: Structured autonomous pipeline
# ═══════════════════════════════════════════════════════════════

class AgentCodeRunner:
    """
    v3 Architecture:
      PHASE 1: PLANNER (JSON structured plan)
      PHASE 2: FOR EACH STEP:
                 → BUILDER writes code
                 → REVIEWER validates (JSON)
                 → If fail → REPAIR → Re-validate
      PHASE 3: Integration check
      PHASE 4: Final stability loop

    Key improvements:
    - All planning/review outputs are JSON-structured
    - One-step execution with per-step validation
    - Self-correction loop (Generate → Validate → Fix → Continue)
    - Deterministic generation settings
    - Strict role reinforcement
    - Memory checkpoints after every step
    """

    MAX_RETRIES = 3
    MAX_ITERATIONS = 80
    REPAIR_THRESHOLD = 2

    PROJECT_STACKS = {
        'CLI': {'tech': ['Python'], 'desc': 'command-line application'},
        'API': {'tech': ['Python', 'Flask', 'SQLite'], 'desc': 'REST API backend'},
        'WEB': {'tech': ['Python', 'Flask', 'HTML', 'CSS', 'JavaScript', 'SQLite'],
                'desc': 'web application with frontend'},
        'LIB': {'tech': ['Python'], 'desc': 'reusable library/module'},
        'GAME': {'tech': ['Python', 'pygame'], 'desc': 'game'},
        'DATA': {'tech': ['Python', 'SQLite'], 'desc': 'data processing tool'},
    }

    WEB_KEYWORDS = [
        'website', 'web app', 'webapp', 'web page', 'webpage',
        'frontend', 'front-end', 'html', 'css', 'browser',
        'landing page', 'dashboard', 'web interface', 'web ui',
    ]

    def __init__(self, request: str, project_root: str,
                 models: ModelManager, memory: MemoryManager):
        self.request = request
        self.project_root = project_root
        self.models = models
        self.memory = memory
        self.pm = ProjectMemory(project_root)
        self.checkpoint = StepCheckpoint(project_root)
        self._stop = False
        self._consecutive_failures = 0
        self._error_counts: Dict[str, int] = {}
        self._dep_graph: Dict[str, set] = {}

        self._old_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._on_stop)

    def _on_stop(self, sig, frame):
        print("\n\n  🛑 Stop requested — finishing current task then exiting.")
        self._stop = True

    def _restore_sigint(self):
        signal.signal(signal.SIGINT, self._old_sigint)

    def _banner(self, text: str):
        print(f"\n {'═'*62}")
        print(f"  {text}")
        print(f" {'═'*62}")

    def _step(self, n: int, total: int, title: str):
        print(f"\n  [{n}/{total}] {title}")

    def _error_fingerprint(self, error: str) -> str:
        e = error.lower().strip()
        e = re.sub(r'line\s+\d+', 'line N', e)
        e = re.sub(r"'[^']{1,40}'", "'X'", e)
        e = re.sub(r'\s+', ' ', e)
        return e[:120]

    def _seen_before(self, task_id: str, error: str) -> bool:
        key = f"{task_id}::{self._error_fingerprint(error)}"
        self._error_counts[key] = self._error_counts.get(key, 0) + 1
        return self._error_counts[key] >= 2

    # ═════════════════════════════════════════════════════════
    # Dependency graph
    # ═════════════════════════════════════════════════════════

    def _refresh_dep_graph(self):
        self._dep_graph = {}
        for rel in self._list_project_files():
            content = self._read_file(rel)
            if not content:
                continue
            deps = set()
            ext = os.path.splitext(rel)[1].lower()

            if ext == '.py':
                for m in re.finditer(r'^(?:from|import)\s+([\w.]+)', content, re.M):
                    mod = m.group(1).replace('.', '/') + '.py'
                    full = os.path.join(self.project_root, mod)
                    if os.path.exists(full):
                        deps.add(mod)
            elif ext in ('.html', '.htm'):
                for m in re.finditer(
                    r'(?:src|href|action)\s*=\s*["\']([^"\']+)["\']', content
                ):
                    ref = m.group(1).split('?')[0]
                    if not ref.startswith(('http', '//', 'data:', '#', 'mailto:')):
                        deps.add(ref.lstrip('/'))
            elif ext == '.js':
                for m in re.finditer(
                    r'(?:import|require)\s*\(?["\']([^"\']+)["\']', content
                ):
                    ref = m.group(1)
                    if not ref.startswith(('http', '//')):
                        deps.add(ref.lstrip('./'))

            self._dep_graph[rel] = deps

    def _get_reverse_deps(self, filename: str) -> List[str]:
        basename = os.path.basename(filename)
        dependents = []
        for src, deps in self._dep_graph.items():
            if src == filename:
                continue
            for dep in deps:
                if dep == filename or dep == basename or dep.endswith('/' + basename):
                    dependents.append(src)
                    break
        return dependents

    def _check_dependencies(self) -> List[str]:
        self._refresh_dep_graph()
        issues = []
        written = set(self._list_project_files())
        for src, deps in self._dep_graph.items():
            for dep in deps:
                candidates = [dep]
                if '.' not in dep:
                    candidates += [dep + '.py', dep + '.js', dep + '.html']
                found = any(c in written for c in candidates)
                if not found:
                    issues.append(f"{src} references '{dep}' which does not exist")
        return issues

    def _propagate_changes(self, written_file: str):
        self._refresh_dep_graph()
        dependents = self._get_reverse_deps(written_file)
        if not dependents:
            return

        written_content = self._read_file(written_file)
        if not written_content:
            return

        print(f"  🔗 Checking {len(dependents)} dep(s) of {written_file}…")

        for dep_file in dependents:
            dep_content = self._read_file(dep_file)
            if not dep_content:
                continue

            # ⚙️ CHANGE 1: JSON-structured dependency check
            check_result = self.models.chat_complete_json(
                [
                    {'role': 'system', 'content': (
                        "You are a STRICT DEPENDENCY CHECKER. Output ONLY valid JSON.\n"
                        'Schema: {"status": "ok|needs_fix", "fix": "description or null"}'
                    )},
                    {'role': 'user', 'content': (
                        f"UPDATED FILE ({written_file}):\n{written_content[:1200]}\n\n"
                        f"DEPENDENT FILE ({dep_file}):\n{dep_content[:1200]}\n\n"
                        "Check: import paths, function names, API routes, old name refs."
                    )},
                ],
                num_predict=150,
                temperature=DEFAULT_REVIEW_TEMP,
            )

            if not check_result or check_result.get('status') == 'ok':
                continue

            fix_desc = check_result.get('fix', 'unknown fix needed')
            print(f"  🔧 Propagating to {dep_file}: {fix_desc[:80]}")

            ext = os.path.splitext(dep_file)[1].lower()
            primer = '<' if ext in ('.html', '.htm') else ''
            lang = {'.py': 'Python', '.html': 'HTML', '.css': 'CSS', '.js': 'JavaScript'}.get(ext, 'code')

            system = f"Expert {lang} developer. Apply the fix. Output ONLY complete fixed file."
            fix_prompt = (
                f"FILE ({dep_file}):\n{dep_content[:2000]}\n\n"
                f"REFERENCED FILE ({written_file}):\n{written_content[:800]}\n\n"
                f"FIX NEEDED:\n{fix_desc}\n\n"
                f"All files same directory. Write COMPLETE fixed {dep_file}."
            )

            raw = self._builder(system, fix_prompt, num_predict=3000, primer=primer)
            fixed = self._sanitise(raw, dep_file)

            if fixed and len(fixed.strip()) > 20:
                self._write_file(dep_file, fixed)
                print(f"  ✅ Updated {dep_file}")
            else:
                print(f"  ⚠️ Could not auto-fix {dep_file}")

    def _propagate_tier_brain(self, written_files: List[str]) -> Dict[str, Dict]:
        self._refresh_dep_graph()
        fixes_needed: Dict[str, Dict] = {}

        for written_file in written_files:
            dependents = self._get_reverse_deps(written_file)
            if not dependents:
                continue

            written_content = self._read_file(written_file)
            if not written_content:
                continue

            for dep_file in dependents:
                if dep_file in fixes_needed:
                    continue

                dep_content = self._read_file(dep_file)
                if not dep_content:
                    continue

                # ⚙️ CHANGE 1: JSON dependency check
                check_result = self.models.chat_complete_json(
                    [
                        {'role': 'system', 'content': (
                            "You are a STRICT DEPENDENCY CHECKER. Output ONLY valid JSON.\n"
                            'Schema: {"status": "ok|needs_fix", "fix": "description or null"}'
                        )},
                        {'role': 'user', 'content': (
                            f"UPDATED ({written_file}):\n{written_content[:1000]}\n\n"
                            f"DEPENDENT ({dep_file}):\n{dep_content[:1000]}\n\n"
                            "Check: import paths, function names, routes, old refs."
                        )},
                    ],
                    num_predict=120,
                    temperature=DEFAULT_REVIEW_TEMP,
                )

                if check_result and check_result.get('status') == 'needs_fix':
                    fixes_needed[dep_file] = {
                        'source': written_file,
                        'fix': check_result.get('fix', 'needs update'),
                        'content': dep_content,
                    }

        return fixes_needed

    def _propagate_tier_builder(self, fixes_needed: Dict[str, Dict]):
        if not fixes_needed:
            return

        print(f"  🔧 Applying {len(fixes_needed)} propagation fix(es)…")

        for dep_file, info in fixes_needed.items():
            ext = os.path.splitext(dep_file)[1].lower()
            primer = '<' if ext in ('.html', '.htm') else ''
            lang = {'.py': 'Python', '.html': 'HTML', '.css': 'CSS', '.js': 'JavaScript'}.get(ext, 'code')
            src_content = self._read_file(info['source'])

            system = f"Expert {lang} developer. Apply the fix. Output ONLY complete fixed file."
            fix_prompt = (
                f"FILE ({dep_file}):\n{info['content'][:2000]}\n\n"
                f"REFERENCED FILE ({info['source']}):\n{src_content[:800]}\n\n"
                f"FIX NEEDED:\n{info['fix']}\n\n"
                f"All files same directory. Write COMPLETE fixed {dep_file}."
            )

            raw = self._builder(system, fix_prompt, num_predict=3000, primer=primer)
            fixed = self._sanitise(raw, dep_file)

            if fixed and len(fixed.strip()) > 20:
                self._write_file(dep_file, fixed)
                print(f"  ✅ Propagated fix → {dep_file}")
            else:
                print(f"  ⚠️ Propagation failed for {dep_file}")

    def _compute_tiers(self, tier_size: int = 3) -> List[List[Dict]]:
        all_tasks = {t['id']: t for t in self.pm.tasks}
        done_ids = set(self.pm.completed)
        remaining = [t for t in self.pm.tasks
                     if t['status'] == ProjectMemory.STATUS_PENDING]

        tiers: List[List[Dict]] = []

        while remaining:
            ready = [t for t in remaining
                     if all(d in done_ids for d in t.get('depends_on', []))]

            if not ready:
                ready = [remaining[0]]
                logger.warning("Dependency deadlock — forcing %s into next tier", ready[0]['id'])

            for chunk_start in range(0, len(ready), tier_size):
                chunk = ready[chunk_start:chunk_start + tier_size]
                tiers.append(chunk)
                done_ids.update(t['id'] for t in chunk)

            for t in ready:
                remaining.remove(t)

        return tiers

    # ⚙️ CHANGE 7+8: Strict brain calls with deterministic settings
    def _brain(self, prompt: str, num_predict: int = 1024,
               temperature: float = DEFAULT_PLAN_TEMP) -> str:
        self.models.ensure_chat()
        mem_block = self.memory.build_context_block()
        system = BASE_SYSTEM_PROMPT.format(date=datetime.now().strftime('%Y-%m-%d'))
        if mem_block:
            system = mem_block + system
        return self.models.chat_complete(
            [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': prompt},
            ],
            num_predict=num_predict,
            temperature=temperature,
        )

    # ⚙️ CHANGE 7: Builder with deterministic settings
    def _builder(self, system: str, task_prompt: str,
                 num_predict: int = 4096, primer: str = '') -> str:
        self.models.ensure_code()
        if not primer:
            ext_match = re.search(r'FILE[:\s]+\S+\.(\w+)', task_prompt)
            if ext_match and ext_match.group(1).lower() in ('html', 'htm'):
                primer = '<'
        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': task_prompt},
        ]
        if primer:
            messages.append({'role': 'assistant', 'content': primer})
        resp = ollama.chat(
            model=CODE_MODEL,
            messages=messages,
            options={'num_predict': num_predict, 'temperature': DEFAULT_CODE_TEMP,
                     'top_p': 0.9, 'repeat_penalty': 1.1},
            keep_alive=-1,
        )
        raw = resp['message']['content'].strip()
        if primer and not raw.startswith(primer):
            raw = primer + raw
        return raw

    # ── File helpers ─────────────────────────────────────────
    def _read_file(self, rel_path: str) -> str:
        try:
            with open(os.path.join(self.project_root, rel_path),
                       'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except FileNotFoundError:
            return ''

    def _write_file(self, rel_path: str, content: str):
        abs_path = os.path.join(self.project_root, rel_path)
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if os.path.exists(abs_path):
            try:
                shutil.copy2(abs_path,
                             abs_path + f".{datetime.now().strftime('%H%M%S')}.bak")
            except Exception:
                pass
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info("AgentCode wrote: %s (%d chars)", rel_path, len(content))

    def _list_project_files(self) -> List[str]:
        result = []
        skip = {'.agentcode_state.json', '.step_checkpoint.json',
                '.code_backups', '__pycache__',
                'node_modules', '.git', 'venv', '.venv'}
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in skip]
            for f in files:
                if f.endswith('.bak'):
                    continue
                result.append(
                    os.path.relpath(os.path.join(root, f), self.project_root)
                )
        return sorted(result)

    # ⚙️ CHANGE 5: Reduced context per step
    def _read_existing_files_context(self, max_chars: int = 10000) -> str:
        files = self._list_project_files()
        if not files:
            return '(no files written yet)'
        parts = []
        budget = max_chars
        per_file = max(500, budget // max(len(files), 1))
        for rel in files:
            content = self._read_file(rel)
            if not content:
                continue
            snippet = content[:per_file]
            if len(content) > per_file:
                snippet += '\n... (truncated)'
            parts.append(f"--- {rel} ---\n{snippet}")
            budget -= len(snippet)
            if budget <= 0:
                break
        return '\n\n'.join(parts)

    def _parse_plan(self, raw: str) -> Optional[Dict]:
        match = re.search(r'\{[\s\S]+\}', raw)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    # ═════════════════════════════════════════════════════════
    # ⚙️ CHANGE 1: JSON-structured project type detection
    # ═════════════════════════════════════════════════════════

    def _detect_project_type(self) -> str:
        request_lower = self.request.lower()
        for kw in self.WEB_KEYWORDS:
            if kw in request_lower:
                return 'WEB'

        result = self.models.chat_complete_json(
            [
                {'role': 'system', 'content': (
                    "You are a STRICT PROJECT CLASSIFIER. Output ONLY valid JSON.\n"
                    'Schema: {"type": "CLI|API|WEB|LIB|GAME|DATA", "reason": "one sentence"}'
                )},
                {'role': 'user', 'content': (
                    f"Classify: \"{self.request}\"\n\n"
                    "Rules:\n"
                    "- CLI = command-line, script, utility (DEFAULT)\n"
                    "- WEB = ONLY if user explicitly asks for web page/website\n"
                    "- API = ONLY if explicitly requested\n"
                    "- GAME = ONLY if user says 'game'"
                )},
            ],
            num_predict=50,
            temperature=DEFAULT_PLAN_TEMP,
        )

        if result and result.get('type'):
            ptype = result['type'].upper()
            if ptype in self.PROJECT_STACKS:
                return ptype
        return 'CLI'

    # ═════════════════════════════════════════════════════════
    # PHASE 1 — ⚙️ CHANGE 1: JSON-structured planning
    # ═════════════════════════════════════════════════════════

    def _phase_plan(self) -> Optional[Dict]:
        self._banner("PHASE 1 — PLANNER: Structured project plan (JSON)")

        project_type = self._detect_project_type()
        stack_info = self.PROJECT_STACKS.get(project_type, self.PROJECT_STACKS['CLI'])

        print(f"  🏗️ Project type: {project_type} ({stack_info['desc']})")
        print(f"  🔧 Tech stack: {', '.join(stack_info['tech'])}")

        if project_type == 'WEB':
            structure_note = (
                "For web projects: ALL files in SAME directory (flat).\n"
                "Flask(__name__, template_folder='.', static_folder='.')\n"
                "No templates/ or static/ subfolders."
            )
        elif project_type == 'CLI':
            structure_note = "CLI app. NO HTML/CSS/JS/Flask. Python only. Same directory."
        elif project_type == 'API':
            structure_note = "Backend API. No HTML pages unless explicitly requested."
        else:
            structure_note = "All files in same directory (flat structure)."

        # ⚙️ CHANGE 1: Enforce JSON plan output
        plan_json = self.models.chat_complete_json(
            [
                {'role': 'system', 'content': (
                    "You are a STRICT PROJECT PLANNER. Output ONLY valid JSON.\n"
                    "Failure to output valid JSON = system crash.\n\n"
                    "Schema:\n"
                    '{"goal": "one sentence", '
                    f'"tech_stack": {json.dumps(stack_info["tech"])}, '
                    '"file_structure": ["file1.py", ...], '
                    '"tasks": [{"id": "t01", "title": "short title", '
                    '"file": "filename.py", '
                    '"description": "detailed description", '
                    '"depends_on": []}]}'
                )},
                {'role': 'user', 'content': (
                    f"Build: \"{self.request}\"\n"
                    f"Type: {project_type} — {stack_info['desc']}\n"
                    f"Tech: {', '.join(stack_info['tech'])}\n\n"
                    f"{structure_note}\n\n"
                    "Rules:\n"
                    "- 4-10 small tasks, ONE file per task\n"
                    "- No web frontend for CLI/API projects\n"
                    "- ALL files same directory\n"
                    "- List dependencies correctly"
                )},
            ],
            num_predict=1500,
            temperature=DEFAULT_PLAN_TEMP,
        )

        if not plan_json:
            print("  ⚠️ JSON plan failed — retrying…")
            plan_json = self.models.chat_complete_json(
                [
                    {'role': 'system', 'content': (
                        "OUTPUT ONLY VALID JSON. Nothing else. No text before or after.\n"
                        '{"goal": "...", "tech_stack": [...], "file_structure": [...], '
                        '"tasks": [{"id": "t01", "title": "...", "file": "...", '
                        '"description": "...", "depends_on": []}]}'
                    )},
                    {'role': 'user', 'content': f"Build: \"{self.request}\""},
                ],
                num_predict=1500,
                temperature=DEFAULT_PLAN_TEMP,
            )

        if not plan_json:
            print("  ❌ [ERROR] Planning failed after 2 attempts.")
            return None

        plan = plan_json

        # Enforce flat structure for non-web
        if project_type != 'WEB':
            for t in plan.get('tasks', []):
                if t.get('file'):
                    t['file'] = os.path.basename(t['file'])
            plan['file_structure'] = [
                os.path.basename(f) for f in plan.get('file_structure', [])
            ]

        # Remove web files from non-web projects
        if project_type in ('CLI', 'LIB', 'DATA'):
            web_exts = {'.html', '.css', '.js', '.htm'}
            plan['tasks'] = [
                t for t in plan.get('tasks', [])
                if os.path.splitext(t.get('file', ''))[1].lower() not in web_exts
            ]
            plan['file_structure'] = [
                f for f in plan.get('file_structure', [])
                if os.path.splitext(f)[1].lower() not in web_exts
            ]

        tasks = []
        for i, t in enumerate(plan.get('tasks', []), 1):
            tasks.append({
                'id': t.get('id', f't{i:02d}'),
                'title': t.get('title', f'Task {i}'),
                'file': t.get('file', ''),
                'description': t.get('description', ''),
                'depends_on': t.get('depends_on', []),
                'status': ProjectMemory.STATUS_PENDING,
                'error': '',
                'strategy': 'normal',
            })
        plan['tasks'] = tasks
        return plan

    # ═════════════════════════════════════════════════════════
    # PHASE 2 — ⚙️ CHANGE 2+3: Per-step build with structured spec
    # ═════════════════════════════════════════════════════════

    def _build_task(self, task: Dict, context: str,
                    strategy: str = 'normal') -> Tuple[bool, str]:
        coding_prompt = self._plan_task(task, context, strategy)
        return self._code_task(task, coding_prompt, context, strategy)

    # ⚙️ CHANGE 2: PLANNER phase — structured spec per task
    def _plan_task(self, task: Dict, context: str,
                   strategy: str = 'normal') -> str:
        ext = os.path.splitext(task['file'])[1].lower()
        lang_map = {'.py': 'Python', '.html': 'HTML', '.css': 'CSS',
                    '.js': 'JavaScript', '.json': 'JSON', '.txt': 'text',
                    '.sh': 'bash'}
        lang = lang_map.get(ext, 'code')

        strategy_note = {
            'normal': '',
            'alternative': (
                '\n⚠️ PREVIOUS ATTEMPTS FAILED. Use COMPLETELY DIFFERENT approach.'
            ),
            'simplified': (
                '\n⚠️ ALL ATTEMPTS FAILED. Write SIMPLEST possible version.'
            ),
        }[strategy]

        # ⚙️ CHANGE 5: Only pass relevant existing exports, not full files
        existing_exports = []
        for rel in self._list_project_files():
            if rel == task['file']:
                continue
            content = self._read_file(rel)
            if not content:
                continue

            fns = re.findall(r'^(?:def|class)\s+(\w+)', content, re.M)
            routes = re.findall(r"@app\.route\(['\"]([^'\"]+)['\"]", content)
            jsfns = re.findall(r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\()', content)
            summary = f"  • {rel}"
            if fns:
                summary += f" — defines: {', '.join(fns[:6])}"
            if routes:
                summary += f" — routes: {', '.join(routes[:4])}"
            jsfns_flat = [x[0] or x[1] for x in jsfns if x[0] or x[1]]
            if jsfns_flat:
                summary += f" — functions: {', '.join(jsfns_flat[:4])}"
            existing_exports.append(summary)

        exports_block = '\n'.join(existing_exports) if existing_exports else '  (none yet)'

        # ⚙️ CHANGE 6+8: Strict constraints in spec prompt
        prompt = f"""You are a STRICT {lang} SPECIFICATION WRITER.
Your ONLY job is to write a technical spec. Do NOT write code.
{strategy_note}

PROJECT: {self.pm.goal}
TECH: {', '.join(self.pm.tech_stack)}
FILE: {task['file']} ({lang})
TASK: {task['description']}

EXISTING FILES:
{exports_block}

CONTEXT:
{context[:3000] if context else '(none)'}

Write a CONCRETE spec with:
1. IMPORTS: every import needed
2. CONSTANTS: any config values
3. FUNCTIONS/CLASSES: exact name, signature, return type, behavior
4. CONNECTIONS: how it links to other files
5. MAIN: entry point

Rules:
- Be SPECIFIC — the coder must write complete code from this alone
- No code, only specification
- No explanations outside the spec"""

        print(f"  📋 Spec [{strategy}]: {task['file']}…")
        spec = self._brain(prompt, num_predict=900,
                           temperature=DEFAULT_PLAN_TEMP if strategy == 'normal' else 0.4)
        return spec

    # ⚙️ CHANGE 3: BUILDER phase — one step execution
    def _code_task(self, task: Dict, coding_prompt: str, context: str,
                   strategy: str = 'normal') -> Tuple[bool, str]:
        ext = os.path.splitext(task['file'])[1].lower()
        lang_map = {'.py': 'Python', '.html': 'HTML', '.css': 'CSS',
                    '.js': 'JavaScript', '.json': 'JSON', '.txt': 'text',
                    '.md': 'Markdown', '.sql': 'SQL', '.sh': 'bash'}
        lang = lang_map.get(ext, 'code')
        primer = '<' if ext in ('.html', '.htm') else ''

        # ⚙️ CHANGE 6+8: Strict builder constraints
        builder_system = (
            f"You are a STRICT {lang} DEVELOPER in an autonomous build system.\n"
            "RULES (violation = system crash):\n"
            "1. Output ONLY raw file content — no explanations, no markdown\n"
            "2. File MUST be COMPLETE — every function fully implemented\n"
            "3. NO placeholders: no 'TODO', no 'pass # implement', no '...'\n"
            "4. NO truncation — write every line\n"
            "5. ALL files same directory — simple filenames for imports\n"
            f"6. Start immediately with first character of {lang} file\n"
            "7. Failure to follow these rules = system crash"
        )

        builder_prompt = (
            f"PROJECT: {self.pm.goal}\n"
            f"FILE: {task['file']}\n\n"
            f"SPECIFICATION:\n{coding_prompt}\n\n"
            f"EXISTING FILES:\n{context[:2500] if context else '(none)'}\n\n"
            f"Write COMPLETE {task['file']}. Follow spec exactly.\n"
            f"Implement EVERY function. Output ONLY {lang} code."
        )

        print(f"  🔨 Building [{strategy}]: {task['file']}…")
        raw = self._builder(builder_system, builder_prompt, primer=primer)
        content = self._sanitise(raw, task['file'])

        if not content or len(content.strip()) < 30:
            return False, "Output empty or too short (< 30 chars)"

        real_placeholders = re.findall(
            r'^(?!.*#.*(?:TODO|FIXME)).*\b(?:raise\s+NotImplementedError|pass\s*$)',
            content, re.MULTILINE | re.IGNORECASE,
        )
        stub_markers = re.findall(
            r'#\s*(?:TODO|FIXME|IMPLEMENT|ADD HERE|your code here)',
            content, re.IGNORECASE,
        )
        total_stubs = len(real_placeholders) + len(stub_markers)
        if total_stubs > 3:
            return False, (
                f"File has {total_stubs} unimplemented stubs "
                f"(pass/NotImplementedError/TODO comments)"
            )

        return True, content

    # ═════════════════════════════════════════════════════════
    # Sanitise
    # ═════════════════════════════════════════════════════════

    def _sanitise(self, raw: str, filename: str) -> str:
        ext = os.path.splitext(filename)[1].lower()
        raw = raw.strip()

        if raw.startswith('```'):
            lines = raw.splitlines()
            raw = '\n'.join(lines[1:])
            fence = raw.rfind('```')
            if fence != -1:
                raw = raw[:fence]

        if ext in ('.html', '.htm'):
            m = re.search(r'<(!DOCTYPE|html|head|body)', raw, re.I)
            if m and m.start() > 0:
                raw = raw[m.start():]
            m2 = re.search(r'</html>', raw, re.I)
            if m2:
                raw = raw[:m2.end()]
        elif ext == '.css':
            idx = raw.rfind('}')
            if idx != -1:
                raw = raw[:idx + 1]
        elif ext in ('.js', '.py'):
            lines = raw.splitlines()
            while lines and ' ' in lines[-1] and not any(
                c in lines[-1] for c in ['{', '}', '(', ')', '=', ':', '#', '/']
            ):
                lines.pop()
            raw = '\n'.join(lines)

        return raw.strip()

    # ═════════════════════════════════════════════════════════
    # ⚙️ CHANGE 4+10: Review with JSON-structured output
    # ═════════════════════════════════════════════════════════

    def _review_task(self, task: Dict, content: str) -> Tuple[bool, str]:
        context = self._read_existing_files_context(max_chars=2500)

        # ⚙️ CHANGE 1: JSON-structured review
        review = self.models.chat_complete_json(
            [
                {'role': 'system', 'content': (
                    "You are a STRICT CODE REVIEWER. Output ONLY valid JSON.\n"
                    'Schema: {"status": "pass|fail", '
                    '"issues": [{"type": "completeness|imports|syntax|implementation", '
                    '"description": "what is wrong"}]}'
                )},
                {'role': 'user', 'content': (
                    f"PROJECT: {self.pm.goal}\n"
                    f"FILE: {task['file']}\n"
                    f"TASK: {task['description']}\n\n"
                    f"CODE:\n{content[:1800]}\n\n"
                    f"OTHER FILES:\n{context[:1500]}\n\n"
                    "Check: completeness, imports, syntax, no stubs/TODOs."
                )},
            ],
            num_predict=200,
            temperature=DEFAULT_REVIEW_TEMP,
        )

        if not review:
            # Fallback: assume pass if can't parse
            return True, 'OK (review parse failed)'

        if review.get('status') == 'pass':
            return True, 'OK'

        issues = review.get('issues', [])
        feedback = '; '.join(
            f"[{i.get('type', '?')}] {i.get('description', '?')}"
            for i in issues
        ) or 'Unknown issue'
        return False, feedback

    # ═════════════════════════════════════════════════════════
    # ⚙️ CHANGE 4: Repair with structured feedback
    # ═════════════════════════════════════════════════════════

    def _repair_task(self, task: Dict, error: str, prev_content: str,
                     strategy: str = 'normal') -> Tuple[bool, str]:
        instruction = self._repair_plan(task, error, prev_content, strategy)
        return self._repair_code(task, instruction, prev_content, strategy)

    def _repair_plan(self, task: Dict, error: str, prev_content: str,
                     strategy: str = 'normal') -> str:
        context = self._read_existing_files_context(max_chars=2000)
        strategy_note = {
            'normal': '',
            'alternative': '⚠️ Normal repair failed. Use COMPLETELY DIFFERENT approach.',
            'simplified': '⚠️ All repairs failed. Write SIMPLEST possible version.',
        }[strategy]

        # ⚙️ CHANGE 6: Strict repair prompt
        prompt = f"""You are a STRICT DEBUGGING EXPERT. Produce an exact repair instruction.

FILE: {task['file']}
TASK: {task['description']}
ERROR: {error}
{strategy_note}

BROKEN CODE:
{prev_content[:1500]}

OTHER FILES:
{context[:1000]}

Rules:
- Be SPECIFIC about what is wrong (reference code content, not line numbers)
- Describe EXACTLY what to change
- List missing imports
- List incomplete functions
- No code, only repair instruction"""

        print(f"  🔍 Repair-spec [{strategy}]: {error[:60]}…")
        return self._brain(
            prompt, num_predict=600,
            temperature=DEFAULT_PLAN_TEMP if strategy == 'normal' else 0.4,
        )

    def _repair_code(self, task: Dict, repair_instruction: str,
                     prev_content: str, strategy: str = 'normal') -> Tuple[bool, str]:
        ext = os.path.splitext(task['file'])[1].lower()
        lang_map = {'.py': 'Python', '.html': 'HTML', '.css': 'CSS', '.js': 'JavaScript'}
        lang = lang_map.get(ext, 'code')
        primer = '<' if ext in ('.html', '.htm') else ''

        print(f"  🔧 Repair-code [{strategy}]: {task['file']}…")

        # ⚙️ CHANGE 8: Strict role in repair
        system = (
            f"You are a STRICT {lang} REPAIR SPECIALIST.\n"
            "Rules (violation = system crash):\n"
            "- Output ONLY the complete fixed file\n"
            "- Every function fully implemented\n"
            "- No TODOs, no placeholders\n"
            "- No explanations, no markdown"
        )
        prompt = (
            f"BROKEN {task['file']}:\n{prev_content[:1500]}\n\n"
            f"FIX INSTRUCTION:\n{repair_instruction}\n\n"
            f"Write COMPLETE fixed {task['file']}. Output ONLY the file."
        )
        raw = self._builder(system, prompt, num_predict=3000, primer=primer)
        content = self._sanitise(raw, task['file'])

        if not content or len(content.strip()) < 20:
            return False, "Repair output empty"
        return True, content

    # ═════════════════════════════════════════════════════════
    # Integration check — ⚙️ CHANGE 1: JSON structured
    # ═════════════════════════════════════════════════════════

    def _phase_integration_check(self) -> List[str]:
        self._banner("Integration Check")

        dep_issues = self._check_dependencies()
        if dep_issues:
            print(f"  ⚠️ Dependency issues: {len(dep_issues)}")
            for iss in dep_issues[:5]:
                print(f"    • {iss}")

        context = self._read_existing_files_context(max_chars=5000)
        files = self._list_project_files()

        # ⚙️ CHANGE 1: JSON-structured integration check
        check_result = self.models.chat_complete_json(
            [
                {'role': 'system', 'content': (
                    "You are a STRICT INTEGRATION CHECKER. Output ONLY valid JSON.\n"
                    'Schema: {"status": "all_clear|issues_found", '
                    '"issues": [{"file": "filename", "description": "what is wrong", '
                    '"fix": "how to fix it"}]}'
                )},
                {'role': 'user', 'content': (
                    f"PROJECT: {self.pm.goal}\n"
                    f"TECH: {', '.join(self.pm.tech_stack)}\n"
                    f"FILES: {', '.join(files)}\n"
                    f"All files in same directory.\n\n"
                    f"CODE:\n{context}\n\n"
                    "Check: imports resolve, no circular deps, no TODOs, "
                    "functions implemented, correct linking."
                )},
            ],
            num_predict=600,
            temperature=DEFAULT_REVIEW_TEMP,
        )

        if check_result and check_result.get('status') == 'all_clear':
            print("  ✅ Integration check passed.")
            all_issues = dep_issues
            return all_issues

        issues = []
        if check_result and check_result.get('issues'):
            for issue in check_result['issues']:
                desc = issue.get('description', '')
                if desc:
                    issues.append(desc)

        issues.extend(dep_issues)
        issues = list(dict.fromkeys(issues))
        print(f"  ⚠️ {len(issues)} integration issue(s) found.")
        return issues

    def _fix_integration_issue(self, issue: str) -> bool:
        context = self._read_existing_files_context(max_chars=4000)
        files = self._list_project_files()

        # ⚙️ CHANGE 1: JSON-structured fix guidance
        guidance = self.models.chat_complete_json(
            [
                {'role': 'system', 'content': (
                    "You are a STRICT FIX PLANNER. Output ONLY valid JSON.\n"
                    'Schema: {"file": "filename to fix", "fix": "description of change"}'
                )},
                {'role': 'user', 'content': (
                    f"ISSUE: {issue}\n"
                    f"PROJECT: {self.pm.goal}\n"
                    f"FILES: {', '.join(files)}\n\n"
                    f"CODE:\n{context[:2000]}\n\n"
                    "Which file needs fixing and what change is needed?"
                )},
            ],
            num_predict=200,
            temperature=DEFAULT_PLAN_TEMP,
        )

        if not guidance or not guidance.get('file'):
            print(f"  ⚠️ Could not determine fix target for: {issue[:60]}")
            return False

        target_file = guidance['file']
        fix_instr = guidance.get('fix', issue)

        existing = self._read_file(target_file)
        if not existing:
            print(f"  ⚠️ File not found: {target_file}")
            return False

        ext = os.path.splitext(target_file)[1].lower()
        primer = '<' if ext in ('.html', '.htm') else ''
        lang = {'.py': 'Python', '.html': 'HTML', '.css': 'CSS', '.js': 'JavaScript'}.get(ext, 'code')

        system = (
            f"STRICT {lang} FIXER. Output ONLY complete fixed file.\n"
            "No placeholders. No explanations."
        )
        prompt = (
            f"FILE ({target_file}):\n{existing[:2000]}\n\n"
            f"FIX NEEDED:\n{fix_instr}\n\n"
            f"Write COMPLETE fixed {target_file}."
        )
        raw = self._builder(system, prompt, num_predict=3000, primer=primer)
        content = self._sanitise(raw, target_file)

        if content and len(content.strip()) > 20:
            self._write_file(target_file, content)
            print(f"  ✅ Fixed {target_file}")
            self._propagate_changes(target_file)
            return True

        print(f"  ❌ Fix failed for {target_file}")
        return False

    # ═════════════════════════════════════════════════════════
    # Final stability loop
    # ═════════════════════════════════════════════════════════

    def _phase_final_stability(self):
        self._banner("Final Stability Loop")
        MAX_STABILITY_ROUNDS = 3

        for rnd in range(1, MAX_STABILITY_ROUNDS + 1):
            if self._stop:
                break
            print(f"\n  🔄 Round {rnd}/{MAX_STABILITY_ROUNDS}…")
            issues = self._phase_integration_check()

            if not issues:
                print("  ✅ Project is stable.")
                break

            fixed = 0
            for issue in issues[:5]:
                if self._stop:
                    break
                print(f"\n  🔧 Fixing: {issue[:80]}")
                if self._fix_integration_issue(issue):
                    fixed += 1
            print(f"\n  Round {rnd} done — fixed {fixed}/{len(issues)} issues.")

    # ═════════════════════════════════════════════════════════
    # Emergency brain review
    # ═════════════════════════════════════════════════════════

    def _emergency_brain_review(self):
        print("\n  🚨 Emergency brain review…")
        context = self._read_existing_files_context(max_chars=3500)
        failed = self.pm.failed_tasks()
        errors = self.pm.errors[-3:]

        prompt = f"""Senior developer debugging a failing build.

PROJECT: {self.pm.goal}
FAILED: {json.dumps(failed, indent=2)[:500]}
ERRORS: {json.dumps(errors, indent=2)[:300]}

FILES:\n{context[:2000]}

Give a 5-bullet action plan. Be specific."""

        advice = self._brain(prompt, num_predict=400, temperature=DEFAULT_CHAT_TEMP)
        print(f"\n  💡 Brain advice:\n{advice}\n")

    # ═════════════════════════════════════════════════════════
    # Documentation
    # ═════════════════════════════════════════════════════════

    def _generate_docs(self):
        self._banner("Generating documentation")

        files = [f for f in self._list_project_files()
                 if not f.endswith('.bak') and f != 'documentation.txt']

        errors_summary = '\n'.join(
            f"  • {e['task']}: {e['error'][:70]} → {e.get('fix', 'no fix')[:50]}"
            for e in self.pm.errors[-6:]
        ) or '  None'

        self._refresh_dep_graph()
        dep_summary = '\n'.join(
            f"  {src} → {', '.join(deps)}"
            for src, deps in self._dep_graph.items() if deps
        ) or '  (no cross-file dependencies detected)'

        prompt = f"""Write clean documentation.

PROJECT: {self.pm.goal}
TECH: {', '.join(self.pm.tech_stack)}
FILES:
{chr(10).join('  • ' + f for f in files)}
TASKS: {len(self.pm.completed)}/{len(self.pm.tasks)} completed
ITERATIONS: {self.pm.iteration}

DEPENDENCY GRAPH:
{dep_summary}

ERRORS AND FIXES:
{errors_summary}

Sections:
# Project Overview
# Features
# File Structure
# Tech Stack
# How to Run
# Problems & Fixes
# Final Status"""

        print("  📝 Generating docs…")
        content = self._brain(prompt, num_predict=1200, temperature=DEFAULT_CHAT_TEMP)
        self._write_file('documentation.txt', content)
        print("  ✅ documentation.txt written.")

    # ═════════════════════════════════════════════════════════
    # MAIN RUN LOOP — ⚙️ CHANGE 3+9+10: Step-by-step with checkpoints
    # ═════════════════════════════════════════════════════════

    def run(self):
        try:
            self._run_inner()
        finally:
            self._restore_sigint()

    def _run_inner(self):
        self._banner("🚀 AGENTCODE v3 — Structured Pipeline Builder")
        print(f"  Request : {self.request}")
        print(f"  Folder  : {self.project_root}")
        print(f"  Hardware: RTX 4050 6GB VRAM | 8GB RAM | Ryzen 5")
        print(f"  Models  : Brain={CHAT_MODEL.split(':')[0]} Builder={CODE_MODEL.split(':')[0]}")
        print(f"  Mode    : Structured JSON pipeline (v3)")
        print(f"  Strategy: normal → alternative → simplified")
        print(f"  Ctrl+C  : stop after current phase\n")

        # ── Resume check ─────────────────────────────────────
        if self.pm.load():
            done = len(self.pm.completed)
            pending = len(self.pm.pending_tasks())
            print(f"  📦 Found existing state: {done} done, {pending} pending")
            print(f"     Goal: {self.pm.goal}")
            if input("  Resume? (y/n): ").strip().lower() != 'y':
                print("  Starting fresh…")
                self.pm.reset()
                self.checkpoint.clear()

        # ── Phase 1: Plan ────────────────────────────────────
        if not self.pm.has_state():
            plan = self._phase_plan()
            if not plan:
                print("  ❌ [FATAL] Planning failed.")
                return

            self.pm.init(
                goal=plan.get('goal', self.request),
                tech_stack=plan.get('tech_stack', []),
                file_structure=plan.get('file_structure', []),
                tasks=plan['tasks'],
            )

            print(f"\n  📋 Plan:")
            print(f"     Goal : {self.pm.goal}")
            print(f"     Tech : {', '.join(self.pm.tech_stack)}")
            print(f"     Tasks: {len(self.pm.tasks)}")
            print()
            for t in self.pm.tasks:
                deps = f" (needs: {','.join(t['depends_on'])})" if t['depends_on'] else ''
                print(f"     [{t['id']}] {t['title']} → {t['file']}{deps}")
            print()

            # ⚙️ CHANGE 9: Checkpoint after planning
            self.checkpoint.save_step('plan', 'done', json.dumps(plan)[:500])

        total_tasks = len(self.pm.tasks)

        # ── Phase 2-4: Tiered build ─────────────────────────
        MAX_REPAIR_CYCLES = 3

        tiers = self._compute_tiers(tier_size=3)
        self._banner(
            f"Build: {len(tiers)} tier(s), {total_tasks} tasks"
        )

        swap_log: List[str] = []

        for tier_idx, tier_tasks in enumerate(tiers, 1):
            if self._stop:
                break

            if all(t['status'] == ProjectMemory.STATUS_DONE for t in tier_tasks):
                print(f"  ✅ Tier {tier_idx}: already complete.")
                continue

            tier_pending = [t for t in tier_tasks
                           if t['status'] == ProjectMemory.STATUS_PENDING]
            if not tier_pending:
                continue

            self._banner(
                f"TIER {tier_idx}/{len(tiers)} — "
                f"{', '.join(t['file'] for t in tier_pending)}"
            )
            print(self.pm.summary())

            context = self._read_existing_files_context()

            # ── PHASE A — PLANNER: specs ─────────────────────
            print(f"\n  [Phase A] 🧠 PLANNER: {len(tier_pending)} spec(s)…")
            self.models.ensure_chat()
            swap_log.append(f"T{tier_idx}-A: LLaMA")

            plans: Dict[str, Dict] = {}
            for task in tier_pending:
                if self._stop:
                    break
                strategy = task.get('strategy', 'normal')
                self.pm.set_task_status(task['id'], ProjectMemory.STATUS_RUNNING)
                coding_prompt = self._plan_task(task, context, strategy)
                plans[task['id']] = {'prompt': coding_prompt, 'strategy': strategy}

            if self._stop:
                break

            # ── PHASE B — BUILDER: code ──────────────────────
            print(f"\n  [Phase B] 🔨 BUILDER: {len(tier_pending)} file(s)…")
            self.models.ensure_code()
            swap_log.append(f"T{tier_idx}-B: DeepSeek")

            built: Dict[str, Dict] = {}
            for task in tier_pending:
                if self._stop:
                    break
                plan_data = plans.get(task['id'], {})
                strategy = plan_data.get('strategy', 'normal')
                ok, result = self._code_task(
                    task, plan_data.get('prompt', ''), context, strategy
                )
                built[task['id']] = {
                    'content': result if ok else '',
                    'strategy': strategy,
                    'ok': ok,
                    'error': '' if ok else result,
                }
                if not ok:
                    self.pm.log_error(task['id'], result)
                    self._consecutive_failures += 1

                # ⚙️ CHANGE 9: Checkpoint after each build
                self.checkpoint.save_step(
                    f"build_{task['id']}",
                    'done' if ok else 'failed',
                    result[:200] if ok else f"ERROR: {result[:200]}",
                )

            # Write successful files
            written_this_tier: List[str] = []
            for task in tier_pending:
                data = built.get(task['id'], {})
                if data.get('ok') and data.get('content'):
                    self._write_file(task['file'], data['content'])
                    written_this_tier.append(task['file'])

            if self._stop:
                break

            # ── PHASE C — REVIEWER: validate ─────────────────
            print(f"\n  [Phase C] 🧠 REVIEWER: {len(tier_pending)} file(s)…")
            self.models.ensure_chat()
            swap_log.append(f"T{tier_idx}-C: LLaMA (review)")

            prop_fixes = self._propagate_tier_brain(written_this_tier)

            needs_repair: List[Dict] = []
            for task in tier_pending:
                data = built.get(task['id'], {})
                content = data.get('content', '')
                if not content:
                    needs_repair.append({
                        'task': task,
                        'content': '',
                        'error': data.get('error', 'Build failed'),
                    })
                    continue

                ok, feedback = self._review_task(task, content)
                if ok:
                    self.pm.set_task_status(task['id'], ProjectMemory.STATUS_DONE)
                    self._consecutive_failures = 0
                    print(f"  ✅ {task['file']} ({len(content):,} chars)")
                    # ⚙️ CHANGE 9: Checkpoint
                    self.checkpoint.save_step(f"review_{task['id']}", 'done')
                else:
                    print(f"  ⚠️ {task['file']}: {feedback[:80]}")
                    needs_repair.append({
                        'task': task,
                        'content': content,
                        'error': feedback,
                    })
                    self.checkpoint.save_step(f"review_{task['id']}", 'failed', feedback[:200])

            if self._stop:
                break

            # ── PHASES D+E — ⚙️ CHANGE 4+10: Repair loop ────
            for repair_cycle in range(1, MAX_REPAIR_CYCLES + 1):
                if self._stop or not needs_repair:
                    break

                print(f"\n  [Phase D{repair_cycle}] 🔨 REPAIR: {len(needs_repair)} file(s)…")
                self.models.ensure_code()
                swap_log.append(f"T{tier_idx}-D{repair_cycle}: DeepSeek")

                repaired: Dict[str, Dict] = {}
                for item in needs_repair:
                    if self._stop:
                        break
                    task = item['task']
                    error = item['error']
                    prev = item['content']
                    strategy = task.get('strategy', 'normal')

                    if self._seen_before(task['id'], error):
                        if strategy == 'normal':
                            strategy = 'alternative'
                            print(f"  ⚠️ Same error → ALTERNATIVE: {task['file']}")
                        elif strategy == 'alternative':
                            strategy = 'simplified'
                            print(f"  ⚠️ Still stuck → SIMPLIFIED: {task['file']}")
                        task['strategy'] = strategy

                    repair_instr = self._repair_plan(task, error, prev, strategy)
                    ok, content = self._repair_code(task, repair_instr, prev, strategy)

                    repaired[task['id']] = {
                        'content': content if ok else prev,
                        'strategy': strategy,
                        'ok': ok,
                        'error': error if not ok else '',
                    }
                    if not ok:
                        self.pm.log_error(task['id'], error)

                    # ⚙️ CHANGE 9: Checkpoint
                    self.checkpoint.save_step(
                        f"repair_{task['id']}_c{repair_cycle}",
                        'done' if ok else 'failed',
                    )

                # Write repaired files
                repaired_files: List[str] = []
                for task in [i['task'] for i in needs_repair]:
                    data = repaired.get(task['id'], {})
                    if data.get('ok') and data.get('content'):
                        self._write_file(task['file'], data['content'])
                        repaired_files.append(task['file'])
                        built[task['id']]['content'] = data['content']

                if self._stop:
                    break

                # Phase E: Re-review
                print(f"\n  [Phase E{repair_cycle}] 🧠 RE-REVIEW: {len(needs_repair)} file(s)…")
                self.models.ensure_chat()
                swap_log.append(f"T{tier_idx}-E{repair_cycle}: LLaMA")

                prop_fixes.update(self._propagate_tier_brain(repaired_files))

                still_failing: List[Dict] = []
                for item in needs_repair:
                    task = item['task']
                    data = repaired.get(task['id'], {})
                    content = data.get('content', '')
                    strategy = data.get('strategy', 'normal')

                    if not content:
                        still_failing.append(item)
                        continue

                    ok, feedback = self._review_task(task, content)
                    if ok:
                        self.pm.set_task_status(task['id'], ProjectMemory.STATUS_DONE)
                        self._consecutive_failures = 0
                        self.pm.log_error(
                            task['id'], item['error'],
                            fix=f'Repaired [{strategy}] cycle {repair_cycle}'
                        )
                        print(f"  ✅ {task['file']} fixed [{strategy}]")
                        self.checkpoint.save_step(f"fixed_{task['id']}", 'done')
                    else:
                        still_failing.append({
                            'task': task,
                            'content': content,
                            'error': feedback,
                        })
                        print(f"  ⚠️ {task['file']} still failing: {feedback[:60]}")

                needs_repair = still_failing
                if not needs_repair:
                    break

            # Mark remaining as failed
            for item in needs_repair:
                task = item['task']
                strategy = task.get('strategy', 'normal')
                content = item.get('content', '')

                if content and len(content.strip()) > 30:
                    self._write_file(task['file'], content)
                    print(f"  ⚠️ {task['file']} written (best-effort)")
                    self.pm.set_task_status(
                        task['id'], ProjectMemory.STATUS_FAILED,
                        f"Issues after {MAX_REPAIR_CYCLES} cycles [{strategy}]",
                    )
                else:
                    self.pm.set_task_status(
                        task['id'], ProjectMemory.STATUS_FAILED,
                        f"No output after {MAX_REPAIR_CYCLES} cycles [{strategy}]",
                    )
                    print(f"  ❌ {task['file']} FAILED")
                self._consecutive_failures += 1

            self.pm._save()

            if self._consecutive_failures >= self.REPAIR_THRESHOLD:
                self.models.ensure_chat()
                self._emergency_brain_review()
                self._consecutive_failures = 0

            # Propagation
            if prop_fixes and not self._stop:
                print(f"\n  [Propagation] 🔧 {len(prop_fixes)} fix(es)…")
                self.models.ensure_code()
                swap_log.append(f"T{tier_idx}-P: DeepSeek")
                self._propagate_tier_builder(prop_fixes)

            self._refresh_dep_graph()
            self.pm.bump_iteration()

        # ── Retry pass for failed tasks ──────────────────────
        failed_tasks = self.pm.failed_tasks()
        if failed_tasks and not self._stop:
            self._banner(f"RETRY — {len(failed_tasks)} failed task(s)")

            for t in failed_tasks:
                t['status'] = ProjectMemory.STATUS_PENDING
                t['strategy'] = 'simplified'
                t['error'] = ''
            self.pm._save()

            retry_tiers = self._compute_tiers(tier_size=3)
            for tier_idx, tier_tasks in enumerate(retry_tiers, 1):
                if self._stop:
                    break
                tier_pending = [t for t in tier_tasks
                               if t['status'] == ProjectMemory.STATUS_PENDING]
                if not tier_pending:
                    continue

                self._banner(f"RETRY TIER {tier_idx}")
                context = self._read_existing_files_context()

                self.models.ensure_chat()
                retry_plans: Dict[str, str] = {}
                for task in tier_pending:
                    if self._stop:
                        break
                    self.pm.set_task_status(task['id'], ProjectMemory.STATUS_RUNNING)
                    retry_plans[task['id']] = self._plan_task(task, context, 'simplified')

                if self._stop:
                    break

                self.models.ensure_code()
                for task in tier_pending:
                    if self._stop:
                        break
                    spec = retry_plans.get(task['id'], '')
                    ok, content = self._code_task(task, spec, context, 'simplified')
                    if ok and content:
                        self._write_file(task['file'], content)
                        self.pm.set_task_status(task['id'], ProjectMemory.STATUS_DONE)
                        print(f"  ✅ Retry succeeded: {task['file']}")
                    else:
                        if content and len(content.strip()) > 30:
                            self._write_file(task['file'], content)
                            print(f"  ⚠️ Retry partial: {task['file']}")
                        self.pm.set_task_status(
                            task['id'], ProjectMemory.STATUS_FAILED,
                            "Failed in retry pass"
                        )
                self.pm._save()

        # ── Final stability ──────────────────────────────────
        if not self._stop:
            self._phase_final_stability()

        # ── Documentation ────────────────────────────────────
        if not self._stop:
            self._generate_docs()

        # ── Summary ──────────────────────────────────────────
        self._banner("🏁 AGENTCODE v3 COMPLETE")
        done = len(self.pm.completed)
        total = len(self.pm.tasks)
        failed = len(self.pm.failed_tasks())

        print(f"  ✅ Completed : {done}/{total}")
        print(f"  ❌ Failed    : {failed}")
        print(f"  📦 Tiers     : {len(tiers)}")
        print(f"  🔄 Swaps     : {len(swap_log)}")
        print(f"  📁 Files     : {len(self._list_project_files())}")
        print(f"\n  Project : {self.project_root}")
        print(f"  Docs    : documentation.txt")

        if len(swap_log) <= 20:
            print(f"\n  Swap log: {' → '.join(swap_log)}")

        if failed:
            print(f"\n  ⚠️ Failed:")
            for t in self.pm.failed_tasks():
                print(f"    • {t['file']} [{t.get('strategy', '?')}]"
                      f" — {t.get('error', '?')[:60]}")
        print()

    def _pick_next_task(self, pending: List[Dict]) -> Optional[Dict]:
        done_ids = set(self.pm.completed)
        for task in pending:
            if all(d in done_ids for d in task.get('depends_on', [])):
                return task
        return None


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def main():
    assistant = AIAssistant()
    try:
        assistant.run()
    except KeyboardInterrupt:
        print("\n👋 Exiting.")
    except Exception as e:
        logger.exception("Unhandled exception in main loop")
        print(f"\n  ❌ [FATAL] {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
