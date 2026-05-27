"""
AI Assistant Web Server v2
─────────────────────────
Bridges ai_router.py to a browser UI with:
 • Full AIAssistant integration (same pipeline, same DB, same models)
 • Per-request activity streaming via SSE
 • SQLite database viewer
 • Main chat shows ONLY final answers; all intermediate steps → Activity tab
"""

import os, sys, json, time, re, queue, sqlite3, logging, threading, subprocess
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from flask import Flask, request, jsonify, Response, send_from_directory, stream_with_context
from flask_cors import CORS

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PORT       = int(os.environ.get('PORT', 5001))
DB_PATH    = os.path.join(SCRIPT_DIR, 'conversations.db')
MEMORY_DB  = os.path.join(SCRIPT_DIR, 'memory.db')

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger('ai_web')

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# ── Per-thread activity queue ────────────────────────────────────────────────
_tl = threading.local()   # .queue → Queue | None  /  .tokens_emitted → int

def emit(event: dict):
    q = getattr(_tl, 'queue', None)
    if q is not None:
        q.put(event)

def emit_act(icon: str, text: str):
    emit({'type': 'activity', 'icon': icon, 'text': text})

# ── Stdout capture ───────────────────────────────────────────────────────────
_ANSI = re.compile(r'\x1b\[[0-9;]*m')
_SKIP_PREFIXES = ('You: ', 'Assistant: ', '\nAssistant:')

class _Tee:
    """Route print() calls to per-thread activity queues without losing console output."""
    def __init__(self, orig):
        self._orig = orig
        self._buf  = ''

    def write(self, text):
        # Best-effort write to original stdout; fall back to safe encoding on Windows consoles
        try:
            self._orig.write(text)
        except UnicodeEncodeError:
            # Replace characters that cannot be represented in the current code page
            enc = getattr(self._orig, "encoding", None) or "utf-8"
            safe_text = text.encode(enc, errors="replace").decode(enc, errors="replace")
            self._orig.write(safe_text)
        q = getattr(_tl, 'queue', None)
        if q is None:
            return
        self._buf += text
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            clean = _ANSI.sub('', line).strip()
            if not clean:
                continue
            if any(clean.startswith(p.strip()) for p in _SKIP_PREFIXES):
                continue
            if len(clean) > 300:
                clean = clean[:300] + ' …'
            q.put({'type': 'activity', 'text': clean})

    def flush(self):   self._orig.flush()
    def isatty(self):  return False
    def fileno(self):  return self._orig.fileno()

_orig_stdout = sys.stdout
sys.stdout   = _Tee(_orig_stdout)

# ── ai_router integration ────────────────────────────────────────────────────
AI_ROUTER_OK = False
_ai          = None
_ai_lock     = threading.Lock()

sys.path.insert(0, SCRIPT_DIR)

def _patch(ModelManager, CHAT_MODEL, CODE_MODEL, ollama_mod):
    """Patch ModelManager so _stream emits token events instead of printing."""

    orig_stream = ModelManager._stream
    orig_load   = ModelManager.load

    def new_stream(self, model, messages, num_predict, temperature):
        emit_act('🧠', f'Generating with {model.split(":")[0]} …')
        chunks = []
        try:
            for chunk in ollama_mod.chat(
                model=model, messages=messages,
                options={'num_predict': num_predict, 'temperature': temperature,
                         'top_p': 0.9, 'repeat_penalty': 1.1},
                keep_alive=-1, stream=True,
            ):
                token = chunk.get('message', {}).get('content', '')
                if token:
                    chunks.append(token)
                    emit({'type': 'token', 'text': token})
                    setattr(_tl, 'tokens_emitted', getattr(_tl, 'tokens_emitted', 0) + 1)
        except Exception as e:
            emit_act('❌', f'Stream error: {e}')
        result = ''.join(chunks).strip()
        emit_act('✅', f'Done — {len(result)} chars')
        _orig_stdout.write('\n')
        return result

    def new_load(self, model):
        if self.current != model:
            emit_act('🔄', f'Loading model: {model.split(":")[0]} …')
        orig_load(self, model)

    ModelManager._stream = new_stream
    ModelManager.load    = new_load

try:
    import ai_router as _ar
    import ollama as _ollama_mod
    _patch(_ar.ModelManager, _ar.CHAT_MODEL, _ar.CODE_MODEL, _ollama_mod)
    _ai = _ar.AIAssistant()
    AI_ROUTER_OK = True
    log.info("✅ ai_router.py loaded and patched")
except Exception as e:
    log.warning("⚠️  ai_router not available: %s — using standalone fallback", e)
    try:
        import ollama as _ollama_sa
    except ImportError:
        _ollama_sa = None

# ── Standalone helpers (fallback when ai_router not available) ───────────────
def _standalone_stream(messages, model=None, temperature=0.3):
    if not _ollama_sa:
        yield "❌ Ollama not installed."
        return
    model = model or os.environ.get('CHAT_MODEL', 'llama3.1:8b-instruct-q4_K_M')
    try:
        for chunk in _ollama_sa.chat(
            model=model, messages=messages,
            options={'temperature': temperature, 'top_p': 0.9},
            keep_alive='10m', stream=True,
        ):
            t = chunk.get('message', {}).get('content', '')
            if t:
                yield t
    except Exception as ex:
        yield f"\n\n❌ {ex}"

def _web_search(query):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as d:
            results = list(d.text(query, max_results=5))
        return "\n\n".join(
            f"**{r.get('title','?')}**\n{r.get('href','')}\n{r.get('body','')[:200]}"
            for r in results
        ) or f"No results for '{query}'."
    except Exception as ex:
        return f"Search error: {ex}"

# ── DB helpers ───────────────────────────────────────────────────────────────
def _db(path=DB_PATH):    return sqlite3.connect(path)

def _ensure_dbs():
    with _db() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY, name TEXT,
                created_at TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT NOT NULL, timestamp TEXT NOT NULL,
                FOREIGN KEY (conv_id) REFERENCES conversations(id)
            );
        """)
    with _db(MEMORY_DB) as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS memory (
                key TEXT PRIMARY KEY, value TEXT NOT NULL,
                created_at TEXT, updated_at TEXT
            );
        """)

def _load_history(conv_id, limit=60):
    try:
        with _db() as c:
            rows = c.execute(
                "SELECT role, content FROM messages WHERE conv_id=? ORDER BY id DESC LIMIT ?",
                (conv_id, limit)
            ).fetchall()
        return [{'role': r, 'content': cnt} for r, cnt in reversed(rows)]
    except Exception:
        return []

def _new_conv(name=''):
    cid  = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    now  = datetime.now().isoformat()
    name = name or f"Chat {datetime.now().strftime('%b %d %H:%M')}"
    with _db() as c:
        c.execute("INSERT INTO conversations VALUES (?,?,?,?)", (cid, name, now, now))
    return cid

def _save_msg(conv_id, role, content):
    now = datetime.now().isoformat()
    with _db() as c:
        c.execute("INSERT INTO messages (conv_id,role,content,timestamp) VALUES (?,?,?,?)",
                  (conv_id, role, content, now))
        c.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conv_id))

def _rename_conv(conv_id, name):
    with _db() as c:
        c.execute("UPDATE conversations SET name=? WHERE id=?", (name, conv_id))

def _delete_conv(conv_id):
    with _db() as c:
        c.execute("DELETE FROM messages WHERE conv_id=?", (conv_id,))
        c.execute("DELETE FROM conversations WHERE id=?", (conv_id,))

# ── Help text ────────────────────────────────────────────────────────────────
HELP_TEXT = """\
## Commands

| Command | Description |
|---|---|
| `/search <query>` | Web search + AI summary |
| `/research <topic>` | Deep 3-phase research pipeline |
| `/news [topic]` | Top news |
| `/code <instruction>` | Code generation |
| `/agentcode <request>` | Autonomous project builder |
| `/analyze <files> <question>` | Audit files |
| `/task <goal>` | Autonomous multi-step browser agent |
| `/memory add <fact>` | Store persistent memory |
| `/memory recall` | Show memories |
| `/memory clear` | Clear memories |
| `/models` | List Ollama models |
| `/help` | This help |

Chat normally for conversation. Activity tab shows all background steps.
"""

# ── Command dispatch ─────────────────────────────────────────────────────────
def _dispatch(conv_id: str, message: str, q: queue.Queue):
    """
    Run in a background thread. Emits events to q:
      {type:'token', text:...}     → main chat bubble
      {type:'activity', text:...}  → Activity tab
      {type:'done', conv_id:...}   → close stream
    """
    setattr(_tl, 'queue', q)
    setattr(_tl, 'tokens_emitted', 0)
    response           = ''
    recording_done     = False   # True if handler already recorded to DB

    try:
        lower = message.lower().strip()

        # ── Commands that don't need the AI ────────────────────────────────
        if lower == '/help' or lower == 'help':
            q.put({'type': 'token', 'text': HELP_TEXT})
            response = HELP_TEXT

        elif lower.startswith('/memory'):
            args = message.strip()[7:].strip()
            parts = args.split(None, 1)
            sub   = parts[0].lower() if parts else 'recall'
            if sub == 'add' and len(parts) > 1:
                if AI_ROUTER_OK:
                    response = _ai.memory.add(parts[1])
                else:
                    now = datetime.now().isoformat()
                    with _db(MEMORY_DB) as c:
                        c.execute("INSERT OR REPLACE INTO memory VALUES (?,?,?,?)",
                                  (f"note_{datetime.now().strftime('%H%M%S')}", parts[1], now, now))
                    response = f"🧠 Stored: {parts[1]}"
            elif sub == 'clear':
                if AI_ROUTER_OK:
                    response = _ai.memory.clear()
                else:
                    with _db(MEMORY_DB) as c:
                        c.execute("DELETE FROM memory")
                    response = "🗑️ Memory cleared."
            else:
                if AI_ROUTER_OK:
                    response = _ai.memory.recall()
                else:
                    with _db(MEMORY_DB) as c:
                        rows = c.execute("SELECT key,value FROM memory ORDER BY updated_at DESC").fetchall()
                    response = "\n".join(f"• {k}: {v}" for k, v in rows) or "📭 No memories."
            q.put({'type': 'token', 'text': response})

        elif lower.startswith('/models'):
            try:
                mod = _ollama_mod if AI_ROUTER_OK else _ollama_sa
                names = [m['model'] for m in mod.list().get('models', [])]
                response = "**Available models:**\n" + "\n".join(f"• {n}" for n in names)
            except Exception as ex:
                response = f"❌ Ollama: {ex}"
            q.put({'type': 'token', 'text': response})

        # ── AI_ROUTER commands ──────────────────────────────────────────────
        elif AI_ROUTER_OK:
            _ai.conv_id = conv_id
            _ai.history = _load_history(conv_id)

            if lower.startswith('/search '):
                query = message[8:].strip()
                emit_act('🔍', f'Web search: {query}')
                # _cmd_search with stream=True (no return_string) streams via patched _stream
                _ai._cmd_search(query)
                # _cmd_search records internally when return_string=False
                recording_done = True
                # Grab the recorded response
                new_h = _ai.history[len(_load_history(conv_id)):]
                for m in new_h:
                    if m['role'] == 'assistant':
                        response = m['content']
                        break

            elif lower.startswith('/research '):
                topic = message[10:].strip()
                emit_act('🔬', f'Deep research: {topic}')
                emit_act('📝', 'Phase 1 — Planning initial draft …')
                hist_before = len(_ai.history)
                _ai._cmd_research(topic)   # prints phases → stdout → activity events
                recording_done = True
                # Grab recorded assistant message
                for m in _ai.history[hist_before:]:
                    if m['role'] == 'assistant':
                        response = m['content']
                        if _tl.tokens_emitted == 0:
                            q.put({'type': 'token', 'text': response})
                        break

            elif lower.startswith('/news'):
                topic = message[5:].strip()
                emit_act('📰', f'Fetching news: {topic or "top stories"}')
                response = _ai._cmd_news(topic, return_string=True) or ''
                q.put({'type': 'token', 'text': response})

            elif lower.startswith('/code '):
                instruction = message[6:].strip()
                emit_act('💻', f'Code task: {instruction[:80]}')
                emit_act('📋', 'Planning …')
                response = _ai._cmd_code(instruction, return_string=True) or ''
                if response and _tl.tokens_emitted == 0:
                    q.put({'type': 'token', 'text': response})

            elif lower.startswith('/agentcode'):
                req = message[10:].strip()
                emit_act('🤖', f'AgentCode: {req[:80]}')
                response = _ai._cmd_agentcode(req, return_string=True) or ''
                if response and _tl.tokens_emitted == 0:
                    q.put({'type': 'token', 'text': response})

            elif lower.startswith('/analyze') or lower.startswith('/analyse'):
                args = message.split(None, 1)[1].strip() if ' ' in message else ''
                emit_act('🔍', f'Analyzing: {args[:80]}')
                response = _ai._cmd_analyze(args, return_string=True) or ''
                if response and _tl.tokens_emitted == 0:
                    q.put({'type': 'token', 'text': response})

            elif lower.startswith('/task ') or lower.startswith('/agent '):
                goal = message.split(None, 1)[1].strip()
                emit_act('🕵️', f'Task agent: {goal[:80]}')
                try:
                    agent = _ar.TaskAgent(
                        goal=goal, models=_ai.models,
                        memory=_ai.memory, project_root=_ai.project_root,
                    )
                    response = agent.run() or ''
                    q.put({'type': 'token', 'text': response})
                except Exception as ex:
                    response = f"❌ Agent error: {ex}"
                    q.put({'type': 'token', 'text': response})

            elif lower.startswith('/undo'):
                fn = message[5:].strip()
                response = _ai._cmd_undo(fn, return_string=True) or ''
                q.put({'type': 'token', 'text': response})

            elif lower.startswith('/cd '):
                response = _ai._cmd_cd(message[4:], return_string=True) or ''
                q.put({'type': 'token', 'text': response})

            elif lower == '/refresh':
                response = _ai._cmd_refresh(return_string=True) or ''
                q.put({'type': 'token', 'text': response})

            else:
                intent = _ai.route_intent(message)
                if intent and intent.get('command') and intent['command'] != 'CHAT':
                    cmd = intent['command']
                    emit_act('🔀', f'Router decided: {cmd}')
                    q.put({'type': 'token', 'text': f'_(Router selected **{cmd}** based on your request.)_\n\n'})
                    
                    if cmd == '/search':
                        query = intent.get('param', message)
                        emit_act('🔍', f'Web search: {query[:80]}')
                        _ai._cmd_search(query)
                        recording_done = True
                        for m in reversed(_ai.history):
                            if m['role'] == 'assistant':
                                response = m['content']
                                if getattr(_tl, 'tokens_emitted', 0) == 0:
                                    q.put({'type': 'token', 'text': response})
                                break
                    elif cmd == '/news':
                        topic = intent.get('param', message)
                        emit_act('📰', f'News: {topic[:80]}')
                        response = _ai._cmd_news(topic, return_string=True) or ''
                        if response and getattr(_tl, 'tokens_emitted', 0) == 0:
                            q.put({'type': 'token', 'text': response})
                    elif cmd == '/research':
                        topic = intent.get('param', message)
                        emit_act('🔬', f'Deep research: {topic}')
                        emit_act('📝', 'Phase 1 — Planning initial draft …')
                        hist_before = len(_ai.history)
                        _ai._cmd_research(topic)
                        recording_done = True
                        for m in _ai.history[hist_before:]:
                            if m['role'] == 'assistant':
                                response = m['content']
                                if getattr(_tl, 'tokens_emitted', 0) == 0:
                                    q.put({'type': 'token', 'text': response})
                                break
                    elif cmd == '/code':
                        req = intent.get('param', message)
                        emit_act('💻', f'Code task: {req[:80]}')
                        response = _ai._cmd_code(req, return_string=True) or ''
                        if response and getattr(_tl, 'tokens_emitted', 0) == 0:
                            q.put({'type': 'token', 'text': response})
                    elif cmd == '/agent':
                        req = intent.get('param', message)
                        emit_act('🕵️', f'Task agent: {req[:80]}')
                        try:
                            agent = _ar.TaskAgent(
                                goal=req, models=_ai.models,
                                memory=_ai.memory, project_root=_ai.project_root,
                            )
                            response = agent.run() or ''
                            if getattr(_tl, 'tokens_emitted', 0) == 0:
                                q.put({'type': 'token', 'text': response})
                        except Exception as ex:
                            response = f"❌ Agent error: {ex}"
                            q.put({'type': 'token', 'text': response})
                    elif cmd == '/analyze':
                        req = intent.get('param', message)
                        emit_act('🔍', f'Analyzing: {req[:80]}')
                        response = _ai._cmd_analyze(req, return_string=True) or ''
                        if response and getattr(_tl, 'tokens_emitted', 0) == 0:
                            q.put({'type': 'token', 'text': response})
                    else:
                        emit_act('💬', 'Processing …')
                        response = _ai._process_with_tools(message)
                        if response and getattr(_tl, 'tokens_emitted', 0) == 0:
                            q.put({'type': 'token', 'text': response})
                else:
                    # Regular chat — call _process_with_tools directly
                    emit_act('💬', 'Processing …')
                    response = _ai._process_with_tools(message)
                    if response and getattr(_tl, 'tokens_emitted', 0) == 0:
                        q.put({'type': 'token', 'text': response})

        # ── Standalone fallback ─────────────────────────────────────────────
        else:
            history = _load_history(conv_id)
            if lower.startswith('/search '):
                emit_act('🔍', f'Web search: {message[8:]}')
                res = _web_search(message[8:])
                emit_act('📄', 'Summarizing …')
                msgs = [
                    {'role': 'system', 'content': 'Summarize these search results clearly.'},
                    {'role': 'user', 'content': f"Query: {message[8:]}\n\nResults:\n{res}"}
                ]
                for t in _standalone_stream(msgs):
                    response += t
                    q.put({'type': 'token', 'text': t})
            else:
                sys_msg = {'role': 'system',
                           'content': f"You are a helpful AI assistant. Today is {datetime.now().strftime('%B %d, %Y')}."}
                msgs = [sys_msg] + history[-10:] + [{'role': 'user', 'content': message}]
                emit_act('🧠', 'Generating response …')
                for t in _standalone_stream(msgs):
                    response += t
                    q.put({'type': 'token', 'text': t})

        # ── Record to DB ────────────────────────────────────────────────────
        if not recording_done and response:
            _save_msg(conv_id, 'user', message)
            _save_msg(conv_id, 'assistant', response)
            # Auto-name conversation from first real message
            with _db() as c:
                row = c.execute("SELECT name FROM conversations WHERE id=?", (conv_id,)).fetchone()
            if row and row[0].startswith('Chat '):
                _rename_conv(conv_id, message[:60])

    except Exception as ex:
        log.exception("dispatch error")
        err = f"\n\n❌ Error: {ex}"
        q.put({'type': 'activity', 'icon': '❌', 'text': str(ex)})
        q.put({'type': 'token', 'text': err})
    finally:
        q.put({'type': 'done', 'conv_id': conv_id})
        setattr(_tl, 'queue', None)

# ── Status ───────────────────────────────────────────────────────────────────
def _check_ollama():
    mod = (_ollama_mod if AI_ROUTER_OK else _ollama_sa) if (AI_ROUTER_OK or '_ollama_sa' in globals()) else None
    if not mod:
        return {'ok': False, 'error': 'ollama not installed'}
    try:
        models = [m['model'] for m in mod.list().get('models', [])]
        return {'ok': True, 'models': models}
    except Exception as ex:
        return {'ok': False, 'error': str(ex)}

# ── Flask routes ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json')

@app.route('/sw.js')
def sw():
    return send_from_directory('static', 'sw.js')


@app.route('/api/status')
def status():
    info = _check_ollama()
    chat_m = _ar.CHAT_MODEL if AI_ROUTER_OK else os.environ.get('CHAT_MODEL', '?')
    code_m = _ar.CODE_MODEL if AI_ROUTER_OK else os.environ.get('CODE_MODEL', '?')
    return jsonify({
        'ollama': info,
        'router': AI_ROUTER_OK,
        'chat_model': chat_m,
        'code_model': code_m,
    })

@app.route('/api/health')
def health():
    info = _check_ollama()
    return jsonify({
        'ok': True,
        'router': AI_ROUTER_OK,
        'ollama_ok': info.get('ok', False),
    })


# ── Conversations ─────────────────────────────────────────────────────────────
@app.route('/api/conversations', methods=['GET'])
def list_convs():
    with _db() as c:
        rows = c.execute(
            "SELECT id, name, updated_at, "
            "(SELECT COUNT(*) FROM messages WHERE conv_id=conversations.id) AS cnt "
            "FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return jsonify([{'id': r[0], 'name': r[1], 'updated_at': r[2][:16], 'count': r[3]} for r in rows])

@app.route('/api/conversations', methods=['POST'])
def create_conv():
    d = request.json or {}
    return jsonify({'id': _new_conv(d.get('name', ''))})

@app.route('/api/conversations/<cid>', methods=['DELETE'])
def del_conv(cid):
    _delete_conv(cid)
    return jsonify({'ok': True})

@app.route('/api/conversations/<cid>/rename', methods=['POST'])
def rename(cid):
    _rename_conv(cid, (request.json or {}).get('name', ''))
    return jsonify({'ok': True})

@app.route('/api/conversations/<cid>/messages', methods=['GET'])
def get_messages(cid):
    return jsonify(_load_history(cid, limit=200))


# ── Chat (SSE streaming) ──────────────────────────────────────────────────────
@app.route('/api/chat', methods=['POST'])
def chat():
    d       = request.json or {}
    message = d.get('message', '').strip()
    conv_id = d.get('conv_id', '')

    if not message:
        return jsonify({'error': 'empty'}), 400
    if not conv_id:
        conv_id = _new_conv()

    q = queue.Queue()

    def run():
        try:
            with _ai_lock:
                _dispatch(conv_id, message, q)
        except Exception as e:
            log.exception("Fatal error in dispatch")
            q.put({'type': 'activity', 'icon': '💀', 'text': f'Internal error: {e}'})
            q.put({'type': 'done', 'conv_id': conv_id})
        finally:
            q.put({'type': 'done', 'conv_id': conv_id})  # ensure it's sent

    threading.Thread(target=run, daemon=True).start()

    def generate():
        while True:
            try:
                ev = q.get(timeout=120)
            except queue.Empty:
                yield f"data: {json.dumps({'type':'activity','text':'⏱ Timeout'})}\n\n"
                yield f"data: {json.dumps({'type':'done','conv_id':conv_id})}\n\n"
                break
            yield f"data: {json.dumps(ev)}\n\n"
            if ev.get('type') == 'done':
                break

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no', 'Connection': 'keep-alive'},
    )


# ── Database viewer ──────────────────────────────────────────────────────────
@app.route('/api/db/overview')
def db_overview():
    def file_size(p):
        try:
            return os.path.getsize(p)
        except Exception:
            return 0

    with _db() as c:
        n_convs = c.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        n_msgs  = c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        oldest  = c.execute("SELECT MIN(created_at) FROM conversations").fetchone()[0]
    try:
        with _db(MEMORY_DB) as c:
            n_mem = c.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
    except Exception:
        n_mem = 0
    return jsonify({
        'conversations': n_convs,
        'messages':      n_msgs,
        'memory_items':  n_mem,
        'oldest_conv':   (oldest or '')[:10],
        'db_size_bytes': file_size(DB_PATH),
        'mem_size_bytes': file_size(MEMORY_DB),
    })

@app.route('/api/db/conversations')
def db_conversations():
    with _db() as c:
        rows = c.execute("""
            SELECT c.id, c.name, c.created_at, c.updated_at,
                   COUNT(m.id) AS msg_count
            FROM conversations c
            LEFT JOIN messages m ON m.conv_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        """).fetchall()
    return jsonify([{
        'id': r[0], 'name': r[1],
        'created_at': r[2][:16] if r[2] else '',
        'updated_at': r[3][:16] if r[3] else '',
        'msg_count': r[4],
    } for r in rows])

@app.route('/api/db/conversations/<cid>/messages')
def db_messages(cid):
    with _db() as c:
        rows = c.execute(
            "SELECT id, role, content, timestamp FROM messages WHERE conv_id=? ORDER BY id",
            (cid,)
        ).fetchall()
    return jsonify([{
        'id': r[0], 'role': r[1],
        'content': r[2][:500] + ('…' if len(r[2]) > 500 else ''),
        'full_content': r[2],
        'timestamp': r[3][:16] if r[3] else '',
        'chars': len(r[2]),
    } for r in rows])

@app.route('/api/db/memory')
def db_memory():
    try:
        with _db(MEMORY_DB) as c:
            rows = c.execute("SELECT key, value, updated_at FROM memory ORDER BY updated_at DESC").fetchall()
        return jsonify([{'key': r[0], 'value': r[1], 'updated_at': (r[2] or '')[:16]} for r in rows])
    except Exception:
        return jsonify([])

@app.route('/api/db/memory', methods=['DELETE'])
def clear_memory():
    with _db(MEMORY_DB) as c:
        c.execute("DELETE FROM memory")
    return jsonify({'ok': True})


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    _ensure_dbs()
    print(f"\n{'═'*56}")
    print(f"  🤖  AI Assistant Web UI v2")
    print(f"  http://localhost:{PORT}")
    if AI_ROUTER_OK:
        print(f"  Chat  : {_ar.CHAT_MODEL}")
        print(f"  Code  : {_ar.CODE_MODEL}")
        print(f"  DB    : {DB_PATH}")
    else:
        print(f"  Mode  : standalone (ai_router.py not found in {SCRIPT_DIR})")
    print(f"{'═'*56}\n")

    st = _check_ollama()
    if st['ok']:
        print(f"  ✅ Ollama — {len(st['models'])} model(s) available\n")
    else:
        print(f"  ⚠️  Ollama: {st['error']}\n  Run: ollama serve\n")

    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
