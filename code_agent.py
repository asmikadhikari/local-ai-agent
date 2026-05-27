"""
CodeAgent — Full dependency graph and multi-file code generation.
Now uses raw generation for web files (HTML/CSS/JS) to avoid JSON parsing issues.
"""

import os
import re
import json
import shutil
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Set, Tuple

try:
    import ollama
except ImportError:
    ollama = None

# Token counting
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text, disallowed_special=()))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text) // 4

logger = logging.getLogger(__name__)

# Settings
_SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.mypy_cache'}
_SKIP_EXTS = {'.pyc', '.pyo', '.o', '.obj', '.class', '.exe', '.dll', '.so',
              '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg',
              '.pdf', '.zip', '.tar', '.gz', '.bz2', '.7z', '.whl',
              '.ttf', '.woff', '.woff2', '.eot', '.mp3', '.mp4', '.wav'}
_MAX_FILE_CHARS = 8_000
_MAX_FILES = 500

# Web file extensions that should be generated as raw code (no JSON)
_WEB_EXTS = {'.html', '.htm', '.css', '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'}

class CodeAgent:
    CODING_MODEL = 'deepseek-coder:6.7b-instruct-q4_K_M'

    def __init__(self, project_root: str, max_depth: int = 3, max_context_tokens: int = 6_000):
        self.project_root = os.path.abspath(project_root)
        self.max_depth = max_depth
        self.max_context_tokens = max_context_tokens
        self.backup_dir = os.path.join(self.project_root, ".code_backups")
        os.makedirs(self.backup_dir, exist_ok=True)

        self.file_list = []
        self.depends_on = {}
        self.depended_by = {}
        self.file_content_cache = {}
        self.symbols_defined = {}
        self.symbols_used = {}
        self.last_scan = None

        self._build_graph()

    # ------------------------------------------------------------------
    # Graph building (full detection layers) – unchanged
    # ------------------------------------------------------------------
    def _build_graph(self):
        logger.info("Building dependency graph for %s", self.project_root)
        self.file_list.clear()
        self.depends_on.clear()
        self.depended_by.clear()
        self.file_content_cache.clear()
        self.symbols_defined.clear()
        self.symbols_used.clear()

        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs
                       if not d.startswith('.')
                       and d not in _SKIP_DIRS
                       and os.path.join(root, d) != self.backup_dir]
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in _SKIP_EXTS:
                    continue
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, self.project_root)
                self.file_list.append(rel)
                self.depends_on[rel] = {}
                self.depended_by[rel] = set()
                self.symbols_defined[rel] = set()
                self.symbols_used[rel] = set()

        if len(self.file_list) > _MAX_FILES:
            logger.warning("Too many files (%d). Truncating to %d.", len(self.file_list), _MAX_FILES)
            self.file_list = self.file_list[:_MAX_FILES]

        for rel in self.file_list:
            abs_path = os.path.join(self.project_root, rel)
            content = self._read_file_safe(abs_path)
            if content is None:
                continue
            self.file_content_cache[rel] = content
            self.symbols_defined[rel] = self._extract_definitions(rel, content)
            self.symbols_used[rel] = self._extract_usages(rel, content)

        for rel in self.file_list:
            content = self.file_content_cache.get(rel, "")
            if not content:
                continue
            path_refs = self._extract_path_references(rel, content)
            for target in path_refs:
                self._add_edge(rel, target, "import", "strong", f"reference to {target}")
            siblings = self._find_naming_siblings(rel)
            for sib in siblings:
                self._add_edge(rel, sib, "naming_sibling", "weak", "naming convention")
            config_refs = self._extract_config_references(rel, content)
            for target in config_refs:
                self._add_edge(rel, target, "config", "medium", "config reference")
            data_refs = self._extract_data_flow_references(rel, content)
            for target in data_refs:
                self._add_edge(rel, target, "data_flow", "medium", "data flow")

        self._build_symbol_edges()
        self._build_api_edges()

        self.last_scan = datetime.now()
        edge_count = sum(len(v) for v in self.depends_on.values())
        logger.info("Graph built: %d files, %d edges", len(self.file_list), edge_count)

    def _add_edge(self, src: str, dst: str, edge_type: str, strength: str, what: str):
        if src == dst or dst not in self.depends_on:
            return
        metadata = {"type": edge_type, "strength": strength, "what": what}
        self.depends_on[src][dst] = metadata
        self.depended_by[dst].add(src)

    # ------------------------------------------------------------------
    # Detection methods – full implementations (unchanged)
    # ------------------------------------------------------------------
    def _extract_path_references(self, rel_path: str, content: str) -> Set[str]:
        ext = os.path.splitext(rel_path)[1].lower()
        base_dir = os.path.dirname(rel_path)
        refs = set()

        def resolve(ref: str) -> Optional[str]:
            ref = ref.strip().strip('\'"')
            if not ref or ref.startswith(('http://', 'https://', '//', 'data:')):
                return None
            ref = ref.split('?')[0].split('#')[0]
            if not ref:
                return None
            candidate = os.path.normpath(os.path.join(base_dir, ref))
            try:
                rel_cand = os.path.relpath(candidate, self.project_root)
            except ValueError:
                return None
            if rel_cand.startswith('..'):
                return None
            if rel_cand in self.depends_on:
                return rel_cand
            if '.' not in os.path.basename(rel_cand):
                for try_ext in ('.py', '.js', '.ts', '.jsx', '.tsx',
                                '.c', '.h', '.cpp', '.go', '.rb', '.php',
                                '.css', '.html', '.json', '.yaml', '.yml'):
                    with_ext = rel_cand + try_ext
                    if with_ext in self.depends_on:
                        return with_ext
            return None

        if ext in ('.html', '.htm', '.jinja', '.jinja2', '.j2'):
            for m in re.finditer(r'(?:href|src|action|data-src)\s*=\s*["\']([^"\']+)["\']', content):
                r = resolve(m.group(1))
                if r: refs.add(r)
            for m in re.finditer(r'@import\s+url\(["\']([^"\']+)["\']\)', content):
                r = resolve(m.group(1))
                if r: refs.add(r)

        elif ext in ('.css', '.scss', '.less', '.sass'):
            for m in re.finditer(r"@import\s+['\"]([^'\"]+)['\"]", content):
                r = resolve(m.group(1))
                if r: refs.add(r)
            for m in re.finditer(r"url\(['\"]?([^'\")\s]+)['\"]?\)", content):
                r = resolve(m.group(1))
                if r: refs.add(r)

        elif ext in ('.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.cs'):
            for m in re.finditer(r'#include\s+"([^"]+)"', content):
                r = resolve(m.group(1))
                if r: refs.add(r)

        elif ext in ('.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'):
            for m in re.finditer(r'(?:import|export)\s+.*?from\s+["\']([^"\']+)["\']', content):
                r = resolve(m.group(1))
                if r: refs.add(r)
            for m in re.finditer(r'require\(["\']([^"\']+)["\']\)', content):
                r = resolve(m.group(1))
                if r: refs.add(r)
            for m in re.finditer(r'import\s*\(["\']([^"\']+)["\']\)', content):
                r = resolve(m.group(1))
                if r: refs.add(r)

        elif ext == '.py':
            for m in re.finditer(r'^(?:from|import)\s+([\w.]+)', content, re.MULTILINE):
                mod = m.group(1).replace('.', os.sep)
                r = resolve(mod)
                if r: refs.add(r)

        elif ext == '.go':
            for m in re.finditer(r'import\s+"([^"]+)"', content):
                r = resolve(m.group(1).split('/')[-1])
                if r: refs.add(r)

        elif ext == '.rs':
            for m in re.finditer(r'mod\s+(\w+)\s*;', content):
                r = resolve(m.group(1))
                if r: refs.add(r)

        elif ext == '.rb':
            for m in re.finditer(r'require(?:_relative)?\s+["\']([^"\']+)["\']', content):
                r = resolve(m.group(1))
                if r: refs.add(r)

        elif ext == '.php':
            for m in re.finditer(r'(?:include|require)(?:_once)?\s*[("\']([^"\'()]+)["\')]', content):
                r = resolve(m.group(1))
                if r: refs.add(r)

        elif os.path.basename(rel_path).lower() in ('makefile', 'gnumakefile'):
            for m in re.finditer(r'include\s+(\S+)', content):
                r = resolve(m.group(1))
                if r: refs.add(r)

        for m in re.finditer(r'["\']([^"\']{3,}?)["\']', content):
            r = resolve(m.group(1))
            if r: refs.add(r)

        return refs

    def _extract_definitions(self, rel_path: str, content: str) -> Set[str]:
        syms = set()
        ext = os.path.splitext(rel_path)[1].lower()
        if ext == '.py':
            for m in re.finditer(r'^(?:def|class)\s+(\w+)', content, re.MULTILINE):
                syms.add(m.group(1))
        elif ext in ('.js', '.jsx', '.ts', '.tsx', '.mjs'):
            for m in re.finditer(
                r'(?:function\s+(\w+)|(?:const|let|var|class)\s+(\w+)\s*=?\s*(?:function|\(|class))',
                content
            ):
                syms.add(m.group(1) or m.group(2))
            for m in re.finditer(r'export\s+(?:default\s+)?(?:function|class)\s+(\w+)', content):
                syms.add(m.group(1))
        elif ext in ('.c', '.cpp', '.h', '.hpp'):
            for m in re.finditer(r'^\w[\w\s\*]+\s+(\w+)\s*\(', content, re.MULTILINE):
                syms.add(m.group(1))
        elif ext == '.go':
            for m in re.finditer(r'^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(', content, re.MULTILINE):
                syms.add(m.group(1))
        syms.discard('')
        return syms

    def _extract_usages(self, rel_path: str, content: str) -> Set[str]:
        return set(re.findall(r'\b([a-zA-Z_]\w{2,})\b', content))

    def _build_symbol_edges(self):
        all_defs = {}
        for rel, syms in self.symbols_defined.items():
            for sym in syms:
                all_defs[sym] = rel
        for rel, usages in self.symbols_used.items():
            for sym in usages:
                if sym in all_defs and all_defs[sym] != rel:
                    self._add_edge(rel, all_defs[sym], "shared_symbol", "medium", f"uses symbol '{sym}'")

    def _find_naming_siblings(self, rel_path: str) -> Set[str]:
        siblings = set()
        basename = os.path.basename(rel_path)
        stem = re.sub(
            r'(?:\.test|\.spec|\.model|\.view|\.controller|\.route|\.service|'
            r'\.handler|\.util|\.helper|Test|Spec|Service|Controller|Repository)$',
            '', os.path.splitext(basename)[0], flags=re.IGNORECASE
        ).lower()
        if len(stem) < 3:
            return siblings
        for other in self.file_list:
            if other == rel_path:
                continue
            other_base = os.path.basename(other)
            other_stem = os.path.splitext(other_base)[0].lower()
            if other_stem.startswith(stem) or stem.startswith(other_stem[:max(3, len(stem)-2)]):
                siblings.add(other)
        return siblings

    def _extract_config_references(self, rel_path: str, content: str) -> Set[str]:
        refs = set()
        basename = os.path.basename(rel_path).lower()
        if basename in ('package.json', 'package-lock.json'):
            for m in re.finditer(r'"(?:main|module|browser)"\s*:\s*"([^"]+)"', content):
                t = self._resolve_rel(os.path.dirname(rel_path), m.group(1))
                if t: refs.add(t)
        elif basename in ('docker-compose.yml', 'docker-compose.yaml'):
            for m in re.finditer(r'build:\s*(\S+)', content):
                t = self._resolve_rel(os.path.dirname(rel_path), m.group(1))
                if t: refs.add(t)
            for m in re.finditer(r'dockerfile:\s*(\S+)', content, re.IGNORECASE):
                t = self._resolve_rel(os.path.dirname(rel_path), m.group(1))
                if t: refs.add(t)
        elif basename in ('makefile', 'gnumakefile'):
            for m in re.finditer(r'include\s+(\S+)', content):
                t = self._resolve_rel(os.path.dirname(rel_path), m.group(1))
                if t: refs.add(t)
        elif basename in ('tsconfig.json', 'jsconfig.json'):
            for m in re.finditer(r'"(?:include|exclude|files)"\s*:\s*\[([^\]]+)\]', content):
                for item in re.findall(r'"([^"]+)"', m.group(1)):
                    t = self._resolve_rel(os.path.dirname(rel_path), item)
                    if t: refs.add(t)
        elif basename.endswith(('.yml', '.yaml')):
            for m in re.finditer(r'(?:file|path|source):\s*(\S+)', content):
                t = self._resolve_rel(os.path.dirname(rel_path), m.group(1))
                if t: refs.add(t)
        return refs

    def _extract_data_flow_references(self, rel_path: str, content: str) -> Set[str]:
        return set()

    def _resolve_rel(self, base_dir: str, ref: str) -> Optional[str]:
        ref = ref.strip().strip('\'"')
        if not ref:
            return None
        candidate = os.path.normpath(os.path.join(base_dir, ref))
        try:
            rel = os.path.relpath(candidate, self.project_root)
        except ValueError:
            return None
        if rel.startswith('..'):
            return None
        return rel if rel in self.depends_on else None

    def _build_api_edges(self):
        endpoint_defs = {}
        env_defs = {}
        for rel, content in self.file_content_cache.items():
            for m in re.finditer(r'(?:app|router|route)\.\w+\(["\'](/[^"\']{1,80})["\']', content):
                endpoint_defs[m.group(1)] = rel
            if rel.endswith('.env') or '.env.' in rel:
                for m in re.finditer(r'^([A-Z_][A-Z0-9_]+)\s*=', content, re.MULTILINE):
                    env_defs[m.group(1)] = rel
        for rel, content in self.file_content_cache.items():
            for ep, def_rel in endpoint_defs.items():
                if def_rel != rel and ep in content:
                    self._add_edge(rel, def_rel, "api", "medium", f"calls endpoint '{ep}'")
            for var, def_rel in env_defs.items():
                if def_rel != rel and var in content:
                    self._add_edge(rel, def_rel, "config", "medium", f"reads env var '{var}'")

    def _get_related(self, primary: str, visited: Optional[Set[str]] = None, depth: int = 0) -> Set[str]:
        if visited is None:
            visited = set()
        if depth > self.max_depth:
            return set()
        visited.add(primary)
        result = set()
        for dep in self.depends_on.get(primary, {}).keys():
            if dep not in visited:
                result.add(dep)
                result.update(self._get_related(dep, visited, depth + 1))
        for dep in self.depended_by.get(primary, set()):
            if dep not in visited:
                result.add(dep)
                result.update(self._get_related(dep, visited, depth + 1))
        return result

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    def _read_file_safe(self, abs_path: str) -> Optional[str]:
        try:
            with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception as e:
            logger.error("Failed to read %s: %s", abs_path, e)
            return None

    def _read_file(self, rel_path: str) -> Optional[str]:
        return self._read_file_safe(os.path.join(self.project_root, rel_path))

    def _write_file(self, rel_path: str, content: str, create_backup: bool = True):
        abs_path = os.path.join(self.project_root, rel_path)
        if create_backup and os.path.exists(abs_path):
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{os.path.basename(abs_path)}.{ts}.bak"
            backup_path = os.path.join(self.backup_dir, backup_name)
            shutil.copy2(abs_path, backup_path)
            logger.info("Backup created: %s", backup_path)
        os.makedirs(os.path.dirname(abs_path) or '.', exist_ok=True)
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info("Written: %s", abs_path)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _is_valid_html_structure(self, text: str) -> bool:
        t = text.lower().strip()

        # Accept any of these as "complete enough"
        # Rule 1: full document with <html>...</html>
        if '<html' in t and '</html>' in t and len(text) > 100:
            return True

        # Rule 2: doctype-declared document (modern minimalist HTML5)
        # e.g. <!DOCTYPE html><head>...</head><body>...</body>
        if '<!doctype' in t and '<body' in t and '</body>' in t and len(text) > 100:
            return True

        # Rule 3: partial but has both head and body sections (fragment editor output)
        if '<head' in t and '<body' in t and len(text) > 100:
            return True

        # Reject tiny outputs and pure-text explanations
        return False

    def _looks_like_code(self, text: str, filename: str) -> bool:
        """Validate that model output looks like a complete file, not a fragment."""
        text = text.strip()
        ext = os.path.splitext(filename)[1].lower()

        if ext in ('.html', '.htm'):
            result = self._is_valid_html_structure(text)
            if not result:
                # Print first 300 chars so user/logs can see what was rejected
                preview = text[:300].replace('\n', '↵')
                logger.warning("HTML completeness check FAILED. Output preview: %s", preview)
                print(f"\n  [DEBUG] DeepSeek output rejected (not a complete HTML file).")
                print(f"  [DEBUG] First 300 chars: {preview}\n")
            return result

        if ext == '.css':
            return '{' in text and '}' in text and len(text) > 50

        if ext in ('.js', '.mjs', '.cjs'):
            return any(kw in text for kw in ['function', 'const', 'let', 'var', '=>']) or len(text) > 100

        # Generic fallback — accept anything that looks like real source code
        if any(kw in text for kw in ['def ', 'class ', 'int main', '#include', 'function ', 'return ']):
            return True
        if '{' in text and '}' in text and len(text) > 100:
            return True
        # Last resort: if it's long enough and has no newlines it's probably an explanation, not code
        if len(text) > 200 and '\n' in text:
            return True
        return False

    def _is_web_file(self, filename: str) -> bool:
        """Return True if the file extension is in the web file set."""
        ext = os.path.splitext(filename)[1].lower()
        return ext in _WEB_EXTS

    # ------------------------------------------------------------------
    # Core API: analyze_and_generate (with raw mode for web files)
    # ------------------------------------------------------------------
    def _call_model(self, system_prompt: str, instruction: str) -> str:
        """
        Call DeepSeek.  We prime the assistant turn with the first character
        of expected output so the model cannot start with prose like
        "Sure, here is your updated...".  For HTML we prime with '<',
        for JSON with '{', otherwise leave blank.
        """
        # Detect expected output type from the system prompt
        if '"primary"' in system_prompt or 'JSON' in system_prompt:
            primer = '{'
        elif 'raw code' in system_prompt.lower() or 'html' in system_prompt.lower():
            primer = '<'
        else:
            primer = ''

        messages = [
            {'role': 'system',    'content': system_prompt},
            {'role': 'user',      'content': instruction},
        ]
        if primer:
            # Inject a partial assistant turn — Ollama will continue from here
            messages.append({'role': 'assistant', 'content': primer})

        resp = ollama.chat(
            model=self.CODING_MODEL,
            messages=messages,
            options={'num_predict': 4096, 'temperature': 0.2},
        )
        raw = resp['message']['content'].strip()
        # Re-attach the primer if the model didn't echo it back
        if primer and not raw.startswith(primer):
            raw = primer + raw
        return raw

    def analyze_and_generate(
        self,
        primary_file: str,
        instruction: str,
        extra_files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if extra_files is None:
            extra_files = []
        if ollama is None:
            return {'error': 'ollama package is not installed'}

        abs_primary = os.path.join(self.project_root, primary_file)
        is_new = not os.path.exists(abs_primary)

        # For web files, use raw generation (no JSON)
        if self._is_web_file(primary_file):
            logger.info("Using raw generation for web file: %s", primary_file)

            # Build context from graph
            if primary_file not in self.depends_on:
                self.depends_on[primary_file] = {}
                self.depended_by[primary_file] = set()

            related = self._get_related(primary_file)
            all_ctx = (set(extra_files) | related) - {primary_file}

            primary_content = self._read_file(primary_file) if not is_new else ""
            context_parts = []
            token_budget = self.max_context_tokens

            for rel in sorted(all_ctx):
                content = self._read_file(rel)
                if not content:
                    continue
                if len(content) > _MAX_FILE_CHARS:
                    content = content[:_MAX_FILE_CHARS] + "\n... (truncated)"
                tok = count_tokens(content)
                if tok > token_budget:
                    logger.warning("Skipping %s — exceeds token budget (%d tokens)", rel, tok)
                    continue
                token_budget -= tok
                context_parts.append((rel, content))

            # Build prompt
            lines = [
                "You are an expert programmer. You have access to the following project files.\n",
            ]
            if primary_content:
                lines.append(f"PRIMARY FILE ({primary_file}):\n```\n{primary_content}\n```\n")
            else:
                lines.append(f"PRIMARY FILE ({primary_file}) does not exist yet — create it.\n")
            if context_parts:
                lines.append("RELATED FILES (for context, do NOT modify them unless explicitly requested):\n")
                for fname, fcontent in context_parts:
                    lines.append(f"--- {fname} ---\n```\n{fcontent}\n```\n")
            lines.append(f"TASK: {instruction}\n")
            lines.append(
                "IMPORTANT RULES:\n"
                "- You MUST output the ENTIRE file content with the changes applied. Do NOT output only the changed parts.\n"
                "- If the instruction is vague (like 'fix the mistake'), examine the file and figure out what's wrong.\n"
                "- For HTML, preserve all existing content, only add what's missing.\n"
                "- Output ONLY the raw code (no markdown fences, no extra text).\n"
                "- Include helpful comments using the appropriate comment syntax.\n"
                "- Use proper indentation and follow language conventions.\n"
            )
            system = "\n".join(lines)

            # Generate raw code
            raw = self._call_model(system, instruction)
            logger.info("----- MODEL RAW OUTPUT (web file) START -----")
            logger.info(raw[:1000])
            logger.info("----- MODEL RAW OUTPUT END -----")

            # Validate completeness
            if self._looks_like_code(raw, primary_file):
                logger.info("Raw code passed validation.")
                return {'primary': raw}
            else:
                logger.warning("Raw code failed validation; retrying with stricter prompt.")
                stricter = system + """
CRITICAL:
Your previous response was incomplete.

You MUST:
- Return the FULL file content (not a fragment)
- Include <html>, <head>, <body>, </html> (for HTML)
- Output COMPLETE working code

If you output partial code again, it will be rejected.
"""
                raw2 = self._call_model(stricter, instruction)
                if self._looks_like_code(raw2, primary_file):
                    return {'primary': raw2}
                else:
                    return {'error': 'Model failed to output a complete file after retry'}

        # For non-web files, continue with JSON plan (existing logic)
        # ... (keep the existing code for JSON generation here)
        # For brevity, I'll include the existing JSON branch (same as before)
        # But to keep the file complete, I'll copy the rest from your previous version.

        # ----- Existing JSON branch (unchanged) -----
        # New file: generate directly (though web files already handled above)
        if is_new:
            logger.info("New file requested: %s", primary_file)
            system = (
                "You are an expert programmer.\n"
                "Output ONLY the complete file content with no extra explanation, "
                "no markdown fences, and no JSON. Just the raw code.\n"
                "Include descriptive comments in the code using the appropriate comment syntax for the language.\n"
            )
            try:
                code = self._call_model(system, instruction)
                code = self._strip_fences(code)
                return {'primary': code}
            except Exception as e:
                logger.exception("Code generation failed (new file)")
                return {'error': str(e)}

        # Existing file: dependency-aware JSON plan
        if primary_file not in self.depends_on:
            self.depends_on[primary_file] = {}
            self.depended_by[primary_file] = set()

        related = self._get_related(primary_file)
        all_ctx = (set(extra_files) | related) - {primary_file}

        primary_content = self._read_file(primary_file) or ""
        context_parts = []
        token_budget = self.max_context_tokens

        for rel in sorted(all_ctx):
            content = self._read_file(rel)
            if not content:
                continue
            if len(content) > _MAX_FILE_CHARS:
                content = content[:_MAX_FILE_CHARS] + "\n... (truncated)"
            tok = count_tokens(content)
            if tok > token_budget:
                logger.warning("Skipping %s — exceeds token budget (%d tokens)", rel, tok)
                continue
            token_budget -= tok
            context_parts.append((rel, content))

        # Build system prompt (strong)
        lines = [
            "You are an expert programmer. You have access to the following project files.\n",
        ]
        if primary_content:
            lines.append(f"PRIMARY FILE ({primary_file}):\n```\n{primary_content}\n```\n")
        else:
            lines.append(f"PRIMARY FILE ({primary_file}) does not exist yet — create it.\n")
        if context_parts:
            lines.append("RELATED FILES (for context, do NOT modify them unless explicitly requested):\n")
            for fname, fcontent in context_parts:
                lines.append(f"--- {fname} ---\n```\n{fcontent}\n```\n")
        lines.append(f"TASK: {instruction}\n")
        lines.append(
            "IMPORTANT RULES:\n"
            "- You MUST output the ENTIRE file content with the changes applied. Do NOT output only the changed parts.\n"
            "- If the instruction is vague (like 'fix the mistake'), examine the file and figure out what's wrong.\n"
            "- For HTML/CSS/JS, preserve all existing content, only add what's missing.\n"
            "- Output ONLY a valid JSON object (no markdown, no extra text).\n"
            "- Include helpful comments in the code using the appropriate comment syntax for the language.\n"
            "- Use proper indentation and follow language conventions.\n"
            "JSON structure:\n"
            "{\n"
            '  "primary": "<complete new content for the primary file>",\n'
            '  "extra": {\n'
            '    "<filename>": "<complete new content>"\n'
            "  }\n"
            "}\n"
            "If no extra files need changes, omit the 'extra' field.\n"
            "All string values must use JSON string escaping (\\n for newlines, etc.).\n"
            "\n"
            "EXAMPLE for an HTML file that adds a missing CSS link:\n"
            "{\n"
            '  "primary": "<!DOCTYPE html>\\n<html>\\n<head>\\n    <link rel=\\"stylesheet\\" href=\\"style.css\\">\\n    ...</head>\\n<body>...</body>\\n</html>"\n'
            "}\n"
        )
        system = "\n".join(lines)

        # First attempt
        raw = self._call_model(system, instruction)
        logger.info("----- MODEL RAW OUTPUT (JSON) START -----")
        logger.info(raw[:1000])
        logger.info("----- MODEL RAW OUTPUT END -----")

        plan = self._parse_json_plan(raw)
        logger.info(f"Parsed plan keys: {list(plan.keys())}")

        # Helper to check if content is complete
        def is_complete(content, filename):
            if not content:
                return False
            return self._looks_like_code(content, filename)

        # If JSON parsing failed, try to accept raw code
        if 'error' in plan and ('cannot parse' in plan['error'] or 'Invalid JSON' in plan['error']):
            if is_complete(raw, primary_file):
                logger.info("Model output raw code; treating as primary file content.")
                return {'primary': raw}
            else:
                logger.warning("Raw output is incomplete; retrying with stronger prompt.")
                stricter = system + """
CRITICAL:
Your previous response was INVALID.

You MUST:
- Return FULL FILE (not fragment)
- Include <html>, <head>, <body>, </html> (for HTML)
- Output COMPLETE working code

If you output partial code again, it will be rejected.
"""
                raw2 = self._call_model(stricter, instruction)
                logger.info("----- RETRY MODEL RAW OUTPUT (JSON) START -----")
                logger.info(raw2[:1000])
                logger.info("----- RETRY MODEL RAW OUTPUT END -----")
                if is_complete(raw2, primary_file):
                    return {'primary': raw2}
                else:
                    return {'error': 'Model failed to output a complete file after retry'}

        # JSON succeeded, validate the primary content
        if 'primary' in plan:
            if not is_complete(plan['primary'], primary_file):
                logger.warning("JSON plan contains incomplete content; retrying with stricter prompt.")
                stricter = system + """
CRITICAL:
Your previous plan was incomplete.

You MUST:
- Return the ENTIRE file content inside the JSON "primary" field
- Include <html>, <head>, <body>, </html> (for HTML)
- Output COMPLETE working code

If you output partial code again, it will be rejected.
"""
                raw2 = self._call_model(stricter, instruction)
                plan2 = self._parse_json_plan(raw2)
                if 'error' in plan2:
                    return plan2
                if 'primary' in plan2 and is_complete(plan2['primary'], primary_file):
                    return plan2
                else:
                    return {'error': 'Model still produced incomplete content after retry'}
        else:
            # No primary field – treat as error
            return {'error': 'Model output missing "primary" field'}

        return plan

    def _strip_fences(self, text: str) -> str:
        """
        Remove markdown code fences and any trailing explanation text.

        Handles patterns like:
            ```html
            <html>...</html>
            ```
            This is the corrected file...   ← must be stripped
        """
        text = text.strip()

        # ── Strip opening fence ────────────────────────────────
        if text.startswith('```'):
            lines = text.splitlines()
            # Skip the opening fence line (e.g. ```html or just ```)
            start = 1 if lines[0].strip().startswith('```') else 0
            text = '\n'.join(lines[start:])

        # ── Strip closing fence + anything after it ────────────
        # Find the LAST ``` in the text; discard it and everything after
        fence_idx = text.rfind('```')
        if fence_idx != -1:
            text = text[:fence_idx]

        # ── Strip trailing explanation sentences ───────────────
        # If the last line(s) look like prose (no HTML/code chars) remove them
        lines = text.splitlines()
        while lines:
            last = lines[-1].strip()
            # Keep if it looks like code/markup
            if last.startswith('<') or last.startswith('{') or last.startswith('/') \
                    or last.endswith('>') or last.endswith('}') or last.endswith(';') \
                    or not last:
                break
            # Drop if it looks like a prose sentence (contains spaces and no tag chars)
            if ' ' in last and '<' not in last and '{' not in last:
                lines.pop()
            else:
                break
        text = '\n'.join(lines)

        return text.strip()

    def _parse_json_plan(self, raw: str) -> Dict[str, Any]:
        raw = self._strip_fences(raw)
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(raw)
            return obj
        except json.JSONDecodeError as e:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                candidate = match.group(0)
                try:
                    obj, _ = decoder.raw_decode(candidate)
                    return obj
                except json.JSONDecodeError as e2:
                    return {'error': f'Cannot parse model JSON: {e2}'}
            else:
                return {'error': f'Cannot parse model JSON: {e}'}

    # ------------------------------------------------------------------
    # Apply plan (with final validation guard)
    # ------------------------------------------------------------------
    def apply_plan(self, plan: Dict[str, Any], primary_file: str, extra_files: Optional[List[str]] = None) -> str:
        if 'error' in plan:
            return f"Error: {plan['error']}"

        if 'primary' not in plan:
            return "No changes to apply."

        content = plan['primary']

        # ── Sanitise: strip anything after the last closing tag ──
        content = self._sanitise_output(content, primary_file)

        # 🔥 FINAL SAFETY CHECK – block incomplete files
        if not self._looks_like_code(content, primary_file):
            logger.error("❌ BLOCKED: Model tried to write incomplete file!")
            return "❌ Blocked: Model output was incomplete. File NOT updated."

        self._write_file(primary_file, content, create_backup=True)
        report = f"✅ Primary file updated: {primary_file}\n"

        if 'extra' in plan:
            for fname, fcontent in plan['extra'].items():
                fcontent = self._sanitise_output(fcontent, fname)
                if not self._looks_like_code(fcontent, fname):
                    logger.warning(f"Skipped invalid extra file: {fname}")
                    continue
                self._write_file(fname, fcontent, create_backup=True)
                report += f"✅ Extra file updated: {fname}\n"

        return report

    def _sanitise_output(self, content: str, filename: str) -> str:
        """Strip trailing prose/explanation that models sometimes append after the code,
        and prose introductions before the actual code starts."""
        ext = os.path.splitext(filename)[1].lower()
        content = self._strip_fences(content)

        if ext in ('.html', '.htm'):
            # Strip any prose before the first HTML tag
            first_tag = re.search(r'<(!DOCTYPE|html|head|body)', content, re.IGNORECASE)
            if first_tag and first_tag.start() > 0:
                content = content[first_tag.start():]
            # Cut everything after the last </html>
            m = re.search(r'</html\s*>', content, re.IGNORECASE)
            if m:
                content = content[:m.end()]

        elif ext == '.css':
            # Strip prose before first selector/rule
            first_rule = re.search(r'[a-zA-Z\.\#\*\[\:]', content)
            if first_rule and first_rule.start() > 0:
                # Only strip if it's actually prose (contains a full sentence)
                prefix = content[:first_rule.start()]
                if len(prefix.split()) > 4:
                    content = content[first_rule.start():]
            # Cut after last closing brace
            idx = content.rfind('}')
            if idx != -1:
                content = content[:idx + 1]

        elif ext in ('.js', '.mjs', '.cjs', '.ts', '.tsx', '.jsx'):
            # Cut after last closing brace if followed only by prose lines
            idx = content.rfind('}')
            if idx != -1:
                tail = content[idx + 1:].strip()
                if '\n' not in tail or all(
                    not l.strip().startswith(('<', '{', '/'))
                    for l in tail.splitlines() if l.strip()
                ):
                    content = content[:idx + 1]

        return content.strip()

    def refresh(self):
        self._build_graph()

    # ------------------------------------------------------------------
    # Stage-2 Executor: apply a planner's plain-text plan
    # ------------------------------------------------------------------
    def apply_plan_with_code_model(
        self,
        primary_file: str,
        plan: str,
        instruction: str,
        extra_files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Two-stage executor (Stage 2).

        Receives:
          • primary_file  – the file to edit
          • plan          – plain-text plan produced by the Planner (LLaMA)
          • instruction   – the original user instruction (for context)
          • extra_files   – additional files explicitly mentioned by the user

        Combines them into the combined prompt format:
            FILE:  <content>
            PLAN:  <plan>
            TASK:  Apply the plan and fix the file.

        Returns a result dict compatible with apply_plan():
            {'primary': '<complete new file content>'}    on success
            {'error':   '<message>'}                      on failure
        """
        if extra_files is None:
            extra_files = []
        if ollama is None:
            return {'error': 'ollama package is not installed'}

        abs_primary = os.path.join(self.project_root, primary_file)
        is_new       = not os.path.exists(abs_primary)
        primary_content = (self._read_file(primary_file) or "") if not is_new else ""

        # Gather context from EXISTING files only (skip files that don't exist yet)
        if primary_file not in self.depends_on:
            self.depends_on[primary_file]  = {}
            self.depended_by[primary_file] = set()

        related  = self._get_related(primary_file)
        all_ctx  = (set(extra_files) | related) - {primary_file}

        context_parts: List[Tuple[str, str]] = []
        token_budget = self.max_context_tokens
        for rel in sorted(all_ctx):
            abs_rel = os.path.join(self.project_root, rel)
            if not os.path.exists(abs_rel):
                continue           # silently skip files that don't exist yet
            content = self._read_file(rel)
            if not content:
                continue
            if len(content) > _MAX_FILE_CHARS:
                content = content[:_MAX_FILE_CHARS] + "\n... (truncated)"
            tok = count_tokens(content)
            if tok > token_budget:
                logger.warning("Executor: skipping %s — exceeds token budget", rel)
                continue
            token_budget -= tok
            context_parts.append((rel, content))

        is_web = self._is_web_file(primary_file)

        # ── Build the combined prompt ──────────────────────────
        lines = ["You are a coding assistant.\n"]

        # FILE section
        if primary_content:
            lines.append(f"FILE ({primary_file}):\n{primary_content}\n")
        else:
            lines.append(f"FILE ({primary_file}) does not exist yet — CREATE IT FROM SCRATCH.\n")

        # Related files (only ones that already exist)
        if context_parts:
            lines.append("RELATED FILES (already written — use these for reference and linking):\n")
            for fname, fcontent in context_parts:
                lines.append(f"--- {fname} ---\n```\n{fcontent}\n```\n")

        # PLAN / SPEC section
        lines.append(f"SPECIFICATION:\n{plan}\n")

        # TASK section
        task_verb = "Create" if is_new else "Apply the plan and fix"
        lines.append(f"TASK:\n{task_verb} {primary_file}.\n")

        # Output rules — stricter for new files
        if is_web:
            new_file_extra = (
                "- This is a NEW file — write the ENTIRE file from the first line to the last\n"
                "- For HTML: start with <!DOCTYPE html> and end with </html>\n"
                "- For CSS: include ALL rules from the spec\n"
                "- For JS: include ALL functions from the spec\n"
            ) if is_new else ""
            lines.append(
                "Rules:\n"
                + new_file_extra +
                "- Follow the specification exactly\n"
                "- Output ONLY raw code — NO prose, NO markdown fences, NO explanations\n"
                "- Your response must start with the very first character of the file\n"
            )
        else:
            lines.append(
                "Rules:\n"
                "- Follow the specification exactly\n"
                "- Return the FULL file (no fragments)\n"
                'Output ONLY JSON: { "primary": "<complete file content>" }\n'
                "Important: no explanations, no partial output.\n"
            )

        system_prompt = "\n".join(lines)
        logger.info("Executor: sending combined prompt for %s", primary_file)

        # ── First attempt ──────────────────────────────────────
        raw = self._call_model(system_prompt, "Apply the plan.")
        logger.info("----- EXECUTOR RAW OUTPUT START -----")
        logger.info(raw[:1000])
        logger.info("----- EXECUTOR RAW OUTPUT END -----")

        result = self._parse_executor_output(raw, primary_file, is_web)
        if 'error' not in result:
            return result

        # ── Retry with a stricter prompt ───────────────────────
        logger.warning("Executor first attempt failed (%s) — retrying.", result['error'])
        stricter = system_prompt + (
            "\nCRITICAL — your previous response was rejected as incomplete.\n"
            "You MUST return the ENTIRE file content.\n"
            + ("Include <html>, <head>, <body>, </html>.\n" if is_web else "")
            + "No partial output. No explanations. FULL file only.\n"
        )
        raw2 = self._call_model(stricter, "Apply the plan.")
        logger.info("----- EXECUTOR RETRY OUTPUT START -----")
        logger.info(raw2[:1000])
        logger.info("----- EXECUTOR RETRY OUTPUT END -----")

        result2 = self._parse_executor_output(raw2, primary_file, is_web)
        if 'error' not in result2:
            return result2

        return {'error': f'Executor failed after retry: {result2["error"]}'}

    def _parse_executor_output(
        self, raw: str, primary_file: str, is_web: bool
    ) -> Dict[str, Any]:
        """
        Try to extract a complete file from the model's raw output.
        For web files: accept raw code directly.
        For all others: try JSON first, fall back to raw code.
        """
        stripped = self._strip_fences(raw)

        if is_web:
            if self._looks_like_code(stripped, primary_file):
                return {'primary': stripped}
            return {'error': 'Executor output failed completeness check'}

        # Try JSON
        plan_obj = self._parse_json_plan(raw)
        if 'error' not in plan_obj and 'primary' in plan_obj:
            content = plan_obj['primary']
            if self._looks_like_code(content, primary_file):
                return plan_obj
            return {'error': 'JSON primary field failed completeness check'}

        # Fallback: treat raw output as code
        if self._looks_like_code(stripped, primary_file):
            logger.info("Executor: JSON parse failed, accepted raw code as fallback.")
            return {'primary': stripped}

        return {'error': plan_obj.get('error', 'Output is not valid code and not valid JSON')}

    def show_graph_summary(self) -> str:
        lines = [
            f"Project root : {self.project_root}",
            f"Files scanned: {len(self.file_list)}",
            f"Total edges  : {sum(len(v) for v in self.depends_on.values())}",
            f"Last scan    : {self.last_scan}",
            "",
            "Top dependencies:",
        ]
        by_degree = sorted(self.depended_by.items(), key=lambda kv: len(kv[1]), reverse=True)
        for fname, dependents in by_degree[:10]:
            if dependents:
                lines.append(f"  {fname}  ← needed by {len(dependents)} file(s)")
        return '\n'.join(lines)

    def get_related_files(self, primary: str) -> List[str]:
        return sorted(self._get_related(primary))