#!/usr/bin/env bash
# ═══════════════════════════════════════════════
#  AI Assistant Web UI — One-click launcher
#  Double-click this file or run: ./start.sh
# ═══════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=5001
URL="http://localhost:$PORT"

echo ""
echo "  ══════════════════════════════════════"
echo "   🤖  AI Assistant Web UI"
echo "  ══════════════════════════════════════"
echo ""

# 1. Check Python
if ! command -v python3 &>/dev/null; then
  echo "  ❌ Python3 not found. Please install Python 3.9+"
  exit 1
fi

# 2. Check / start Ollama
if ! pgrep -x ollama &>/dev/null; then
  echo "  🔄 Starting Ollama..."
  ollama serve &>/dev/null &
  sleep 3
  echo "  ✅ Ollama started"
else
  echo "  ✅ Ollama already running"
fi

# 3. Install Python deps if needed
cd "$SCRIPT_DIR"
NEEDED=0
python3 -c "import flask" 2>/dev/null || NEEDED=1
python3 -c "import flask_cors" 2>/dev/null || NEEDED=1
if [ $NEEDED -eq 1 ]; then
  echo "  📦 Installing Python dependencies..."
  pip3 install flask flask-cors --quiet --break-system-packages 2>/dev/null \
    || pip3 install flask flask-cors --quiet
fi

# 4. Kill any existing server on same port
lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
sleep 0.5

# 5. Start Flask server in background
echo "  🚀 Starting server on $URL"
python3 "$SCRIPT_DIR/server.py" &
SERVER_PID=$!
sleep 2

# 6. Open browser
echo "  🌐 Opening browser..."

if command -v google-chrome &>/dev/null; then
  google-chrome --app="$URL" --no-first-run &
elif command -v chromium-browser &>/dev/null; then
  chromium-browser --app="$URL" --no-first-run &
elif command -v chromium &>/dev/null; then
  chromium --app="$URL" &
elif [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS — try Chrome first, then default browser
  if open -a "Google Chrome" "$URL" 2>/dev/null; then
    echo "  ✅ Opened in Chrome"
  else
    open "$URL"
  fi
else
  xdg-open "$URL" 2>/dev/null || echo "  ℹ️  Open browser at: $URL"
fi

echo ""
echo "  ✅ AI Assistant running at $URL"
echo "  Press Ctrl+C to stop"
echo ""

# Wait for server
wait $SERVER_PID
