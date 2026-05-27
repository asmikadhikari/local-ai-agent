import os
import sys
import sqlite3
import subprocess
import time

def check_file(path):
    exists = os.path.exists(path)
    print(f"[FILE] {path}: {'✅ Found' if exists else '❌ MISSING'}")
    return exists

def check_package(name):
    try:
        __import__(name)
        print(f"[PKG]  {name}: ✅ Installed")
        return True
    except ImportError:
        print(f"[PKG]  {name}: ❌ NOT INSTALLED")
        return False

def check_db(path):
    if not os.path.exists(path):
        return
    try:
        conn = sqlite3.connect(path)
        c = conn.cursor()
        if 'conversations.db' in path:
            c.execute("SELECT COUNT(*) FROM conversations")
            count = c.fetchone()[0]
            print(f"[DB]   {path}: ✅ Accessible ({count} conversations)")
        elif 'memory.db' in path:
            c.execute("SELECT COUNT(*) FROM memory")
            count = c.fetchone()[0]
            print(f"[DB]   {path}: ✅ Accessible ({count} items)")
        conn.close()
    except Exception as e:
        print(f"[DB]   {path}: ❌ Error: {e}")

def check_ollama():
    try:
        import ollama
        models = ollama.list().get('models', [])
        print(f"[LLM]  Ollama: ✅ Running ({len(models)} models)")
        for m in models[:3]:
             print(f"       - {m['model']}")
    except Exception as e:
        print(f"[LLM]  Ollama: ❌ Error/Not Running: {e}")

if __name__ == "__main__":
    print("--- AIAgent Diagnostic ---")
    check_file("server.py")
    check_file("ai_router.py")
    check_file("static/index.html")
    
    check_package("flask")
    check_package("flask_cors")
    check_package("duckduckgo_search")
    check_package("ollama")
    
    check_db("conversations.db")
    check_db("memory.db")
    
    check_ollama()
    print("--- Diagnostic Complete ---")
