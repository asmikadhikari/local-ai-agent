import sqlite3
import os

def check_memory():
    db_path = 'memory.db'
    if not os.path.exists(db_path):
        return "memory.db not found"
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    output = []
    try:
        cur.execute("SELECT * FROM memory")
        rows = cur.fetchall()
        output.append("--- Memory Table ---")
        for row in rows:
            output.append(str(row))
    except Exception as e:
        output.append(f"Error reading memory: {e}")
    conn.close()
    return "\n".join(output)

def check_conversations():
    db_path = 'conversations.db'
    if not os.path.exists(db_path):
        return "conversations.db not found"
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    output = []
    try:
        cur.execute("SELECT role, content FROM messages ORDER BY id DESC LIMIT 100")
        msgs = cur.fetchall()
        output.append("\n--- Recent Messages (Last 100) ---")
        for role, content in msgs:
            # Look for name-related strings
            content_lower = content.lower()
            if "name" in content_lower or "i am" in content_lower or "i'm" in content_lower:
                 output.append(f"[{role}]: {content[:200]}...") # truncate for readability
    except Exception as e:
        output.append(f"Error reading conversations: {e}")
    conn.close()
    return "\n".join(output)

if __name__ == "__main__":
    res_m = check_memory()
    res_c = check_conversations()
    with open("db_dump.txt", "w", encoding="utf-8") as f:
        f.write(res_m + "\n" + res_c)
