import sqlite3
import os
from datetime import datetime

# Mock the regex behavior from ai_router.py to test extraction
import re

_EXTRACT_PATTERNS = [
    (r"(?:my|the user(?:'s)?) name is ['\"]?([\w][\w\s]*?)(?:['\"])?(?:\s|$)", 'user_name'),
    (r"(?:your|the assistant(?:'s)?) name is ['\"]?([\w][\w\s]*?)(?:['\"])?(?:\s|$)", 'assistant_name'),
    (r"act (?:like|as) (?:a |an )?(.+?)(?:\s*$)", 'behavior'),
    (r"(?:i am|i'm) (?:a |an )?([\w][\w\s]{1,30}?)(?:\s*$)", 'user_role'),
    (r"(?:i (?:work|works) (?:at|for)|employed (?:at|by)) ([^.\n]+)", 'user_employer'),
]

def test_extraction(text):
    stored = []
    for pattern, key in _EXTRACT_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip().rstrip("'\"")
            if value:
                stored.append((key, value))
    return stored

def verify_persistence():
    db_path = 'memory.db'
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Test adding
    name = "Asmik"
    now = datetime.now().isoformat()
    cur.execute(
        "INSERT INTO memory (key, value, created_at, updated_at) VALUES (?,?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        ('user_name', name, now, now),
    )
    conn.commit()
    
    # Test retrieval
    cur.execute("SELECT value FROM memory WHERE key='user_name'")
    row = cur.fetchone()
    conn.close()
    
    if row and row[0] == name:
        return True
    return False

if __name__ == "__main__":
    print(f"Extraction test ('my name is asmik'): {test_extraction('my name is asmik')}")
    print(f"Extraction test ('i am a student'): {test_extraction('i am a student')}")
    if verify_persistence():
        print("Persistence test: PASSED")
    else:
        print("Persistence test: FAILED")
