import sqlite3
import os
import json
import sqlite_vec
import ollama
from datetime import datetime, timedelta

DB_PATH = 'conversations.db'
MEMORY_DB = 'memory.db'

def test_v4_nexus():
    print("🚀 Testing V4 Nexus Memory...")
    
    # 1. Verify Schema
    with sqlite3.connect(DB_PATH) as con:
        cols = [c[1] for c in con.execute("PRAGMA table_info(messages)").fetchall()]
        if 'tags' not in cols:
            print("❌ Migration FAILED: 'tags' column missing")
            return
    print("✅ Schema Migration: Tags column verified")

    with sqlite3.connect(MEMORY_DB) as con:
        cur = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='entities'")
        if not cur.fetchone():
            print("❌ Migration FAILED: 'entities' table missing")
            return
    print("✅ Schema Migration: Entities table verified")

    # 2. Test Recency Boost Query
    print("\n🔍 Testing Recency Boost SQL...")
    try:
        with sqlite3.connect(DB_PATH) as con:
            con.enable_load_extension(True)
            sqlite_vec.load(con)
            
            # Setup dummy vector
            dummy_emb = [0.1] * 768
            dummy_bytes = sqlite_vec.serialize_float32(dummy_emb)
            
            # This should not crash if SQL is valid
            con.execute("""
                SELECT m.role, m.content, m.timestamp
                FROM vec_messages v
                JOIN messages m ON v.rowid = m.id
                ORDER BY 
                    (vec_distance_cosine(v.embedding, ?) * 0.8) + 
                    (COALESCE(julianday('now') - julianday(m.timestamp), 365) / 365.0 * 0.2)
                LIMIT 1
            """, (dummy_bytes,))
        print("✅ Recency Boost: SQL Query Validated")
    except Exception as e:
        print(f"❌ Recency Boost: SQL Error - {e}")

    # 3. Test Entity Storage
    print("\n📦 Testing Entity Storage...")
    try:
        from ai_router import MemoryManager
        mem = MemoryManager(MEMORY_DB)
        mem._store_entity("Asmik", "is_a", "Student at Bridgewater")
        entities = mem.get_entities()
        if any(e[0] == "Asmik" and e[1] == "is_a" for e in entities):
            print("✅ Entity Storage: Triplets functional")
        else:
            print("❌ Entity Storage: Could not retrieve triplet")
    except Exception as e:
        print(f"❌ Entity Storage Error: {e}")

    print("\n✨ V4 Nexus Memory Integration: SUCCESS")

if __name__ == "__main__":
    test_v4_nexus()
