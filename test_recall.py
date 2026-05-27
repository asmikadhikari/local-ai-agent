import sqlite3
import os
import sqlite_vec
import ollama
from typing import List, Dict

DB_PATH = 'conversations.db'
EMBED_MODEL = 'nomic-embed-text'
EMBED_DIM = 768

def test_recall(query: str):
    print(f"Testing recall for: {query}")
    try:
        resp = ollama.embed(model=EMBED_MODEL, input=query)
        query_emb = resp['embeddings'][0]
        query_emb_bytes = sqlite_vec.serialize_float32(query_emb)
    except Exception as e:
        return f"Embedding failed: {e}"

    try:
        with sqlite3.connect(DB_PATH) as con:
            con.enable_load_extension(True)
            sqlite_vec.load(con)

            # Check if vec_messages table exists
            cur = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_messages'")
            if not cur.fetchone():
                return "vec_messages table NOT found in conversations.db"

            rows = con.execute(
                """
                SELECT m.role, m.content, m.timestamp
                FROM vec_messages v
                JOIN messages m ON v.rowid = m.id
                ORDER BY vec_distance_cosine(v.embedding, ?)
                LIMIT 5
                """,
                (query_emb_bytes,),
            ).fetchall()
            
            items = []
            for r in rows:
                items.append(f"[{r[0]} at {r[2]}]: {r[1][:100]}...")
            return items
    except Exception as e:
        return f"Query failed: {e}"

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print("conversations.db not found")
    else:
        results = test_recall("bank security and passwords")
        if isinstance(results, list):
            print("Found relevant context:")
            for res in results:
                print(f"  - {res}")
        else:
            print(results)
