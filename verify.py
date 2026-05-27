import requests
import time

def check(url, name):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            print(f"[VERIFY] {name}: ✅ OK ({url})")
            return True
        else:
            print(f"[VERIFY] {name}: ⚠️ Status {r.status_code} ({url})")
            return False
    except Exception as e:
        print(f"[VERIFY] {name}: ❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    print("--- Final Verification ---")
    # Check Ollama
    o_ok = check("http://127.0.0.1:11434/api/tags", "Ollama API")
    
    # Check Server (it should be running in the background from previous run_command)
    s_ok = check("http://127.0.0.1:5001/api/health", "AIAgent Server")
    
    if o_ok and s_ok:
        print("\n🎉 Everything is working properly!")
    else:
        print("\n⚠️ Some components are still not responding as expected.")
