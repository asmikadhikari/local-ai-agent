
import os
import sys
import time
import logging

# Set up logging for visibility
logging.basicConfig(level=logging.INFO)

# Import the modified BrowserManager from ai_router.py
try:
    from ai_router import BrowserManager, CHROME_USER_DATA_DIR
    print(f"Successfully imported BrowserManager.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def verify_fix():
    print("--- Verifying Chrome Fixes ---")
    print(f"User Data Dir: {CHROME_USER_DATA_DIR}")
    
    browser = BrowserManager(headless=False)
    try:
        print("1. Testing Browser Launch and 'Restore pages' fix...")
        # This will trigger _ensure which calls _fix_chrome_exit_type
        page = browser.get_active_page()
        
        print(f"2. Navigating to Google...")
        browser.open_url("https://www.google.com")
        
        print("3. Testing Typing Fix (sequential typing)...")
        # Google search box selector
        selector = 'textarea[name="q"]'
        result = browser.type_text(selector, "AIAgent automated test", press_enter=False)
        print(f"Result: {result}")
        
        print("Verification steps complete. Please check the browser window.")
        time.sleep(5)
    except Exception as e:
        print(f"Verification failed: {e}")
    finally:
        browser.close()

if __name__ == "__main__":
    verify_fix()
