
import os
import json
import time
import platform
import subprocess
from playwright.sync_api import sync_playwright

def get_chrome_user_data_dir():
    if platform.system() == 'Windows':
        return os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Google', 'Chrome', 'User Data')
    return None

def fix_chrome_profile(user_data_dir):
    pref_path = os.path.join(user_data_dir, 'Default', 'Preferences')
    if not os.path.exists(pref_path):
        # Try without 'Default' if it's a different profile, but 'Default' is typical
        return
    
    try:
        with open(pref_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if data.get('profile', {}).get('exit_type') != 'Normal':
            print(f"Fixing exit_type in {pref_path}")
            data.setdefault('profile', {})['exit_type'] = 'Normal'
            data.setdefault('profile', {})['exited_cleanly'] = True
            with open(pref_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
    except Exception as e:
        print(f"Failed to fix profile: {e}")

def test_typing():
    user_data_dir = get_chrome_user_data_dir()
    if not user_data_dir:
        print("Could not find Chrome user data dir")
        return

    # Fix the "Restore pages" bubble
    fix_chrome_profile(user_data_dir)

    with sync_playwright() as p:
        chrome_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        if not os.path.exists(chrome_exe):
             chrome_exe = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"

        print(f"Launching Chrome from: {chrome_exe}")
        try:
            context = p.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                executable_path=chrome_exe,
                headless=False,
                args=[
                    '--no-first-run',
                    '--no-default-browser-check',
                    '--disable-infobars',
                ]
            )
            page = context.pages[0]
            print(f"Navigating to google.com...")
            page.goto("https://www.google.com", wait_until="domcontentloaded")
            
            # Try to type
            selector = 'textarea[name="q"]' # Google's search box
            print(f"Waiting for selector: {selector}")
            page.wait_for_selector(selector, timeout=5000)
            
            print("Attempting to fill...")
            page.fill(selector, "Playwright test")
            time.sleep(2)
            
            print("Attempting to type (press_sequentially)...")
            page.locator(selector).clear()
            page.type(selector, "Typing test with delay", delay=100)
            time.sleep(2)
            
            print("Typing complete. Check the browser window.")
            time.sleep(5)
            context.close()
        except Exception as e:
            print(f"Error during test: {e}")

if __name__ == "__main__":
    test_typing()
