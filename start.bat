@echo off
:: ═══════════════════════════════════════════════
::  AI Assistant Web UI — Windows Launcher
::  Double-click this file to start
:: ═══════════════════════════════════════════════

title AI Assistant
set PORT=5001
set URL=http://localhost:%PORT%
cd /d "%~dp0"

echo.
echo   ══════════════════════════════════════
echo    AI Assistant Web UI
echo   ══════════════════════════════════════
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo   ERROR: Python not found. Install from python.org
    pause
    exit /b 1
)

:: Start Ollama if not running
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe" >NUL
if errorlevel 1 (
    echo   Starting Ollama...
    start /min "" ollama serve
    timeout /t 3 /nobreak >nul
    echo   Ollama started
) else (
    echo   Ollama already running
)

:: Install deps if needed
echo   Checking dependencies...
python -c "import flask, flask_cors" >nul 2>&1
if errorlevel 1 (
    echo   Installing Python packages...
    pip install flask flask-cors --quiet
)

:: Kill old server
for /f "tokens=5" %%a in ('netstat -aon ^| find ":%PORT%"') do taskkill /F /PID %%a >nul 2>&1
timeout /t 1 /nobreak >nul

:: Start server in background
echo   Starting server at %URL%
start /min "" python server.py
timeout /t 2 /nobreak >nul

:: Open in Chrome app mode (pinnable)
echo   Opening browser...
set CHROME_PATH=
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" (
    set CHROME_PATH="%ProgramFiles%\Google\Chrome\Application\chrome.exe"
) else if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" (
    set CHROME_PATH="%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
) else if exist "%LocalAppData%\Google\Chrome\Application\chrome.exe" (
    set CHROME_PATH="%LocalAppData%\Google\Chrome\Application\chrome.exe"
)

if defined CHROME_PATH (
    start "" %CHROME_PATH% --app=%URL% --no-first-run
) else (
    start "" %URL%
)

echo.
echo   AI Assistant is running at %URL%
echo   Close this window to stop the server.
echo.
pause
