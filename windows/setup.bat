@echo off
setlocal
cd /d "%~dp0\.."
set "P300_PY_VERSION=3.12"
if not "%~1"=="" set "P300_PY_VERSION=%~1"
if exist ".venv\Scripts\python.exe" goto install
where py >nul 2>nul
if errorlevel 1 goto missing_python
py -%P300_PY_VERSION% -m venv ".venv"
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -c "import sys, tkinter; assert (3,11) <= sys.version_info[:2] < (3,14)"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -e ".[test]"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m bsense_p300_pilot --plan
if errorlevel 1 goto failed
echo [OK] Setup complete. Open windows\run.bat next.
if defined CI exit /b 0
pause
exit /b 0
:missing_python
echo [ERROR] Install Python 3.12 or run this script with 3.13 as the argument.
:failed
echo [ERROR] Setup failed. Read the messages above.
if defined CI exit /b 1
pause
exit /b 1
