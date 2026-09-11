@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m bsense_p300_pilot %*
if errorlevel 1 goto failed
exit /b 0
:missing
echo [ERROR] Run windows\setup.bat first.
:failed
if defined CI exit /b 1
pause
exit /b 1
