@echo off
cd /d "%~dp0"
start "" "%~dp0.venv310\Scripts\pythonw.exe" "%~dp0main.py"
