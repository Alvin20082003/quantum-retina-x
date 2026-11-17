@echo off
echo Starting QuantumRetina-X Backend...
cd /d J:\research\QuantumRetina-X
J:\research\QuantumRetina-X\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
pause