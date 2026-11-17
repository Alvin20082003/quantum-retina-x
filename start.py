"""
QuantumRetina-X — one-command launcher
Starts FastAPI backend (port 8000) and Vite frontend (port 5173) in parallel.
Run from project root: python start.py
"""
import subprocess, sys, os, time, signal, threading

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(VENV_PY):
    VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")  # Linux/macOS

def stream(proc, prefix):
    for line in proc.stdout:
        print(f"[{prefix}] {line}", end="", flush=True)

print("=" * 56)
print("  QuantumRetina-X Launcher")
print("  Backend  → http://127.0.0.1:8000")
print("  Frontend → http://127.0.0.1:5173")
print("  Press Ctrl+C to stop both")
print("=" * 56)

backend = subprocess.Popen(
    [VENV_PY, "-m", "uvicorn", "backend.main:app", "--reload",
     "--host", "127.0.0.1", "--port", "8000"],
    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    env={**os.environ, "PYTHONUNBUFFERED": "1"}
)

time.sleep(2)

# Find npm-cli.js
import shutil
node_exe = shutil.which("node") or "node"
npm_script = shutil.which("npm") or "npm"

frontend = subprocess.Popen(
    [npm_script, "run", "dev"],
    cwd=os.path.join(ROOT, "frontend"),
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=False
)

t1 = threading.Thread(target=stream, args=(backend, "BACKEND"), daemon=True)
t2 = threading.Thread(target=stream, args=(frontend, "FRONTEND"), daemon=True)
t1.start(); t2.start()

try:
    t1.join(); t2.join()
except KeyboardInterrupt:
    print("\\nShutting down...")
    backend.terminate(); frontend.terminate()
