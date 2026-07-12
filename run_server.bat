@echo off
REM DocLoom Server — double-click to launch backend/app.py's FastAPI app with
REM every route it mounts (embedding, neuro_visualizer, and the legacy loom/ai
REM routers if their dependencies are present — app.py mounts each best-effort
REM and just logs a warning for any that fail to import).
REM
REM Runs ONE uvicorn worker, deliberately — NOT --workers N. WeaveBrainCoordinator
REM and the .loom storage layer (backend/services/loom_service/weaver/substrate_layout.py)
REM are single-writer-per-process by design: each uvicorn worker is a separate OS
REM process with its own independent coordinator and RAM ledger, so multiple workers
REM would each open the SAME universe.loom/crystal_*.loom files and could interleave
REM writes into corruption. "Maximum capacity" here means tuning ONE process's
REM concurrency ceiling, not horizontal worker scaling — see the scalability notes
REM this was delivered alongside for the longer version of this tradeoff.

setlocal
set SCRIPT_DIR=%~dp0
set VENV_PY=%SCRIPT_DIR%.venv\Scripts\python.exe

if exist "%VENV_PY%" (
    set PYTHON_EXE=%VENV_PY%
) else (
    set PYTHON_EXE=python
)

cd /d "%SCRIPT_DIR%"

echo Starting DocLoom server on http://0.0.0.0:8000 (single worker — see comments in this file)...
echo Press Ctrl+C to stop.
echo.

"%PYTHON_EXE%" -m uvicorn backend.app:app ^
    --host 0.0.0.0 --port 8000 ^
    --workers 1 ^
    --timeout-keep-alive 75 ^
    --ws-ping-interval 20 --ws-ping-timeout 20 ^
    --limit-concurrency 1000 --backlog 2048

echo.
echo Server stopped.
pause
