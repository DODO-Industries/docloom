@echo off
REM DocLoom Loom Module — double-click to launch module_loom/app.py standalone.
REM Single uvicorn worker: the .loom storage layer (module_loom/services/weaver)
REM is single-writer-per-process, so --workers 1 is deliberate, not a default.

setlocal
set SCRIPT_DIR=%~dp0
set VENV_PY=%SCRIPT_DIR%..\.venv\Scripts\python.exe

if exist "%VENV_PY%" (
    set PYTHON_EXE=%VENV_PY%
) else (
    set PYTHON_EXE=python
)

cd /d "%SCRIPT_DIR%.."

echo Starting DocLoom Loom Module on http://localhost:8000 ...
echo Testing Sandbox: http://localhost:8000/loom/testing
echo Visualizer:      http://localhost:8000/loom/neuro
echo Press Ctrl+C to stop.
echo.

"%PYTHON_EXE%" -m uvicorn module_loom.app:app ^
    --host 0.0.0.0 --port 8000 ^
    --reload ^
    --timeout-keep-alive 75 ^
    --ws-ping-interval 20 --ws-ping-timeout 20

echo.
echo Server stopped.
pause
