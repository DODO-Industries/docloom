@echo off
REM DocLoom Brain Decoder — double-click to launch.
REM Browses assets\.brain_data (or a path you pass as the first argument),
REM lists every cluster (crystal), and lets you view/export any one of them
REM or the whole brain as human-readable JSON. Read-only.

setlocal
set SCRIPT_DIR=%~dp0
set VENV_PY=%SCRIPT_DIR%.venv\Scripts\python.exe

if exist "%VENV_PY%" (
    set PYTHON_EXE=%VENV_PY%
) else (
    set PYTHON_EXE=python
)

cd /d "%SCRIPT_DIR%"
"%PYTHON_EXE%" -m backend.services.loom_service.decoder.cli %*

echo.
pause
