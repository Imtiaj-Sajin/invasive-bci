@echo off
rem Reproduce the study on Windows. Usage:  reproduce.bat [setup ^| data ^| ladders ^| analyses ^| electrodes ^| simulator ^| summaries ^| check ^| all ^| --list]
rem The first run of "reproduce.bat setup" creates the virtual environment .venv and installs the pinned packages.
setlocal
cd /d "%~dp0"

if /I "%~1"=="setup" (
    if not exist ".venv\Scripts\python.exe" python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    ".venv\Scripts\python.exe" -m pip install -e .
    echo Setup finished. Next: reproduce.bat data
    goto :eof
)

if not exist ".venv\Scripts\python.exe" (
    echo The environment is missing. Run "reproduce.bat setup" first.
    exit /b 1
)
if "%~1"=="" (
    ".venv\Scripts\python.exe" scripts\reproduce.py --list
) else (
    ".venv\Scripts\python.exe" scripts\reproduce.py %*
)
