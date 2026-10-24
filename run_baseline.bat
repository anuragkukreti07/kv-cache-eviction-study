@echo off
REM Runs the whole baseline and builds the report.   run_baseline.bat [--quick] [--fresh] [--skip needle,ppl]
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Run setup.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
python scripts\run_all.py --open %*
pause
