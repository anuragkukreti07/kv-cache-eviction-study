@echo off
REM One-time setup: virtual environment + PyTorch (CUDA 12.8, needed for RTX 50-series / sm_120) + dependencies.
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10-3.13 from python.org and tick "Add to PATH".
  exit /b 1
)
if not exist .venv\Scripts\python.exe python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 exit /b 1
pip install -r requirements.txt
if errorlevel 1 exit /b 1
echo.
echo ---- checking GPU and model access ----
set PYTHONUTF8=1
python scripts\s0_check_env.py
echo.
echo If the model check failed: accept the licence at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
echo then run:  .venv\Scripts\hf auth login      (the script falls back to an un-gated mirror otherwise)
pause
