@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto run
echo First run: setting up FreshStock 2.0. This needs internet once.
set "PY="
for %%V in (3.14 3.13 3.12) do if not defined PY py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
if not defined PY python -c "import sys; sys.exit(sys.version_info < (3, 12))" >nul 2>&1 && set "PY=python"
if not defined PY goto nopython
%PY% -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --only-binary=:all: -r requirements.txt
if errorlevel 1 (
  rmdir /s /q .venv
  goto failed
)
:run
".venv\Scripts\python.exe" freshstock_desktop_2.py %*
set "CODE=%errorlevel%"
echo.
pause
exit /b %CODE%
:nopython
echo Python 3.12 or newer was not found. Install it from https://www.python.org/downloads/ and run this file again.
pause
exit /b 1
:failed
echo Setup did not finish. Check your internet connection and run this file again.
pause
exit /b 1
