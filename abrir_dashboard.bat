@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" launch_dashboard.py %*
    goto :done
)

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 launch_dashboard.py %*
    goto :done
)

where python >nul 2>nul
if %errorlevel%==0 (
    python launch_dashboard.py %*
    goto :done
)

echo [ERRO] Python 3.11+ nao encontrado no PATH. Instale o Python e tente novamente.
pause
exit /b 1

:done
if %errorlevel% neq 0 (
    echo.
    echo [ERRO] O launcher encerrou com codigo %errorlevel%.
    pause
    exit /b %errorlevel%
)
