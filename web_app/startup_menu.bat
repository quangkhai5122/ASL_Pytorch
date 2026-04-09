@echo off
REM Master Startup Guide - Windows

echo.
echo ============================================================
echo  Sign Language Recognition - Full Stack Startup
echo ============================================================
echo.
echo This will guide you through starting both backend and frontend.
echo.
echo You will need TWO terminal windows:
echo   Terminal 1: Backend (FastAPI)
echo   Terminal 2: Frontend (React/Vite)
echo.
echo ============================================================
echo.

:menu
cls
echo ============================================================
echo  STARTUP OPTIONS
echo ============================================================
echo.
echo  1. Start Backend (Terminal 1)
echo  2. Start Frontend (Terminal 2) 
echo  3. Show credentials / test URLs
echo  4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Starting Backend in new window...
    start "SLR Backend" cmd /k start_backend.bat
    echo.
    echo Backend starting... Please wait ~30-60 seconds for model to load.
    echo When ready, it will show: "Uvicorn running on http://0.0.0.0:8000"
    echo.
    pause
    goto menu
)

if "%choice%"=="2" (
    echo.
    echo Starting Frontend in new window...
    start "SLR Frontend" cmd /k start_frontend.bat
    echo.
    echo Frontend starting... Open http://localhost:5173 when ready.
    echo.
    pause
    goto menu
)

if "%choice%"=="3" (
    cls
    echo ============================================================
    echo  TEST CREDENTIALS AND URLS
    echo ============================================================
    echo.
    echo LOGIN CREDENTIALS:
    echo   Username: testuser
    echo   Password: testpass123
    echo.
    echo   OR
    echo.
    echo   Username: demo
    echo   Password: demo123
    echo.
    echo URLS (after starting both services):
    echo   Frontend:     http://localhost:5173
    echo   Backend API:  http://localhost:8000
    echo   API Docs:     http://localhost:8000/api/v1/docs
    echo   Health Check: http://localhost:8000/api/v1/health
    echo.
    echo ============================================================
    echo.
    pause
    goto menu
)

if "%choice%"=="4" (
    echo.
    echo Goodbye!
    echo.
    exit /b 0
)

echo Invalid choice. Please try again.
pause
goto menu
