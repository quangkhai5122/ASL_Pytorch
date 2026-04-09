@echo off
REM Start Backend - Windows Batch Script

echo ============================================================
echo Starting Sign Language Recognition Backend
echo ============================================================

cd /d "%~dp0"

REM Check if backend_env exists
if not exist "backend_env" (
    echo.
    echo Creating virtual environment...
    python -m venv backend_env
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Make sure Python 3.9+ is installed and in PATH
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call backend_env\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Create .env if it doesn't exist
if not exist "backend\.env" (
    echo.
    echo Creating backend\.env from template...
    copy "backend\.env.example" "backend\.env"
    echo Backend configured with defaults. You can edit backend\.env to change settings.
)

REM Install requirements
echo.
echo Installing requirements...
pip install -q -r backend/requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

REM Start server
echo.
echo ============================================================
echo Backend starting on http://localhost:8000
echo API Docs:  http://localhost:8000/api/v1/docs
echo Health:    http://localhost:8000/api/v1/health
echo ============================================================
echo.
echo Press Ctrl+C to stop the server
echo.

cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

REM Deactivate on exit
deactivate
