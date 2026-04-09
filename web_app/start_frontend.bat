@echo off
REM Start Frontend - Windows Batch Script

echo ============================================================
echo Starting Sign Language Recognition Frontend
echo ============================================================

cd /d "%~dp0frontend"

REM Create .env.local if it doesn't exist
if not exist ".env.local" (
    echo.
    echo Creating .env.local from template...
    copy ".env.example" ".env.local"
    echo Frontend configured with defaults.
)

REM Check if node_modules exists
if not exist "node_modules" (
    echo.
    echo Installing npm packages...
    call npm install
    if errorlevel 1 (
        echo ERROR: Failed to install npm packages
        echo Make sure Node.js 18+ is installed and in PATH
        pause
        exit /b 1
    )
)

REM Start dev server
echo.
echo ============================================================
echo Frontend starting on http://localhost:5173
echo ============================================================
echo.
echo Press Ctrl+C to stop the server
echo.

call npm run dev
