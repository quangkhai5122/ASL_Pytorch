@echo off
REM Frontend-Backend Integration Setup Script for Windows
REM Run this from the web_app directory

setlocal enabledelayedexpansion

echo ==========================================
echo Sign Language Recognition - Setup Script
echo ==========================================
echo.

REM Check if Docker is running
echo Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo Docker is not running. Please start Docker Desktop.
    exit /b 1
)

REM Setup Backend
echo.
echo ==========================================
echo Setting up Backend...
echo ==========================================

if not exist "backend\.env" (
    echo Creating backend .env file...
    copy backend\.env.example backend\.env
    echo Edit backend\.env if needed
) else (
    echo Backend .env already exists
)

echo Starting Docker containers...
docker-compose up -d

echo.
echo Waiting for backend to load model (30-60 seconds^)...
setlocal enabledelayedexpansion
set RETRY=0
:wait_backend
if !RETRY! lss 30 (
    timeout /t 2 /nobreak >nul
    curl -s http://localhost:8000/api/v1/health >nul 2>&1
    if errorlevel 1 (
        set /a RETRY=!RETRY!+1
        echo -n "."
        goto wait_backend
    )
)

if not !RETRY! lss 30 (
    echo Backend failed to start. Check logs: docker-compose logs backend
    exit /b 1
)

echo Backend is ready!
curl -s http://localhost:8000/api/v1/health

REM Setup Frontend
echo.
echo ==========================================
echo Setting up Frontend...
echo ==========================================

cd frontend

if not exist ".env.local" (
    echo Creating frontend .env.local...
    copy .env.example .env.local
    echo Configuration:
    type .env.local | findstr /v "^#" | findstr /v "^$"
) else (
    echo Frontend .env.local already exists
)

echo.
echo Installing frontend dependencies...
call npm install --legacy-peer-deps

REM Summary
echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo.
echo 1. Start frontend development server:
echo    cd frontend ^&^& npm run dev
echo.
echo 2. Open http://localhost:5173 in your browser
echo.
echo 3. Login with:
echo    Username: testuser
echo    Password: testpass123
echo.
echo 4. Backend API docs available at:
echo    http://localhost:8000/api/v1/docs
echo.
echo Troubleshooting:
echo - Backend logs: docker-compose logs backend
echo - Frontend logs: Check browser console (F12)
echo - Backend health: curl http://localhost:8000/api/v1/health
echo.
