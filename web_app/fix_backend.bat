@echo off
REM Fix Backend Dependencies - Windows Batch Script
REM Removes corrupted environment and reinstalls with fixed dependencies

echo.
echo ============================================================
echo Sign Language Recognition - Backend Dependency Fix
echo ============================================================
echo.
echo This script will:
echo   1. Remove the old backend_env (if it exists)
echo   2. Create a fresh virtual environment
echo   3. Install updated dependencies (fixed versions)
echo   4. Verify the installation
echo.
pause

cd /d "%~dp0"

REM Step 1: Remove old environment
echo.
echo [1/4] Removing old backend_env...
if exist "backend_env" (
    echo    - Deleting backend_env directory...
    rmdir /s /q backend_env
    if errorlevel 1 (
        echo ERROR: Could not remove backend_env
        echo Please close any Python processes using backend_env and try again.
        pause
        exit /b 1
    )
    echo    - Done!
)

REM Step 2: Create fresh environment
echo.
echo [2/4] Creating fresh virtual environment...
python -m venv backend_env
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo    - Done!

REM Step 3: Activate and install dependencies
echo.
echo [3/4] Installing dependencies (this may take 2-3 minutes)...
call backend_env\Scripts\activate.bat
pip install --upgrade pip setuptools wheel
pip install -r backend/requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo    - Done!

REM Step 4: Verify installation
echo.
echo [4/4] Verifying installation...
python -c "import bcrypt; print(f'✓ bcrypt {bcrypt.__version__} installed')"
python -c "import passlib; print('✓ passlib installed')"
python -c "import fastapi; print('✓ fastapi installed')"
python -c "import app.main; print('✓ app modules import successfully')"

if errorlevel 1 (
    echo ERROR: Verification failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo ✓ ALL DEPENDENCIES FIXED!
echo ============================================================
echo.
echo Next steps:
echo   1. Close this window
echo   2. Run: start_backend.bat
echo   3. The server should start without errors
echo.
pause
