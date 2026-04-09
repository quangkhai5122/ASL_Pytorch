# Backend Startup Issues - Troubleshooting Guide

## The Problem

You encountered a **bcrypt/passlib compatibility error** when starting the backend. This happened because:

```
(trapped) error reading bcrypt version
AttributeError: module 'bcrypt' has no attribute '__about__'
ValueError: password cannot be longer than 72 bytes
```

### Root Cause
- **Old passlib version** (`1.7.4`) doesn't work with modern bcrypt (`4.x`)
- **Passlib expects older bcrypt structure** with `__about__` attribute
- **Password hashing fails** when systems try to validate credentials

## The Solution

All necessary files have been **already fixed**:

✅ **`requirements.txt`** - Updated to compatible versions:
- `passlib[bcrypt]==1.7.4.1` (was 1.7.4)
- `bcrypt==4.1.2` (now explicitly included)

✅ **`app/core/auth.py`** - Enhanced bcrypt configuration:
- Added explicit bcrypt import
- Configured `bcrypt__rounds=12` for compatibility

✅ **Response schemas** - Fixed Pydantic warning:
- Renamed `model_loaded` → `is_model_loaded` (avoids protected namespace)
- Updated all routes that reference this field

---

## Steps to Fix Your Local Backend

### **Option 1: Automatic Fix (Recommended)**

```bash
cd D:\Python_Project\CV_GISLR\web_app
fix_backend.bat
```

This script will:
1. Delete the corrupted `backend_env` folder
2. Create a fresh virtual environment
3. Install all dependencies with correct versions
4. Verify everything works

**Takes ~3 minutes. Just watch the output.**

### **Option 2: Manual Fix**

If you prefer to do it manually:

**Step 1: Stop the current backend**
```bash
# Close the terminal running the backend
# Press Ctrl+C
```

**Step 2: Remove old environment**
```bash
cd D:\Python_Project\CV_GISLR\web_app
rmdir /s /q backend_env
# Type 'Y' to confirm deletion
```

**Step 3: Create fresh environment**
```bash
python -m venv backend_env
backend_env\Scripts\activate.bat
```

**Step 4: Reinstall dependencies**
```bash
pip install --upgrade pip setuptools wheel
pip install -r backend/requirements.txt
# Wait for installation to complete (2-3 minutes)
```

**Step 5: Verify it works**
```bash
python -c "import bcrypt; print(f'✓ bcrypt {bcrypt.__version__} OK')"
python -c "import app.main; print('✓ app modules OK')"
```

**Step 6: Start backend**
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## What Changed

### **requirements.txt**

**Before:**
```
passlib[bcrypt]==1.7.4
```

**After:**
```
passlib[bcrypt]==1.7.4.1
bcrypt==4.1.2
```

**Why:** Explicit bcrypt version + newer passlib fixes version detection

### **app/core/auth.py**

**Before:**
```python
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
```

**After:**
```python
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12
)
```

**Why:** Explicit bcrypt configuration for compatibility

### **Response Schemas**

**Before:**
```python
model_loaded: bool = Field(...)
```

**After:**
```python
is_model_loaded: bool = Field(...)
```

**Why:** Avoids Pydantic v2's "model_" protected namespace warning

---

## Verification Checklist

After running `fix_backend.bat` (or manual steps), verify:

✅ **Can you see this message?**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

✅ **No red ERROR lines** in console output

✅ **Can you access the API?**
```bash
curl http://localhost:8000/api/v1/health
# Should return: {"status":"healthy","is_model_loaded":true,...}
```

✅ **No bcrypt warnings** in console

---

## If It Still Doesn't Work

### Problem: "Cannot find module app"

**Fix:**
```bash
# Make sure you're in the backend folder
cd D:\Python_Project\CV_GISLR\web_app\backend

# Check app folder exists
dir app
# Should show: main.py, config.py, api/, etc.
```

### Problem: Port 8000 already in use

**Fix:**
```bash
# Find what's using port 8000
netstat -ano | findstr :8000

# Kill the process (replace XXXX with PID from above)
taskkill /PID XXXX /F

# Then try starting backend again
```

### Problem: Still getting bcrypt error

**Fix:**
```bash
# Completely remove and reinstall bcrypt
pip uninstall bcrypt -y
pip install bcrypt==4.1.2

# Then reinstall passlib
pip uninstall passlib -y
pip install "passlib[bcrypt]==1.7.4.1"
```

### Problem: "AttributeError: module 'bcrypt' has no attribute '__about__'"

**This means old passlib is still cached.** Do this:

```bash
# Complete clean reinstall
pip uninstall bcrypt passlib cryptography -y
pip install cryptography==41.0.7 bcrypt==4.1.2 "passlib[bcrypt]==1.7.4.1"
```

---

## Production Deployment Note

For production, you should:

1. **Use environment-based configuration** instead of test credentials
2. **Increase bcrypt rounds** (currently 12, can go to 14+)
3. **Use proper password management** (don't hash passwords at startup)

See the `.env` file for production configuration options.

---

## What's Next?

Once backend is running:

```bash
# Terminal 1: Backend (keep running)
cd D:\Python_Project\CV_GISLR\web_app
start_backend.bat

# Terminal 2: Frontend (in separate window)
cd D:\Python_Project\CV_GISLR\web_app\frontend
npm run dev
```

Then open http://localhost:5173 and test the integration!

---

## Questions?

**Check these files for more info:**
- [QUICK_INTEGRATION_GUIDE.md](QUICK_INTEGRATION_GUIDE.md)
- [FRONTEND_INTEGRATION.md](FRONTEND_INTEGRATION.md)
- [backend/.env.example](backend/.env.example)

**API Documentation (once backend is running):**
- Swagger UI: http://localhost:8000/api/v1/docs
- ReDoc: http://localhost:8000/api/v1/redoc
