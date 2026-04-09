#!/bin/bash
# Frontend-Backend Integration Setup Script
# Run this from the web_app directory

set -e

echo "=========================================="
echo "Sign Language Recognition - Setup Script"
echo "=========================================="
echo ""

# Check if Docker is running
echo "✓ Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo "✗ Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Setup Backend
echo ""
echo "=========================================="
echo "Setting up Backend..."
echo "=========================================="

if [ ! -f "backend/.env" ]; then
    echo "✓ Creating backend .env file..."
    cp backend/.env.example backend/.env
    echo "  Edit backend/.env if needed"
else
    echo "✓ Backend .env already exists"
fi

echo "✓ Starting Docker containers..."
docker-compose up -d

echo ""
echo "Waiting for backend to load model (30-60 seconds)..."
RETRY=0
while [ $RETRY -lt 30 ]; do
    if curl -s http://localhost:8000/api/v1/health > /dev/null; then
        echo "✓ Backend is ready!"
        curl -s http://localhost:8000/api/v1/health | jq .
        break
    fi
    echo -n "."
    sleep 2
    RETRY=$((RETRY+1))
done

if [ $RETRY -eq 30 ]; then
    echo "✗ Backend failed to start. Check logs: docker-compose logs backend"
    exit 1
fi

# Setup Frontend
echo ""
echo "=========================================="
echo "Setting up Frontend..."
echo "=========================================="

cd frontend

if [ ! -f ".env.local" ]; then
    echo "✓ Creating frontend .env.local..."
    cp .env.example .env.local
    echo "  Configuration:"
    cat .env.local | grep -v "^#" | grep -v "^$"
else
    echo "✓ Frontend .env.local already exists"
fi

echo ""
echo "✓ Installing frontend dependencies..."
npm install --legacy-peer-deps

# Summary
echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start frontend development server:"
echo "   cd frontend && npm run dev"
echo ""
echo "2. Open http://localhost:5173 in your browser"
echo ""
echo "3. Login with:"
echo "   Username: testuser"
echo "   Password: testpass123"
echo ""
echo "4. Backend API docs available at:"
echo "   http://localhost:8000/api/v1/docs"
echo ""
echo "Troubleshooting:"
echo "- Backend logs: docker-compose logs backend"
echo "- Frontend logs: Check browser console (F12)"
echo "- Backend health: curl http://localhost:8000/api/v1/health"
echo ""
