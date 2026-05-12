#!/bin/bash
# =============================================================================
# ASL Sign Language Web App - Docker Compose Launcher
# =============================================================================
# Usage:
#   bash start.sh          → Start all services
#   bash start.sh build    → Rebuild and start (after Dockerfile/deps changes)
#   bash start.sh stop     → Stop all services
#   bash start.sh logs     → Follow logs
#   bash start.sh status   → Show container status
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "${1:-start}" in
  start)
    echo "🚀 Starting backend + frontend..."
    docker compose up -d
    echo ""
    docker compose ps
    echo ""
    echo "✅ Backend:  http://localhost:8000"
    echo "✅ Frontend: http://localhost:8080"
    echo "📄 API Docs: http://localhost:8000/api/v1/docs"
    ;;
  build)
    echo "🔨 Rebuilding and starting..."
    docker compose up --build -d
    echo ""
    docker compose ps
    ;;
  stop)
    echo "🛑 Stopping all services..."
    docker compose down
    ;;
  logs)
    docker compose logs -f
    ;;
  status)
    docker compose ps
    ;;
  *)
    echo "Usage: bash start.sh [start|build|stop|logs|status]"
    exit 1
    ;;
esac
