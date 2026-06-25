#!/usr/bin/env bash
# start.sh — one-command project launcher
# Usage: bash start.sh

set -e
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   🛡️  Warranty Fraud Detector — Startup      ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# 1. Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt --break-system-packages -q

# 2. Check for .env
if [ ! -f ".env" ]; then
    echo "⚠️  .env not found — copying from .env.example"
    cp .env.example .env
fi

# 3. Start FastAPI in background
echo ""
echo "🚀 Starting FastAPI backend on http://localhost:8000 ..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!
sleep 2

# 4. Start Streamlit
echo ""
echo "🎨 Starting Streamlit frontend on http://localhost:8501 ..."
streamlit run app/frontend/app.py --server.port 8501 --server.headless true &
STREAMLIT_PID=$!

echo ""
echo "✅ Both servers running!"
echo "   FastAPI  → http://localhost:8000"
echo "   API Docs → http://localhost:8000/docs"
echo "   Frontend → http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop."

# Wait for Ctrl+C
trap "kill $FASTAPI_PID $STREAMLIT_PID 2>/dev/null; echo 'Stopped.'; exit 0" INT
wait