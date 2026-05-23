#!/bin/bash

ROOT="$(cd "$(dirname "$0")" && pwd)"

if [ "${1:-}" = "stop" ]; then
  exec "$ROOT/stop.sh"
fi

# Kill both processes when Ctrl+C is pressed
trap 'kill 0; exit' SIGINT SIGTERM

"$ROOT/stop.sh"
sleep 1

# Activate the new Python 3.12 environment
source /Users/palashjauhari/Desktop/Projects/rag_env_1/bin/activate

echo "Starting Backend (FastAPI)..."
uvicorn api.main:app --reload &

echo "Starting Frontend (Dash)..."
python3 -m ui.dash_app &

wait
