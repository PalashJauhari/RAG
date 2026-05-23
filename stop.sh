#!/bin/bash

echo "Stopping Backend (FastAPI) and Frontend (Dash)..."

for port in 8000 8050; do
  pids=$(lsof -ti:"$port" 2>/dev/null)
  if [ -n "$pids" ]; then
    echo "  Killing port $port (PIDs: $pids)"
    kill -9 $pids 2>/dev/null
  fi
done

pkill -f 'uvicorn api.main:app' 2>/dev/null
pkill -f 'python3 -m ui.dash_app' 2>/dev/null
pkill -f 'ui.dash_app' 2>/dev/null

sleep 0.5

if lsof -ti:8000 >/dev/null 2>&1 || lsof -ti:8050 >/dev/null 2>&1; then
  echo "Warning: something is still bound to port 8000 or 8050."
  lsof -i:8000 -i:8050 2>/dev/null
  exit 1
fi

echo "Done."
