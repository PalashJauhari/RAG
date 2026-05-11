#!/bin/bash

# Kill both processes when Ctrl+C is pressed
trap 'kill 0; exit' SIGINT SIGTERM

# Kill any existing processes on ports 8000 and 8050
echo "Cleaning up existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8050 | xargs kill -9 2>/dev/null
sleep 1

# Activate the new Python 3.12 environment
source /Users/palashjauhari/Desktop/Projects/rag_env_1/bin/activate

echo "Starting Backend (FastAPI)..."
uvicorn api.main:app --reload &

echo "Starting Frontend (Dash)..."
python3 -m ui.dash_app &

wait
