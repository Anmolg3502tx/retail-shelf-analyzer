#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON=$(command -v python3 || command -v python)

if [ -z "$PYTHON" ]; then echo "python3 not found"; exit 1; fi

if [ ! -d "$ROOT/.venv" ]; then
    echo "creating virtualenv..."
    $PYTHON -m venv "$ROOT/.venv"
fi
source "$ROOT/.venv/bin/activate"

echo "installing dependencies..."
pip install -q --upgrade pip
pip install -q -r "$ROOT/flask_server/requirements.txt"
pip install -q -r "$ROOT/detector_service/requirements.txt"
pip install -q -r "$ROOT/grouping_service/requirements.txt"

mkdir -p "$ROOT/visualizations"

export DETECTOR_URL="http://localhost:5001"
export GROUPING_URL="http://localhost:5002"
export OUTPUT_DIR="$ROOT/visualizations"

cd "$ROOT/detector_service"
FLASK_APP=app.py flask run --port 5001 --no-debugger &
DET=$!

cd "$ROOT/grouping_service"
FLASK_APP=app.py flask run --port 5002 --no-debugger &
GRP=$!

echo "waiting for services to start..."
sleep 5

cd "$ROOT/flask_server"
FLASK_APP=app.py flask run --port 5000 --no-debugger &
GW=$!

echo ""
echo "all services running:"
echo "  http://localhost:5000  <- open this in your browser"
echo ""
echo "press Ctrl+C to stop"

trap "kill $DET $GRP $GW 2>/dev/null" SIGINT SIGTERM
wait
