#!/bin/bash
# Convenient runner script for Pattern Recognition Lab 07 - Classifiers Studio
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Check if venv exists
if [ -d "venv" ]; then
    echo "[+] Using dedicated virtual environment: venv"
    PYTHON="$DIR/venv/bin/python"
else
    echo "[+] Using system python3"
    PYTHON="python3"
fi

echo "[+] Starting Pattern Recognition Lab 07: Classifiers Studio..."
echo "[+] URL: http://127.0.0.1:5001"
echo "--------------------------------------------------------"
exec "$PYTHON" app.py
