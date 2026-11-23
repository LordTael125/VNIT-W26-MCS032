#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Look for virtual environment
if [ -d "$DIR/venv" ]; then
    VENV_PY="$DIR/venv/bin/python"
elif [ -d "$DIR/../../PR LAB 7/Classifiers/venv" ]; then
    VENV_PY="$DIR/../../PR LAB 7/Classifiers/venv/bin/python"
elif [ -d "$DIR/../venv" ]; then
    VENV_PY="$DIR/../venv/bin/python"
else
    echo "Creating virtual environment in $DIR/venv..."
    python3 -m venv venv
    "$DIR/venv/bin/pip" install --upgrade pip
    "$DIR/venv/bin/pip" install -r requirements.txt
    VENV_PY="$DIR/venv/bin/python"
fi

echo "Starting Classifiers Web Application (Parzen Window, Bayes, KNN, Ho-Kashyap)..."
PORT=${PORT:-5002}
echo "Open in your browser: http://127.0.0.1:$PORT"
exec "$VENV_PY" app.py

