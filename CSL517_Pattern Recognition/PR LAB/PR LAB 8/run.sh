#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR/Classifiers"

# Look for virtual environment
if [ -d "$DIR/Classifiers/venv" ]; then
    VENV_PY="$DIR/Classifiers/venv/bin/python"
elif [ -d "$DIR/../PR LAB 7/Classifiers/venv" ]; then
    VENV_PY="$DIR/../PR LAB 7/Classifiers/venv/bin/python"
else
    echo "Creating virtual environment in $DIR/Classifiers/venv..."
    python3 -m venv "$DIR/Classifiers/venv"
    "$DIR/Classifiers/venv/bin/pip" install --upgrade pip
    "$DIR/Classifiers/venv/bin/pip" install -r requirements.txt
    VENV_PY="$DIR/Classifiers/venv/bin/python"
fi

echo "=================================================================="
echo " Starting Classifiers Web Application (Bayes, KNN, Ho-Kashyap)..."
echo " Open in your browser: http://127.0.0.1:5001"
echo " Press Ctrl+C to stop the server."
echo "=================================================================="
PORT=${PORT:-5001}
exec "$VENV_PY" app.py

