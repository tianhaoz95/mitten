#!/bin/bash
set -e

if [ -d ".venv" ]; then
    echo ">> Virtual environment already exists. Skipping creation."
else
    echo ">> Creating virtual environment with uv..."
    uv venv .venv --seed
fi

echo ">> Activating environment and installing huggingface-hub[cli]..."
source .venv/bin/activate
uv pip install "huggingface-hub[cli]"

echo ">> Environment initialized successfully."
