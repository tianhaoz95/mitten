#!/bin/bash
set -e

MODEL_DIR="$(dirname "$0")/../model/gemma-4-E2B-it"

if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "ERROR: model not found at $MODEL_DIR/model.safetensors"
    exit 1
fi

echo ">> Starting Mitten API Server (model: $MODEL_DIR)..."
cargo build -p inference-api
./target/debug/inference-api &
SERVER_PID=$!

cleanup() {
    echo ">> Shutting down server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    echo ">> Server stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

wait $SERVER_PID
