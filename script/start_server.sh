#!/bin/bash
set -e

MODEL_DIR="${1:-$(dirname "$0")/../model/gemma-4-E2B-it}"
USE_GPU_ARG=""
if [ "$2" == "--use-gpu" ]; then
    USE_GPU_ARG="--use-gpu"
fi

if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "ERROR: model not found at $MODEL_DIR/model.safetensors"
    exit 1
fi

echo ">> Starting Mitten API Server (model: $MODEL_DIR, GPU: ${USE_GPU_ARG:-false})..."
cargo build -p inference-api
./target/debug/inference-api --model-dir "$MODEL_DIR" $USE_GPU_ARG &
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
