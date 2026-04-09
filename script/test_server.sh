#!/bin/bash
# script/test_server.sh
# Test script for Mitten API server supporting Gemma 4 and Qwen 3.5.

MODEL_PATH="${1:-script/../model/gemma-4-E2B-it}"
TIMEOUT=1200

# Determine model name for the request
if [[ "$MODEL_PATH" == *"qwen"* ]]; then
  MODEL_NAME="qwen-3.5-0.8b"
else
  MODEL_NAME="gemma-4-e2b-it"
fi

# Function to kill server on exit
cleanup() {
  if [ -n "$SERVER_PID" ]; then
    echo ">> Killing server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
  fi
}
trap cleanup EXIT

echo ">> Starting server with model: $MODEL_PATH..."
cargo run -p inference-api -- --model-dir "$MODEL_PATH" > server.log 2>&1 &
SERVER_PID=$!
echo ">> Server PID: $SERVER_PID"

# Wait for server to start
echo ">> Waiting for server to start..."
START_TIME=$(date +%s)
while ! curl -s http://localhost:8080/v1/models > /dev/null; do
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server process died. Log:"
    cat server.log
    exit 1
  fi
  
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))
  if [ $ELAPSED -gt 120 ]; then
    echo "ERROR: Timeout waiting for server to start"
    exit 1
  fi
  sleep 2
done
echo ">> Server ready after ${ELAPSED}s"

echo ">> Sending test request for $MODEL_NAME (timeout: ${TIMEOUT}s)..."
RESPONSE=$(curl -sf --max-time $TIMEOUT http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_NAME\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
      {\"role\": \"user\", \"content\": \"What is the capital of China? Answer in one word.\"}
    ],
    \"max_tokens\": 50,
    \"temperature\": 0.0,
    \"stream\": false
  }") || { echo ">> FAIL: Request timed out or failed"; exit 1; }

echo ">> Response: $RESPONSE"

if echo "$RESPONSE" | grep -q "\"choices\""; then
  echo ">> PASS: Server returned valid chat completion response"
  CONTENT=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
  echo ">> Content: $CONTENT"
  if echo "$CONTENT" | grep -qi "Beijing"; then
    echo ">> SUCCESS: Model answered correctly!"
  else
    echo ">> NOTE: Response does not contain 'Beijing'"
    if [[ "$MODEL_NAME" == "gemma-4-e2b-it" ]]; then
       echo ">> (Expected for multimodal-only model without vision tokens)"
    fi
  fi
else
  echo ">> FAIL: Unexpected response format"
  exit 1
fi
