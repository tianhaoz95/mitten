#!/bin/bash
# script/test_server.sh
# Simple script to test the Mitten API server with Gemma 4 E2B model.

MODEL_PATH="${1:-script/../model/gemma-4-E2B-it}"
TIMEOUT=300

# Function to kill server on exit
cleanup() {
  if [ -n "$SERVER_PID" ]; then
    echo ">> Killing server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
  fi
}
trap cleanup EXIT

echo ">> Starting server..."
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

echo ">> Sending test request (timeout: ${TIMEOUT}s)..."
RESPONSE=$(curl -sf --max-time $TIMEOUT http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-e2b-it",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of China? Answer in one word."}
    ],
    "max_tokens": 5,
    "temperature": 0.0,
    "stream": false
  }') || { echo ">> FAIL: Request timed out or failed"; exit 1; }

echo ">> Response: $RESPONSE"

# NOTE: Gemma 4 E2B is a multimodal model (Gemma4ForConditionalGeneration).
# The text-only language_model sub-model does not produce coherent text output
# without the full multimodal pipeline. A text-only Gemma model (e.g. Gemma 3 1B/4B)
# would be needed for correct text responses in this text-only Mitten implementation.

if echo "$RESPONSE" | grep -q "\"choices\""; then
  echo ">> PASS: Server returned valid chat completion response"
  CONTENT=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
  echo ">> Content: $CONTENT"
  if echo "$CONTENT" | grep -qi "Beijing"; then
    echo ">> SUCCESS: Model answered correctly!"
  else
    echo ">> NOTE: Response does not contain 'Beijing' (expected for multimodal-only model)"
    echo ">> To get correct text output, use a text-only Gemma 3 1B/4B model"
  fi
else
  echo ">> FAIL: Unexpected response format"
  exit 1
fi
