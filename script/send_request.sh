#!/bin/bash

# Usage: ./script/send_request.sh [prompt] [stream]
# Example: ./script/send_request.sh "What is the capital of France?" true

PROMPT=${1:-"What is the capital of France?"}
STREAM=${2:-"false"}

echo ">> Sending request to http://localhost:8080/v1/chat/completions"
echo ">> Prompt: $PROMPT"
echo ">> Streaming: $STREAM"
echo ""

# Escape double quotes in the prompt for JSON
ESCAPED_PROMPT=$(echo "$PROMPT" | sed 's/"/\\"/g')

curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"qwen-3.5-0.8b\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
      {\"role\": \"user\", \"content\": \"$ESCAPED_PROMPT\"}
    ],
    \"max_tokens\": 512,
    \"temperature\": 0.7,
    \"stream\": $STREAM
  }"
