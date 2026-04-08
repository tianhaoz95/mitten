#!/bin/bash
source .venv/bin/activate
./script/start_server.sh &
SCRIPT_PID=$!
sleep 10
echo "Sending SIGINT to $SCRIPT_PID"
kill -INT $SCRIPT_PID
wait $SCRIPT_PID
echo "Script finished"
