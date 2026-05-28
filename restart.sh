#!/bin/bash
cd /tmp/fleet-gap-test

# Kill existing server
if [ -f ui.pid ]; then
    kill $(cat ui.pid) 2>/dev/null
    sleep 1
    kill -9 $(cat ui.pid) 2>/dev/null
fi

# Also kill anything on port 8765
lsof -ti:8765 | xargs kill -9 2>/dev/null
sleep 1

# Restart
source .venv/bin/activate
fleet ui --port 8765 > ui.log 2>&1 &
echo $! > ui.pid
echo "Server restarted with PID $!"
