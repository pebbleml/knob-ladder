#!/usr/bin/env bash
# Restart only the dashboard server on port 8008; the Cloudflare tunnel (and its URL) keeps running.
cd ~/knob-ladder
for p in $(pgrep -f "python3 scripts/watch.py --port 8008"); do kill "$p"; done
sleep 1
nohup python3 scripts/watch.py --port 8008 > runs/watch.log 2>&1 < /dev/null &
