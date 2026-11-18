#!/usr/bin/env bash
# Stop the running E5 jobs, set their outputs aside in runs/aborted/, and start scripts/e5_queue.sh again.
cd ~/knob-ladder
for p in $(pgrep -f "python3 scripts/r0_pilot.py"); do kill "$p"; done
sleep 10
mkdir -p runs/aborted
for d in runs/live/e5-inside* runs/live/e5-blind*; do [ -e "$d" ] && mv "$d" runs/aborted/; done
for f in runs/e5-inside*.jsonl runs/e5-blind*.jsonl runs/e5q_*.log; do [ -e "$f" ] && mv "$f" runs/aborted/; done
nohup scripts/e5_queue.sh > runs/e5_queue.log 2>&1 < /dev/null &
