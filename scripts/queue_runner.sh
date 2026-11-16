#!/usr/bin/env bash
# GPU job queue for the box, so the GPU never waits on a Claude session.
#
#   nohup setsid bash ~/knob-ladder/scripts/queue_runner.sh > ~/queue/runner.log 2>&1 < /dev/null &
#
# Jobs are shell files in ~/queue/todo, run one at a time in name order (010_x.sh before 020_y.sh) from
# ~/knob-ladder. Each moves to ~/queue/running, then ~/queue/done or ~/queue/failed; its output goes to
# ~/knob-ladder/runs/logs/<job>.log. ~/queue/heartbeat holds the time, the state and the running job.
# The queue lives outside ~/knob-ladder because remote.sh syncs that folder with --delete.
set -u
Q=~/queue
mkdir -p $Q/todo $Q/running $Q/done $Q/failed ~/knob-ladder/runs/logs
if [ -f $Q/runner.pid ] && kill -0 "$(cat $Q/runner.pid)" 2>/dev/null; then echo "runner already running"; exit 0; fi
echo $$ > $Q/runner.pid
idle_since=""
while true; do
  job=$(ls $Q/todo/*.sh 2>/dev/null | sort | head -1)
  if [ -z "$job" ]; then
    [ -z "$idle_since" ] && idle_since=$(date -u +%FT%TZ)
    echo "$(date -u +%FT%TZ) idle since $idle_since" > $Q/heartbeat
    sleep 30
    continue
  fi
  idle_since=""
  name=$(basename "$job" .sh)
  mv "$job" $Q/running/
  echo "$(date -u +%FT%TZ) running $name" > $Q/heartbeat
  echo "$(date -u +%FT%TZ) start $name"
  ( cd ~/knob-ladder && bash $Q/running/$name.sh ) > ~/knob-ladder/runs/logs/$name.log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    echo "$(date -u +%FT%TZ) running $name" > $Q/heartbeat
    sleep 30
  done
  wait $pid; rc=$?
  if [ $rc -eq 0 ]; then mv $Q/running/$name.sh $Q/done/; else mv $Q/running/$name.sh $Q/failed/; fi
  echo "$(date -u +%FT%TZ) end $name rc=$rc"
done
