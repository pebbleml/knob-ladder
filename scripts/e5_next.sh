#!/usr/bin/env bash
# E5 v1, rungs B and C, queued behind rung A: waits until rung A's processes exit (their scoring included),
# then runs rung B (inside) on GPUs 0-3 and rung C (blind) on GPUs 4-7, paste arm, 32 sessions each.
# Same as rung A otherwise, except: up to 30 knob turns (a safety ceiling; the model ends when it wants),
# and --choice, which adds a scoring question that includes "nothing changed" (grades sham and random knobs).
#   nohup scripts/e5_next.sh > runs/e5_next.log 2>&1 &
set -uo pipefail
cd ~/knob-ladder
while pgrep -f "[r]0_pilot.py .*--frame explicit" > /dev/null; do sleep 30; done
echo "rung A finished $(date -u); starting rungs B and C"
group="E5 v1 rungs B and C: less told, reasoning pasted back ($(date -u +%Y-%m-%d))"
common=(--model Qwen/Qwen3.8-27B --dtype bfloat16 --screen latest --layer 0.7 --alpha 0.5 --window observation
        --thinking --think-tokens 1024 --paste-thinking --max-turns 30 --no-yoked --choice --episodes 32 --batch 16
        --group "$group")
launch() {  # gpus frame shard
  CUDA_VISIBLE_DEVICES=$1 nohup python3 scripts/r0_pilot.py "${common[@]}" --frame "$2" --shard "$3" \
    > "runs/e5_$2_${3%/*}.log" 2>&1 < /dev/null &
  echo "rung $2 shard $3 on GPUs $1: pid $!"
}
launch 0,1 inside 0/2
launch 2,3 inside 1/2
launch 4,5 blind 0/2
launch 6,7 blind 1/2
