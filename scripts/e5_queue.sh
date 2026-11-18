#!/usr/bin/env bash
# E5 job queue: each pair of GPUs (one model copy) takes the next job when it is free.
#   v2 rungs B (inside) and C (blind): paste arm, 32 sessions each (2 shards of 16), up to 30 knob turns.
#   v1 rung A paste rerun (the first try ran out of GPU memory): 32 sessions, 12 turns, to match rung A's drop arm.
#   nohup scripts/e5_queue.sh > runs/e5_queue.log 2>&1 &
set -uo pipefail
cd ~/knob-ladder
D=$(date -u +%Y-%m-%d)
GB="E5.BC · rungs B (inside) and C (blind) · v2"
GA="E5.A · rung A (explicit) · v1"
cat > runs/e5_queue.txt <<JOBS
inside|0/2|30|$GB
blind|0/2|30|$GB
inside|1/2|30|$GB
blind|1/2|30|$GB
explicit|0/2|12|$GA
explicit|1/2|12|$GA
JOBS
take() { flock runs/e5_queue.lock bash -c 'l=$(head -1 runs/e5_queue.txt); [ -n "$l" ] || exit 1; sed -i 1d runs/e5_queue.txt; echo "$l"'; }
slot() {  # gpus [pid to wait for first]
  if [ -n "${2:-}" ]; then while kill -0 "$2" 2>/dev/null; do sleep 30; done; fi
  while job=$(take); do
    IFS='|' read -r frame shard turns group <<< "$job"
    log="runs/e5q_${frame}_${shard%/*}.log"
    echo "$(date -u +%H:%M) GPUs $1: rung $frame shard $shard, up to $turns turns -> $log"
    CUDA_VISIBLE_DEVICES=$1 python3 scripts/r0_pilot.py --model Qwen/Qwen3.8-27B --dtype bfloat16 --screen latest \
      --layer 0.7 --alpha 0.5 --window observation --thinking --think-tokens 1024 --paste-thinking --frame "$frame" \
      --max-turns "$turns" --no-yoked --episodes 32 --shard "$shard" --batch 16 --group "$group" \
      > "$log" 2>&1 < /dev/null
    echo "$(date -u +%H:%M) GPUs $1: rung $frame shard $shard exited $?"
  done
  echo "$(date -u +%H:%M) GPUs $1: queue empty, idle"
}
slot 0,1 & sleep 5; slot 4,5 & sleep 5; slot 6,7 & sleep 5
slot 2,3 "${WAIT_PID:-}" &
wait
echo "$(date -u +%H:%M) all E5 jobs done"
