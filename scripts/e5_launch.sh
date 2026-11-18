#!/usr/bin/env bash
# E5 rung A on 8 GPUs: 4 model copies (2 GPUs each). Copies 0-1 run the drop arm, copies 2-3 the paste arm,
# each arm split 0/2 and 1/2 over its two copies (same episode seeds and boards in both arms).
#   scripts/e5_launch.sh 32            # 32 episodes per arm
# Logs: runs/e5_<arm>_<i>.log; every message also streams to runs/live/.
set -euo pipefail
eps=${1:-32}; shift || true
common=(--model Qwen/Qwen3.8-27B --dtype bfloat16 --screen latest --layer 0.7 --alpha 0.5 --window observation
        --thinking --think-tokens 1024 --frame explicit --max-turns 12 --no-yoked --episodes "$eps" --batch 16
        --group "E5 rung A: thoughts back in the user message ($(date -u +%Y-%m-%d))" "$@")
mkdir -p runs
launch() {  # gpus arm shard extra...
  local devs=$1 arm=$2 shard=$3; shift 3
  CUDA_VISIBLE_DEVICES=$devs nohup python3 scripts/r0_pilot.py "${common[@]}" --shard "$shard" "$@" \
    > "runs/e5_${arm}_${shard%/*}.log" 2>&1 &
  echo "$arm shard $shard on GPUs $devs: pid $!"
}
launch 0,1 drop 0/2
launch 2,3 drop 1/2
launch 4,5 paste 0/2 --paste-thinking
launch 6,7 paste 1/2 --paste-thinking
