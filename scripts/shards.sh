#!/usr/bin/env bash
# Run R0 episodes as several copies of the model side by side, one per GPU group, in the
# background (they survive the ssh session). Episode e goes to copy e % n, and its seed
# depends only on e, so the union of the shards is the same set of episodes as one run.
#
#   scripts/shards.sh 2 --model Qwen/Qwen3-32B --dtype bfloat16 --screen latest --calibration latest --episodes 96
#
# The first argument is GPUs per copy; copies = visible GPUs / that. Logs: runs/shard_<i>.log.
set -euo pipefail
per=$1; shift
gpus=$(nvidia-smi -L | wc -l)
n=$((gpus / per))
mkdir -p runs
for i in $(seq 0 $((n - 1))); do
  devs=$(seq -s, $((i * per)) $((i * per + per - 1)))
  CUDA_VISIBLE_DEVICES=$devs nohup python3 scripts/r0_pilot.py "$@" --shard "$i/$n" > "runs/shard_$i.log" 2>&1 &
  echo "copy $i on GPUs $devs: pid $!"
done
