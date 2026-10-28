#!/usr/bin/env bash
# Sync this repo to a GPU box and run a command there. Models download on the box, never locally.
#
#   scripts/remote.sh ubuntu@1.2.3.4 setup
#   scripts/remote.sh ubuntu@1.2.3.4 run python3 -m pytest tests -q
#   scripts/remote.sh ubuntu@1.2.3.4 pull        # copy runs/ back
#   scripts/remote.sh ubuntu@1.2.3.4 watch       # live episodes at http://localhost:8008
set -euo pipefail
host=$1; cmd=$2; shift 2
here=$(cd "$(dirname "$0")/.." && pwd)
key=~/.ssh/id_ed25519
ssh_="ssh -i $key -o StrictHostKeyChecking=accept-new"

# The box gets no .git, so each sync writes REVISION: the commit, and "+dirty" with uncommitted changes.
sync() {
  rsync -az --delete --exclude runs/ --exclude .git/ --exclude __pycache__/ --exclude .claude/ --exclude REVISION -e "$ssh_" "$here/" "$host:knob-ladder/"
  rev=$(git -C "$here" rev-parse HEAD)$([ -n "$(git -C "$here" status --porcelain)" ] && echo +dirty)
  echo "$rev" | $ssh_ "$host" 'cat > knob-ladder/REVISION'
}

case $cmd in
  setup)
    sync
    $ssh_ "$host" 'pip install -q "transformers==4.57.6" "numpy<2" "pillow>=10" accelerate scikit-learn pytest && python3 -c "import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.get_device_name())"'
    ;;
  run)
    sync
    $ssh_ "$host" "cd knob-ladder && $*"
    ;;
  watch)
    sync
    echo "open http://localhost:8008"
    $ssh_ -t -L 8008:localhost:8008 "$host" "cd knob-ladder && python3 scripts/watch.py --port 8008"
    ;;
  pull)
    rsync -az -e "$ssh_" "$host:knob-ladder/runs/" "$here/runs/"
    ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
