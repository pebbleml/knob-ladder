#!/usr/bin/env bash
# Sync this repo to a GPU box and run a command there. Models download on the box, never locally.
#
#   scripts/remote.sh ubuntu@1.2.3.4 setup
#   scripts/remote.sh ubuntu@1.2.3.4 run python3 -m pytest tests -q
#   scripts/remote.sh ubuntu@1.2.3.4 pull        # copy runs/ back
set -euo pipefail
host=$1; cmd=$2; shift 2
here=$(cd "$(dirname "$0")/.." && pwd)
key=~/.ssh/id_ed25519
ssh_="ssh -i $key -o StrictHostKeyChecking=accept-new"

sync() { rsync -az --delete --exclude runs/ --exclude .git/ --exclude __pycache__/ --exclude .claude/ -e "$ssh_" "$here/" "$host:knob-ladder/"; }

case $cmd in
  setup)
    sync
    $ssh_ "$host" 'pip install -q "transformers==4.57.6" "numpy<2" "pillow>=10" accelerate scikit-learn pytest && python3 -c "import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.get_device_name())"'
    ;;
  run)
    sync
    $ssh_ "$host" "cd knob-ladder && $*"
    ;;
  pull)
    rsync -az -e "$ssh_" "$host:knob-ladder/runs/" "$here/runs/"
    ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
