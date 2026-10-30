#!/usr/bin/env bash
# Sync this repo to a GPU box and run a command there. Models download on the box, never locally.
#
#   scripts/remote.sh ubuntu@1.2.3.4 setup
#   scripts/remote.sh ubuntu@1.2.3.4 run python3 -m pytest tests -q
#   scripts/remote.sh ubuntu@1.2.3.4 pull        # copy runs/ back
#   scripts/remote.sh ubuntu@1.2.3.4 watch       # live episodes at http://localhost:8008 (--blind hides labels)
#   scripts/remote.sh ubuntu@1.2.3.4 share       # the same viewer at a secret public https link; share stop
set -euo pipefail
host=$1; cmd=$2; shift 2
here=$(cd "$(dirname "$0")/.." && pwd)
key=~/.ssh/id_ed25519
ssh_="ssh -i $key -o StrictHostKeyChecking=accept-new"

# The box gets no .git, so each sync writes REVISION: the commit, and "+dirty" with uncommitted changes.
sync() {
  rsync -az --delete --exclude runs/ --exclude .git/ --exclude __pycache__/ --exclude .claude/ --exclude REVISION -e "$ssh_" "$here/" "$host:knob-ladder/"
  rev=$(git -C "$here" rev-parse HEAD)
  if [ -n "$(git -C "$here" status --porcelain)" ]; then rev=$rev+dirty; fi
  echo "$rev" | $ssh_ "$host" 'cat > knob-ladder/REVISION'
}

case $cmd in
  setup)
    sync
    # transformers 5.20.2 has Qwen3.8 (qwen3_5); it needs python >= 3.10. flash-linear-attention is the fast
    # Gated DeltaNet kernel (pure triton, no torch upgrade). causal-conv1d only ships as source, so its build is
    # best-effort: without it the conv falls back to torch, exact but slower.
    $ssh_ "$host" 'python3 -c "import sys; assert sys.version_info >= (3, 10), sys.version" &&
      pip install -q "transformers==5.20.2" "flash-linear-attention==0.5.2" "numpy<2" "pillow>=10" accelerate scikit-learn pytest "jinja2>=3.1" &&
      (timeout 1200 pip install -q --no-build-isolation "causal-conv1d==1.6.2.post1" || echo "causal-conv1d not built; torch fallback") &&
      python3 -c "import torch,transformers,importlib.util as u;print(torch.__version__,transformers.__version__,torch.cuda.get_device_name(),{m:bool(u.find_spec(m)) for m in (\"fla\",\"causal_conv1d\")})"'
    ;;
  run)
    sync
    $ssh_ "$host" "cd knob-ladder && $*"
    ;;
  watch)
    sync
    echo "open http://localhost:8008"
    $ssh_ -t -L 8008:localhost:8008 "$host" "cd knob-ladder && python3 scripts/watch.py --port 8008 $*"
    ;;
  share)
    # The live viewer at a public https URL (Cloudflare quick tunnel), gated by a random token in the link.
    # Anyone with the link can read the episodes; `share stop` takes it down.
    sync
    if [ "${1:-}" = stop ]; then
      $ssh_ "$host" 'kill $(cat ~/knob-ladder/runs/share.pids 2>/dev/null) 2>/dev/null || true; rm -f ~/knob-ladder/runs/share.pids'
      exit 0
    fi
    $ssh_ "$host" 'set -e
      cf=~/bin/cloudflared
      if [ ! -x $cf ]; then
        mkdir -p ~/bin
        curl -sSL -o $cf.tmp https://github.com/cloudflare/cloudflared/releases/download/2026.10.0/cloudflared-linux-amd64
        echo "d33ff2d14475178d2012c2c56beba87389ac5ded27649519f198a7d3134a99db  $cf.tmp" | sha256sum -c --quiet
        chmod +x $cf.tmp && mv $cf.tmp $cf
      fi
      # PIDs, not pkill -f: the command line of this very shell contains the patterns.
      kill $(cat ~/knob-ladder/runs/share.pids 2>/dev/null) 2>/dev/null || true
      sleep 1; rm -f ~/knob-ladder/runs/share_tunnel.log  # the old tunnel writes its exit into the log
      token=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
      cd knob-ladder && mkdir -p runs
      nohup python3 scripts/watch.py --port 8009 --token $token '"$*"' > runs/share_watch.log 2>&1 < /dev/null &
      echo $! > runs/share.pids
      : > ~/bin/cloudflared-empty.yml  # else it reads the box config at /etc/cloudflared, whose ingress 404s us
      nohup $cf --config ~/bin/cloudflared-empty.yml tunnel --no-autoupdate --protocol http2 --url http://127.0.0.1:8009 > runs/share_tunnel.log 2>&1 < /dev/null &
      echo $! >> runs/share.pids
      for i in $(seq 30); do
        url=$(grep -ao "https://[a-z0-9-]*\.trycloudflare\.com" runs/share_tunnel.log | head -1 || true)
        [ -n "$url" ] && break; sleep 1
      done
      [ -n "$url" ] || { tail runs/share_tunnel.log; exit 1; }
      echo "$url/?k=$token"'
    ;;
  pull)
    rsync -az -e "$ssh_" "$host:knob-ladder/runs/" "$here/runs/"
    ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
