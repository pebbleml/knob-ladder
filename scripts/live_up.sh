#!/usr/bin/env bash
# Restart the live dashboard (port 8008) and a public Cloudflare quick tunnel to it; print the tunnel URL.
set -e
cd ~/knob-ladder
for p in $(pgrep -f "scripts/watch.py --port 8008"); do [ "$p" != "$$" ] && kill $p || true; done
for p in $(pgrep -f "cloudflared-empty.yml"); do kill $p || true; done
sleep 1
nohup python3 scripts/watch.py --port 8008 > runs/watch.log 2>&1 < /dev/null &
cf=~/bin/cloudflared
if [ ! -x $cf ]; then
  mkdir -p ~/bin
  curl -sSL -o $cf.tmp https://github.com/cloudflare/cloudflared/releases/download/2026.10.0/cloudflared-linux-amd64
  echo "d33ff2d14475178d2012c2c56beba87389ac5ded27649519f198a7d3134a99db  $cf.tmp" | sha256sum -c --quiet
  chmod +x $cf.tmp && mv $cf.tmp $cf
fi
: > ~/bin/cloudflared-empty.yml
rm -f runs/live_tunnel.log
nohup $cf --config ~/bin/cloudflared-empty.yml tunnel --no-autoupdate --protocol http2 --url http://127.0.0.1:8008 > runs/live_tunnel.log 2>&1 < /dev/null &
for i in $(seq 40); do
  url=$(grep -ao "https://[a-z0-9-]*\.trycloudflare\.com" runs/live_tunnel.log | head -1 || true)
  [ -n "$url" ] && break; sleep 1
done
echo "TUNNEL $url"
