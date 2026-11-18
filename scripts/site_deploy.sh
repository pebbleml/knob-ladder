#!/usr/bin/env bash
# Rebuild the site with a saved copy of every session board, and deploy it to Cloudflare Pages.
#   BOX=ubuntu@1.2.3.4 scripts/site_deploy.sh LIVE_JSON     (LIVE_JSON: {origin: tunnel URL, ...}; BOX defaults to none)
# The saved copy = the E-series boards pulled off the GPU box + the older session runs kept on this Mac
# (runs/live) + sessions rebuilt from session logs (runs/recovered_live). The worker serves the box's live
# state when the box is up and adds every experiment the box does not hold from this copy.
set -euo pipefail
B=/Volumes/1TB_SSD/pebble/machine-consciousness/.build
cd "$(dirname "$0")/.."
mkdir -p $B/live_box
# BOX=user@host pulls the E-series boards from a running GPU box; BOX=none (no box up) uses this Mac's runs/live/e5-* copy.
BOX=${BOX:-none}
if [ "$BOX" = none ]; then rsync -a --delete --include 'e5-*/***' --exclude '*' runs/live/ $B/live_box/
else rsync -az --delete -e "ssh -i $HOME/.ssh/id_ed25519" --include 'e5-*/***' --exclude '*' "$BOX":knob-ladder/runs/live/ $B/live_box/; fi
rm -rf $B/live_snap && mkdir -p $B/live_snap
rsync -a $B/live_box/ $B/live_snap/
rsync -a --exclude 'e5-*' --exclude 'share*' --exclude '*.ckpt.pt' runs/live/ $B/live_snap/
rsync -a runs/recovered_live/ $B/live_snap/
python3 scripts/build_site.py --out $B/site --live "$1" --live-snapshot $B/live_snap 2>&1 | grep -v -i warn | tail -1
cd $B && npm_config_cache=$B/npm-cache npx --yes wrangler pages deploy site --project-name knob-ladder-runs --branch main 2>&1 | tail -1
