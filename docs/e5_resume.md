# Resuming E5: everything needed to run it again

Paused 2026-11-17. WIRK ticket: **f3f64d4c** ("E5 Thoughts back in the user message…"). Next version: [e5_v3_design.md](e5_v3_design.md).

## Where everything is

| What | Where |
|---|---|
| Code | GitHub `pebbleml/knob-ladder` (this repo) |
| Raw runs | Hugging Face dataset `pebbleml/knob-ladder-runs`: `live/e5-*/` (every message, one file per session), `e5-*.jsonl` (end-of-session scoring), `logs/e5*.log`, `screen_Qwen3.8-27B_20261030-012358.json` (concept screen), `calibrate_gen_Qwen3.8-27B_20261030-032016.json` (dose calibration) |
| Readable sessions, rungs B and C | `analysis/e5_sessions/<B|C>-epNNN.txt`, with the answer key at the top |
| Analysis | `analysis/e5_analyze.py` (text measures and scoring), `analysis/e5_export_sessions.py` (makes the readable sessions), `analysis/e5_review/` (per-session Haiku judge plus Sonnet write-ups: `SUMMARY.md`, `ALL_WRITEUPS.md`, `review.json`) |
| Dashboard | https://knob-ladder-runs.pages.dev/live (a saved copy is served when no box is up) |
| Design and prompts | `docs/e5_design.md`, `docs/e5_prompts.md` (rendered by `scripts/e5_render.py`) |

## Runs (Qwen3.8-27B, revision 1d4bf0f2, bf16, layer 44, alpha 0.5, dose 55.5)

| Run | Sessions | State |
|---|---|---|
| Rung A, reasoning dropped (v1, 12 turns) | 32 | complete, scored |
| Rung A, reasoning pasted (v1) | 32 | experiments complete; scoring ran out of GPU memory |
| Rung B inside, pasted (v2, 30 turns) | 16 + 16 | half complete; the other half stopped by Sam before scoring |
| Rung C blind, pasted (v2, 30 turns) | 32 | complete, scored |
| Rung A pasted, rerun | 32 | stopped by Sam partway |

## Running it on a fresh GPU box (8× A100 40GB, or any box with 2+ GPUs per model copy)

New compute needs Sam's OK. Models download on the box, never on Sam's Mac.

```bash
scripts/remote.sh ubuntu@HOST setup        # rsync code (not runs/), pip install transformers 5.20.2, fla, causal-conv1d
rsync -az runs/screen_Qwen3.8-27B_20261030-012358.json runs/calibrate_gen_Qwen3.8-27B_20261030-032016.json ubuntu@HOST:knob-ladder/runs/
```

(`--screen latest` needs the screen file in the box's `runs/`. `remote.sh` never syncs `runs/`, so you must copy it as above. You can also get it from Hugging Face.)

**The 64 parallel sessions (rungs B and C, v2):** `scripts/e5_queue.sh` gives each GPU pair one model copy, and each copy takes the next job from the queue. Each job is 16 sessions in lockstep (`--batch 16`, which drops to 8 or 4 on out-of-memory). Edit the job list at the top of the script, then:

```bash
ssh ubuntu@HOST 'cd knob-ladder && nohup scripts/e5_queue.sh > runs/e5_queue.log 2>&1 < /dev/null &'
```

**Rung A, dropped vs pasted:** `scripts/e5_launch.sh 32` (4 copies: drop 0/2 and 1/2, paste 0/2 and 1/2). `scripts/e5_next.sh` queues rungs B and C behind it (v1 settings).

The core command line, which every script wraps:

```
python3 scripts/r0_pilot.py --model Qwen/Qwen3.8-27B --dtype bfloat16 --screen latest --layer 0.7 --alpha 0.5
  --window observation --thinking --think-tokens 1024 [--paste-thinking] --frame explicit|inside|blind
  --max-turns 12|30 --no-yoked --episodes 32 --shard 0/2 --batch 16 --group "<dashboard group name>"
```

**Live dashboard and public link:** `scripts/live_up.sh` on the box starts `watch.py` on port 8008 plus a Cloudflare quick tunnel, and prints the tunnel URL. `scripts/watch_restart.sh` restarts only the server.

To publish the site from the Mac:

```bash
BOX=ubuntu@HOST scripts/site_deploy.sh live.json
```

`live.json` is `{"id","title","question","origin": <tunnel URL>,"started"}`. With `BOX=none`, the deploy uses the Mac's copy. Publishing needs a Cloudflare wrangler login on the Mac.

**Restart a stuck queue:** `scripts/e5_restart.sh`, run on the box. Never use `pkill -f` over ssh: the pattern matches the ssh shell itself. Use the scripts on the box.

**On every check:** pull the runs.

```bash
rsync -az --exclude '*.ckpt.pt' -e "ssh -i ~/.ssh/id_ed25519" ubuntu@HOST:knob-ladder/runs/ runs/
```

Then upload to Hugging Face. Exclude `share*` and `*.ckpt.pt`. Tell Sam when the box is idle.

## Analysing a finished run

```bash
python3 analysis/e5_export_sessions.py      # readable sessions with the answer key
python3 analysis/e5_analyze.py              # text measures and the end-of-session scoring
```

Per-session review: Claude Code's Workflow tool with `scripts/e5_review_workflow.js`. It runs one Haiku judge per session, one Sonnet write-up per positive session, and one Sonnet ranker.
