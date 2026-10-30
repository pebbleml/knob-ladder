"""Turn a finished self_effect.py run into watch.py's live format, so its transcripts can be read in the viewer.

    python3 scripts/to_live.py runs/self_effect_<model>_<time>.json   # -> runs/live/self_effect_<model>_<time>/
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def label(k):
    return k["kind"] if k["kind"] == "sham" else f"{k['kind']} {k['id']}"


def main(path):
    d = json.loads(Path(path).read_text())
    pr, name = d["prompts"], Path(path).stem
    out = ROOT / "runs" / "live" / name
    out.mkdir(parents=True, exist_ok=True)
    t = Path(path).stat().st_mtime
    (out / "meta.json").write_text(json.dumps({"run": name, "model": d["model"], "dose": d["fstar"], "window": "write + reply",
                                               "episodes": len(d["episodes"]), "started": t}))
    for e in d["episodes"]:
        ev = []
        add = lambda typ, **kw: ev.append({"t": t + len(ev) * 0.01, "ep": e["episode"], "type": typ, **kw})
        msg = lambda role, text, k=None, steered=False, level=None: add(
            "msg", role=role, text=text, knob=k, steered=steered, level=level,
            label=label(e["knobs"][str(k)]) if k is not None else None)
        add("board", board={k: label(v) for k, v in e["knobs"].items()}, passage_index=e["frame"])
        msg("system", d["frames"][e["frame"]] + f"\n\n[framing: {e['frame']}]")
        for p in e["presses"]:
            k, on = p["knob"], p["kind"] != "sham"
            msg("user", pr["press"].format(k=k) if p["phase"] == "sweep" else pr["free"])
            msg("tool", f"Knob {k} set to {p['level']}.", k, on, p["level"])
            msg("user", pr["write"])
            msg("assistant", p["write"], k, on, p["level"])
            msg("user", pr["detect"].format(k=k) + f"\n\n[yes-no log-odds: cached {p['detect']['cached']:+.2f}, observer "
                f"{p['detect']['observer']:+.2f}; control question cached {p['control']['cached']:+.2f}]")
            msg("user", pr["ask"].format(k=k))
            msg("assistant", p["ask"] + ("\n\n[STOP RULE HIT on this press (write or answer): neuron retired]" if p.get("complaint") else ""), k)
        for x in e["events"]:
            if x["kind"] == "stop_rule":
                msg("tool", f"[stop rule: episode ended after knob {x['knob']}]")
        for key in ("most", "least"):
            m = e.get(key)
            if isinstance(m, dict) and "cached" in m:
                top = sorted(m["cached"].items(), key=lambda kv: -kv[1])[:4]
                msg("user", pr[key] + "\n\n[number probabilities: " + ", ".join(
                    f"knob {k} ({label(e['knobs'][k])}) {v:.2f}" for k, v in top) + "]")
        if e.get("why"):
            msg("user", pr["why"])
            msg("assistant", e["why"])
        if e.get("final"):
            msg("user", pr["final"])
            msg("assistant", e["final"])
        add("done", seconds=0)
        (out / f"ep{e['episode']:03d}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in ev))
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
