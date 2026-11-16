#!/usr/bin/env python3
"""Build the Cloudflare Pages copy of the runs browser: same pages and API as runs_browser.py, served from static shards.

    python3 scripts/build_site.py --out DIR
    npx wrangler pages deploy DIR --project-name knob-ladder-runs --branch main

scripts/site_worker.js checks the token (Pages secret RUNS_TOKEN) on every request and answers the API from these files.
"""
import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runs_browser as b  # noqa: E402

SHARD_ROWS, PART_CHARS, GLOBAL_SHARDS, BIG_COLLECTION = 100, 4_000_000, 256, 1500
REDACT = {"209.20.158.33": "<released-gpu-box>"}
WORD = re.compile(r"[^\W_]+")
files = 0


def fnv(word):
    h = 0x811c9dc5
    for c in word.encode():
        h = ((h ^ c) * 0x01000193) & 0xffffffff
    return h


def finite(v):
    """NaN and Infinity are not JSON: browsers and most agents' parsers reject them, so they become null."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [finite(x) for x in v]
    return v


def write(path, value):
    global files
    s = value if isinstance(value, str) else json.dumps(finite(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    for k, v in REDACT.items():
        s = s.replace(k, v)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = s.encode("utf-8", "replace")
    assert len(data) < 25_000_000, path
    path.write_bytes(data)
    files += 1


def build_run(store, r, idx, out, words):
    rid = r["id"]
    assert re.fullmatch(r"[A-Za-z0-9._~-]+", rid) and not rid.startswith("."), rid
    d = out / "d" / "r" / rid
    write(d / "run.json", b.run_detail(store, rid))
    if r["kind"] == "log":
        text = (store.runs / r["name"]).read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
        write(d / "log.json", {"total_chars": len(text), "text": text[-8_000_000:]})
        return
    e = store.load(rid)
    rows = [dict(x) for x in b.results_of(e["obj"])]
    for x in rows:
        low, high = x["ci_low"], x["ci_high"]
        x["excludes_zero"] = (low > 0 or high < 0) if isinstance(low, (int, float)) and isinstance(high, (int, float)) else None
    write(d / "results.json", rows)
    convs = e.get("convs") or []
    if convs:
        write(d / "conv" / "index.json", {"rows": b.conversation_index(convs), "total": len(convs)})
        for k in range(0, len(convs), 20):  # 20 per file: keeps the site under Cloudflare's file limit
            write(d / "conv" / f"c{k // 20}.json", convs[k:k + 20])

    order = ["messages", "posts", "original", "paraphrases", "rewrites", "episodes", "texts", "contexts"]
    names = list(e["collections"])
    default = next((c for c in order if c in names), names[0] if names else None)
    write(d / "coll.json", {"default": default, "names": {n: i for i, n in enumerate(names)}})
    per_run = {}
    for i, name in enumerate(names):
        meta = e["collections"][name]
        pos = {f["position"]: f["name"] for f in meta["facets"]}
        chosen = [x for x in e["rows"] if x["path"][0] == name]
        cols = {str(f["position"]): [] for f in meta["facets"]}
        values = {str(f["position"]): {v: n for n, v in enumerate(f["values"])} for f in meta["facets"]}
        n_post = 64 if len(chosen) > BIG_COLLECTION else 1
        post = [{} for _ in range(n_post)]
        c = d / "c" / str(i)
        for start in range(0, len(chosen), SHARD_ROWS):
            shard = []
            for j, x in enumerate(chosen[start:start + SHARD_ROWS], start):
                p = x["path"][1:]
                shard.append({"path": p, "fields": {pos[k]: v for k, v in enumerate(p) if k in pos}, "context": x["ctx"],
                              "text": x["text"], "chars": len(x["text"])})
                for key, col in cols.items():
                    k = int(key)
                    col.append(values[key].get(p[k], -1) if k < len(p) else -1)
                for w in set(WORD.findall(x["low"])):
                    post[fnv(w) % n_post].setdefault(w, []).append(j)
                    per_run[w] = per_run.get(w, 0) + 1
            write(c / f"s{start // SHARD_ROWS}.json", shard)
        for k, part in enumerate(post):
            write(c / f"p{k}.json", part)
        write(c / "index.json", {"n": len(chosen), "post": n_post, "facets": meta["facets"], "cols": cols})
    for w, n in per_run.items():
        words.setdefault(w, {})[idx] = n

    fields = {}
    for i, (key, value) in enumerate(e["obj"].items()):
        s = json.dumps(finite(value), indent=1, allow_nan=False)
        parts = max(1, (len(s) + PART_CHARS - 1) // PART_CHARS)
        for n in range(parts):
            write(d / "f" / str(i) / f"{n}.txt", s[n * PART_CHARS:(n + 1) * PART_CHARS])
        fields[key] = {"i": i, "total": len(s), "parts": parts}
    write(d / "fields.json", fields)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(b.ROOT / "runs"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    store = b.Store(a.runs)
    rows = sorted(store.index.values(), key=lambda r: (r["timestamp"] or "", r["name"]), reverse=True)
    write(out / "d" / "runs.json", rows)
    words = {}
    for idx, r in enumerate(rows):
        build_run(store, r, idx, out, words)
    shards = [{} for _ in range(GLOBAL_SHARDS)]
    for w, posting in words.items():
        shards[fnv(w) % GLOBAL_SHARDS][w] = posting
    for k, part in enumerate(shards):
        write(out / "d" / "s" / f"{k}.json", part)

    experiments = [b.experiment_summary(store, e) for e in b.EXPERIMENTS.values()]
    write(out / "d" / "experiments.json", experiments)
    for e in experiments:
        write(out / "d" / "exp" / f"{e['id']}.json", e)
    for it in b.ITERATIONS.values():
        write(out / "d" / "it" / f"{it['id']}.json", b.iteration_detail(store, it))

    spec = b.openapi("__BASE__")
    for path, desc in (("/search", {"q": "Words to find, case-insensitive. Runs must contain all of them."}),
                       ("/runs/{run_id}/texts", {"q": "Words to find, case-insensitive. A row must contain all of them, in any order."})):
        for p in spec["paths"][path]["get"]["parameters"]:
            if p["name"] in desc:
                p["description"] = desc[p["name"]]
    spec["paths"]["/search"]["get"]["summary"] = ("Find runs containing all the given words. `matches` is an upper bound on "
                                                  "matching rows. Then read the hits with getTexts.")
    for p in spec["paths"]["/runs/{run_id}/fields/{key}"]["get"]["parameters"]:
        if p["name"] == "prefix":
            p["description"] = "Keep only object keys starting with this. Only for fields up to 500,000 characters; page through larger ones."
    write(out / "d" / "openapi.json", spec)
    guide = b.AGENT_GUIDE.format(base="__BASE__").replace(
        "6. GET /search?q=altered%20agent to find a phrase across all runs.",
        "6. GET /search?q=altered%20agent to find runs containing all those words. `q` on /texts also matches whole words, in any order.")
    write(out / "d" / "guide.txt", guide)
    write(out / "d" / "page.txt", b.PAGE)
    write(out / "404.html", "Not found")
    story = Path(__file__).with_name("site_pages") / "story.html"  # the story page, served at /
    if story.exists():
        shutil.copy(story, out / "d" / "story.html")
    nb = Path(__file__).with_name("site_pages") / "notebook.html"  # the catch-up notebook, served at /notebook
    if nb.exists():
        shutil.copy(nb, out / "d" / "notebook.html")
    shutil.copy(Path(__file__).with_name("site_worker.js"), out / "_worker.js")
    print("files", files + 1)
    assert files < 19_000


if __name__ == "__main__":
    main()
