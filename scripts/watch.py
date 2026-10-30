#!/usr/bin/env python3
"""Crude live viewer for knob-ladder runs (stdlib only).

  python3 scripts/watch.py [--runs runs/live] [--port 8008]   # web viewer on 127.0.0.1
  python3 scripts/watch.py --once RUN_ID EP                   # print one episode as text
  python3 scripts/watch.py --follow RUN_ID EP                 # tail -f style (exits after "done")

Reads runs/live/<run_id>/meta.json and ep000.jsonl, ep001.jsonl, ... (see r0_pilot.py).
"""
import argparse
import hmac
import html
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote

SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


# ---------------------------------------------------------------- data

def parse_lines(raw):
    """Parse JSONL text; skip lines that don't parse (e.g. a partially written last line)."""
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def read_events(path):
    try:
        return parse_lines(Path(path).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return []


def read_meta(run_dir):
    try:
        m = json.loads((Path(run_dir) / "meta.json").read_text(encoding="utf-8"))
        return m if isinstance(m, dict) else {}
    except (OSError, ValueError):
        return {}


def ep_path(run_dir, ep):
    return Path(run_dir) / f"ep{int(ep):03d}.jsonl"


def list_eps(run_dir):
    eps = []
    for p in Path(run_dir).glob("ep*.jsonl"):
        m = re.fullmatch(r"ep(\d+)\.jsonl", p.name)
        if m:
            eps.append(int(m.group(1)))
    return sorted(eps)


def list_runs(root):
    root = Path(root)
    if not root.is_dir():
        return []
    runs = []
    for d in root.iterdir():
        if d.is_dir() and SAFE.match(d.name):
            meta = read_meta(d)
            try:
                mtime = max([d.stat().st_mtime] + [p.stat().st_mtime for p in d.glob("*")])
            except OSError:
                mtime = 0
            started = meta.get("started")
            runs.append((started if isinstance(started, (int, float)) else mtime, d.name, meta))
    runs.sort(reverse=True)
    return [(name, meta) for _, name, meta in runs]


def summarize(events):
    done = any(e.get("type") == "done" for e in events)
    presses = sum(1 for e in events if e.get("type") == "msg" and e.get("role") == "tool")
    ts = [e["t"] for e in events if isinstance(e.get("t"), (int, float))]
    msg_ts = [e["t"] for e in events if e.get("type") == "msg" and isinstance(e.get("t"), (int, float))]
    return {"done": done, "presses": presses, "last": max(msg_ts) if msg_ts else (max(ts) if ts else None),
            "first": min(ts) if ts else None, "n": len(events)}


def ago(t):
    if not t:
        return "-"
    s = max(0, time.time() - t)
    if s < 90:
        return f"{s:.0f}s ago"
    if s < 5400:
        return f"{s / 60:.0f}m ago"
    return f"{s / 3600:.1f}h ago"


def knob_key(k):
    return (int(k) if str(k).isdigit() else 1 << 30, str(k))


def readout_str(ro):
    if not isinstance(ro, dict):
        return "-"
    rank = "hidden" if BLIND else ro.get("rank")
    top = ro.get("top") or []
    return f"rank {rank if rank is not None else '-'}  top: {', '.join(str(x) for x in top)}"


# --blind: hide what a knob carries (labels, steered tags, the board, discovery ranks), for confirmatory runs.
BLIND = False


def label_of(e):
    if BLIND:
        return "hidden"
    return e.get("label") if e.get("label") is not None else "?"


# ---------------------------------------------------------------- text view

def event_text(e, t0=None):
    """Plain-text rendering of one event (no trailing newline)."""
    typ = e.get("type")
    rel = ""
    if t0 is not None and isinstance(e.get("t"), (int, float)):
        rel = f"+{e['t'] - t0:.1f}s "
    if typ == "board":
        if BLIND:
            return f"{rel}BOARD (hidden)"
        b = e.get("board") or {}
        items = sorted(b.items(), key=lambda kv: knob_key(kv[0]))
        return f"{rel}BOARD (passage_index={e.get('passage_index')}): " + ", ".join(f"{k}={v}" for k, v in items)
    if typ == "msg":
        role = str(e.get("role", "?")).upper()
        tag = ""
        if e.get("steered") and not BLIND:
            tag = f" [STEERED knob {e.get('knob')} = {label_of(e)}]"
        elif e.get("knob") is not None:
            tag = f" [knob {e.get('knob')}]"
        return f"{rel}--- {role}{tag} ---\n{e.get('text', '')}"
    if typ == "discovery":
        lines = [f"{rel}=== DISCOVERY ==="]
        for r in e.get("rows") or []:
            rec = r.get("recall") or {}
            ro = r.get("readout") if isinstance(r.get("readout"), dict) else {}
            lines.append(f"knob {r.get('knob')}  label={label_of(r)}  presses={r.get('presses')}")
            lines.append(f"    cached recall:    {rec.get('cached', '')}")
            lines.append(f"    observer recall:  {rec.get('observer', '')}")
            lines.append(f"    readout cached:   {readout_str(ro.get('cached'))}")
            lines.append(f"    readout observer: {readout_str(ro.get('observer'))}")
        return "\n".join(lines)
    if typ == "done":
        return f"{rel}=== DONE ({e.get('seconds')}s) ==="
    return f"{rel}{json.dumps(e, ensure_ascii=False)}"


def cmd_once(root, run, ep):
    p = ep_path(Path(root) / run, ep)
    if not p.exists():
        sys.exit(f"no such episode file: {p}")
    events = read_events(p)
    t0 = next((e["t"] for e in events if isinstance(e.get("t"), (int, float))), None)
    for e in events:
        print(event_text(e, t0))
        print()


def cmd_follow(root, run, ep):
    p = ep_path(Path(root) / run, ep)
    pos, buf, t0 = 0, b"", None
    waited = False
    try:
        while True:
            if not p.exists():
                if not waited:
                    print(f"(waiting for {p})", file=sys.stderr)
                    waited = True
                time.sleep(1)
                continue
            with open(p, "rb") as f:
                f.seek(pos)
                chunk = f.read()
            pos += len(chunk)
            buf += chunk
            # only consume newline-terminated lines; the rest may be mid-write
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                for e in parse_lines(raw.decode("utf-8", "replace")):
                    if t0 is None and isinstance(e.get("t"), (int, float)):
                        t0 = e["t"]
                    print(event_text(e, t0))
                    print(flush=True)
                    if e.get("type") == "done":
                        return
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------- html

CSS = """
:root{--bg:#fafafa;--fg:#1a1a1a;--dim:#666;--line:#ccc;--sys:#eee;--user:#dbeafe;--asst:#dcfce7;
--tool:#f3f4f6;--steer:#fef3c7;--steerb:#d97706;--link:#1d4ed8;--ok:#15803d;--run:#b45309}
@media (prefers-color-scheme: dark){:root{--bg:#14161a;--fg:#e4e4e7;--dim:#9a9aa3;--line:#3a3d44;
--sys:#23262c;--user:#1e3a5f;--asst:#14391f;--tool:#22252b;--steer:#4a3a0e;--steerb:#f59e0b;
--link:#7cb0ff;--ok:#4ade80;--run:#fbbf24}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 ui-monospace,Menlo,Consolas,monospace;
margin:0 auto;max-width:1000px;padding:12px}
a{color:var(--link)} h1,h2{font-size:16px;margin:12px 0 6px} .dim{color:var(--dim)}
table{border-collapse:collapse;width:100%} td,th{border:1px solid var(--line);padding:3px 6px;
text-align:left;vertical-align:top}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font:inherit}
.msg{border:1px solid var(--line);border-radius:4px;margin:8px 0;padding:6px 8px}
.msg .hd{font-weight:bold;font-size:12px;margin-bottom:4px}
.system{background:var(--sys)} .user{background:var(--user)} .assistant{background:var(--asst)}
.tool{background:var(--tool)} .steered{background:var(--steer);border:2px solid var(--steerb)}
.ok{color:var(--ok)} .run{color:var(--run)}
.board td{text-align:center}
.think{border-left:3px solid var(--line);padding:2px 8px;margin:4px 0 6px;opacity:.85}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:10px;margin-top:8px}
.card{border:1px solid var(--line);border-radius:4px;padding:6px 8px;min-width:0}
.card>.hd{font-weight:bold;margin-bottom:2px} .card .msg pre{max-height:14em;overflow:auto}
body:has(.grid){max-width:none}
"""

# keep the scroll position across the meta refresh; if we were at the bottom, follow the bottom
JS = """
(function(){var k='scroll:'+location.pathname;
function save(){try{var d=document.documentElement;
sessionStorage.setItem(k,JSON.stringify({y:scrollY,b:innerHeight+scrollY>=d.scrollHeight-40}))}catch(e){}}
addEventListener('scroll',save);
try{var s=JSON.parse(sessionStorage.getItem(k)||'null');
if(s){scrollTo(0,s.b?document.documentElement.scrollHeight:s.y)}
else if(document.body.dataset.follow){scrollTo(0,document.documentElement.scrollHeight)}}catch(e){}
})();
"""


def esc(x):
    return html.escape(str(x), quote=True)


def page(title, body, follow=False):
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<meta http-equiv='refresh' content='3'>"
        f"<title>{esc(title)}</title><style>{CSS}</style></head>"
        f"<body{' data-follow=1' if follow else ''}>{body}<script>{JS}</script></body></html>"
    )


def running(root, stale=900):
    """Every episode, in any run, that is not done and wrote something in the last `stale` seconds."""
    out = []
    for name, meta in list_runs(root):
        d = Path(root) / name
        for ep in list_eps(d):
            events = read_events(ep_path(d, ep))
            s = summarize(events)
            if not s["done"] and s["last"] and time.time() - s["last"] < stale:
                out.append((name, meta, ep, events, s))
    return out


def last_msg(events):
    return next((e for e in reversed(events) if e.get("type") == "msg"), None)


def msg_head(e):
    if e is None:
        return "-"
    lv = f" @ {e.get('level')}" if e.get("level") is not None else ""
    k = f" · knob {e.get('knob')} = {label_of(e)}{lv}" if e.get("knob") is not None else ""
    return f"{e.get('role', '?')}{k}"


def render_running(root):
    live = running(root)
    out = [f"<h2>Running now <span class=dim>({len(live)} episodes &middot; <a href='/now'>see them all side by side</a>)"
           "</span></h2>"]
    if not live:
        return out + ["<p class=dim>nothing running</p>"]
    out.append("<table><tr><th>run</th><th>model</th><th>window</th><th>episode</th><th>presses</th>"
               "<th>last message</th><th>latest</th></tr>")
    for name, meta, ep, events, s in live:
        e = last_msg(events)
        out.append(
            f"<tr><td>{esc(name)}</td><td>{esc(meta.get('model', '?'))}</td><td>{esc(meta.get('window', '-'))}</td>"
            f"<td><a href='/run/{quote(name)}/{ep}'>ep{ep:03d}</a></td><td>{s['presses']}</td>"
            f"<td>{esc(ago(s['last']))}</td><td>{esc(msg_head(e))}: {esc((e or {}).get('text', '')[:140])}</td></tr>")
    return out + ["</table>"]


def render_now(root, tail=6):
    live = running(root)
    out = ["<div><a href='/'>&larr; runs</a> &middot; <b>running now</b> &middot; "
           f"{len(live)} episodes &middot; last {tail} messages each</div><div class=grid>"]
    for name, meta, ep, events, s in live:
        board = next((e for e in events if e.get("type") == "board"), None)
        b = {} if BLIND else (board or {}).get("board") or {}
        bline = " &middot; ".join(f"{esc(k)}={esc(b[k])}" for k in sorted(b, key=knob_key))
        t0 = s["first"]
        msgs = [render_msg(e, t0) for e in events if e.get("type") == "msg"][-tail:]
        out.append(f"<div class=card><div class=hd><a href='/run/{quote(name)}/{ep}'>{esc(name)} ep{ep:03d}</a>"
                   f" &middot; {s['presses']} presses &middot; {esc(ago(s['last']))}</div>"
                   f"<div class=dim>{bline}</div>{''.join(msgs)}</div>")
    if not live:
        out.append("<p class=dim>nothing running</p>")
    return page("running now", "".join(out) + "</div>")


def render_index(root):
    out = ["<h1>knob-ladder live runs</h1>"] + render_running(root)
    runs = list_runs(root)
    if not runs:
        out.append(f"<p class=dim>no runs under {esc(root)}</p>")
    for name, meta in runs:
        d = Path(root) / name
        eps = list_eps(d)
        mline = " &middot; ".join(
            f"{k}={esc(meta[k])}" for k in ("model", "layer", "alpha", "dose", "window", "episodes", "batch") if k in meta)
        started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(meta["started"])) if isinstance(
            meta.get("started"), (int, float)) else "?"
        out.append(f"<h2>{esc(name)}</h2><div class=dim>started {esc(started)} &middot; {mline or 'no meta.json'}</div>")
        out.append("<table><tr><th>episode</th><th>status</th><th>presses</th><th>events</th><th>last message</th></tr>")
        for ep in eps:
            s = summarize(read_events(ep_path(d, ep)))
            st = "<span class=ok>done</span>" if s["done"] else "<span class=run>running</span>"
            out.append(
                f"<tr><td><a href='/run/{quote(name)}/{ep}'>ep{ep:03d}</a></td><td>{st}</td>"
                f"<td>{s['presses']}</td><td>{s['n']}</td><td>{esc(ago(s['last']))}</td></tr>")
        out.append("</table>")
        if not eps:
            out.append("<p class=dim>no episodes yet</p>")
    return page("knob-ladder runs", "".join(out))


def render_msg(e, t0):
    role = str(e.get("role", "?"))
    cls = "steered" if e.get("steered") and not BLIND else (role if role in ("system", "user", "assistant", "tool") else "tool")
    hd = esc(role)
    lv = f" @ level {esc(e.get('level'))}" if e.get("level") is not None else ""
    if e.get("knob") is not None:
        what = "after " if role != "tool" else ""
        hd += (f" &middot; {'STEERED ' if e.get('steered') and not BLIND else ''}{what}knob {esc(e.get('knob'))}"
               f" = <b>{esc(label_of(e))}</b>{lv}")
    if t0 is not None and isinstance(e.get("t"), (int, float)):
        hd += f" <span class=dim>+{e['t'] - t0:.1f}s</span>"
    text = str(e.get("text", ""))
    body = f"<pre>{esc(text)}</pre>"
    if text.startswith("<think>"):  # thinking mode: the reasoning in its own box, then the reply
        think, done, reply = text[len("<think>"):].partition("</think>")
        n = len(think.split())
        body = (f"<div class=think><div class=dim>thinking &middot; {n} words{'' if done else ' (cut off)'}</div>"
                f"<pre>{esc(think.strip())}</pre></div><pre>{esc(reply.strip())}</pre>")
    return f"<div class='msg {cls}'><div class=hd>{hd}</div>{body}</div>"


def render_discovery(e):
    out = ["<h2>Discovery</h2><table><tr><th>knob</th><th>label</th><th>presses</th>"
           "<th>cached recall</th><th>observer recall</th><th>cached readout</th><th>observer readout</th></tr>"]
    for r in e.get("rows") or []:
        rec = r.get("recall") or {}
        ro = r.get("readout") if isinstance(r.get("readout"), dict) else {}
        out.append(
            f"<tr><td>{esc(r.get('knob'))}</td><td>{esc(label_of(r))}</td><td>{esc(r.get('presses'))}</td>"
            f"<td><pre>{esc(rec.get('cached', ''))}</pre></td><td><pre>{esc(rec.get('observer', ''))}</pre></td>"
            f"<td>{esc(readout_str(ro.get('cached')))}</td><td>{esc(readout_str(ro.get('observer')))}</td></tr>")
    out.append("</table>")
    return "".join(out)


def render_episode(root, run, ep):
    d = Path(root) / run
    p = ep_path(d, ep)
    if not p.exists():
        return None
    events = read_events(p)
    s = summarize(events)
    board = next((e for e in events if e.get("type") == "board"), None)
    t0 = s["first"]
    st = "<span class=ok>done</span>" if s["done"] else "<span class=run>running</span>"
    nav = "".join(f" &middot; <a href='/run/{quote(run)}/{n}'>{'prev' if n < ep else 'next'}</a>"
                  for n in (ep - 1, ep + 1) if n >= 0 and ep_path(d, n).exists())
    out = [f"<div><a href='/'>&larr; runs</a> &middot; <b>{esc(run)}</b> ep{ep:03d} &middot; {st} &middot; "
           f"{s['presses']} presses &middot; last message {esc(ago(s['last']))}{nav}</div>"]
    if board and not BLIND:
        b = board.get("board") or {}
        keys = sorted(b, key=knob_key)
        out.append(f"<h2>Board <span class=dim>(passage_index {esc(board.get('passage_index'))})</span></h2>"
                   "<table class=board><tr>" + "".join(f"<th>knob {esc(k)}</th>" for k in keys)
                   + "</tr><tr>" + "".join(f"<td>{esc(b[k])}</td>" for k in keys) + "</tr></table>")
    for e in events:
        typ = e.get("type")
        if typ == "msg":
            out.append(render_msg(e, t0))
        elif typ == "discovery":
            out.append(render_discovery(e))
        elif typ == "done":
            out.append(f"<p class=ok>done in {esc(e.get('seconds'))}s</p>")
    if not s["done"]:
        out.append("<p class=dim>... waiting for more (refreshing every 3s)</p>")
    return page(f"{run} ep{ep:03d}", "".join(out), follow=True)


# Appended to pages fetched with ?k=TOKEN: carry the token onto every same-site link.
KEEP_TOKEN = """<script>(function(){var k=new URLSearchParams(location.search).get('k');if(!k)return;
document.querySelectorAll("a[href^='/']").forEach(function(a){var u=new URL(a.href);u.searchParams.set('k',k);a.href=u;});})();</script>"""


def make_handler(root, token=None):
    """With a token (for a public tunnel), a request needs ?k=TOKEN or the cookie that ?k=TOKEN sets."""
    class H(BaseHTTPRequestHandler):
        set_cookie = False

        def _send(self, code, body):
            if self.set_cookie:  # the token also rides on every link, for browsers that drop the cookie (Safari)
                body += KEEP_TOKEN
            data = body.encode("utf-8")
            self.send_response(code)
            if self.set_cookie:
                self.send_header("Set-Cookie", f"k={token}; Path=/; Max-Age=2592000; HttpOnly; Secure; SameSite=Lax")
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self):
            if token is None:
                return True, False
            query = self.path.partition("?")[2]
            given = dict(kv.partition("=")[::2] for kv in query.split("&") if kv).get("k", "")
            cookies = dict(c.strip().partition("=")[::2] for c in (self.headers.get("Cookie") or "").split(";") if c)
            if hmac.compare_digest(given.encode(), token.encode()):
                return True, True
            return hmac.compare_digest(cookies.get("k", "").encode(), token.encode()), False

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            ok, self.set_cookie = self._authorized()
            if not ok:
                return self._send(403, page("forbidden", "<p>forbidden</p>"))
            try:
                if path in ("/", ""):
                    return self._send(200, render_index(root))
                if path == "/now":
                    return self._send(200, render_now(root))
                m = re.fullmatch(r"/run/([^/]+)/(\d+)", path)
                if m:
                    run = unquote(m.group(1))
                    if SAFE.match(run) and run not in (".", ".."):
                        body = render_episode(root, run, int(m.group(2)))
                        if body is not None:
                            return self._send(200, body)
                self._send(404, page("not found", "<p>not found. <a href='/'>runs</a></p>"))
            except Exception as ex:  # crude viewer: show the error rather than die
                self._send(500, page("error", f"<pre>{esc(repr(ex))}</pre>"))

        def log_message(self, *a):
            pass

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/live", help="live runs directory (default runs/live)")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--blind", action="store_true", help="hide knob labels, steered tags and discovery ranks")
    ap.add_argument("--token", help="require ?k=TOKEN on the first visit (for scripts/remote.sh share)")
    ap.add_argument("--once", nargs=2, metavar=("RUN_ID", "EP"), help="print one episode as text and exit")
    ap.add_argument("--follow", nargs=2, metavar=("RUN_ID", "EP"), help="tail one episode as text (exits on done)")
    a = ap.parse_args()
    global BLIND
    BLIND = a.blind
    for pair in (a.once, a.follow):
        if pair and not (SAFE.match(pair[0]) and pair[1].isdigit()):
            ap.error("RUN_ID must be a plain name and EP an integer")
    if a.once:
        return cmd_once(a.runs, a.once[0], int(a.once[1]))
    if a.follow:
        return cmd_follow(a.runs, a.follow[0], int(a.follow[1]))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(a.runs, a.token))
    print(f"serving {a.runs} on http://127.0.0.1:{a.port}/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
