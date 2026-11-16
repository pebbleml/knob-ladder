#!/usr/bin/env python3
"""Build the story page (served at / on the runs site): every experiment in order, with one real conversation each.

    python3 scripts/story/build_story.py            # writes scripts/site_pages/story.html

Content lives in content.py next to this file. Example conversations come from scripts/conversations.py, applied to
each chapter's primary run (the run the catalog marks as the one to read)."""
import html
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import runs_browser as b  # noqa: E402
from content import CHAPTERS, GLOSSARY, LEDE, LEGEND, LOOKALIKES, SWARM, TITLE, TRACKS, VERDICTS  # noqa: E402

OUT_LABEL = {"positive": "pre-registered test passed", "null": "no effect found", "mixed": "mixed", "not_replicated": "did not replicate",
             "control_failed": "control failed", "not_run": "not run", "setup": "setup", "running": "running now", "lost": "lost"}
WHO = {"system": "System prompt", "user": "User turn (us)", "assistant": "Model reply", "thinking": "Model thinking",
       "tool": "Tool response", "hidden": "Experimenter side · not shown to the model", "judge": "Judge (separate call)"}
e = html.escape


def md(s):
    s = e(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"'([^']{2,80})'", r"‘\1’", s)
    return s


def turn_html(m, long=1600):
    role = m.get("role", "user")
    text = m.get("text") or ""
    flags = []
    if m.get("steered"):
        flags.append('<span class="flag steer">steering on</span>')
    if m.get("scored"):
        flags.append('<span class="flag score">scored from first word</span>')
    tag = f'<span class="tag">{e(m["tag"])}</span>' if m.get("tag") else ""
    if role == "thinking" or len(text) > long:
        head = text[:500] if role != "thinking" else ""
        body = (f'<div class="txt">{e(head)}</div>' if head else "") + \
               f'<details><summary>{"Show its thinking" if role == "thinking" else "Show the rest"} ({len(text):,} characters)</summary>' \
               f'<div class="txt">{e(text[len(head):])}</div></details>'
    else:
        body = f'<div class="txt">{e(text)}</div>'
    note = f'<aside class="note">{md(m["note"])}</aside>' if m.get("note") else '<aside class="note empty"></aside>'
    return (f'<div class="turn r-{role}"><div class="bubble"><div class="who">{WHO.get(role, role)}{tag}{"".join(flags)}</div>'
            f'{body}</div>{note}</div>')


def conv_html(c, rid):
    badge = '<span class="badge exact">Exact: what the model saw</span>' if c.get("exact") else \
        '<span class="badge rebuilt">Rebuilt from stored prompts and replies</span>'
    meta = c.get("meta") or {}
    meta_html = "".join(f"<div><b>{e(str(k))}</b> {e(str(v))[:300]}</div>" for k, v in meta.items())
    raw = (f'<details class="raw"><summary>Show the raw text the model received (chat template)</summary>'
           f'<pre>{e(c["raw"])}</pre></details>') if c.get("raw") else ""
    return (f'<div class="conv"><div class="convhead"><div class="convtitle">{e(c["title"])}</div>{badge}'
            f'<p class="muted small">{md(c.get("note") or "")}</p>'
            + (f'<details class="meta"><summary>Hidden setup for this conversation</summary>{meta_html}</details>' if meta_html else "")
            + f'</div><div class="turns">{"".join(turn_html(m) for m in c["messages"])}</div>{raw}'
            f'<a class="more" href="/runs#run={e(rid)}">Read every conversation from this run →</a></div>')


def build():
    store = b.Store(str(b.ROOT / "runs"))
    its = b.ITERATIONS
    when, last = {}, {}
    for r in store.index.values():
        if r["iteration_id"] and r["timestamp"] and r["role"] in ("full", "early"):
            when[r["iteration_id"]] = min(when.get(r["iteration_id"], "9"), r["timestamp"])
            last[r["iteration_id"]] = max(last.get(r["iteration_id"], ""), r["timestamp"])
    toc, body, n = [], [], 0
    legend = "".join(f'<span class="lg r-{k}"><i></i>{e(name)}</span>' for k, name, _ in LEGEND)
    legend_full = "".join(f'<div class="lgrow r-{k}"><i></i><b>{e(name)}</b> {e(d)}</div>' for k, name, d in LEGEND)
    for t in TRACKS:
        toc.append(f'<li class="tt"><a href="#{t["id"]}">{e(t["title"])}</a></li>')
        body.append(f'<section class="track" id="{t["id"]}"><div class="kicker">{e(t["dates"])}</div><h2>{e(t["title"])}</h2>'
                    f'<p class="lede">{md(t["summary"])}</p></section>')
        if t["chapters"] == "swarm":
            rows = "".join(f"<tr><td class='mono'>{e(a)}</td><td>{e(q)}</td><td>{e(r)}</td></tr>" for a, q, r in SWARM)
            ex = ""
            try:
                rid = b.run_id(t["example_run"])
                cv = store.load(rid).get("convs") or []
                pick = next((x for x in cv if t["example"].lower() in x["title"].lower()), None)
                if pick:
                    ex = (f'<h4 class="exhead">One real prompt from this kind of experiment</h4><p class="exnote">{md(t["example_note"])}</p>'
                          + conv_html(pick, rid))
            except Exception as exc:
                print("swarm example failed", exc)
            body.append(f'<div class="chapter"><div class="scroll"><table><thead><tr><th>Experiments</th><th>Question</th>'
                        f'<th>What we found</th></tr></thead><tbody>{rows}</tbody></table></div>'
                        f'<p><a href="/runs">Every multi-agent experiment and its numbers →</a> (their conversations are not yet shown turn by turn)</p>{ex}</div>')
            continue
        chs = CHAPTERS[t["chapters"]]
        for k, ch in enumerate(chs):
            n += 1
            cov = [c for c in ch["covers"] if c in its]
            d0 = min((when[c] for c in cov if c in when), default="")[:10]
            d1 = max((last[c] for c in cov if c in last), default="")[:10]
            date = d0 if d0 == d1 else f"{d0} to {d1}"
            chips = "".join(f'<a class="chip o-{its[c]["outcome"]}" href="/runs#i={e(c)}">{e(c)} · {OUT_LABEL.get(its[c]["outcome"], its[c]["outcome"])}</a>'
                            for c in cov)
            example, rid = "", None
            for c in ([None] if ch.get("example_run") else []) + cov:
                rid = b.run_id(ch["example_run"]) if c is None else b.iteration_summary(store, its[c]).get("primary_run_id")
                if not rid:
                    continue
                try:
                    convs = store.load(rid).get("convs") or []
                except Exception as exc:  # a run file that fails to load just has no example
                    print("skip", rid, exc)
                    convs = []
                if convs:
                    pick = next((x for x in convs if ch.get("example") and ch["example"].lower() in x["title"].lower()), convs[0])
                    example = ((f'<p class="exnote">{md(ch["example_note"])}</p>' if ch.get("example_note") else "")
                               + conv_html(pick, rid) + f'<p class="muted small">This run has {len(convs)} conversations shown on the site.</p>')
                    break
            if not example:
                example = ('<p class="muted">No stored conversation for this experiment: its files keep numbers only, or the '
                           'texts are not yet converted. <a href="/runs#i=' + e(cov[0] if cov else "") + '">See its files →</a></p>')
            nxt = chs[k + 1]["id"] if k + 1 < len(chs) else None
            toc.append(f'<li><a href="#{ch["id"]}">{e(ch["title"])}</a></li>')
            body.append(
                f'<article class="chapter" id="{ch["id"]}"><div class="kicker">Chapter {n} · {e(date)}</div><h3>{md(ch["title"])}</h3>'
                f'<p class="verdict"><b>In short:</b> {md(VERDICTS.get(ch["id"], ""))}</p><div class="chips">{chips}</div>'
                f'<div class="grid4"><div><h4>Where we were</h4><p>{md(ch["where"])}</p></div>'
                f'<div><h4>What we did</h4><p>{md(ch["did"])}</p></div></div>'
                f'<h4 class="exhead">One real conversation from this experiment</h4><div class="legend">{legend}</div>{example}'
                f'<div class="grid4"><div><h4>What happened</h4><p>{md(ch["result"])}</p></div>'
                f'<div class="next"><h4>So next</h4><p>{md(ch["next"])}</p>'
                + (f'<a href="#{nxt}">Next chapter ↓</a>' if nxt else "") + '</div></div></article>')
    look = "".join(f'<tr><td><b>{e(a)}</b><div class="muted small">{e(w)}</div></td><td>{e(d)}</td><td>{e(r)}</td></tr>'
                   for a, w, d, r in LOOKALIKES["rows"])
    gloss = "".join(f"<dt>{e(a)}</dt><dd>{e(d)}</dd>" for a, d in GLOSSARY)
    page = PAGE.format(title=e(TITLE), lede=md(LEDE), toc="".join(toc), body="".join(body), legend=legend_full,
                       look_title=e(LOOKALIKES["title"]), look_intro=e(LOOKALIKES["intro"]), look=look, gloss=gloss)
    out = HERE.parent / "site_pages" / "story.html"
    out.write_text(page, encoding="utf-8")
    print("wrote", out, len(page))


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Self-Knowledge Experiments</title>
<script>if(/^#(run|i|search)=|^#other$/.test(location.hash))location.replace('/runs'+location.hash)</script>
<style>
:root{{--bg:#F6F4EF;--panel:#FFFFFF;--ink:#1D232A;--muted:#5E6873;--rule:#DAD5CB;--accent:#1F5C8A;
--c-system:#6B4FA0;--c-user:#1F5C8A;--c-assistant:#2F7A4D;--c-thinking:#8A8F96;--c-tool:#A0661E;--c-hidden:#B23A48;--c-judge:#5B6B2E;
--t-system:#F1ECFA;--t-user:#EAF2FA;--t-assistant:#EAF6EE;--t-thinking:#F2F3F4;--t-tool:#FBF2E6;--t-hidden:#FCEDEF;--t-judge:#F2F5E8;
--serif:"Iowan Old Style","Palatino Linotype",Georgia,serif;--sans:-apple-system,"Segoe UI",Roboto,sans-serif;--mono:ui-monospace,Menlo,monospace}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14181C;--panel:#1C2126;--ink:#E7EAEE;--muted:#9AA4AE;--rule:#2F363D;--accent:#7FB6E2;
--c-system:#B9A2E8;--c-user:#7FB6E2;--c-assistant:#7FCB98;--c-thinking:#9AA0A8;--c-tool:#E0A866;--c-hidden:#F08A96;--c-judge:#B7C77F;
--t-system:#231F2E;--t-user:#18232E;--t-assistant:#18261E;--t-thinking:#1E2226;--t-tool:#2A2218;--t-hidden:#2D1B1F;--t-judge:#22261A;color-scheme:dark}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 var(--sans)}}a{{color:var(--accent)}}
.top{{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--rule);padding:10px max(16px,env(safe-area-inset-left));display:flex;gap:18px;flex-wrap:wrap;align-items:center}}
.top b{{font-family:var(--serif);font-size:18px}}
.wrap{{max-width:1180px;margin:0 auto;padding:24px max(16px,env(safe-area-inset-left));display:grid;grid-template-columns:250px minmax(0,1fr);gap:36px}}
nav.toc{{position:sticky;top:64px;align-self:start;max-height:calc(100vh - 80px);overflow:auto;font-size:13.5px}}
nav.toc ol{{list-style:none;margin:0;padding:0;display:grid;gap:4px}}nav.toc .tt{{margin-top:14px;font-weight:700}}nav.toc a{{color:var(--ink);text-decoration:none}}nav.toc a:hover{{color:var(--accent)}}
main{{min-width:0;display:grid;gap:28px}}h1{{font-family:var(--serif);font-size:40px;line-height:1.1;margin:0 0 10px;text-wrap:balance}}
h2{{font-family:var(--serif);font-size:30px;margin:0;text-wrap:balance}}h3{{font-family:var(--serif);font-size:24px;margin:2px 0 8px;text-wrap:balance}}
h4{{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 4px}}p{{margin:0;max-width:72ch}}
.lede{{font-size:17.5px;max-width:70ch}}.kicker{{font:600 12px var(--mono);color:var(--muted);letter-spacing:.04em}}.muted{{color:var(--muted)}}.small{{font-size:13.5px}}.mono{{font-family:var(--mono);font-size:13px}}
.track{{border-top:3px solid var(--ink);padding-top:16px;display:grid;gap:8px;margin-top:20px}}
.chapter{{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:22px;display:grid;gap:16px}}
.grid4{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}.grid4>div{{min-width:0}}
.next{{border-left:3px solid var(--accent);padding-left:14px}}.verdict{{font-size:17px;background:var(--bg);border-radius:6px;padding:10px 14px;max-width:none}}.exnote{{font-size:14px;max-width:none;border-left:3px solid var(--c-hidden);padding-left:10px}}
.chips{{display:flex;flex-wrap:wrap;gap:6px}}.chip{{font:12px var(--mono);border:1px solid var(--rule);border-radius:99px;padding:2px 9px;text-decoration:none;color:var(--ink)}}
.chip.o-positive{{border-color:#2F7A4D}}.chip.o-null,.chip.o-control_failed,.chip.o-not_replicated{{border-color:#B23A48}}.chip.o-mixed{{border-color:#A0661E}}
.legend{{display:flex;flex-wrap:wrap;gap:10px;font-size:12.5px;color:var(--muted)}}.lg i,.lgrow i{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}}
{role_css}
.conv{{border:1px solid var(--rule);border-radius:8px;padding:14px;display:grid;gap:10px;background:var(--bg)}}.convtitle{{font-weight:650;font-size:15.5px}}
.badge{{display:inline-block;font:600 11.5px var(--mono);border-radius:4px;padding:1px 7px;margin:4px 0}}.badge.exact{{background:#2F7A4D;color:#fff}}.badge.rebuilt{{background:var(--rule);color:var(--ink)}}
.turns{{display:grid;gap:8px}}.turn{{display:grid;grid-template-columns:minmax(0,1fr) 250px;gap:14px;align-items:start}}
.bubble{{border-left:4px solid var(--rule);border-radius:0 6px 6px 0;padding:8px 12px;min-width:0}}
.who{{font:600 12px var(--mono);display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:3px}}
.txt{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14.5px}}.r-thinking .txt{{font-size:13.5px;color:var(--muted)}}
.tag{{font-weight:500;border:1px solid currentColor;border-radius:3px;padding:0 5px}}.flag{{font-weight:500;border-radius:3px;padding:0 5px;color:#fff}}.flag.steer{{background:#B23A48}}.flag.score{{background:#5E6873}}
.note{{font-size:13px;color:var(--muted);border-left:2px solid var(--rule);padding-left:10px;font-style:italic}}.note.empty{{border:0}}
details summary{{cursor:pointer;color:var(--accent);font-size:13.5px}}details.meta div{{font-size:13px}}.raw pre{{white-space:pre-wrap;font:12px/1.5 var(--mono);max-height:420px;overflow:auto;background:var(--panel);padding:10px;border-radius:6px}}
.more{{font-weight:600;font-size:14px}}.box{{background:var(--panel);border:2px solid var(--accent);border-radius:10px;padding:20px;display:grid;gap:12px}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th{{text-align:left;font:600 12px var(--mono);color:var(--muted);border-bottom:1px solid var(--ink);padding:6px 10px 6px 0}}td{{border-bottom:1px solid var(--rule);padding:8px 10px 8px 0;vertical-align:top}}
dl{{display:grid;grid-template-columns:200px 1fr;gap:6px 16px;margin:0}}dt{{font-weight:650}}dd{{margin:0}}.lgrow{{font-size:14px}}
@media (max-width:860px){{.wrap{{grid-template-columns:1fr}}nav.toc{{position:static;max-height:none}}.grid4{{grid-template-columns:1fr}}.turn{{grid-template-columns:1fr}}.note{{margin-left:16px}}dl{{grid-template-columns:1fr}}h1{{font-size:30px}}}}
</style></head><body>
<div class="top"><b>Self-knowledge experiments</b><a href="#start">The story</a><a href="#lookalikes">Things that sound alike</a><a href="#glossary">Glossary</a><a href="/runs">All runs and conversations</a><a href="/notebook">Detailed notebook</a></div>
<div class="wrap"><nav class="toc"><ol>{toc}</ol></nav><main>
<div id="start"><h1>{title}</h1><p class="lede">{lede}</p></div>
<div class="box"><h4>How to read a conversation</h4><p>Each turn is coloured by who produced it. Notes in the right margin explain what the turn is for. <b>Red “steering on”</b> marks text the model read or wrote while we were changing its activations. <b>Grey “scored from first word”</b> marks yes/no questions where we read its probabilities instead of letting it write.</p>{legend}</div>
<div class="box" id="lookalikes"><h4>{look_title}</h4><p>{look_intro}</p><div class="scroll"><table><thead><tr><th>Experiment</th><th>What it is</th><th>Result</th></tr></thead><tbody>{look}</tbody></table></div></div>
{body}
<section class="track" id="glossary"><h2>Glossary</h2><dl>{gloss}</dl></section>
</main></div></body></html>"""

ROLES = ["system", "user", "assistant", "thinking", "tool", "hidden", "judge"]
PAGE = PAGE.replace("{role_css}", "".join(
    f".r-{r} .bubble{{{{border-left-color:var(--c-{r});background:var(--t-{r})}}}}.r-{r} .who{{{{color:var(--c-{r})}}}}"
    f".lg.r-{r} i,.lgrow.r-{r} i{{{{background:var(--c-{r})}}}}" for r in ROLES))

if __name__ == "__main__":
    build()
