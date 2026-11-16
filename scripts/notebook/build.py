"""Builds catchup.html from content.py + the runs catalog (appendix)."""
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from content import HEAD, PROGRAM, DOCTRINE, FINDINGS, SECTIONS, NOW, LITERATURE, GLOSSARY  # noqa: E402

CAT = json.load(open("/Volumes/1TB_SSD/pebble/machine-consciousness/knob-ladder/scripts/runs_catalog.json"))
E = html.escape
OUT_LABEL = {"positive": "Passed", "null": "Null", "mixed": "Mixed", "not_replicated": "Did not replicate",
             "control_failed": "Control failed", "not_run": "Not run", "setup": "Setup", "running": "Running"}


def md(s):
    """Tiny inline markup: **bold**, `code`, paragraphs on blank lines, '- ' bullets."""
    def inline(t):
        t = E(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
        return t
    out = []
    for block in s.strip().split("\n\n"):
        para, items = [], []
        for l in block.split("\n") + [None]:
            if l is not None and l.startswith("- "):
                if para:
                    out.append(f"<p>{inline(' '.join(para))}</p>")
                    para = []
                items.append(l[2:])
                continue
            if items:
                out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>")
                items = []
            if l is not None:
                para.append(l)
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out)


def chip(outcome):
    return f'<span class="chip {outcome}">{OUT_LABEL.get(outcome, outcome)}</span>'


def iteration(it):
    prompts = "".join(f'<figure class="prompt"><figcaption>{E(lbl)}</figcaption><pre>{E(txt)}</pre></figure>'
                      for lbl, txt in it.get("prompts", []))
    quotes = "".join(
        f'<figure class="quote"><figcaption>{E(q["cond"])}</figcaption><blockquote>{E(q["text"])}</blockquote>'
        + (f'<div class="note">{md(q["note"])}</div>' if q.get("note") else "") + "</figure>"
        for q in it.get("quotes", []))
    nums = ""
    if it.get("numbers"):
        rows = "".join(f"<tr><th>{E(a)}</th><td>{E(b)}</td><td>{chip(c) if c else ''}</td></tr>"
                       for a, b, c in it["numbers"])
        nums = f'<div class="tablewrap"><table class="nums"><tbody>{rows}</tbody></table></div>'
    return f"""
<article class="iter" id="{it['id']}">
  <header class="iter-head">
    <div class="iter-id">{E(it['id'])}</div>
    <h3>{E(it['name'])}</h3>
    {chip(it['outcome'])}
    <div class="meta">{E(it.get('meta', ''))}</div>
  </header>
  <div class="cols">
    <section><h4>Why this construction</h4>{md(it['why'])}</section>
    <section><h4>What the model was given</h4>{md(it['built'])}{prompts}</section>
  </div>
  <section class="result"><h4>Result</h4>{md(it['result'])}{nums}</section>
  {f'<section class="quotes"><h4>In its own words</h4>{quotes}</section>' if quotes else ''}
</article>"""


def section(s):
    its = "".join(iteration(it) for it in s["iterations"])
    return f"""
<section class="exp" id="{s['id']}">
  <div class="exp-head">
    <div class="exp-tag">{E(s['tag'])}</div>
    <h2>{E(s['title'])}</h2>
    <p class="exp-q">{E(s['question'])}</p>
    {md(s.get('intro', '')) if s.get('intro') else ''}
  </div>
  {its}
</section>"""


def appendix():
    series = {s["id"]: s for s in CAT.get("series", [])} if isinstance(CAT.get("series"), list) else CAT.get("series", {})
    rows = []
    for e in CAT["experiments"]:
        if e["series"] == "introspect":
            continue
        for it in e["iterations"]:
            rows.append(f"<tr><td class='mono'>{E(it['id'])}</td><td>{E(e['title'])}</td><td>{chip(it['outcome'])}</td>"
                        f"<td>{E(it['headline'])}</td></tr>")
    return "".join(rows)


def toc():
    items = []
    for s in SECTIONS:
        sub = "".join(f'<li><a href="#{it["id"]}"><span class="mono">{E(it["id"])}</span> {chip(it["outcome"])}</a></li>'
                      for it in s["iterations"])
        items.append(f'<li><a href="#{s["id"]}">{E(s["tag"])} · {E(s["short"])}</a><ul>{sub}</ul></li>')
    return "".join(items)


CSS = (HERE / "style.css").read_text()
page = f"""<title>Introspection Program Notebook</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..125,400..800&family=Literata:ital,opsz,wght@0,7..72,400..700;1,7..72,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header class="top">
  <div class="kicker">{E(HEAD['kicker'])}</div>
  <h1>{E(HEAD['title'])}</h1>
  <p class="lede">{E(HEAD['lede'])}</p>
  <dl class="facts">{''.join(f'<div><dt>{E(a)}</dt><dd>{E(b)}</dd></div>' for a, b in HEAD['facts'])}</dl>
</header>
<div class="layout">
<nav class="toc" aria-label="Contents"><div class="toc-title">Contents</div><ol>
  <li><a href="#short">The short version</a></li><li><a href="#terms">Terms</a></li><li><a href="#how">How we ask</a></li>{toc()}
  <li><a href="#prior">Has anyone done this?</a></li><li><a href="#now">Running now</a></li><li><a href="#earlier">Earlier work</a></li></ol></nav>
<main>
<section id="short" class="block"><h2>The short version</h2>
  <ol class="findings">{''.join(f'<li><strong>{E(a)}</strong> {md(b)}</li>' for a, b in FINDINGS)}</ol>
</section>
<section id="terms" class="block"><h2>Terms used below</h2>
  <dl class="gloss">{''.join(f'<div><dt>{E(a)}</dt><dd>{E(b)}</dd></div>' for a, b in GLOSSARY)}</dl>
</section>
<section id="program" class="block"><h2>Where this came from</h2>{md(PROGRAM)}</section>
<section id="how" class="block"><h2>How we ask</h2>{md(DOCTRINE['intro'])}
  <ol class="doctrine">{''.join(f'<li><strong>{E(a)}</strong> {E(b)}</li>' for a, b in DOCTRINE['rules'])}</ol>
  {md(DOCTRINE['outro'])}
</section>
{''.join(section(s) for s in SECTIONS)}
<section id="prior" class="block"><h2>Has anyone done this?</h2>
  <p>A literature check (10 Nov 2026) on the neuron results and the preference idea. The two key references were opened and checked by hand.</p>
  <dl class="gloss">{''.join(f'<div><dt>{E(a)}</dt><dd>{E(b)}</dd></div>' for a, b in LITERATURE)}</dl>
</section>
<section id="now" class="block"><h2>Running now</h2>{md(NOW)}</section>
<section id="earlier" class="block"><h2>Earlier work</h2>
  <p>Before this program: the multi-agent experiments (labels, rank, self, spread, detection) and the original knob ladder (R0/R1). One line each, from the runs catalog. Full runs are on <a href="https://knob-ladder-runs.pages.dev" target="_blank" rel="noopener">knob-ladder-runs.pages.dev</a>. To read the early knob conversations turn by turn (system prompt, every user and tool turn, every reply, with the hidden knob identities marked), open <a href="https://knob-ladder-runs.pages.dev/#i=R0" target="_blank" rel="noopener">iteration R0</a> (or R1.c, R1.d, R0.d, R0.o) and then its full run. The iteration page lists every prompt, including the concept-vector templates.</p>
  <div class="tablewrap"><table class="appendix"><thead><tr><th>Run</th><th>Experiment</th><th>Outcome</th><th>Headline</th></tr></thead><tbody>{appendix()}</tbody></table></div>
</section>
</main></div></div>
"""
(HERE / "catchup.html").write_text(page)
# Standalone copy for the public runs site (served at /notebook): its own document head, and a link back to the runs.
standalone = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
              '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
              '<style>*{box-sizing:border-box}body{margin:0}</style>'
              + page.replace('<div class="kicker">', '<div class="kicker"><a href="/">← All runs</a> · ', 1)
              + "</html>")
SITE = Path("/Volumes/1TB_SSD/pebble/machine-consciousness/knob-ladder/scripts/site_pages")
SITE.mkdir(exist_ok=True)
(SITE / "notebook.html").write_text(standalone)
print("wrote", HERE / "catchup.html", len(page))
