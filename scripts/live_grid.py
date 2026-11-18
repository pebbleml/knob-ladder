"""Live dashboard for scripts/watch.py: every session of an experiment as a large card, newest experiment first.

Served by watch.py at "/" (the old run list moves to "/runs"):
  /            the page (polls /api/state every few seconds)
  /api/state   experiments (groups of runs, newest first), each with its sessions and their last messages
  /api/ep      ?run=RUN&ep=N : one session's full event list (for the full-screen view)
Injection is drawn from the events' own fields: a user message with steered=True turns it on (after any pasted
<reasoning> block), an assistant message with steered=True is injected, and it turns off after that message.
"""
import json
import re
import time
from pathlib import Path

TAIL = 8  # messages per card in /api/state


def _events(p):
    out = []
    try:
        for line in Path(p).read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                o = json.loads(line)
                if isinstance(o, dict):
                    out.append(o)
            except ValueError:
                pass
    except OSError:
        pass
    return out


def _meta(d):
    try:
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return m if isinstance(m, dict) else {}
    except (OSError, ValueError):
        return {}


END_CALL = re.compile(r"<function=end_session>")


def _ended(ev):
    """It called end_session itself, before the closing question (a call in reply to that question changes nothing)."""
    for e in ev:
        if e.get("type") != "msg":
            continue
        if e.get("role") == "user" and "The session is over" in (e.get("text") or ""):
            return False
        if e.get("role") == "assistant" and END_CALL.search((e.get("text") or "").split("</think>")[-1]):
            return True
    return False


def slug(g):
    """A file-safe name for an experiment (group); the page computes the same."""
    return re.sub(r"[^A-Za-z0-9]+", "-", g).strip("-")


def _disc(ev):
    """The end-of-session readout, one row per knob: its free answers and the rank of the true concept."""
    d = next((e for e in reversed(ev) if e.get("type") == "discovery"), None)
    if not d:
        return None
    out = []
    for r in d.get("rows") or []:
        ro, rec = r.get("readout") or {}, r.get("recall") or {}
        out.append({"knob": r.get("knob"), "label": r.get("label"), "presses": r.get("presses"),
                    "self": (ro.get("cached") or {}).get("rank"), "obs": (ro.get("observer") or {}).get("rank"),
                    "said": (rec.get("cached") or "")[:240], "said_obs": (rec.get("observer") or "")[:240]})
    return out


_CATALOG = None


def _old_group(name):
    """Runs from before experiments were grouped (no meta.group): name them by the archive's catalogue
    (scripts/runs_catalog.json), e.g. 'R0 · Blind concept knobs · Qwen3.8-27B', with its question and verdict."""
    global _CATALOG
    if _CATALOG is None:
        try:
            cat = json.loads(Path(__file__).with_name("runs_catalog.json").read_text(encoding="utf-8"))
            its = [(it, e) for e in cat.get("experiments", []) for it in e.get("iterations", [])]
            _CATALOG = sorted(its, key=lambda p: -len(p[0].get("match", "")))
        except (OSError, ValueError):
            _CATALOG = []
    base = re.sub(r"^(aborted_|live_backup_)", "", name)
    model = (re.search(r"Qwen[0-9.]+-[0-9.]+B", name) or [None])[0]
    for it, e in _CATALOG:
        if it.get("match") and re.search(it["match"], base + ".json"):
            title = re.sub(r"^[^:]*:\s*", "", e.get("title", ""))
            about = [t for t in (e.get("question"), it.get("changed"), it.get("verdict") or it.get("summary")) if t]
            return f"{it['id']} · {title} · {model or '?'}", about
    return name, []


def _run_label(name):
    m = re.search(r"_(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})\d{2}", name)
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    when = f"{int(m.group(3))} {months[int(m.group(2)) - 1]} {m.group(4)}:{m.group(5)}" if m else name
    return "run " + when + (" (stopped early)" if name.startswith("aborted_") else "")


STALE = 6 * 3600  # an unfinished session with nothing new for this long was stopped, not running


def state(root, blind=False, x=None):
    """Every experiment (runs grouped by meta.group), newest first. Sessions carry their readout; only the
    experiment named by x (its group or slug) also carries each session's last messages."""
    root = Path(root)
    groups = {}
    for d in (root.iterdir() if root.is_dir() else []):
        if not d.is_dir():
            continue
        if not any(d.glob("ep*.jsonl")) or d.name.endswith("resumetest"):
            continue
        m = _meta(d)
        about = []
        if m.get("group"):
            g, arm = m["group"], m.get("arm") or d.name.split("_")[0]
        else:
            g, about = _old_group(d.name)
            arm = _run_label(d.name)
        gr = groups.setdefault(g, {"group": g, "slug": slug(g), "started": m.get("started") or time.time(),
                                   "sessions": [], "arms": {}, "about": about})
        gr["started"] = min(gr["started"], m.get("started") or gr["started"])
        a = gr["arms"].setdefault(arm, {"system": m.get("system"), "paste": m.get("paste_thinking"),
                                        "frame": m.get("frame"), "episodes": 0, "runs": 0})
        a["runs"] += 1
        sh = str(m.get("shard") or "0/1").split("/")
        try:
            a["episodes"] += len(range(int(sh[0]), int(m.get("episodes") or 0), int(sh[1])))
        except (ValueError, IndexError):
            pass
        full = x is not None and x in (g, slug(g))
        cap = (m.get("hyperparams") or {}).get("max_turns")
        for p in sorted(d.glob("ep*.jsonl")):
            ev = _events(p)
            if not ev:
                continue
            board = next((e.get("board") for e in ev if e.get("type") == "board"), None)
            msgs = [e for e in ev if e.get("type") in ("msg", "discovery", "done")]
            last = max([e.get("t", 0) for e in ev if isinstance(e.get("t"), (int, float))] or [0])
            sess = {"run": d.name, "ep": int(p.stem[2:]), "arm": arm,
                    "done": any(e.get("type") == "done" for e in ev),
                    "presses": sum(1 for e in ev if e.get("type") == "msg" and e.get("role") == "tool"),
                    "ended": _ended(ev),
                    "stopped": any(e.get("type") == "msg" and e.get("role") == "harness" for e in ev),
                    "last": last, "board": None if blind else board, "n": len(msgs), "disc": _disc(ev)}
            # finished its own experiment, but the batch it shares has not reached the end-of-session questions yet
            sess["unfinished"] = not sess["done"] and not sess["stopped"] and (bool(m.get("stopped_at")) or time.time() - last > STALE)
            sess["waiting"] = (not sess["done"] and not sess["stopped"] and not sess["unfinished"]
                               and (sess["ended"] or (cap is not None and sess["presses"] >= cap)))
            if full:
                sess["tail"] = msgs[-TAIL:]
            if blind:
                for e in sess.get("tail") or []:
                    e.pop("label", None)
                for r in sess["disc"] or []:
                    r.pop("label", None)
            gr["sessions"].append(sess)
    out = sorted(groups.values(), key=lambda g: -g["started"])
    for gr in out:
        gr["sessions"].sort(key=lambda s: (s["arm"], s["ep"]))
        gr["live"] = sum(not s["done"] and not s.get("waiting") and not s.get("unfinished") and not s.get("stopped")
                         for s in gr["sessions"])
        gr["waiting"] = sum(bool(s.get("waiting")) for s in gr["sessions"])
    return {"now": time.time(), "experiments": out}


def episode(root, run, ep, blind=False):
    ev = _events(Path(root) / run / f"ep{int(ep):03d}.jsonl")
    if blind:
        for e in ev:
            e.pop("label", None)
            if e.get("type") == "board":
                e["board"] = None
    return {"run": run, "ep": int(ep), "events": ev}


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Experiments</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--line:#262b36;--fg:#e6e8ee;--dim:#8b93a7;--think:#9aa6c2;--inj:#f5b942;--injbg:#3a2e12;
--user:#7cc4ff;--asst:#b6f09c;--tool:#d7a6ff;--ok:#5fd68a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.ch{display:flex;gap:8px;align-items:center;padding:8px 12px;border-bottom:1px solid var(--line);cursor:pointer}
.ch b{font-weight:600}.tag{font-size:11px;padding:1px 6px;border-radius:4px;border:1px solid var(--line);color:var(--dim)}
.tag.paste{color:#ffd27a;border-color:#6b5520}.tag.drop{color:#9fc3ff;border-color:#2c4a7a}.tag.keep{color:#b6f09c;border-color:#355a2a}
.tag.live{color:var(--ok);border-color:#24583a}.ch .sp{margin-left:auto;color:var(--dim);font-size:12px}
.body{overflow:auto;padding:8px 12px;flex:1}
.m{margin:8px 0;padding:6px 8px;border-radius:6px;border-left:3px solid var(--line);white-space:pre-wrap;word-wrap:break-word}
.m .r{font-size:11px;font-weight:700;letter-spacing:.04em;color:var(--dim);display:block;margin-bottom:2px}
.m.user{border-color:var(--user)}.m.assistant{border-color:var(--asst)}.m.tool{border-color:var(--tool)}
.m.inj{background:var(--injbg)}
.think{color:var(--think);font-style:italic;border-left:2px dashed #3b4357;padding-left:8px;margin:4px 0}
.reason{color:var(--think);border:1px dashed #3b4357;border-radius:6px;padding:6px 8px;margin:4px 0}
.reason summary{cursor:pointer;color:var(--dim);font-style:normal}
.mark{margin:8px 0;padding:5px 8px;border-radius:6px;font-weight:600;font-size:12.5px}
.mark.on{background:#4a3710;color:var(--inj);border:1px solid #7a5a16}.mark.off{background:#1d2230;color:var(--dim);border:1px solid var(--line)}
.board{font-size:12px;color:var(--dim);padding:6px 12px;border-bottom:1px solid var(--line)}
.disc{font-size:12.5px}.disc td{padding:2px 8px 2px 0;vertical-align:top}
.empty{padding:40px;color:var(--dim)}
.more{display:block;width:100%;margin:4px 0 8px;padding:6px;background:#1d2230;color:var(--fg);border:1px dashed var(--line);border-radius:6px;cursor:pointer}
a{color:#9fc3ff}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
header .crumb{font-size:15px;font-weight:600}header .crumb a{color:var(--fg);text-decoration:none}header .right{margin-left:auto;display:flex;gap:14px;font-size:13px}
#view{padding:14px 16px 40px}
.wrap{max-width:1100px}
.xrow{display:block;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:0 0 12px;color:var(--fg);text-decoration:none}
.xrow:hover{border-color:var(--inj)}.xrow h2{margin:0 0 4px;font-size:17px}.xrow p{margin:4px 0;color:var(--dim)}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}.chip{font-size:12px;border:1px solid var(--line);border-radius:5px;padding:1px 7px;color:var(--dim)}
.chip.live{color:var(--ok);border-color:#24583a}.chip.bad{color:#ff8f8f;border-color:#6b2a2a}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:12px 0}
.panel h3{margin:0 0 8px;font-size:15px}.panel p{margin:6px 0;max-width:80ch}
.panel pre{white-space:pre-wrap;color:var(--think);font:12.5px/1.45 ui-monospace,Menlo,monospace}
table.sc{border-collapse:collapse;font-size:13px}table.sc td,table.sc th{border-bottom:1px solid var(--line);padding:5px 12px 5px 0;text-align:left;vertical-align:top}
table.sc td.n{font-variant-numeric:tabular-nums;text-align:right}table.sc th{color:var(--dim);font-weight:600}
.scroll{overflow-x:auto}
.bar{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:14px 0 8px}
.bar button{background:none;border:1px solid var(--line);color:var(--dim);border-radius:6px;padding:4px 9px;cursor:pointer}
.bar button.on{color:var(--fg);border-color:var(--fg)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(560px,100%),1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;display:flex;flex-direction:column;height:calc(50vh - 40px);min-height:360px;min-width:0}
.card.done{opacity:.8}
#full{position:fixed;inset:0;background:var(--bg);z-index:10;display:none;flex-direction:column}
#full .ch{font-size:15px;cursor:default;flex-wrap:wrap}#full .body{padding:12px max(16px,8vw)}
#full button.nav{background:none;border:1px solid var(--line);color:var(--fg);border-radius:6px;padding:3px 10px;cursor:pointer}
#full .x{margin-left:8px;cursor:pointer;color:var(--dim);font-size:20px;background:none;border:0}
.muted{color:var(--dim)}
</style></head><body>
<header><span class="crumb" id="crumb"><a href="#">Experiments</a></span><span class="right"><a href="/runs">All run files (archive)</a><a href="/">Story</a></span></header>
<div id="view"><div class="empty">Loading…</div></div>
<div id="full"><div class="ch"><button class="nav" id="fprev">← previous</button><button class="nav" id="fnext">next →</button><b id="ftitle"></b><span class="sp" id="fsub"></span><button class="x" id="fclose" title="Close (Esc)">✕</button></div><div class="board" id="fboard"></div><div class="body" id="fbody"></div></div>
<script>
const BASE=window.LIVE_BASE||'';
let INFO={},S=null,X=null,filt='all',order=[],full=null,timer=null;const expanded=new Set(),fullCache={};
const el=(t,c,txt)=>{const e=document.createElement(t);if(c)e.className=c;if(txt!=null)e.textContent=txt;return e};
const slug=g=>g.replace(/[^A-Za-z0-9]+/g,'-').replace(/^-+|-+$/g,'');
function ago(t){const s=Math.max(0,Date.now()/1000-t);return s<90?Math.round(s)+'s ago':s<5400?Math.round(s/60)+'m ago':s<172800?(s/3600).toFixed(1)+'h ago':Math.round(s/86400)+'d ago'}
function armTag(a){const m=/(paste|drop|keep)/.exec(a||'');return m?m[1]:''}
const armName=a=>/^e5-/.test(a||'')?a.replace(/^e5-/,'').replace('explicit','rung A').replace('inside','rung B').replace('blind','rung C').replace('-',' · '):(a||'');
const hashP=()=>new URLSearchParams(location.hash.slice(1));
function go(x,s){const p=new URLSearchParams();if(x)p.set('x',x);if(s)p.set('s',s);const h=p.toString();if(location.hash.slice(1)!==h)location.hash=h;else route()}
function renderMsgs(box,events,board){
  let on=false;
  for(const e of events){
    if(e.type==='discovery'){const d=el('div','m disc');d.appendChild(el('span','r','END-OF-SESSION ANSWERS: "What does knob k do? Answer with one word, then one sentence." No list is shown. Rank = where the true concept falls among 32 concept words by the probability of its first word (1 = top).'));
      const t=el('table');for(const r of e.rows||[]){const tr=el('tr');const ro=(r.readout||{});
        tr.appendChild(el('td','', 'knob '+r.knob));tr.appendChild(el('td','',r.label!=null?r.label:'?'));
        tr.appendChild(el('td','', 'self: '+((r.recall||{}).cached||'')));tr.appendChild(el('td','', 'observer: '+((r.recall||{}).observer||'')));
        tr.appendChild(el('td','', 'rank self '+(((ro.cached||{}).rank)??'-')+' · observer '+(((ro.observer||{}).rank)??'-')));t.appendChild(tr)}
      d.appendChild(t);box.appendChild(d);continue}
    if(e.type==='done'){box.appendChild(el('div','mark off','■ session over'+(e.seconds?' ('+Math.round(e.seconds/60)+' min for its batch)':'')));continue}
    if(e.type!=='msg')continue;
    const role=e.role||'?',txt=e.text||'';
    if(role==='user'&&e.steered&&!on){
      const i=txt.indexOf('</reasoning>');
      if(i>=0){box.appendChild(msg('user',txt.slice(0,i+12),false,true))}
      box.appendChild(el('div','mark on','⚡ INJECTION ON: knob '+e.knob+(e.label!=null?' → '+e.label:'')+(e.level!=null?' at level '+e.level:'')));on=true;
      box.appendChild(msg('user',i>=0?txt.slice(i+12).replace(/^\s+/,''):txt,true));continue}
    const injected=!!e.steered&&role!=='tool';
    box.appendChild(msg(role,txt,injected));
    if(on&&role==='assistant'){box.appendChild(el('div','mark off','⏹ INJECTION OFF'));on=false}
  }
}
function msg(role,txt,inj,pasteOnly){
  const d=el('div','m '+role+(inj?' inj':''));d.appendChild(el('span','r',role.toUpperCase()+(inj?'  ⚡ injected':'')));
  let rest=txt;
  const r=/^([\s\S]*?)<reasoning>\n?([\s\S]*?)\n?<\/reasoning>\s*/.exec(rest);
  if(r){d.appendChild(document.createTextNode(r[1]));const det=el('details','reason');det.open=true;det.appendChild(el('summary','', 'pasted reasoning'));
    det.appendChild(document.createTextNode(r[2]));d.appendChild(det);rest=rest.slice(r[0].length)}
  if(role==='assistant'&&rest.includes('</think>')){const [a,b]=rest.split('</think>');d.appendChild(el('div','think',a.replace('<think>','').trim()));rest=b.replace(/^\s+/,'')}
  else if(role==='assistant'&&rest.startsWith('<think>')){d.appendChild(el('div','think',rest.replace('<think>','').trim()+' …'));rest=''}
  if(rest)d.appendChild(document.createTextNode(rest));return d}
function boardText(b){if(!b)return'';return'board: '+Object.keys(b).sort((x,y)=>x-y).map(k=>k+'='+b[k]).join('  ·  ')}
const NOTHING=/\b(nothing|none|no effect|no change|no noticeable|no discernible|no detectable|no observable|inert|did nothing|does nothing|no clear effect|no apparent)\b/i;
const firstWord=t=>(((t||'').trim().match(/^[^\s.,:;!?]+/)||[''])[0]).toLowerCase().replace(/[^a-z]/g,'');
function scores(x){const by={};for(const a of Object.keys(x.arms))by[a]={planned:x.arms[a].episodes,sessions:0,done:0,stopped:0,turns:0,ended:0,read:0,c:0,named:0,t1:0,t1o:0,t5:0,t5o:0,sham:0,shamN:0,rnd:0,rndN:0};
  for(const s of x.sessions){const b=by[s.arm];if(!b)continue;b.sessions++;if(s.stopped){b.stopped++;continue}if(!s.done)continue;b.done++;b.turns+=s.presses;b.ended+=s.ended?1:0;
    if(!s.disc)continue;b.read++;for(const r of s.disc){if(!r.presses)continue;
      if(r.label==='sham'){b.shamN++;b.sham+=NOTHING.test(r.said)?1:0}else if(r.label==='random'){b.rndN++;b.rnd+=NOTHING.test(r.said)?1:0}
      else{b.c++;b.named+=firstWord(r.said)===r.label?1:0;b.t1+=r.self===1?1:0;b.t1o+=r.obs===1?1:0;b.t5+=(r.self&&r.self<=5)?1:0;b.t5o+=(r.obs&&r.obs<=5)?1:0}}}
  return by}
const pct=(k,n)=>n?Math.round(100*k/n)+'%  ('+k+'/'+n+')':'–';
function scoreTable(x){const by=scores(x),arms=Object.keys(by);const t=el('table','sc');const hr=el('tr');hr.appendChild(el('th','',''));for(const a of arms)hr.appendChild(el('th','',armName(a)));t.appendChild(hr);
  const row=(name,f,tip)=>{const tr=el('tr');const td=el('td','',name);if(tip)td.title=tip;tr.appendChild(td);for(const a of arms)tr.appendChild(el('td','n',f(by[a])));t.appendChild(tr)};
  row('Sessions finished (of planned)',b=>b.done+' of '+b.planned+(b.stopped?'  · '+b.stopped+' crashed':''));
  row('Average knob turns',b=>b.done?(b.turns/b.done).toFixed(1):'–');
  row('Ended the session itself',b=>pct(b.ended,b.done));
  row('Concept knobs it turned',b=>b.c?String(b.c):'–');
  row('Its one-word answer was the true concept',b=>pct(b.named,b.c),'First word of its free answer equals the concept');
  row('True concept ranked 1st of 32 · self (chance 3%)',b=>pct(b.t1,b.c),'With the injections still in its activations');
  row('True concept ranked 1st of 32 · observer',b=>pct(b.t1o,b.c),'The same transcript re-read with nothing injected');
  row('True concept in top 5 of 32 · self (chance 16%)',b=>pct(b.t5,b.c));
  row('True concept in top 5 of 32 · observer',b=>pct(b.t5o,b.c));
  row('Sham knob (injects nothing): answer says nothing',b=>pct(b.sham,b.shamN));
  row('Random-direction knob: answer says nothing',b=>pct(b.rnd,b.rndN),'Not graded: a random direction may still do something');
  return t}
function status(s){return s.unfinished?'stopped before the end':s.stopped?'crashed':s.done?(s.ended?'ended itself':'hit turn limit'):s.waiting?(s.ended?'ended itself · waiting for its batch':'hit turn limit · waiting for its batch'):'still experimenting'}
function catalogue(){document.getElementById('crumb').innerHTML='<a href="#">Experiments</a>';document.title='Experiments';
  const v=document.getElementById('view');v.innerHTML='';const w=el('div','wrap');v.appendChild(w);
  w.appendChild(el('p','muted','Every experiment with sessions you can read, newest first: the E-series, and the older R0, R1.c and R1.d session runs. Click an experiment for its write-up, scores and every session. Calibration and other non-session runs are in the archive (link at the top).'));
  for(const x of S.experiments){const a=el('a','xrow');a.href='#x='+x.slug;a.appendChild(el('h2','',x.group));const wu=(INFO.writeups||{})[x.group];
    if(wu&&wu.status)a.appendChild(el('p','',wu.status));const ch=el('div','chips');
    if(x.live)ch.appendChild(el('span','chip live','● '+x.live+' still experimenting'));if(x.waiting)ch.appendChild(el('span','chip',x.waiting+' finished, waiting for their batch'));const by=scores(x);
    for(const k of Object.keys(by)){const b=by[k];ch.appendChild(el('span','chip'+(b.stopped?' bad':''),armName(k)+': '+b.done+'/'+b.planned+' done'+(b.c?' · named the concept '+Math.round(100*b.named/b.c)+'%':'')))}
    ch.appendChild(el('span','chip','started '+ago(x.started)));a.appendChild(ch);w.appendChild(a)}}
function experiment(){const x=S.experiments.find(e=>e.slug===X);const v=document.getElementById('view');
  if(!x){v.innerHTML='<div class="empty">No experiment by that name. <a href="#">All experiments</a></div>';return}
  document.title=x.group;const cr=document.getElementById('crumb');cr.innerHTML='<a href="#">Experiments</a> › ';cr.appendChild(el('span','',x.group));
  let top=v.querySelector('#xtop');const keep={};for(const c of v.querySelectorAll('.card')){const b=c.querySelector('.body');keep[c.dataset.key]={top:b.scrollTop,bottom:b.scrollHeight-b.scrollTop-b.clientHeight<30}}
  if(!top||top.dataset.slug!==X){v.innerHTML='';top=el('div','wrap');top.id='xtop';top.dataset.slug=X;v.appendChild(top);const g=el('div','grid');g.id='grid';const bar=el('div','bar');bar.id='bar';v.appendChild(bar);v.appendChild(g)}
  top.innerHTML='';const wu=(INFO.writeups||{})[x.group]||{};const p=el('div','panel');p.appendChild(el('h3','','Write-up'));
  for(const t of (wu.paras||(x.about&&x.about.length?x.about:null)||['No write-up yet.']))p.appendChild(el('p','',t));top.appendChild(p);
  const sp=el('div','panel');sp.appendChild(el('h3','','Scores'+((x.live||x.waiting)?' (so far; '+(x.live+(x.waiting||0))+' sessions not yet asked the end-of-session questions)':'')));
  sp.appendChild(el('p','muted','From the end-of-session question, asked for each knob it turned: "What does knob k do? Answer with one word, then one sentence." The model is never shown a list. Ranks place the true concept among the 32 concept words by the probability of its first word. Exploratory until every session is done.'));
  const sc=el('div','scroll');sc.appendChild(scoreTable(x));sp.appendChild(sc);top.appendChild(sp);
  const pr=el('details','panel');pr.appendChild(el('summary','','System prompt of each arm'));for(const [k,a] of Object.entries(x.arms)){pr.appendChild(el('h3','',armName(k)+(a.paste?' · reasoning pasted back':' · reasoning dropped')));pr.appendChild(el('pre','',a.system||''))}top.appendChild(pr);
  const bar=document.getElementById('bar');bar.innerHTML='';const opts=[['all','all '+x.sessions.length],['live','still experimenting '+x.live],['wait','finished, waiting '+(x.waiting||0)]].concat(Object.keys(x.arms).map(a=>[a,armName(a)]));
  for(const [k,lab] of opts){const b=el('button',filt===k?'on':'',lab);b.onclick=()=>{filt=k;experiment()};bar.appendChild(b)}
  order=x.sessions.filter(s=>filt==='all'||(filt==='live'?(!s.done&&!s.waiting):filt==='wait'?s.waiting:s.arm===filt)).sort((a,b)=>(a.done-b.done)||(a.ep-b.ep)||a.arm.localeCompare(b.arm));
  const g=document.getElementById('grid');g.innerHTML='';
  for(const s of order){const c=card(s);g.appendChild(c);const b=c.querySelector('.body');const k=keep[c.dataset.key];b.scrollTop=(!k||k.bottom)?b.scrollHeight:k.top}}
function card(s){const c=el('div','card'+(s.done?' done':''));const h=el('div','ch');
  h.appendChild(el('b','',armName(s.arm)+' · session '+s.ep));const t=armTag(s.arm);if(t)h.appendChild(el('span','tag '+t,t));
  h.appendChild(el('span','tag'+(s.done||s.waiting?'':' live'),status(s)));
  h.appendChild(el('span','sp',s.presses+' knob turns · '+ago(s.last)+' · open ⤢'));h.onclick=()=>go(X,s.run+'/'+s.ep);c.appendChild(h);
  if(s.board)c.appendChild(el('div','board',boardText(s.board)));
  const key=s.run+'/'+s.ep;const b=el('div','body');const tail=s.tail||[];
  if(expanded.has(key)&&fullCache[key])renderMsgs(b,fullCache[key],s.board);
  else{if(s.n>tail.length){const more=el('button','more','▲ show from the start ('+(s.n-tail.length)+' earlier messages)');
    more.onclick=async e=>{e.stopPropagation();more.textContent='loading…';expanded.add(key);await fetchFull(s.run,s.ep);experiment()};b.appendChild(more)}
    renderMsgs(b,tail,s.board)}
  c.appendChild(b);c.dataset.key=key;return c}
async function getJSON(u){const r=await fetch(BASE+u,{cache:'no-store'});if(!r.ok)throw new Error(r.status);return r.json()}
async function fetchFull(run,ep){const r=await getJSON('/api/ep?run='+encodeURIComponent(run)+'&ep='+ep);fullCache[run+'/'+ep]=r.events.filter(e=>e.type==='msg'||e.type==='discovery'||e.type==='done');return r}
async function loadFull(){const r=await fetchFull(full.run,full.ep);const b=document.getElementById('fbody');const atBottom=b.scrollHeight-b.scrollTop-b.clientHeight<40;const top=b.scrollTop;b.innerHTML='';
  const board=(r.events.find(e=>e.type==='board')||{}).board;document.getElementById('fboard').textContent=boardText(board);
  renderMsgs(b,r.events,board);b.scrollTop=full.fresh?0:(atBottom?b.scrollHeight:top);full.fresh=false;
  const i=order.findIndex(s=>s.run===full.run&&s.ep===full.ep);const s=order[i];
  document.getElementById('ftitle').textContent=(s?armName(s.arm)+' · session '+s.ep:full.run+' · session '+full.ep);
  document.getElementById('fsub').textContent=i>=0?(i+1)+' of '+order.length+(s?' · '+s.presses+' knob turns · '+status(s):''):'';
  document.getElementById('fprev').disabled=i<=0;document.getElementById('fnext').disabled=i<0||i>=order.length-1}
function step(d){if(!full)return;const i=order.findIndex(s=>s.run===full.run&&s.ep===full.ep);const s=order[i+d];if(s)go(X,s.run+'/'+s.ep)}
document.getElementById('fprev').onclick=()=>step(-1);document.getElementById('fnext').onclick=()=>step(1);document.getElementById('fclose').onclick=()=>go(X,null);
document.addEventListener('keydown',e=>{if(!full)return;if(e.key==='Escape')go(X,null);if(e.key==='ArrowLeft')step(-1);if(e.key==='ArrowRight')step(1)});
async function refresh(){try{S=await getJSON('/api/state'+(X?'?x='+encodeURIComponent(X):''))}catch(e){return}
  await Promise.all([...expanded].map(k=>{const i=k.lastIndexOf('/');return fetchFull(k.slice(0,i),+k.slice(i+1)).catch(()=>{})}));
  if(X)experiment();else catalogue();if(full)await loadFull().catch(()=>{})}
async function route(){const h=hashP();const x=h.get('x'),s=h.get('s');if(x!==X){X=x;filt='all';expanded.clear();S=null}
  if(!S)await refresh();else if(X)experiment();else catalogue();
  const f=document.getElementById('full');
  if(s&&X){const i=s.lastIndexOf('/');full={run:s.slice(0,i),ep:+s.slice(i+1),fresh:true};f.style.display='flex';document.getElementById('fbody').innerHTML='Loading…';await loadFull().catch(()=>{})}
  else{full=null;f.style.display='none'}}
async function loop(){await refresh();const live=S&&S.experiments.some(x=>(x.live||x.waiting)&&(!X||x.slug===X));timer=setTimeout(loop,live?4000:30000)}
window.addEventListener('hashchange',route);
(async()=>{try{INFO=await getJSON('/info')}catch(e){}await route();timer=setTimeout(loop,4000)})();
</script></body></html>"""
