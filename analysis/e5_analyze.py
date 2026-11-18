"""E5 analysis (exploratory, after the fact): text-based naming, loops, chained turns, and the end-of-session readout.
Reads runs/live/e5-*/ep*.jsonl (full messages) and runs/e5-*.jsonl (readout ranks). Run from the repo root."""
import json, glob, re, os, collections, math, random

def forms(w):
    f = {w, w + 's', w + 'es'}
    if w.endswith('y'): f.add(w[:-1] + 'ies')
    return f
def rx(w): return re.compile(r'\b(' + '|'.join(sorted(forms(w), key=len, reverse=True)) + r')\b', re.I)
def strip_paste(t):  # remove the pasted <reasoning> block so we only score what the model writes in this message
    return re.sub(r'<reasoning>.*?</reasoning>', '', t, flags=re.S)

RUNS = collections.OrderedDict()
for d in sorted(glob.glob('runs/live/e5-*')):
    m = json.load(open(d + '/meta.json'))
    arm = f"{m['frame']}-{'paste' if m.get('paste_thinking') else 'drop'}"
    tag = os.path.basename(d).split('_')[2]
    key = arm + (' (stopped)' if m.get('stopped_at') else '') + (' (rerun)' if tag.startswith('20261117') else '')
    RUNS.setdefault(key, []).append(d)

def top_word(t):
    ws = re.findall(r"[^\W\d_]{2,}|[一-鿿]", t.lower())
    if not ws: return '', 0, 0
    w, c = collections.Counter(ws).most_common(1)[0]
    return w, c, len(ws)

out = {}
for key, dirs in RUNS.items():
    S = collections.Counter(); bylev = collections.defaultdict(lambda: [0, 0]); loops = []
    fa = [0, 0]; first_hit = [0, 0]
    for d in dirs:
        for p in sorted(glob.glob(d + '/ep*.jsonl')):
            ev = [json.loads(l) for l in open(p) if l.strip()]
            board = next(e['board'] for e in ev if e['type'] == 'board')
            concepts = [v for v in board.values() if v not in ('random', 'sham')]
            msgs = [e for e in ev if e['type'] == 'msg']
            S['sessions'] += 1
            S['done'] += any(e['type'] == 'done' for e in ev)
            seen = set()
            for i, e in enumerate(msgs):
                if e['role'] == 'tool':
                    S['presses'] += 1
                    prev = msgs[i - 1]
                    if prev['role'] == 'assistant' and prev.get('steered'):
                        S['chained'] += 1  # the turn() came from the injected reply: next observe reads injected reasoning under a new knob
                if e['role'] == 'assistant' and e.get('steered'):
                    lab, lev = e.get('label'), e.get('level') or 0
                    txt = strip_paste(e['text'])
                    if lab in concepts:
                        hit = bool(rx(lab).search(txt))
                        S['obs_c'] += 1; S['obs_hit'] += hit
                        bylev[min(lev, 10) // 4 if lev < 10 else 3][0] += 1; bylev[min(lev, 10) // 4 if lev < 10 else 3][1] += hit
                        if lab not in seen:
                            first_hit[0] += 1; first_hit[1] += hit
                        # false-alarm baseline: other board concepts not yet injected in this session
                        for o in concepts:
                            if o != lab and o not in seen:
                                fa[0] += 1; fa[1] += bool(rx(o).search(txt))
                    elif lab in ('random', 'sham'):
                        S['ctl_obs'] += 1
                    w, c, n = top_word(txt)
                    if c >= 40 and c / max(n, 1) > 0.25:
                        # what did the model say right after? next unsteered assistant message (or next steered one if chained)
                        nxt = next((x for x in msgs[i + 1:] if x['role'] == 'assistant'), None)
                        nt = strip_paste(nxt['text']) if nxt else ''
                        flag = bool(re.search(r'repeat|repetit|loop|over and over|same word|spiral|degenerat|stuck|循环|重复', nt, re.I))
                        named = bool(lab in concepts and rx(lab).search(re.sub(r'<think>.*?</think>', '', nt, flags=re.S)))
                        loops.append(dict(run=os.path.basename(d), ep=p[-9:-6], knob=e.get('knob'), label=lab, level=lev, word=w, count=c,
                                          next_flags_loop=flag, next_names_concept=named, next_chained=nxt.get('steered') if nxt else None,
                                          next_head=nt[:300].replace('\n', ' ')))
                    if lab: seen.add(lab)
                if e['role'] == 'assistant' and e.get('steered') and e.get('label'): pass
    out[key] = dict(S=dict(S), bylev={k: v for k, v in sorted(bylev.items())}, fa=fa, first=first_hit, loops=loops)

# readout from the results files (only runs whose end-of-session scoring finished)
RO = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob('runs/e5-*.jsonl')):
    arm = os.path.basename(f).split('_')[0].replace('e5-', '')
    for l in open(f):
        r = json.loads(l)
        for row in r['discovery']:
            lab = row['label']
            if lab in ('random', 'sham') or not row.get('presses'): continue
            for rd in ('cached', 'observer'):
                rk = row['readout'][rd]['ranks'][lab]
                RO[arm][rd].append(rk)
            RO[arm]['said'].append(bool(rx(lab).search((row.get('recall') or {}).get('cached', '') or '')))

def fmt(a, b): return f"{b}/{a} = {100*b/max(a,1):.0f}%"
print('=== Text measures (from the live logs) ===')
for k, v in out.items():
    S = v['S']
    print(f"\n## {k}: {S.get('sessions')} sessions ({S.get('done')} reached the end), {S.get('presses')} knob turns, "
          f"{fmt(S.get('presses',0), S.get('chained',0))} turned straight from the injected reply")
    print(f"   observation names the injected concept: {fmt(S.get('obs_c',0), S.get('obs_hit',0))}; "
          f"first time a knob is tried: {fmt(*v['first'])}; false-alarm baseline (names a not-yet-injected board concept): {fmt(*v['fa'])}")
    names = {0: 'levels 0-3', 1: 'levels 4-7', 2: 'levels 8-9', 3: 'level 10'}
    print('   by level: ' + '; '.join(f"{names[k2]} {fmt(*x)}" for k2, x in v['bylev'].items()))
    L = v['loops']
    print(f"   loops (one word >=40 times and >25% of words): {len(L)}; next reply mentions repetition/loop: {sum(x['next_flags_loop'] for x in L)}; "
          f"names the concept in its answer: {sum(x['next_names_concept'] for x in L)}")
print('\n=== End-of-session readout (rank of the true concept among 32; chance top-1 3%, top-5 16%, mean 1/rank 0.127) ===')
for arm, d in RO.items():
    for rd in ('cached', 'observer'):
        r = d[rd]; n = len(r)
        print(f"{arm:14s} {rd:8s} n={n}  top1 {100*sum(x==1 for x in r)/n:.1f}%  top5 {100*sum(x<=5 for x in r)/n:.1f}%  1/rank {sum(1/x for x in r)/n:.3f}")
    print(f"{arm:14s} free-text answer contains the concept word: {fmt(len(d['said']), sum(d['said']))}")
json.dump({k: v['loops'] for k, v in out.items()}, open('analysis/e5_loops.json', 'w'), indent=1)
