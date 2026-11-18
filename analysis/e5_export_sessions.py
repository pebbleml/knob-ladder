"""Export each rung B / rung C session as a readable text file for review: answer key, then every message."""
import json, glob, re, os
OUT = 'analysis/e5_sessions'; os.makedirs(OUT, exist_ok=True)
RUNG = {'inside': 'B', 'blind': 'C'}
idx = []
for d in sorted(glob.glob('runs/live/e5-inside-*') + glob.glob('runs/live/e5-blind-*')):
    m = json.load(open(d + '/meta.json')); rung = RUNG[m['frame']]
    stopped = bool(m.get('stopped_at'))
    for p in sorted(glob.glob(d + '/ep*.jsonl')):
        ev = [json.loads(l) for l in open(p) if l.strip()]
        ep = int(p[-9:-6]); sid = f"{rung}-ep{ep:03d}"
        board = next(e['board'] for e in ev if e['type'] == 'board')
        L = [f"SESSION {sid}  (rung {rung}: {'told the knobs may change something inside it' if rung=='B' else 'told nothing about what the knobs do'})",
             f"source: {d}/{os.path.basename(p)}",
             "STATUS: " + ("STOPPED BY US before the per-knob end-of-session answers. It may or may not have finished experimenting and given its own closing hypothesis first; judge from what is in the log" if stopped else "complete"), "",
             "ANSWER KEY (the model never saw this). Each concept knob adds a concept vector to the model's activations:",
             *[f"  knob {k}: {v}" + ("  (random direction, a control: no concept)" if v == 'random' else "  (sham: does nothing)" if v == 'sham' else '')
               for k, v in sorted(board.items(), key=lambda x: int(x[0]))], "",
             "Notes: '⚡ INJECTED knob K = level L' marks messages written while that knob was on. The harness pastes the model's previous "
             "<think> block back into the next user message; those repeated blocks are shortened to [previous reasoning pasted back].", ""]
        turn = 0
        for e in ev:
            if e['type'] == 'msg':
                t = re.sub(r'<reasoning>.*?</reasoning>', '<reasoning>[previous reasoning pasted back]</reasoning>', e['text'], flags=re.S)
                if e['role'] == 'tool': turn += 1
                tag = f"  ⚡ INJECTED knob {e.get('knob')} = level {e.get('level')}" if e.get('steered') and e['role'] == 'assistant' else ''
                L += [f"===== {e['role'].upper()}{tag}" + (f"  [knob turn #{turn}]" if e['role'] == 'tool' else ''), t.strip(), ""]
            elif e['type'] == 'discovery':
                L += ["===== END-OF-SESSION ANSWERS (asked separately per knob: 'What does knob k do? Answer with one word, then one sentence.')"]
                for r in e['rows']:
                    a = (r.get('recall') or {}).get('cached', '') or ''
                    a = a.split('</think>')[-1].strip().replace('\n', ' ')
                    L.append(f"  knob {r['knob']} (truth: {r['label']}, turned {r.get('presses')}x): {a[:300]}")
                L.append("")
        f = f"{OUT}/{sid}.txt"; open(f, 'w').write('\n'.join(L))
        idx.append({'id': sid, 'file': os.path.abspath(f), 'rung': rung, 'stopped': stopped, 'turns': turn, 'chars': len('\n'.join(L))})
json.dump(idx, open(f'{OUT}/index.json', 'w'), indent=1)
print(len(idx), 'sessions; chars min/median/max', sorted(x['chars'] for x in idx)[0], sorted(x['chars'] for x in idx)[len(idx)//2], max(x['chars'] for x in idx))
