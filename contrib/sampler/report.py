#!/usr/bin/env python3
# usage: report.py <samples> <err log> [min percent] : where the main thread's CPU time went while answering (one sample = 10 ms of CPU).
import bisect, collections, re, struct, subprocess, sys
path, err = sys.argv[1], sys.argv[2]
minpct = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
DEPTH, SIZE = 48, 400

maps = []
for line in open(path + '.maps'):
    f = line.split(None, 5)
    if len(f) == 6 and 'x' in f[1]:
        lo, hi = f[0].split('-')
        maps.append((int(lo, 16), int(hi, 16), int(f[2], 16), f[5].strip()))
maps.sort()
syms = {}
def load(lib):
    if lib in syms:
        return syms[lib]
    pairs = {}
    for flag in ([], ['-D']):
        try:
            out = subprocess.run(['nm', '-C', '--defined-only'] + flag + [lib], capture_output=True, text=True).stdout
        except OSError:
            out = ''
        for l in out.splitlines():
            m = re.match(r'([0-9a-f]+) ([tTwWiu]) (.*)', l)
            if m:
                pairs.setdefault(int(m[1], 16), m[3])
    keys = sorted(pairs)
    syms[lib] = (keys, [pairs[k] for k in keys])
    return syms[lib]
def short(name):
    name = re.sub(r'\[clone [^\]]*\]', '', name).strip()
    cut = name.find('(')
    if cut > 0 and not name.startswith('operator') and 'operator()' not in name[:cut + 2]:
        name = name[:cut]
    return name[:72]
cache = {}
def resolve(pc):
    if pc in cache:
        return cache[pc]
    i = bisect.bisect_right([m[0] for m in maps], pc) - 1
    name = f'?{pc:x}'
    if i >= 0 and pc < maps[i][1]:
        lo, hi, off, lib = maps[i]
        base = lib.rsplit('/', 1)[-1]
        name = f'[{base}]'
        if lib.startswith('/'):
            keys, names = load(lib)
            j = bisect.bisect_right(keys, pc - lo + off) - 1
            if j >= 0:
                name = short(names[j])
    cache[pc] = name
    return name

samples = []
data = open(path, 'rb').read()
for at in range(0, len(data) - SIZE + 1, SIZE):
    t, n = struct.unpack_from('<QQ', data, at)
    if t == 0:
        break
    pcs = struct.unpack_from(f'<{n}Q', data, at + 16)
    stack = [resolve(pcs[0])] + [resolve(p - 1) for p in pcs[1:]]
    samples.append((t, stack))

text = open(err, errors='replace').read()
m = re.search(r'(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) runs', text)
eval_ms, runs = float(m[1]), int(m[2])
is_decode = lambda s: any(x.startswith(('llama_decode', 'llama_context::decode')) for x in s)
end = max(t for t, s in samples if is_decode(s))
start = end - eval_ms * 1e6
ans = [s for t, s in samples if start <= t <= end]
print(f'samples {len(samples)}; answering: {eval_ms / 1000:.1f} s for {runs} words = {eval_ms / 1000 / runs:.2f} s/word; main on CPU {len(ans) * .01 / runs:.2f} s/word ({len(ans) * .01 / (eval_ms / 1000) * 100:.0f}% of the time)')
per = lambda c: f'{c * .01 / runs:5.2f} s/word {c / len(ans) * 100:5.1f}%'

print('\nCALL TREE (outermost first, shown at or above', minpct, '% of main CPU)')
root = {}
for s in ans:
    node, prev = root, None
    for name in reversed(s):
        if name == prev:
            continue
        prev = name
        kid = node.setdefault(name, [0, {}])
        kid[0] += 1
        node = kid[1]
def show(node, depth):
    for name, (c, kids) in sorted(node.items(), key=lambda kv: -kv[1][0]):
        if c / len(ans) * 100 >= minpct:
            print(f'{per(c)} {"  " * depth}{name}')
            show(kids, depth + 1)
show(root, 0)

print('\nSELF (the function actually on the CPU)')
for name, c in collections.Counter(s[0] for s in ans).most_common(30):
    print(per(c), name)
