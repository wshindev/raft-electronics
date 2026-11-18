#!/usr/bin/env python3
"""Step 2: render every referenced EPS image to PNG with Ghostscript, and pull
text hints (labels, component types) out of the EPS source.

Usage:
    uv run python3 -I pipeline/render.py --refs data/model/image_refs.jsonl \
        --eps-dir ModEL/eps

Outputs (next to --refs unless --out is given):
    png/<image>.png      rendered image, longest side <= --max-side pixels
    image_meta.jsonl     one record per image: source, image kind, size, text hints
    render_report.txt    summary and failures

Re-running skips PNGs that already exist (use --force to redo).
"""
import argparse
import collections
import concurrent.futures as cf
import json
import multiprocessing
import os
import re
import subprocess
import sys

# --------------------------------------------------------------------------
# PostScript string decoding
# --------------------------------------------------------------------------

# Adobe Symbol font: Latin letter -> Greek glyph
SYMBOL = dict(zip('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
                  'αβχδεφγηιϕκλμνοπθρστυϖωξψζΑΒΧΔΕΦΓΗΙϑΚΛΜΝΟΠΘΡΣΤΥςΩΞΨΖ'))
SYMBOL_OCTAL = {0o260: '°', 0o261: '±', 0o264: '×', 0o270: '÷', 0o273: '≈',
                0o271: '≠', 0o243: '≤', 0o263: '≥', 0o256: '→', 0o254: '←',
                0o255: '↑', 0o257: '↓', 0o253: '↔', 0o245: '∞', 0o326: '√',
                0o362: '∫', 0o266: '∂', 0o327: '·', 0o345: 'Σ', 0o242: '′',
                0o262: '″', 0o361: '〉', 0o341: '〈', 0o246: 'ƒ'}


def ps_string(s, pos):
    """s[pos] == '('. Return (decoded bytes as list of ints, end index)."""
    depth, i, out = 0, pos, []
    while i < len(s):
        c = s[i]
        if c == '\\':
            nxt = s[i + 1:i + 2]
            if nxt and nxt in '01234567':
                m = re.match(r'[0-7]{1,3}', s[i + 1:])
                out.append(int(m.group(0), 8))
                i += 1 + len(m.group(0))
                continue
            out.append(ord({'n': '\n', 't': '\t', 'r': '\r'}.get(nxt, nxt or '\\')))
            i += 2
            continue
        if c == '(':
            depth += 1
            if depth > 1:
                out.append(ord(c))
        elif c == ')':
            depth -= 1
            if depth == 0:
                return out, i + 1
            out.append(ord(c))
        else:
            out.append(ord(c))
        i += 1
    return out, i


def decode(codes, font):
    if 'Symbol' in font:
        return ''.join(SYMBOL_OCTAL.get(c) or SYMBOL.get(chr(c), chr(c)) if c < 256 else chr(c)
                       for c in codes)
    return bytes(c & 0xFF for c in codes).decode('latin-1')


def tokenize(body):
    """Minimal PostScript tokenizer: strings, brace groups (as raw text),
    names, and numbers. Comments are skipped."""
    toks, i, n = [], 0, len(body)
    while i < n:
        c = body[i]
        if c.isspace():
            i += 1
        elif c == '%':
            j = body.find('\n', i)
            i = n if j == -1 else j
        elif c == '(':
            codes, i = ps_string(body, i)
            toks.append(('str', codes))
        elif c == '{':
            depth, j = 0, i
            while j < n:
                if body[j] == '{':
                    depth += 1
                elif body[j] == '}':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            toks.append(('group', body[i + 1:j].strip()))
            i = j + 1
        elif c in '[]':
            toks.append(('name', c))
            i += 1
        else:
            m = re.match(r'[^\s(){}\[\]%]+', body[i:i + 256])
            if not m:      # stray ')' or '}'
                i += 1
                continue
            toks.append(('name', m.group(0)))
            i += m.end()
    return toks


# --------------------------------------------------------------------------
# Text hints
# --------------------------------------------------------------------------

def xcircuit_hints(src):
    """Visible labels and placed component types from an XCircuit EPS."""
    m = re.search(r'^%%Page:.*$', src, re.M)
    body = src[m.end():] if m else src
    end = body.find('pgsave restore')
    body = body[:end] if end != -1 else body
    labels, comps = [], collections.Counter()
    cur = []
    for kind, val in tokenize(body):
        if kind in ('str', 'group'):
            cur.append((kind, val))
            continue
        if re.match(r'^-?[\d.]+$', val) or val.startswith('/'):
            cur.append(('num', val))
            continue
        if val in ('label', 'pinlabel', 'pinglobal'):
            text = xc_label(cur)
            if text.strip():
                labels.append(text.strip())
        elif '::' in val:
            comps[val.split('::', 1)[1]] += 1
        cur = []
    return labels, comps


def xc_label(items):
    """XCircuit stores label segments in reverse order; font and
    sub/superscript switches apply to the strings that follow them when read
    in forward order."""
    segs = [x for x in items if x[0] != 'num']
    font, mode, out = 'Times-Roman', '', []
    for kind, val in reversed(segs):
        if kind == 'group':
            g = val.strip()
            fm = re.match(r'/(\S+)\s+cf', g)
            if fm:
                font = fm.group(1)
            elif g == 'ss':
                mode = '_'
            elif g == 'Ss':
                mode = '^'
            elif g == 'ns':
                mode = ''
            elif g == 'CR':
                out.append(' / ')
            elif g in ('Ts', 'Qs', 'hS', 'qS'):
                out.append(' ')
        else:
            txt = decode(val, font)
            out.append(mode + txt if mode and txt.strip() else txt)
    return re.sub(r'\s+', ' ', ''.join(out))


def generic_hints(src):
    """Fallback for plots (gnuplot, nutmeg, ...): readable strings in the body."""
    m = re.search(r'%%EndProlog', src)
    body = src[m.end():] if m else src
    out = []
    for kind, val in tokenize(body[:2_000_000]):
        if kind != 'str':
            continue
        t = decode(val, '').strip()
        if 0 < len(t) < 60 and re.search(r'[A-Za-z0-9]', t) and \
                not re.fullmatch(r'[A-Za-z-]+(-Bold|-Italic|-Roman|-Oblique)?', t) and \
                sum(ch.isprintable() for ch in t) == len(t):
            out.append(t)
    return list(dict.fromkeys(out))[:80]


def classify(creator):
    c = creator.lower()
    if 'xcircuit' in c:
        return 'xcircuit'
    if any(k in c for k in ('gimp', 'jpeg2ps', 'imagemagick')):
        return 'photo'
    if any(k in c for k in ('gnuplot', 'nutmeg', 'ngspice')):
        return 'plot'
    if 'dvips' in c:
        return 'math_figure'
    if any(k in c for k in ('pcb', 'eeschema')):
        return 'pcb'
    return 'other'


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def process(job):
    img, eps, out_png, max_side, force = job
    rec = {'image': img, 'eps': eps, 'png': out_png, 'status': 'ok'}
    with open(eps, 'rb') as f:
        head = f.read(4096).decode('latin-1')
    creator = re.search(r'^%%Creator:\s*(.*)$', head, re.M)
    rec['creator'] = creator.group(1).strip() if creator else ''
    rec['image_kind'] = classify(rec['creator'])
    bb = re.search(r'^%%BoundingBox:\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)', head, re.M)
    if not bb:
        rec['status'] = 'no_bbox'
        return rec
    x0, y0, x1, y1 = map(float, bb.groups())
    w_pt, h_pt = max(x1 - x0, 1), max(y1 - y0, 1)
    dpi = min(max_side * 72.0 / max(w_pt, h_pt), 600)
    rec['dpi'] = round(dpi, 1)
    rec['width'], rec['height'] = round(w_pt * dpi / 72), round(h_pt * dpi / 72)
    # gnuplot "landscape" output is drawn rotated; Ghostscript ignores the hint
    rec['landscape'] = bool(re.search(r'^%%Orientation:\s*Landscape', head, re.M))
    if rec['landscape']:
        rec['width'], rec['height'] = rec['height'], rec['width']

    # text hints
    if rec['image_kind'] in ('xcircuit', 'plot', 'other', 'math_figure'):
        with open(eps, 'rb') as f:
            src = f.read().decode('latin-1')
        if rec['image_kind'] == 'xcircuit':
            labels, comps = xcircuit_hints(src)
            rec['labels'] = labels
            rec['components'] = dict(comps.most_common())
        else:
            rec['labels'] = generic_hints(src)

    if os.path.exists(out_png) and not force:
        rec['status'] = 'cached'
        return rec
    cmd = ['gs', '-q', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-dEPSCrop',
           '-sDEVICE=png16m', '-r%.2f' % dpi,
           '-dTextAlphaBits=4', '-dGraphicsAlphaBits=4',
           '-sOutputFile=' + out_png, eps]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=120)
        if p.returncode != 0 or not os.path.exists(out_png):
            rec['status'] = 'gs_error'
            rec['error'] = (p.stderr or p.stdout).decode('latin-1')[-500:]
        elif rec['landscape']:
            from PIL import Image
            with Image.open(out_png) as im:
                im.rotate(-90, expand=True).save(out_png)
    except subprocess.TimeoutExpired:
        rec['status'] = 'timeout'
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--refs', required=True)
    ap.add_argument('--eps-dir', action='append', required=True,
                    help='directory to search for EPS files (repeatable, searched recursively)')
    ap.add_argument('--out', default=None,
                    help='output folder (default: the folder containing --refs)')
    ap.add_argument('--max-side', type=int, default=1568,
                    help='longest side in pixels (Claude downsizes anything larger)')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--limit', type=int, default=0, help='only process the first N images')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.dirname(args.refs) or '.'

    index = {}
    for d in args.eps_dir:
        for root, _, files in os.walk(d):
            for fn in files:
                if fn.endswith('.eps'):
                    index.setdefault(fn[:-4], os.path.join(root, fn))

    refs = [json.loads(l) for l in open(args.refs)]
    if args.limit:
        refs = refs[:args.limit]
    png_dir = os.path.join(args.out, 'png')
    os.makedirs(png_dir, exist_ok=True)

    jobs, missing = [], []
    for r in refs:
        eps = index.get(r['image'])
        if not eps:
            missing.append(r['image'])
            continue
        jobs.append((r['image'], eps, os.path.join(png_dir, r['image'] + '.png'),
                     args.max_side, args.force))

    results = []
    # "fork" explicitly: Python 3.14 defaults to forkserver, which needs a Unix
    # socket that sandboxed environments may block
    ctx = multiprocessing.get_context('fork')
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        for i, rec in enumerate(ex.map(process, jobs, chunksize=8), 1):
            results.append(rec)
            if i % 500 == 0:
                print('  %d / %d' % (i, len(jobs)), flush=True)
    for img in missing:
        results.append({'image': img, 'status': 'missing_eps'})
    source = {r['image']: r.get('source') for r in refs}
    for rec in results:
        rec['source'] = source.get(rec['image'])

    with open(os.path.join(args.out, 'image_meta.jsonl'), 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    st = collections.Counter(r['status'] for r in results)
    kinds = collections.Counter(r.get('image_kind', '?') for r in results)
    xc = [r for r in results if r.get('image_kind') == 'xcircuit']
    L = ['images: %d' % len(results),
         'status: %s' % dict(st),
         'image kinds: %s' % dict(kinds),
         'xcircuit images with >=1 label: %d / %d' % (sum(1 for r in xc if r.get('labels')), len(xc))]
    px = sorted(r['width'] * r['height'] for r in results if 'width' in r)
    if px:
        L.append('approx Claude image tokens (w*h/750): total %.1fM, median %d per image' % (
            sum(min(p, 1568 * 1568) for p in px) / 750 / 1e6, px[len(px) // 2] / 750))
    L.append('')
    L.append('failures:')
    for r in results:
        if r['status'] not in ('ok', 'cached'):
            L.append('  %s  %s  %s' % (r['image'], r['status'], r.get('error', '').strip()[:200]))
    with open(os.path.join(args.out, 'render_report.txt'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L[:6]))


if __name__ == '__main__':
    sys.exit(main())
