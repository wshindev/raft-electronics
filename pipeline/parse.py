#!/usr/bin/env python3
"""Step 1: parse ModEL LaTeX sources into section-level chunks, exam Q/A pairs,
and an image-reference index.

Usage:
    uv run python3 -I pipeline/parse.py --content ModEL/content \
        --exammake ModEL/exammake/exammake.c --source model --out data

Every record carries a "source" field, and outputs go to <out>/<source>/ so
that later sources (e.g. OSHA) can sit alongside without colliding.

Outputs (in <out>/<source>/):
    chunks.jsonl       one record per unique chunk (raw LaTeX, comments stripped)
    exam_pairs.jsonl   one record per xq/xa pair
    image_refs.jsonl   one record per referenced image, with every place it is used
    parse_report.txt   summary statistics and warnings
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# common_* files that are boilerplate (dropped). Any other common_* file is
# treated as real content and inlined.
BOILERPLATE_COMMON = re.compile(
    r'^common_(preamble|chapintro_.*|instructor_.*|journaling|license.*|'
    r'list_.*_challenges|outline|philosophy|presentation_.*|proven.*|'
    r'questions|secintro_.*|session_.*|strategies|tools)\.(la)?tex$')

DROP_CHAPTERS = {
    'Version history', 'Creative Commons License', 'Tools used',
    'Instructional philosophy', 'Problem-Solving Strategies', 'Animations',
    'References',
}
DROP_SECTIONS = {'Recommendations for instructors'}
DROP_SUBSECTIONS = {'Reading outline and reflections'}

CHAPTER_TYPES = {
    'Introduction': 'intro',
    'Case Tutorial': 'case_tutorial',
    'Tutorial': 'tutorial',
    'Simplified Tutorial': 'tutorial',
    'Full Tutorial': 'tutorial',
    'Historical References': 'historical',
    'Derivations and Technical References': 'derivation',
    'Programming References': 'programming',
    'Questions': 'question',
    'Projects and Experiments': 'project',
}

MAX_CHUNK_CHARS = 6000   # raw LaTeX characters; larger sections get split
MIN_TEXT_LETTERS = 150   # chunks with fewer letters of prose are dropped
CONTEXT_CHARS = 800      # raw text kept on each side of an image reference

IMG_RE = re.compile(
    r'\\(?:includegraphics|epsfbox)\s*(?:\[[^\]]*\])?\s*\{\s*([^}\s]+?)\s*\}')
INPUT_RE = re.compile(r'\\input\s*(?:\{\s*([^}\s]+)\s*\}|([A-Za-z0-9_.\-]+))')
CHALLENGES_RE = re.compile(r'\\framebox\[\\width\]\s*\{\s*Challenges\s*\}')

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

warnings = []


def warn(msg):
    warnings.append(msg)


def read(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return f.read()


def image_id(arg):
    base = os.path.basename(arg)
    return re.sub(r'\.(eps|pdf|png|jpg)$', '', base)


def strip_comments(text):
    """Remove LaTeX comments, leaving verbatim blocks intact.

    Comments trailing an image reference become \\imgnote{...}, and SELEC tags
    become \\selec{...}, since both carry information worth keeping."""
    out = []
    in_verb = False
    for line in text.split('\n'):
        if in_verb:
            out.append(line)
            if '\\end{verbatim}' in line:
                in_verb = False
            continue
        m = re.search(r'(?<!\\)%', line)
        code_part = line[:m.start()] if m else line
        if '\\begin{verbatim}' in code_part and '\\end{verbatim}' not in line:
            in_verb = True
            out.append(line)
            continue
        if not m:
            out.append(line)
            continue
        code, comment = line[:m.start()], line[m.start() + 1:].strip().lstrip('%').strip()
        extra = ''
        sel = re.match(r'SELEC\s+([\d ,]+)', comment)
        if sel:
            extra = '\\selec{%s}' % sel.group(1).strip()
        elif comment and IMG_RE.search(code):
            extra = '\\imgnote{%s}' % comment.replace('{', '(').replace('}', ')')
        if code.strip() or extra:
            out.append(code.rstrip() + (' ' + extra if extra else ''))
        elif not code.strip():
            # a whole-line comment: drop it without creating a paragraph break
            continue
    return '\n'.join(out)


def balanced_arg(text, i):
    """text[i] must be '{'. Return (content, index after closing brace)."""
    assert text[i] == '{'
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == '\\':
            continue
        if c == '{' and text[j - 1] != '\\':
            depth += 1
        elif c == '}' and text[j - 1] != '\\':
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
    return text[i + 1:], len(text)


def find_cmds(text, name):
    """Find \\name{title} (and \\name*{...}). Return [(start, end, title)]."""
    res = []
    for m in re.finditer(r'\\%s\*?\s*(?=\{)' % name, text):
        title, end = balanced_arg(text, m.end())
        res.append((m.start(), end, clean_title(title)))
    return res


def clean_title(t):
    t = re.sub(r'\\index\{[^}]*\}', '', t)
    t = re.sub(r"\\'([A-Za-z])", r'\1', t)
    t = re.sub(r'\\[a-zA-Z]+\s*', '', t)
    t = t.replace('{', '').replace('}', '').replace('$', '')
    return re.sub(r'\s+', ' ', t).strip()


def letters(text):
    t = re.sub(r'\\[a-zA-Z]+', ' ', text)
    return sum(c.isalpha() for c in t)


def norm_hash(text):
    t = re.sub(r'\s+', ' ', text).strip()
    return hashlib.sha1(t.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------
# Include expansion with provenance markers
# --------------------------------------------------------------------------

class Expander:
    def __init__(self, content_dir):
        self.dir = content_dir
        self.cache = {}
        self.counter = 0
        self.used = collections.Counter()

    def load(self, name):
        if name not in self.cache:
            path = os.path.join(self.dir, name)
            self.cache[name] = strip_comments(read(path)) if os.path.exists(path) else None
        return self.cache[name]

    def resolve(self, raw):
        name = os.path.basename(raw)
        if not re.search(r'\.(la)?tex$', name):
            for ext in ('.latex', '.tex'):
                if os.path.exists(os.path.join(self.dir, name + ext)):
                    return name + ext
            return name + '.latex'
        return name

    def expand(self, text, stack=()):
        def repl(m):
            name = self.resolve(m.group(1) or m.group(2))
            if BOILERPLATE_COMMON.match(name):
                return '\n'
            if name in stack:
                warn('include cycle: %s -> %s' % (' -> '.join(stack), name))
                return '\n'
            body = self.load(name)
            if body is None:
                warn('missing include %s (from %s)' % (name, stack[-1] if stack else '?'))
                return '\n'
            self.used[name] += 1
            self.counter += 1
            k = self.counter
            inner = self.expand(body, stack + (name,))
            return '\n\\srcbegin{%d:%s}\n%s\n\\srcend{%d}\n' % (k, name, inner, k)
        return INPUT_RE.sub(repl, text)


SRC_BEGIN = re.compile(r'\\srcbegin\{(\d+):([^}]*)\}')
SRC_END = re.compile(r'\\srcend\{(\d+)\}')


def source_intervals(text):
    begins = {m.group(1): (m.start(), m.group(2)) for m in SRC_BEGIN.finditer(text)}
    ends = {m.group(1): m.end() for m in SRC_END.finditer(text)}
    return [(s, ends.get(k, len(text)), name) for k, (s, name) in begins.items()]


def strip_markers(text):
    return SRC_END.sub('', SRC_BEGIN.sub('', text))


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def split_long(text, max_chars):
    """Split text at paragraph breaks (outside verbatim / display math)."""
    if len(text) <= max_chars:
        return [text]
    paras, cur, in_verb = [], [], False
    for line in text.split('\n'):
        if '\\begin{verbatim}' in line:
            in_verb = True
        if '\\end{verbatim}' in line:
            in_verb = False
        cur.append(line)
        if not line.strip() and not in_verb and cur and ''.join(cur).count('$$') % 2 == 0:
            paras.append('\n'.join(cur))
            cur = []
    if cur:
        paras.append('\n'.join(cur))
    # paragraphs that are still too long (e.g. long lists) split at line breaks
    fine = []
    for p in paras:
        if len(p) <= max_chars or '\\begin{verbatim}' in p:
            fine.append(p)
            continue
        cur = ''
        for line in p.split('\n'):
            if cur and len(cur) + len(line) > max_chars and cur.count('$$') % 2 == 0:
                fine.append(cur)
                cur = ''
            cur += line + '\n'
        fine.append(cur)
    paras = fine
    parts, buf = [], ''
    for p in paras:
        if buf and len(buf) + len(p) > max_chars:
            parts.append(buf)
            buf = ''
        buf += p + '\n'
    if buf.strip():
        parts.append(buf)
    return parts


def segments(text, start, end, name):
    """Split text[start:end] by \\name headings -> [(seg_start, seg_end, title)].
    The part before the first heading gets title None."""
    heads = [(s, e, t) for s, e, t in find_cmds(text[start:end], name)]
    out = []
    first = heads[0][0] if heads else end - start
    if first > 0:
        out.append((start, start + first, None, start))
    for i, (s, e, t) in enumerate(heads):
        nxt = heads[i + 1][0] if i + 1 < len(heads) else end - start
        out.append((start + e, start + nxt, t, start + s))
    return out


def tidy(text):
    """Remove template placeholder lines and collapse runs of blank lines."""
    text = re.sub(r'^[ \t]*\\item[ \t]*(\?\?\?\.?)?[ \t]*$\n?', '', text, flags=re.M)
    text = re.sub(r'[ \t]+\n', '\n', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def module_title(raw, fname):
    m = re.search(r'^%\s*Module\s*=\s*(.+)$', raw, re.M)
    return m.group(1).strip() if m else fname


def chunk_module(path, expander, report):
    fname = os.path.basename(path)
    mod = fname[:-len('.latex')]
    raw = read(path)
    title = module_title(raw, mod)
    text = expander.expand(strip_comments(raw), (fname,))
    end_doc = text.find('\\end{document}')
    if end_doc != -1:
        text = text[:end_doc]
    intervals = source_intervals(text)
    chapters = find_cmds(text, 'chapter')
    if not chapters:
        warn('no chapters in %s' % fname)
        return []

    pieces = []  # (start, end, chapter, section, subsection)
    for ci, (cs, ce, ctitle) in enumerate(chapters):
        cend = chapters[ci + 1][0] if ci + 1 < len(chapters) else len(text)
        report['chapters'][ctitle] += 1
        if ctitle in DROP_CHAPTERS:
            continue
        for ss, se, stitle, _ in segments(text, ce, cend, 'section'):
            if stitle in DROP_SECTIONS:
                continue
            subs = segments(text, ss, se, 'subsection')
            by_sub = ctitle == 'Questions' or (se - ss) > MAX_CHUNK_CHARS
            if by_sub and len(subs) > 1:
                for us, ue, utitle, hs in subs:
                    if utitle in DROP_SUBSECTIONS:
                        continue
                    # keep the subsection heading in the text when splitting
                    # long sections, so later cleaning can render it
                    pieces.append((us if ctitle == 'Questions' else hs, ue,
                                   ctitle, stitle, utitle))
            else:
                pieces.append((ss, se, ctitle, stitle, None))

    chunks = []
    for s, e, ctitle, stitle, utitle in pieces:
        body = text[s:e]
        sources = sorted({n for (a, b, n) in intervals if a < e and b > s})
        parts = split_long(body, MAX_CHUNK_CHARS)
        if any(t and (t.startswith('(') or ': (' in t) for t in (stitle, utitle)):
            report['dropped_placeholder'] += 1   # unfilled template sections
            continue
        for pi, part in enumerate(parts):
            clean = tidy(strip_markers(part))
            if letters(clean) < MIN_TEXT_LETTERS and not IMG_RE.search(clean):
                report['dropped_tiny'] += 1
                continue
            chunks.append({
                'module': mod, 'module_title': title, 'chapter': ctitle,
                'section': stitle, 'subsection': utitle,
                'part': pi + 1 if len(parts) > 1 else None,
                'type': CHAPTER_TYPES.get(ctitle, 'other'),
                'included_files': [x for x in sources if x != fname],
                'text': clean,
            })
    return chunks


def finalize_chunk(c):
    text = c['text']
    selec = re.findall(r'\\selec\{([^}]*)\}', text)
    c['selec'] = [x for s in selec for x in re.split(r'[ ,]+', s) if x]
    c['images'] = list(dict.fromkeys(image_id(m) for m in IMG_RE.findall(text)))
    if c['type'] == 'question':
        m = CHALLENGES_RE.search(text)
        c['challenges'] = (re.findall(r'\\item\s+(.*?)(?=\\item|\\end\{itemize\}|$)',
                                      text[m.end():], re.S) if m else [])
        c['challenges'] = [x.strip() for x in c['challenges'] if x.strip()]
    return c


# --------------------------------------------------------------------------
# Exams
# --------------------------------------------------------------------------

def parse_exammake(path):
    """Return {xq_number: [ {course, exam, slot, topic} ... ]}."""
    pools = collections.defaultdict(list)
    if not path or not os.path.exists(path):
        warn('exammake.c not found; exam pairs will have no pool metadata')
        return pools
    topics = {}
    entry = re.compile(r'^\s*q\[(\d+)\]\[(\d+)\]\[(\d+)\]\[(\d+)\]\s*=\s*(\d+)\s*;\s*(?://\s*(.*))?$')
    for line in read(path).split('\n'):
        m = entry.match(line)
        if not m:
            continue
        course, exam, slot, idx, val, comment = m.groups()
        key = (int(course), int(exam), int(slot))
        if idx == '0':
            topics[key] = (comment or '').strip()
            continue
        pools[int(val)].append({'course': key[0], 'exam': key[1], 'slot': key[2],
                                'topic': topics.get(key, ''),
                                'note': (comment or '').strip()})
    return pools


def variant_groups(pools):
    """Union-find over questions that share a (course, exam, slot)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_slot = collections.defaultdict(list)
    for q, entries in pools.items():
        for e in entries:
            by_slot[(e['course'], e['exam'], e['slot'])].append(q)
    for qs in by_slot.values():
        for q in qs[1:]:
            parent[find(q)] = find(qs[0])
    groups = collections.defaultdict(list)
    for q in pools:
        groups[find(q)].append(q)
    return {q: 'vg_%04d' % min(members) for members in groups.values() for q in members}


def parse_exams(content_dir, exammake_path):
    pools = parse_exammake(exammake_path)
    vg = variant_groups(pools)
    pairs = []
    for qpath in sorted(glob.glob(os.path.join(content_dir, 'xq_*.tex'))):
        num = re.search(r'xq_(\d+)\.tex$', qpath).group(1)
        apath = os.path.join(content_dir, 'xa_%s.tex' % num)
        q = strip_comments(read(qpath)).strip()
        a = strip_comments(read(apath)).strip() if os.path.exists(apath) else ''
        if letters(q) < 20:
            status = 'blank_question'
        elif not os.path.exists(apath):
            status = 'missing_answer'
        elif letters(a) < 3 and not IMG_RE.search(a):
            status = 'blank_answer'
        else:
            status = 'ok'
        n = int(num)
        pairs.append({
            'id': 'xq_%s' % num, 'status': status,
            'question': q, 'answer': a,
            'question_images': list(dict.fromkeys(image_id(x) for x in IMG_RE.findall(q))),
            'answer_images': list(dict.fromkeys(image_id(x) for x in IMG_RE.findall(a))),
            'pools': pools.get(n, []),
            'variant_group': vg.get(n, 'vg_solo_%s' % num),
        })
    return pairs


# --------------------------------------------------------------------------
# Image index
# --------------------------------------------------------------------------

def image_uses(text, kind, ref, extra_before=''):
    uses = []
    for m in IMG_RE.finditer(text):
        note = re.match(r'[^\n]*?\\imgnote\{([^}]*)\}', text[m.end():])
        uses.append({
            'image': image_id(m.group(1)), 'kind': kind, 'ref': ref,
            'before': (extra_before + text[:m.start()])[-CONTEXT_CHARS:],
            'after': text[m.end():m.end() + CONTEXT_CHARS],
            'note': note.group(1) if note else None,
        })
    return uses


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--content', required=True)
    ap.add_argument('--exammake', default=None)
    ap.add_argument('--source', default='model',
                    help='short source name; tags every record and names the output folder')
    ap.add_argument('--out', required=True, help='data root; outputs go to <out>/<source>/')
    args = ap.parse_args()
    args.out = os.path.join(args.out, args.source)
    os.makedirs(args.out, exist_ok=True)

    report = {'chapters': collections.Counter(), 'dropped_tiny': 0, 'dropped_placeholder': 0}
    expander = Expander(args.content)

    # --- modules ---
    merged = collections.OrderedDict()
    mods = sorted(p for p in glob.glob(os.path.join(args.content, 'mod_*.latex'))
                  if '00template' not in p)
    raw_count = 0
    for path in mods:
        for c in chunk_module(path, expander, report):
            raw_count += 1
            h = norm_hash(c['text'])
            loc = {k: c[k] for k in ('module', 'module_title', 'chapter', 'section',
                                     'subsection', 'part')}
            if h in merged:
                merged[h]['locations'].append(loc)
                continue
            rec = {'id': 'ch_' + h, 'source': args.source, 'type': c['type'], 'locations': [loc],
                   'included_files': c['included_files'], 'text': c['text']}
            merged[h] = finalize_chunk(rec)

    chunks = list(merged.values())
    for c in chunks:
        c['n_chars'] = len(c['text'])

    # --- exams ---
    pairs = parse_exams(args.content, args.exammake)
    for p in pairs:
        p['source'] = args.source

    # --- image index ---
    uses = []
    for c in chunks:
        uses += image_uses(c['text'], 'module', c['id'])
    for p in pairs:
        if p['status'] != 'ok':
            continue
        uses += image_uses(p['question'], 'exam_question', p['id'])
        uses += image_uses(p['answer'], 'exam_answer', p['id'],
                           extra_before='[QUESTION]\n' + p['question'] + '\n[ANSWER]\n')
    images = collections.OrderedDict()
    for u in uses:
        rec = images.setdefault(u.pop('image'), {'uses': []})
        rec['uses'].append(u)
    image_recs = []
    for img, rec in images.items():
        kinds = {u['kind'] for u in rec['uses']}
        image_recs.append({'image': img, 'source': args.source, 'n_uses': len(rec['uses']),
                           'in_exam_question': 'exam_question' in kinds,
                           'in_exam_answer': 'exam_answer' in kinds,
                           'in_module': 'module' in kinds,
                           'uses': rec['uses']})

    # --- write ---
    def dump(name, recs):
        with open(os.path.join(args.out, name), 'w') as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    dump('chunks.jsonl', chunks)
    dump('exam_pairs.jsonl', pairs)
    dump('image_refs.jsonl', image_recs)

    # --- report ---
    L = []
    L.append('modules parsed: %d' % len(mods))
    L.append('chunks before dedup: %d   unique: %d' % (raw_count, len(chunks)))
    L.append('chunks dropped as near-empty: %d   as template placeholders: %d' % (
        report['dropped_tiny'], report['dropped_placeholder']))
    L.append('modules with chunks of type "other": %s' % sorted(
        {c['locations'][0]['module'] for c in chunks if c['type'] == 'other'}))
    L.append('chunks by type: %s' % dict(collections.Counter(c['type'] for c in chunks)))
    sizes = sorted(c['n_chars'] for c in chunks)
    if sizes:
        L.append('chunk size (raw chars) min/median/p90/max: %d / %d / %d / %d' % (
            sizes[0], sizes[len(sizes) // 2], sizes[int(len(sizes) * .9)], sizes[-1]))
    L.append('chunks containing images: %d' % sum(1 for c in chunks if c['images']))
    L.append('question chunks with SELEC tag: %d' % sum(1 for c in chunks if c['selec']))
    L.append('')
    L.append('exam pairs by status: %s' % dict(collections.Counter(p['status'] for p in pairs)))
    ok = [p for p in pairs if p['status'] == 'ok']
    L.append('ok pairs with question image: %d   with answer image: %d   text-only: %d' % (
        sum(1 for p in ok if p['question_images']), sum(1 for p in ok if p['answer_images']),
        sum(1 for p in ok if not p['question_images'] and not p['answer_images'])))
    L.append('ok pairs in an exammake pool: %d   variant groups: %d' % (
        sum(1 for p in ok if p['pools']), len({p['variant_group'] for p in ok})))
    L.append('')
    L.append('unique images referenced: %d  (module %d, exam question %d, exam answer %d)' % (
        len(image_recs), sum(r['in_module'] for r in image_recs),
        sum(r['in_exam_question'] for r in image_recs),
        sum(r['in_exam_answer'] for r in image_recs)))
    L.append('')
    L.append('chapter titles seen (count):')
    for t, n in report['chapters'].most_common():
        L.append('  %4d  %s%s' % (n, t, '   [dropped]' if t in DROP_CHAPTERS else ''))
    unused = sorted(set(os.path.basename(p) for p in glob.glob(os.path.join(args.content, '*.latex')))
                    - set(expander.used) - {os.path.basename(m) for m in mods})
    L.append('')
    L.append('.latex files never included by any module: %d' % len(unused))
    L.append('  ' + ' '.join(unused))
    L.append('')
    L.append('warnings (%d):' % len(warnings))
    L += ['  ' + w for w in warnings]
    with open(os.path.join(args.out, 'parse_report.txt'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L[:16]))


if __name__ == '__main__':
    sys.exit(main())
