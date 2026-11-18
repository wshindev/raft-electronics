#!/usr/bin/env python3
"""Step 4: convert chunk and exam LaTeX into clean text with figure placeholders.

Text becomes light Markdown (*italic*, **bold**, `code`, - lists, | tables |,
fenced verbatim). Math stays LaTeX ($...$, $$...$$). Each
figure becomes a tagged block, for now built from the step 02 labels:
    [Figure image_2041: drawing; labels: V1, R1, R2]

Usage:
    uv run python3 -I pipeline/clean_text.py --data data/model

Reads chunks.jsonl, exam_pairs.jsonl, image_meta.jsonl from --data and writes
chunks_clean.jsonl, exam_pairs_clean.jsonl and clean_report.txt next to them.
"""
import argparse
import collections
import json
import os
import re
import unicodedata

MAX_LABELS = 30

KIND_NAMES = {'xcircuit': 'drawing', 'photo': 'photograph', 'plot': 'plot',
              'math_figure': 'math figure', 'pcb': 'PCB layout'}

# commands whose single {argument} is wrapped in Markdown markers
WRAP = {'textit': '*', 'textsl': '*', 'textbf': '**', 'texttt': '`', 'sout': '~~',
        'textsc': '', 'underline': '', 'centerline': '', 'hbox': '', 'vbox': ''}
# declarations that format the rest of their group: {\bf 22.3 mA}
DECL = {'bf': '**', 'it': '*', 'tt': '`', 'sl': '*', 'large': '', 'Large': ''}
# commands with one argument that is dropped
DROP_ARG = {'index', 'selec', 'label', 'color', 'ref', 'pageref'}
# commands without arguments: paragraph break or nothing
PARA = {'par', 'filbreak', 'eject', 'pagebreak', 'medskip', 'bigskip', 'smallskip',
        'vfil', 'vfill', 'appendix'}
NOTHING = {'noindent', 'hfil', 'hfill', 'raggedright', 'narrower', 'begingroup',
           'endgroup', 'obeyspaces', 'offinterlineskip', 'strut', 'hrule', 'vrule',
           'quad', 'qquad', 'relax', 'cr', 'extrarowheight'}
TEXT_SYMBOLS = {'%': '%', '#': '#', '&': '&', '_': '_', '$': '\\$', '{': '{', '}': '}',
                ' ': ' ', 'textbackslash': '\\', 'backslash': '\\', 'LaTeX': 'LaTeX',
                'bullet': '•', 'Omega': 'Ω', 'omega': 'ω', 'mu': 'μ', 'times': '×',
                'pm': '±', 'approx': '≈', 'Delta': 'Δ', 'pi': 'π', 'circ': '°',
                'rightarrow': '→', 'leftarrow': '←', 'surd': '√', 'angle': '∠',
                'sharp': '♯', 'flat': '♭', 'Rightarrow': '⇒', 'cdot': '·', '\\': '\n',
                '~': '~', '^': '^', '"': '"'}  # accent marks with no letter after them

DIM = r'-?[\d.]+\s*(?:pt|in|cm|mm|em|ex|truein)'
SKIP_RE = re.compile(r'\\[vh]skip\s*' + DIM + r'(?:\s*plus\s*[\d.]*\s*fil+)?')
SIZE_RE = re.compile(r'\\(?:epsfxsize|epsfysize|hsize|extrarowheight|baselineskip)\s*=?\s*' + DIM)
FIG_RE = re.compile(r'\\(?:includegraphics(?:\[[^\]]*\])?|epsfbox)\s*\{(image_\d+)\.eps\}'
                    r'(?:\s*\\imgnote\{([^{}]*)\})?')
# accents: \'e \'{e} -> é (letter + combining mark, then NFC)
ACCENT_ARG_RE = re.compile(r"\s*(?:\{(\w)\}|(\w))")
ACCENTS = {"'": '\u0301', '`': '\u0300', '^': '\u0302', '~': '\u0303', '"': '\u0308'}
FIG_TOKEN = '\x00FIG{}\x00'
FIG_TOKEN_RE = re.compile('\x00FIG(image_\\d+)\x00')


class Converter:
    def __init__(self, meta):
        self.meta = meta
        self.leftover = collections.Counter()  # unknown commands outside math
        self.figures = collections.Counter()   # placeholder kinds written
        self.notes = {}                        # image -> \imgnote text

    # ---------------- figures ----------------
    def figure(self, img):
        m = self.meta.get(img)
        note = self.notes.get(img)
        if m is None:
            self.figures['unavailable'] += 1
            return f'[Figure {img}: figure unavailable]'
        parts = [KIND_NAMES.get(m.get('image_kind'), 'figure')]
        labels = list(dict.fromkeys(l.strip().replace('$', '\\$')
                                    for l in (m.get('labels') or []) if l.strip()))
        if labels:
            more = f' (+{len(labels) - MAX_LABELS} more)' if len(labels) > MAX_LABELS else ''
            parts.append('labels: ' + ', '.join(labels[:MAX_LABELS]) + more)
        if note:
            parts.append('note: ' + note.strip())
        self.figures['labels' if labels else 'kind_only'] += 1
        return f'[Figure {img}: ' + '; '.join(parts) + ']'

    # ---------------- top level ----------------
    def convert(self, text):
        def fig_sub(m):
            if m.group(2):
                self.notes[m.group(1)] = m.group(2)
            return FIG_TOKEN.format(m.group(1))
        text = FIG_RE.sub(fig_sub, text)
        text = SIZE_RE.sub('', text)
        text = SKIP_RE.sub(' ', text)
        text = re.sub(r'\\item\s*\{\$\\bullet\$\}', r'\\item ', text)
        out = self.run(text, 0, len(text), env=[])
        out = FIG_TOKEN_RE.sub(lambda m: '\n\n' + self.figure(m.group(1)) + '\n\n', out)
        return tidy(out)

    # ---------------- scanner ----------------
    def run(self, s, i, end, env):
        """Convert s[i:end]. env is the stack of open list environments."""
        out = []
        while i < end:
            c = s[i]
            if c == '$':
                j, piece = self.math(s, i, end)
                out.append(piece)
                i = j
            elif c == '\\':
                j, piece = self.command(s, i, end, env)
                out.append(piece)
                i = j
            elif c == '{':
                j = match_brace(s, i, end)
                out.append(self.group(s, i + 1, j, env))
                i = j + 1
            elif c == '}':
                i += 1  # stray brace
            elif c == '~':
                out.append(' ')
                i += 1
            elif s.startswith('``', i) or s.startswith("''", i):
                out.append('"')
                i += 2
            else:
                out.append(c)
                i += 1
        return ''.join(out)

    def group(self, s, i, j, env):
        m = re.match(r'\s*\\([A-Za-z]+)(?![A-Za-z])', s[i:j])
        if m and m.group(1) in DECL:
            mark = DECL[m.group(1)]
            inner = self.run(s, i + m.end(), j, env).strip()
            return f'{mark}{inner}{mark}' if inner else ''
        return self.run(s, i, j, env)

    def math(self, s, i, end):
        if s.startswith('$$', i):
            start = i + 2
            box = re.match(r'\s*\\[vh]box\s*\{', s[start:end])
            if box:  # table: its cells may hold inline math, e.g. $\cdot$$^{o}$
                start = match_brace(s, start + box.end() - 1, end)
            j = s.find('$$', start, end)
            j = end if j < 0 else j
            body = s[i + 2:j]
            if '\\halign' in body:
                return j + 2, '\n\n' + self.halign(body) + '\n\n'
            figs = FIG_TOKEN_RE.findall(body)
            if figs:  # figure block, possibly several side by side
                return j + 2, ''.join(FIG_TOKEN.format(f) for f in figs)
            return j + 2, '$$' + body.strip() + '$$'
        j = i + 1
        while j < end and not (s[j] == '$' and s[j - 1] != '\\'):
            j += 1
        return j + 1, s[i:j + 1]

    def command(self, s, i, end, env):
        m = re.match(r'\\([A-Za-z]+|.)', s[i:end], re.S)
        if m is None:  # lone backslash at the end
            return i + 1, ''
        name, j = m.group(1), i + m.end()
        if name in ACCENTS:
            a = ACCENT_ARG_RE.match(s, j, end)
            if a:
                return a.end(), unicodedata.normalize('NFC', (a.group(1) or a.group(2)) + ACCENTS[name])
        if name in ('begin', 'end'):
            g = re.match(r'\s*\{([^}]*)\}', s[j:end])
            envname, j = g.group(1), j + g.end()
            return self.environment(s, j, end, env, name, envname)
        if name == 'item':
            depth = len(env)
            indent = '  ' * max(depth - 1, 0)
            if env and env[-1][0] == 'enumerate':
                env[-1][1] += 1
                return j, f'\n{indent}{env[-1][1]}. '
            return j, f'\n{indent}- '
        if name in ('subitem', 'itemitem'):
            return j, '\n  - '
        if name in ('subsection', 'subsubsection', 'section'):
            j, arg = self.arg(s, j, end, env)
            level = {'section': '#', 'subsection': '##', 'subsubsection': '###'}[name]
            return j, f'\n\n{level} {arg.strip()}\n\n'
        if name == 'footnote':
            j, arg = self.arg(s, j, end, env)
            return j, f' [Footnote: {arg.strip()}]'
        if name == 'underbar':
            j, arg = self.arg(s, j, end, env)
            return j, arg if arg.strip() else '_____'
        if name == 'framebox':
            j = skip_opt(s, j, end)
            j, arg = self.arg(s, j, end, env)
            return j, f'[{arg.strip()}]'
        if name == 'imgnote':  # note not attached to a figure
            j, arg = self.arg(s, j, end, env)
            return j, f' (note: {arg.strip()})'
        if name in WRAP:
            j, arg = self.arg(s, j, end, env)
            mark = WRAP[name]
            return j, f'{mark}{arg.strip()}{mark}' if arg.strip() else ''
        if name in DECL:  # declaration outside a group: no scope to format
            return j, ''
        if name in DROP_ARG:
            j, _ = self.arg(s, j, end, env)
            return j, ''
        if name in ('setlength', 'rule'):
            j, _ = self.arg(s, j, end, env)
            j, _ = self.arg(s, j, end, env)
            return j, ''
        if name == 'noalign':
            j, _ = self.arg(s, j, end, env)
            return j, ''
        if name in PARA:
            return j, '\n\n'
        if name in NOTHING:
            return j, ' '
        if name in TEXT_SYMBOLS:
            return j, TEXT_SYMBOLS[name]
        self.leftover[name] += 1
        return j, '\\' + name

    def arg(self, s, j, end, env):
        k = j
        while k < end and s[k] in ' \t\n':
            k += 1
        if k < end and s[k] == '{':
            close = match_brace(s, k, end)
            return close + 1, self.group(s, k + 1, close, env)
        return j, ''

    def environment(self, s, j, end, env, which, name):
        if which == 'end':
            if name in ('itemize', 'enumerate') and env:
                env.pop()
            return j, '\n\n'
        if name == 'verbatim':
            k = s.find('\\end{verbatim}', j, end)
            k = end if k < 0 else k
            body = s[j:k].strip('\n')
            return k + len('\\end{verbatim}'), f'\n\n```\n{body}\n```\n\n'
        if name == 'tabular':
            k = s.find('\\end{tabular}', j, end)
            k = end if k < 0 else k
            j2 = j
            g = re.match(r'\s*\{', s[j:k])
            if g:  # column spec
                j2 = match_brace(s, j + g.end() - 1, k) + 1
            return k + len('\\end{tabular}'), '\n\n' + self.table(s[j2:k], r'\\\\(?:\[[^\]]*\])?') + '\n\n'
        if name in ('itemize', 'enumerate'):
            env.append([name, 0])
            return j, '\n'
        if name in ('quote', 'quotation', 'framed', 'center'):
            return j, '\n\n'
        self.leftover['env:' + name] += 1
        return j, '\n\n'

    # ---------------- tables ----------------
    def halign(self, body):
        i = body.find('\\halign')
        k = body.find('{', i)
        close = match_brace(body, k, len(body))
        inner = body[k + 1:close]
        rows = re.split(r'\\cr\b', inner)
        rows = [r for r in rows if '#' not in r]  # drop the template row
        return self.table('\\cr'.join(rows), r'\\cr\b')

    def table(self, src, row_sep):
        src = re.sub(r'\\noalign\s*\{[^{}]*\}|\\hline|\\cline\{[^}]*\}', ' ', src)
        out = []
        for row in re.split(row_sep, src):
            if not row.strip():
                continue
            cells = [tidy(self.run(c, 0, len(c), [])).replace('\n', ' ') for c in re.split(r'(?<!\\)&', row)]
            if not any(cells):
                continue
            out.append('| ' + ' | '.join(cells) + ' |')
            if len(out) == 1:
                out.append('|' + '---|' * len(cells))
        return '\n'.join(out)


def match_brace(s, i, end):
    """Index of the brace closing s[i] == '{' (or end)."""
    depth = 0
    k = i
    while k < end:
        c = s[k]
        if c == '\\':
            k += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return k
        k += 1
    return end


def skip_opt(s, j, end):
    m = re.match(r'\s*\[[^\]]*\]', s[j:end])
    return j + m.end() if m else j


def tidy(text):
    """Collapse whitespace, except inside fenced verbatim blocks."""
    parts = re.split(r'(\n```\n.*?\n```\n)', '\n' + text + '\n', flags=re.S)
    return '\n\n'.join(p.strip('\n') if i % 2 else tidy_prose(p)
                         for i, p in enumerate(parts) if p.strip()).strip()


def tidy_prose(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='data/<source> folder from steps 01-02')
    args = ap.parse_args()
    d = args.data
    meta = {m['image']: m for m in load(os.path.join(d, 'image_meta.jsonl'))}
    conv = Converter(meta)

    chunks = load(os.path.join(d, 'chunks.jsonl'))
    lens = []
    with open(os.path.join(d, 'chunks_clean.jsonl'), 'w') as f:
        for c in chunks:
            text = conv.convert(c['text'])
            lens.append(len(text))
            f.write(json.dumps({**c, 'text': text, 'n_chars': len(text)}) + '\n')
    lens.sort()
    chunk_figs = sum(conv.figures.values())

    pairs = load(os.path.join(d, 'exam_pairs.jsonl'))
    with open(os.path.join(d, 'exam_pairs_clean.jsonl'), 'w') as f:
        for x in pairs:
            f.write(json.dumps({**x, 'question': conv.convert(x['question']),
                                'answer': conv.convert(x['answer'])}) + '\n')

    rep = [f'chunks: {len(chunks)}  exam pairs: {len(pairs)}',
           f'clean chunk chars: median {lens[len(lens) // 2]:,}  p90 {lens[int(len(lens) * .9)]:,}  '
           f'max {lens[-1]:,}  total {sum(lens):,}',
           f'figure placeholders: {chunk_figs} in chunks, '
           f'{sum(conv.figures.values()) - chunk_figs} in exam pairs',
           'placeholder kinds: ' + ', '.join(f'{k} {v}' for k, v in conv.figures.most_common()),
           'unknown commands left in text (outside math), count:']
    rep += [f'  \\{k} {v}' for k, v in conv.leftover.most_common()]
    with open(os.path.join(d, 'clean_report.txt'), 'w') as f:
        f.write('\n'.join(rep) + '\n')
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
