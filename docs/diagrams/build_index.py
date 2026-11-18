#!/usr/bin/env python3
"""Rebuild index.html from the .mmd files in this folder: python3 -I docs/diagrams/build_index.py"""
import html
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TITLES = {'stage1_documents': '1. Documents: ModEL → clean text',
          'stage2_eval_set': '2. Eval set: questions, retrieval, key points, judge',
          'stage3_training_data': '3. Training data: Sonnet questions, Gemma 31B answers',
          'stage4_train_evaluate': '4. Train and evaluate'}

parts = []
for name, title in TITLES.items():
    with open(os.path.join(HERE, f'{name}.mmd')) as f:
        src = f.read()
    parts.append(f'<h2>{title}</h2>\n<p class="src">{name}.mmd</p>\n<pre class="mermaid">\n{html.escape(src)}</pre>')
page = f'''<!doctype html>
<html><head><meta charset="utf-8"><title>RAFT-001 pipeline diagrams</title>
<style>body{{max-width:1000px;margin:2em auto;font-family:sans-serif;padding:0 1em}}
.src{{color:#777;font-size:0.85em;margin-top:-0.8em}}.mermaid{{margin:1em 0 3em}}</style>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
mermaid.initialize({{startOnLoad: true}});
</script></head><body>
<h1>RAFT-001 pipeline (final, 31B-teacher run)</h1>
<p>Generated from the .mmd files in this folder by build_index.py. Needs internet (mermaid from jsdelivr).</p>
{chr(10).join(parts)}
</body></html>
'''
with open(os.path.join(HERE, 'index.html'), 'w') as f:
    f.write(page)
print('wrote', os.path.join(HERE, 'index.html'))
