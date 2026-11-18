# Pipeline diagrams

One mermaid sequence diagram per stage of the final pipeline (31B-teacher run). The Claude-teacher first run is told in the write-up, not drawn here. `index.html` shows all four in a browser; it embeds the `.mmd` files, so rebuild it after editing one: `python3 -I docs/diagrams/build_index.py`.

PNG for the website, on a machine with Node.js:
```
for f in *.mmd; do npx -p @mermaid-js/mermaid-cli mmdc -i "$f" -o "${f%.mmd}.png" -s 2 -b white; done
```
Without Node.js: paste a file into https://mermaid.live and export PNG.
