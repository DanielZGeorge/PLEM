# PLEM paper — LaTeX project

This directory is the LaTeX transcription of `PAPER_OUTLINE.md`, ready to upload to a new Overleaf
project (Overleaf itself does the compiling — see below for why).

## Layout

```
paper/
  main.tex                  # \documentclass[sigconf]{acmart}, front matter, \input's each section
  sections/                 # 00_abstract.tex ... 12_conclusion.tex, one file per outline section
  appendix/                 # appendix_a_proofs.tex, appendix_b_complexity.tex
  refs.bib                  # BibTeX entries for every \cite in the paper
  figures/                  # (empty for now — see "Figures" below)
```

## `acmart.cls` is *not* vendored here

The plan for this milestone was to vendor `acmart.cls`/`ACM-Reference-Format.bst` directly into
this directory so the project is self-contained. That step was skipped deliberately: this
environment has no direct internet access from its shell (only a summarizing web-fetch tool that
converts pages to Markdown, which would corrupt a raw `.cls` file rather than vendor it
byte-for-byte), so there was no reliable way to pull the exact CTAN package contents without risking
a subtly-broken class file being silently committed.

This is **not a blocker**: Overleaf's own TeX Live distribution already ships the `acmart` package
(it's one of the most common conference templates on the platform), so
`\documentclass[sigconf]{acmart}` in `main.tex` resolves correctly with zero setup on Overleaf's
side. If you want a fully offline-buildable, vendored copy instead (e.g. for a local LaTeX
toolchain), download `acmart.cls` and `ACM-Reference-Format.bst` yourself from
[CTAN's acmart package page](https://ctan.org/pkg/acmart) and drop them in this directory —
`main.tex` doesn't need any changes either way, since both a vendored copy and Overleaf's own
package resolution satisfy the same `\documentclass`/`\bibliographystyle` calls.

## No local LaTeX toolchain in this environment

No `pdflatex`/`latexmk`/`bibtex`/`biber` binary is installed in this environment (checked via
`which`/`where`, none found), so this project's `.tex`/`.bib` files can only be checked by
`\label`/`\ref`/`\cite` cross-referencing and brace/environment balance scripts, not an actual
compile — the real compile happens on Overleaf.

**A real Overleaf compile has been run and its diagnostics addressed** (`output.log`/`output.blg`/
`output.chktex` reviewed from an Overleaf compile output download). What was found and fixed:
- `\usepackage{amssymb}` in `main.tex` caused a fatal `Command \Bbbk already defined` error
  (`pdflatex` exit code 1) — acmart already loads `amssymb` internally, and re-loading it clashes
  because `\Bbbk` is defined with `\newcommand` (unlike its other symbols, which use
  `\DeclareMathSymbol` and merely warn on redeclaration). Removed the redundant load;
  `\usepackage{amsmath}` was left in place since it loaded cleanly with no conflict.
  **Despite this error, `pdflatex` actually produced a complete 16-page PDF** — the nonzero exit
  code came from the error counter, not a failure to render — but the error is real and worth
  fixing for a clean compile regardless.
- `sections/10_results.tex`'s Table~\ref{tab:joint-test} caption interrupted `\texttt{...}` with
  `\textrm{...}` mid-parenthetical in a way that confused chktex's bracket matcher (2 flagged
  "errors" in `output.chktex`); simplified to avoid the font-switching interruption entirely.
- Several `refs.bib` entries triggered BibTeX warnings (empty publisher/address, missing pages) —
  added `publisher` fields for venues where it's unambiguous (IEEE for CVPR/3DV, Springer for
  ECCV/MICCAI, BMVA Press for BMVC); left page ranges as an explicit `TODO` comment rather than
  guessing exact values. `zhou2019centernet` was reclassified `@article` → `@misc` with
  `eprint`/`archivePrefix` fields, since it's an arXiv preprint with no journal volume/number —
  the "no number and no volume" warning was a symptom of the wrong entry type, not missing data.
- Added `\Description{...}` alt-text to both figures per acmart's own accessibility warning
  ("Some images may lack descriptions").

Not touched (cosmetic/expected, not compile errors): a few `Overfull \hbox` warnings from long
inline code identifiers in narrow single-column text, and a `Package balance Warning: You have
called \balance in second column` — the latter is acmart's own automatic end-of-document call for
sigconf's two-column layout, not something this project's `.tex` files invoke.

If you re-run a compile and hit something new, treat Overleaf's own log as ground truth and fix
forward the same way — see `CLAUDE.md`'s `paper/` entry for how this round was resolved.

## Uploading to Overleaf

1. Create a new Overleaf project → **Upload Project** → zip this `paper/` directory (or its
   contents) and upload.
2. Overleaf should auto-detect `main.tex` as the root file; if not, set it manually in the
   project's menu (⚙ → "Main document").
3. Compile. If `acmart.cls` fails to resolve for any reason, download it from CTAN (see above) and
   upload it alongside `main.tex`.

## Figures

`figures/` currently has two PNGs extracted from `notebooks/train_unet_joint.ipynb`'s executed
output (`joint_training_subterm_curves.png`, `joint_qualitative_panels.png`), both referenced from
`sections/10_results.tex` §10.5 via `\includegraphics`. `notebooks/loss_ablation.ipynb`'s synthetic
ablation figures are not yet exported here — Section 10.4 currently reports its numbers as prose/
tables only; add figures from that notebook if the paper needs them visually illustrated too.

## Placeholder content still to fill in

- `sections/10_results.tex`, §10.5 (`\label{subsec:joint-results}`) is now filled in with real
  numbers from an executed run of `notebooks/train_unet_joint.ipynb` (153 tiles, 2 epochs,
  per-term loss curves, test-set scores by source, two embedded figures). What's still open there:
  an **epoch-matched comparison** against the CE+Dice baseline has not been run — the current
  §10.5 numbers cannot separate "the new loss underperforms" from "2 epochs on a harder task
  underperforms," flagged explicitly in §11/§12 as the highest-priority follow-up.
- Front-matter placeholders that need real values before submission: `\acmDOI`, `\acmISBN` in
  `main.tex` (currently `XXXXXXX` placeholders, standard practice until a DOI is assigned), and
  the reference/citation entries in `refs.bib` that use `@misc` with a generic `howpublished`
  field (SpaceNet APLS, ISPRS Potsdam) — replace with the actual venue/URL once you pin down the
  canonical citation for each.

## Source of truth

This LaTeX project is a transcription of `PAPER_OUTLINE.md` (repo root) — if the two ever
disagree, treat the code and `PAPER_OUTLINE.md` as authoritative (per `CLAUDE.md`) and update this
LaTeX project to match, not the other way around.
