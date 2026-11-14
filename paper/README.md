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
  skeleton.tex              # standalone outline: every section, label, table and figure, no body text
  section_points.md         # the main points each section must make, in skeleton.tex's order
```

`skeleton.tex` compiles by itself (choose it as the main document in Overleaf) and has exactly
the same `\label`s as the full paper. Keep both outline files in step with `sections/` when a
revision pass adds, removes or renames a subsection.

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

**Second compile-log pass (Oct 2026).** The undefined-citation errors came from a *stale
`refs.bib` on Overleaf* (it matched commit `b335855`, before the Related Work citations), so
re-upload every file, not just the sections. Fixed in the repo at the same time:
- `refs.bib`: BibTeX parses `@` even inside `%%` comments, and a comment saying
  `@misc (not @article)` (which also sat above the wrong entry) broke parsing. Comments in
  `refs.bib` must never contain `@`. Csurka et al. listed "Meylan, France" (an affiliation) as an
  author. Added the verified page ranges (clDice, Csurka, RoadTracer) and venue addresses; the
  Sat2Graph page range is still left out because it couldn't be verified.
- Overfull boxes: `main.tex` now makes the text-mode `\_` breakable and sets
  `\emergencystretch=3em`; `::` in code paths is followed by `\allowbreak{}`; the tables use
  ragged-right `p{}` columns (`array` package); the heatmap-loss display uses $H$/$\ell^\pm$
  instead of long `\mathrm{gt\_heatmap}` names; one math-mode list of integers was rewritten as
  text; the `\medskip` acmart warned about was removed.

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

`figures/training_curves.png` (Figure 1, §10.4 Production Training Results) is drawn from the
tracked `results/scaled/plem_history.csv` and `baseline_history.csv`. The two older PNGs from
`notebooks/train_unet_joint.ipynb` (2-epoch loss curves and qualitative panels without points)
were removed in November 2026 together with the old §10.3/§10.5, on review: the paper now keeps
only the matched baseline comparison. A production-model qualitative figure with intersections
(baseline vs. joint) is planned and needs test imagery from CARC's `data/`.

## Placeholder content still to fill in

- Front-matter placeholders that need real values before submission: `\acmDOI`, `\acmISBN` in
  `main.tex` (currently `XXXXXXX` placeholders, standard practice until a DOI is assigned), and
  the reference/citation entries in `refs.bib` that use `@misc` with a generic `howpublished`
  field (SpaceNet APLS, ISPRS Potsdam) — replace with the actual venue/URL once you pin down the
  canonical citation for each.

## Source of truth

Since November 2026 this LaTeX project is the master copy of the paper. It started as a
transcription of `PAPER_OUTLINE.md` (repo root), which is now frozen as a planning record and is
no longer updated. Revise the paper here only. The code and the recorded results (`results/`,
notebook outputs) remain the authority for any number or claim.

## Revision notes

The paper is being revised one section at a time (body first, then §2, then §1/§12/abstract,
then appendices). Items found in one section that need a change elsewhere are listed here until
that section's pass:

- §2.2, §2.3 and §2.6 now have one or two sentences of prose each, using only the citations
  already there. They still need more related work and citations.