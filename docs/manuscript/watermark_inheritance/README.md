# Controlled watermark inheritance: manuscript Section 5.3.1

[`section.tex`](section.tex) is an insertion-ready subsection with separate
run-by-run Dolly and UltraChat tables. It follows the supplied experiment
section's `\myparatight`, `\Cref`, `\citep`, `table*`, `booktabs`, signed-gap
notation, and `Verdict / Correct?` conventions.

## Manuscript integration

This repository contains an older `Causal_Model_Provenance.pdf`, whose LLM
experiments are Section 4.2, but no manuscript `.tex`, `.bib`, or document class.
The author-supplied newer experiment-section source and `Model_Provenance.pdf`
place these experiments in Section 5.3 and include the INT8 subsection.

In that newer manuscript, insert the contents of `section.tex` immediately
after the `\end{table*}` belonging to `\label{tab:target_results}` and before
`\subsubsection{Robustness to INT8 Quantization}`. Alternatively, copy this
folder into the manuscript project and insert:

```latex
\input{docs/manuscript/watermark_inheritance/section}
```

The new subsection becomes 5.3.1 and INT8 becomes 5.3.2 automatically. Preserve
`subsubsec:llm-int8-quantization`; label-based references will follow the new
numbering. The new tables follow the target-results table in source order;
LaTeX may move the two-column floats. No counters or preamble definitions are
changed by the fragment.

The manuscript PDF already lists Sander et al. (2024) as reference [24]. The
actual BibTeX key is unavailable. The fragment uses `sander2024watermarking`;
map this to the manuscript's existing entry, or add the supplied
[`references.bib`](references.bib) entry if absent. Do not create a duplicate.
The entry matches the PDF and the [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2567c95fd41459a98a73ba893775d22a-Abstract-Conference.html).

## Source data and interpretation

The files in `data/` are byte-for-byte snapshots from
[`ngocbh/model_provenance_code` at `08237f7`](https://github.com/ngocbh/model_provenance_code/tree/08237f717acd5d954b0dbd205686134843205f91/experiments/watermark_inheritance).
[`source_manifest.json`](data/source_manifest.json) records full paths, commit,
and SHA-256 hashes. Training code and unrelated upstream changes are not needed
to use this fragment.

- `dolly.csv` preserves `results/2026-10-06/main_results.csv`. Targets `T1`,
  `T2`, and `T` are epochs 1, 2, and 3. Only `T0` at epoch 3 was audited;
  that same control is repeated beside all three target epochs. The source's
  base and parent rows are retained in the CSV but omitted from the Dolly table.
- `ultrachat.csv` preserves `results/2026-10-06-distinct/curve.csv`, the
  **single-pass full UltraChat** trajectory, not the earlier 200k-corpus run.
  The table includes the base/parent baseline and six matched checkpoints per
  run through 12,500 steps = 400,000 unique examples.
- `ultrachat_snapshot.json` records the publication at 2026-10-06 14:00:58 UTC:
  **36 of 42 planned downstream audits** are complete. The six baseline rows
  make the CSV contain 42 rows; they do not make all planned audits complete.
  The 18,750-step / 600,000-example endpoint is not reported as completed.

Table values use the watermark detector's acceptance rates `a` and `a_prime`:
signed gap = `a - a_prime`, advantage = `abs(a - a_prime)`, balanced accuracy =
`(1 + a - a_prime) / 2`. The fixed calibration direction is retained, including
control accuracies below 50%. Display rounding uses decimal half-up (two
decimal places for percentage accuracies and four for gaps); decisions use
unrounded values. The CSV's `threshold` / `watermark_threshold` is the calibrated
**score cutoff**, not the hypothesis-test critical value.

For paired observations, write `D_i = A(S_i) - A(S'_i)` in `[-1, 1]`. Conditional
on the audited model and calibration, independent, mean-zero pair differences
under the null give `Pr(|mean(D)| > t) <= 2 exp(-m t^2 / 2)`. Thus the conservative
paired critical value is `sqrt(2 log(2/0.05) / 8000) = 0.0303680731`.
It is not the manuscript's independent-shard critical value. The exact
within-pair label-swap test additionally requires exchangeability under the
null and gives the same verdicts for all reported watermark measurements.
Reusing audit pairs across checkpoints gives descriptive comparisons, not
simultaneous error control or a completed final-endpoint test. Changing both
corpus and exposure does not identify a dataset-size effect.

Upstream records independently validate the
[Dolly scores](https://github.com/ngocbh/model_provenance_code/blob/08237f717acd5d954b0dbd205686134843205f91/experiments/watermark_inheritance/results/2026-10-06/scoring_validation.json),
[Dolly training](https://github.com/ngocbh/model_provenance_code/blob/08237f717acd5d954b0dbd205686134843205f91/experiments/watermark_inheritance/results/2026-10-06/training_validation.json),
and [available UltraChat scores](https://github.com/ngocbh/model_provenance_code/blob/08237f717acd5d954b0dbd205686134843205f91/experiments/watermark_inheritance/results/2026-10-06-distinct/curve_scoring_validation.json).
The local check below verifies the published aggregates and table transcription;
it does not rerun GPU experiments or reconstruct unavailable per-example scores.

## Validation and preview

From the repository root, with Python 3.9+ and no third-party dependencies:

```bash
python3 docs/manuscript/watermark_inheritance/verify_results.py
```

This checks snapshot hashes, all 60 source measurements, matched exposures,
the incomplete endpoint, exact paired p-values recomputed from saved counts,
and all 30 displayed rows. Use `--write-tables` to regenerate the marked table
bodies after an intentional source update.

To create a standalone preview for a LaTeX editor:

```bash
python3 docs/manuscript/watermark_inheritance/verify_results.py \
  --preview /tmp/watermark-inheritance-preview.tex
```

The self-contained preview uses a two-column article, the fragment's original
tables, and an inline bibliography. It checks compilation without requiring the
unavailable manuscript source, bibliography database, or AISTATS class; final
pagination and the existing bibliography key still require the full manuscript.
