# Table 8 source overlap audit

This was a direct comparison of downloaded content, not an inference from
release dates. The source was the official `EleutherAI/pile` converted Parquet
branch (`refs/convert/parquet`), all ten available
`github/partial/train/0000.parquet`–`0009.parquet` shards. Together they held
416,000 documents. The [Pile construction code](https://github.com/EleutherAI/the-pile/blob/master/the_pile/datasets.py)
lists 19,021,454 documents for the full GitHub component, so these shards
cover about 2.2% **by document count**. They are not asserted to be a random
sample. Pile-CC is a separate Common Crawl component and was not part of this
content comparison.

The entire CodeSearchNet Python **train** split was downloaded from
[`claudios/code_search_net`](https://huggingface.co/datasets/claudios/code_search_net)
at revision `a4ee8053e119ef04ee3d491ffef05bcf2725ae82` (three shards,
412,178 functions). The entire InstructCoder train file was downloaded from
[`likaixin/InstructCoder`](https://huggingface.co/datasets/likaixin/InstructCoder)
at revision `6a778a720284d6520b56bd03d5c3070930d41071` (108,391 rows).

## Prompt/response pair check

For the 243,945 CodeSearchNet train rows eligible under the old pilot's
length filters, all 243,945 normalized `(prompt, response)` pairs are unique.
Normalization strips surrounding whitespace and then removes whitespace
inside each field; it keeps case and punctuation. Matching was unrestricted
by repository. A complete field had to be present inside a Pile document;
short anchor hits alone were not counted. The matcher enumerated overlapping
patterns so that common or nested prompts were not lost.

| Disjoint category, against the 416,000 Pile GitHub documents | Unique pairs |
|---|---:|
| Both prompt and response occur | 1,330 |
| Prompt occurs, complete response does not | 2,005 |
| Complete response occurs, prompt does not | 0 |
| Neither occurs | 240,610 |

Every eligible CodeSearchNet prompt is embedded in its own response code as a
docstring. Consequently, each of the 1,330 complete-response hits also has
its prompt in the **same Pile file**, and the response-only category must be
zero. In total, 3,335 prompts and 1,330 responses occur. `Both` is the most
inclusive pair-overlap count *within these downloaded shards*; the Pile stores
raw text, not labeled prompt–response pairs. It is **not** an upper bound on
overlap with the full Pile. The count output is in
[`pile_github_vs_codesearchnet_pairs.json`](pile_github_vs_codesearchnet_pairs.json),
and the reproducible matcher is `overlap_pairs.py` in the parent directory.
That optional audit script requires `pyahocorasick`; it is not needed to run
Table 8 training or auditing.

| Comparison with the same 416,000 Pile GitHub documents | Observed matches |
|---|---:|
| CodeSearchNet function code verbatim inside a Pile file from the same repository | 591 / 412,178 |
| CodeSearchNet function code with whitespace ignored, same repository | 598 / 412,178 |
| CodeSearchNet old-pilot-eligible functions with whitespace ignored | 303 / 243,945 |
| InstructCoder rows with full input **or** output code present with whitespace ignored (repository unrestricted) | 4 / 108,391 |

Among the 640 CodeSearchNet functions whose repository occurred in the Pile
sample, 591 matched verbatim; among the 329 old-pilot-eligible functions in
those repositories, 303 matched after whitespace normalization. This verifies
that CodeSearchNet and Pythia's GitHub source really do intersect. The 303
count is a **lower bound** on overlap with the full GitHub component, not an
estimate of the full-corpus overlap percentage. It also says nothing about
possible matches in other Pile components.

The InstructCoder check searched all training inputs and outputs for a
distinctive anchor, then verified each candidate against the **complete**
whitespace-normalized code field; anchor-only hits were not counted. The four
known matching rows are excluded from Table 8's keyed selection by stable
record IDs. Absence of further matches in this partial sample is not proof of
absence from Pythia's full pretraining data.

## Filter used by Table 8

Table 8 now uses CodeSearchNet after a stricter, case-insensitive normalized
filter. From 243,937 eligible unique pairs, it removed every row whose complete
prompt or complete response occurred in the 416,000 checked Pile documents.
Normalization applies Unicode NFKC, case folding, and removal of all whitespace
and NUL characters. The filter removed 6,019 rows (including 1,332 full-response
matches) and retained 237,918 rows. The saved pool has zero detected matches
under this rule and is checksum-verified before jobs are submitted.

This means zero detected overlap with the checked sample under the stated
full-field rule. It does not prove zero overlap with the unobserved portion of
Pythia's approximately 95 GiB GitHub component, Pile-CC, or other components.
The raw counts and limited-scope labels are in
[`pile_github_vs_codesearchnet.json`](pile_github_vs_codesearchnet.json) and
[`pile_github_vs_instructcoder.json`](pile_github_vs_instructcoder.json).
The comparison/filter programs are in the parent directory.
