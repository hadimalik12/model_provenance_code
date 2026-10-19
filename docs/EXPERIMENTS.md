# Submitted experiment map

The submitted paper has three main result tables and six appendix tables. The
repository uses purpose-based directory names so appendix numbering changes do
not rename code again.

| Paper location | Experiment | Repository package | Reproduction status |
|---|---|---|---|
| Table 1 | Watermark LLM provenance at final Dolly/UltraChat checkpoints | `experiments/watermark_inheritance/` | Complete code and results |
| Appendix Table 4 | Full Dolly watermark trajectory | `experiments/watermark_inheritance/` | Complete code and results |
| Appendix Table 5 | Full UltraChat watermark trajectory | `experiments/watermark_inheritance/` | Complete code and results |
| Table 2, public derivatives | Fourteen public Pythia targets | `experiments/public_pythia_derivatives/` | Complete code; submitted values bundled |
| Appendix Table 6 | Model-level public Pythia results | `experiments/public_pythia_derivatives/` | Complete code; submitted values bundled |
| Table 2, INT8 | Six quantized Pythia targets | `experiments/pythia_int8_quantization/` | Complete code; submitted values bundled |
| Appendix Table 7 | Model-level Pythia INT8 results | `experiments/pythia_int8_quantization/` | Complete code; submitted values bundled |
| Table 3, public ViTs | Six public derivatives plus controls | `experiments/vit_provenance/` | Complete CLV audit code; submitted values bundled |
| Appendix Table 8 | Model-level public ViT results | `experiments/vit_provenance/` | Complete CLV audit code; submitted values bundled |
| Table 3, transformations | ViT merging, quantization, and pruning | `experiments/vit_provenance/` | Submitted values bundled |
| Appendix Table 9 | Model-level ViT transformation results | `experiments/vit_provenance/` | Original checkpoint-construction source unavailable |

The Appendix Table 9 PDF specifies the transformations and results but not the
three exact merge inputs or all merge hyperparameters. No transformation source
was present in this repository or the teammate repository used to recover the
watermark experiment. The repository therefore publishes the submitted result
table and exact downstream CLV audit, but does not claim an invented
transformation pipeline is an exact reproduction.

## Removed development studies

The publication cleanup removed four independent studies after checking the
submitted PDF and code dependencies:

- parent-only Pythia signal experiments (the retained Pythia runner already
  performs the required parent sanity checks);
- WikiMIA score selection (WikiMIA is absent from the submitted paper);
- multi-domain MIMIR augmentation (the paper uses only MIMIR GitHub); and
- InstructCoder controlled provenance (unrelated to Appendix Table 8, which is
  the public-ViT CLV experiment).

Historical generated logs and superseded table-numbered wrappers were also
removed. Git history remains the source for that development material.
