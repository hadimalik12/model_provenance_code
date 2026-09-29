# Synthetic MQAR Provenance Experiment

## Goal

Replicate the Zoology-style MQAR setup in a provenance experiment.

The question is whether a target model still carries signal from a synthetic
secret shard `S` that was used to train its parent model.

## Data Format

Each sample is one synthetic MQAR sequence. A sequence first defines several
key-value pairs, then repeats some of the keys later in the sequence. The model
should predict the matching value after each repeated key.

Example:

```text
A 7 B 4 C 9 X X X B 4 X X A 7
```

In this example:

```text
A 7, B 4, C 9
```

are the key-value pairs.

```text
X X X
```

are filler tokens.

The repeated keys are:

```text
B
A
```

The correct values after those repeated keys are:

```text
4
7
```

So the model is trained and audited on whether it predicts the correct value
after a repeated key.

## Datasets

Generate three disjoint datasets from the same MQAR generator:

```text
S:  100,000 synthetic MQAR sequences
S': 100,000 synthetic MQAR sequences
U:  100,000 synthetic MQAR sequences
```

`S`, `S'`, and `U` should use different random seeds.

`S` is the secret shard. It is used to train the parent.

`S'` is the control shard. It is generated the same way as `S`, but is never
used for training.

`U` is the target fine-tuning shard. It is used to train the target from the
parent.

## Models

Use the same base model families as the current language-model experiments.

Pythia family:

```text
Base checkpoints:
EleutherAI/pythia-1b
EleutherAI/pythia-1.4b
EleutherAI/pythia-6.9b
EleutherAI/pythia-12b
```

OPT family:

```text
Base checkpoints:
facebook/opt-125m
facebook/opt-350m
facebook/opt-1.3b
facebook/opt-2.7b
facebook/opt-6.7b
```

For the synthetic MQAR experiment, run the training pipeline separately from
each base checkpoint:

```text
B_pythia_1b   = EleutherAI/pythia-1b
B_pythia_1.4b = EleutherAI/pythia-1.4b
B_pythia_6.9b = EleutherAI/pythia-6.9b
B_pythia_12b  = EleutherAI/pythia-12b
B_opt_125m    = facebook/opt-125m
B_opt_350m    = facebook/opt-350m
B_opt_1.3b    = facebook/opt-1.3b
B_opt_2.7b    = facebook/opt-2.7b
B_opt_6.7b    = facebook/opt-6.7b
```

The parent `P` and target `T` are both created by our own fine-tuning runs. We
do not use the public fine-tuned target checkpoints in this synthetic
experiment.

## Parent Training

Start with base model `B`.

Fine-tune `B` on all 100,000 sequences in `S`.

Save the result as the parent model `P`.

Fine-tuning parameters:

```text
Learning rate: TBD
Batch size: TBD
Number of epochs: TBD
Optimizer: TBD
Weight decay: TBD
Maximum sequence length: TBD
Learning-rate schedule: TBD
```

## Target Training

Start with parent model `P`.

Fine-tune `P` on all 100,000 sequences in `U`.

Do not train the target directly on `S` or `S'`.

Save the result as the target model `T`.

Fine-tuning parameters:

```text
Learning rate: TBD
Batch size: TBD
Number of epochs: TBD
Optimizer: TBD
Weight decay: TBD
Maximum sequence length: TBD
Learning-rate schedule: TBD
```

## Audit Data

The audit data is created from sequences in `S` and `S'`.

For each repeated key, cut the sequence right before the answer value.

Example full sequence:

```text
A 7 B 4 C 9 X X X B 4 X X A 7
```

Audit example 1:

```text
Input:  A 7 B 4 C 9 X X X B
Answer: 4
```

Audit example 2:

```text
Input:  A 7 B 4 C 9 X X X B 4 X X A
Answer: 7
```

The model is not judged by free-form generation. Instead, compute the
log-probability or loss of the known correct answer token.

## Audit Split

From `S`, create:

```text
3,000 calibration audit examples
3,000 final audit examples
```

From `S'`, create:

```text
3,000 calibration audit examples
3,000 final audit examples
```

The parent still trains on all of `S`. The calibration/final split is only for
choosing and evaluating the audit threshold.

## Distinguisher

Use answer-token log probability as the main distinguisher.

For each audit example, measure:

```text
How likely does the model think the correct answer value is?
```

Then compare the score distribution for `S` against the score distribution for
`S'`.

## Testing

Test the base model `B` on audit examples from `S` and `S'`.

Expected result: `B` should not prefer `S`.

Test the parent model `P` on audit examples from `S` and `S'`.

Expected result: `P` should score `S` better than `S'`, because `P` trained on
`S`.

Test the target model `T` on audit examples from `S` and `S'`.

Expected result: if provenance signal transfers from parent to target, then `T`
should still score `S` better than `S'`.

## Pipeline

```text
B -> train on S -> P -> train on U -> T
```

Run this once for each base checkpoint listed in the Models section.

Then audit:

```text
B on S vs S'
P on S vs S'
T on S vs S'
```

The main result is whether `T` still distinguishes `S` from `S'`.
