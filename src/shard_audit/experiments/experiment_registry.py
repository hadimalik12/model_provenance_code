"""Central experiment metadata for the paper table pipelines.

The repository contains both paper-replication experiments and augmented
variants.  This module keeps table names, model lists, score keys, and common
threshold calculations in one place so report scripts and sbatch jobs can point
at the same conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


ALPHA = 0.05


def gamma(alpha: float, m_eval_per_shard: int) -> float:
    """Finite-sample level-alpha threshold used in the paper."""
    return math.sqrt(math.log(2 / alpha) / m_eval_per_shard)


def slug(hf_id: str) -> str:
    """Filesystem-safe HuggingFace id."""
    return hf_id.replace("/", "__")


@dataclass(frozen=True)
class TargetModel:
    parent_name: str
    hf_id: str
    display_name: str


@dataclass(frozen=True)
class ParentModel:
    display_name: str
    hf_id: str


PYTHIA_PARENTS = (
    ParentModel("Pythia-1B", "EleutherAI/pythia-1b"),
    ParentModel("Pythia-1.4B", "EleutherAI/pythia-1.4b"),
    ParentModel("Pythia-6.9B", "EleutherAI/pythia-6.9b"),
    ParentModel("Pythia-12B", "EleutherAI/pythia-12b"),
)


PYTHIA_TARGETS = (
    TargetModel("Pythia-1B", "Leogrin/eleuther-pythia1b-hh-sft", "Leogrin Hh Sft"),
    TargetModel("Pythia-1.4B", "herMaster/pythia1.4B-finetuned-on-lamini-docs", "Hermaster1 4B Lamini Docs"),
    TargetModel("Pythia-1.4B", "kykim0/pythia-1.4b-tulu-v2-mix", "Kykim0 Tulu V2 Mix"),
    TargetModel("Pythia-1.4B", "LinguaCustodia/fin-pythia-1.4b", "Linguacustodia Fin"),
    TargetModel("Pythia-1.4B", "lomahony/pythia-1.4b-helpful-dpo", "Lomahony Helpful Dpo"),
    TargetModel("Pythia-1.4B", "lomahony/pythia-1.4b-helpful-sft", "Lomahony Helpful Sft"),
    TargetModel("Pythia-6.9B", "allenai/open-instruct-pythia-6.9b-tulu", "Allenai Tulu"),
    TargetModel("Pythia-6.9B", "lomahony/eleuther-pythia6.9b-hh-dpo", "Lomahony Hh Dpo"),
    TargetModel("Pythia-6.9B", "lomahony/eleuther-pythia6.9b-hh-sft", "Lomahony Hh Sft"),
    TargetModel("Pythia-6.9B", "pkarypis/pythia-ultrachat", "Pkarypis Ultrachat"),
    TargetModel("Pythia-6.9B", "usvsnsp/pythia-6.9b-ppo", "Usvsnsp Ppo"),
    TargetModel("Pythia-12B", "lomahony/eleuther-pythia12b-hh-dpo", "Lomahony Hh Dpo"),
    TargetModel("Pythia-12B", "lomahony/eleuther-pythia12b-hh-sft", "Lomahony Hh Sft"),
)


MIMIR_DOMAINS = ("arxiv", "dm_mathematics", "github", "pile_cc", "wikipedia_en")

MIMIR_DOMAIN_HF_NAMES = {
    "arxiv": "arxiv",
    "dm_mathematics": "dm_mathematics",
    "github": "github",
    "pile_cc": "pile_cc",
    "wikipedia_en": "wikipedia_(en)",
}


VANILLA_MIN_K_20 = "min_k_20_logprob"
AUG_MIN_K_20 = "aug_min_k_20_logprob_mean"
WIKIMIA_SCORE_KEYS = (
    "mean_logprob",
    "min_k_5_logprob",
    "min_k_10_logprob",
    "min_k_20_logprob",
    "min_k_40_logprob",
)


TABLE_DESCRIPTIONS = {
    "table_1": "Paper Table 1: ViT ImageNet shard provenance.",
    "table_2": "Paper Table 2: fine-tuned Pythia targets on MIMIR GitHub.",
    "table_3": "Paper Table 3: Pythia parent shard-signal sanity check on MIMIR GitHub.",
    "table_4": "Paper Table 4: WikiMIA score-selection sanity check.",
    "table_5": "Paper Table 5: MIMIR multi-domain shard-signal sanity check.",
    "aug_table_2": "Augmented Table 2 variant using text/code augmentations.",
    "aug_table_3": "Augmented parent-signal variant on MIMIR GitHub.",
    "aug_table_5": "Augmented Table 5 multi-domain variant.",
}
